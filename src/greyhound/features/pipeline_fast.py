"""Vectorised feature pipeline.

The original `pipeline.build_features` calls `get_form_snapshot` once per
(race, runner), and each call filters the full runs frame. That's
O(rows * dogs) — fine for 1k rows, painful at 200k+. This module computes
the same features using Polars window expressions, which run in Rust
and don't materialise per-call subframes.

Equivalence with the row-by-row pipeline is enforced by
`tests/test_features_pipeline_fast.py` — both implementations are run on
the same synthetic frame and the resulting numeric columns must match.

The "no look-ahead" guarantee here comes from a single discipline:
*every* feature expression uses `.shift(1).over("dog_id", ...)` (or an
analogous lag) so the current race's own row is excluded from its own
feature. `tests/test_features_no_leakage.py` continues to exercise the
primitive on the row-by-row path; the fast-path tests assert per-row
parity, transitively giving the same leakage guarantee.
"""

from __future__ import annotations

import logging

import polars as pl

from greyhound.data.schemas import Config
from greyhound.features.trap import apply_trap_winrate, fit_trap_winrate

log = logging.getLogger(__name__)


def build_features_fast(
    runs: pl.DataFrame,
    cfg: Config,
    *,
    trap_winrate_table: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Vectorised equivalent of build_features.

    Output columns are a superset of the slow path; the slow path is
    retained as a reference and for the leakage tests.
    """
    fcfg = cfg.features.form
    ccfg = cfg.features.connections
    scfg = cfg.features.shape

    # Pre-compute auxiliary columns the windowed aggregations need.
    full = (
        runs
        .with_columns([
            # GBGB convention: calc_time = run_time + going_seconds.
            (pl.col("run_time") + pl.col("going")).alias("_calc_time"),
            # Sectional-1 rank within race (1 = first to first split).
            pl.col("sectional_1")
              .rank(method="min")
              .over("race_id")
              .alias("_sec1_rank"),
            # 1 = won, 0 = otherwise (null finish -> 0).
            (pl.col("finish_position") == 1).cast(pl.Int8).alias("_win"),
        ])
    )

    # GBGB occasionally lists a dog twice at the same (dog_id, race_datetime)
    # — handicap re-allocations, withdrawals, schooling twins. For HISTORY
    # purposes, only one row per (dog, time) should contribute. We dedupe
    # to one canonical row per pair (preferring rows with run_time), do the
    # rolling work on the dedup'd frame, then join the per-row features
    # back onto the full frame so every output row gets the same feature
    # values its same-time peer would have.
    df = (
        full
        .with_columns(pl.col("run_time").is_not_null().cast(pl.Int8).alias("_has_rt"))
        .sort(["dog_id", "race_datetime", "_has_rt", "race_id"],
              descending=[False, False, True, False])
        .unique(subset=["dog_id", "race_datetime"], keep="first", maintain_order=True)
        .drop("_has_rt")
        .sort(["dog_id", "race_datetime"])
    )

    # --------------------------------------------------- per-dog history

    # Past-only convention: every expression is .shift(1).over("dog_id"...)
    # so the row's own race is excluded. We then take rolling / cumulative
    # aggregates of that shifted series.
    by_dog = ["dog_id"]
    by_dog_td = ["dog_id", "track", "distance_m"]

    prev_calc = pl.col("_calc_time").shift(1).over(by_dog_td)
    prev_dt = pl.col("race_datetime").shift(1).over(by_dog)
    prev_win = pl.col("_win").shift(1).over(by_dog_td)
    prev_sec1_rank = pl.col("_sec1_rank").shift(1).over(by_dog)

    form_exprs: list[pl.Expr] = []
    for n in fcfg.last_n_windows:
        form_exprs.append(
            prev_calc.rolling_mean(window_size=n, min_samples=1)
                     .over(by_dog_td)
                     .alias(f"calc_time_last_{n}")
        )

    form_exprs.extend([
        # Best in last 90 days at track+distance.
        # Time-windowed: closed="left" -> window is [t - 90d, t), so the
        # current row is excluded automatically. NO .shift() — that would
        # double-exclude.
        pl.col("_calc_time").rolling_min_by(
            by="race_datetime",
            window_size=f"{fcfg.best_window_days}d",
            closed="left",
        ).over(by_dog_td).alias("calc_time_best_90d"),

        # Std of last N calc_times at track+dist. Count-based rolling
        # needs .shift(1) to exclude the current row from its own window.
        prev_calc.rolling_std(
            window_size=fcfg.trend_window, min_samples=2
        ).over(by_dog_td).alias("calc_time_std_6"),

        # Days since last run (any track).
        ((pl.col("race_datetime") - prev_dt).dt.total_seconds() / 86400.0)
            .alias("days_since_last_run"),

        # Runs in last 28 days (any track). Time-windowed sum of 1.0;
        # closed="left" excludes the current row. Coerce nulls (no prior
        # rows in window) to 0 to match the slow path's convention.
        pl.col("race_datetime").is_not_null().cast(pl.Float64).rolling_sum_by(
            by="race_datetime",
            window_size=f"{fcfg.recent_run_window_days}d",
            closed="left",
        ).over(by_dog).fill_null(0).cast(pl.Int64).alias("runs_28d"),

        # Cumulative count of prior races at track+dist. Position index
        # within (dog, track, dist) — counts withdrawn races as runs.
        pl.int_range(0, pl.len()).over(by_dog_td).alias("_runs_at_td"),

        # n_prior_runs (all runs, any track). Position index within dog.
        pl.int_range(0, pl.len()).over(by_dog).alias("n_prior_runs"),

        # Cumulative wins at track+dist BEFORE this race. Match the slow
        # path's convention: null when this dog has NO prior runs at all
        # (not just at this track+dist).
        pl.when(pl.int_range(0, pl.len()).over(by_dog) > 0)
          .then(prev_win.fill_null(0).cum_sum().over(by_dog_td))
          .otherwise(None)
          .alias("wins_at_track_dist"),

        # Early-pace score: rolling mean of sec1 rank in last 6 races.
        prev_sec1_rank.rolling_mean(window_size=6, min_samples=1)
                      .over(by_dog).alias("early_pace_score"),
    ])

    # Calc-time trend (OLS slope) approximation. We need it after the
    # rolling helpers run, so emit as a follow-on expression below.
    df = df.with_columns(form_exprs)

    # Trend as (last_n_window mean differences) / window — sign-correct
    # approximation of OLS slope.
    df = df.with_columns([
        (
            (
                pl.col("_calc_time").shift(1).over(by_dog_td)
                  .rolling_mean(window_size=2, min_samples=1)
                  .over(by_dog_td)
                - pl.col("_calc_time").shift(fcfg.trend_window - 1).over(by_dog_td)
                    .rolling_mean(window_size=2, min_samples=1)
                    .over(by_dog_td)
            ) / (fcfg.trend_window - 1)
        ).alias("calc_time_trend_6"),
    ])

    # Bayesian-shrunk win rate at track+dist.
    prior_p = 1.0 / 6.0
    prior_n = fcfg.bayesian_shrinkage_prior_runs
    df = df.with_columns([
        (
            (pl.col("wins_at_track_dist").fill_null(0) + prior_n * prior_p)
            / (pl.col("_runs_at_td").fill_null(0) + prior_n)
        ).alias("wins_at_track_dist_rate"),
    ])

    # ----------------------------------------------- trainer rolling

    by_trainer = ["trainer_id"]
    by_trainer_track = ["trainer_id", "track"]

    # Trainer wins in past `window_days` days. Polars time-windowed rolling.
    df = df.sort(["trainer_id", "race_datetime"]).with_columns([
        # Same convention as runs_28d: time-windowed with closed="left"
        # excludes the current row from its own window. NO shift.
        pl.col("_win").cast(pl.Float64).rolling_sum_by(
            by="race_datetime",
            window_size=f"{ccfg.trainer_window_days}d",
            closed="left",
        ).over(by_trainer).alias("_trainer_w_30d"),
        pl.col("race_datetime").is_not_null().cast(pl.Float64).rolling_sum_by(
            by="race_datetime",
            window_size=f"{ccfg.trainer_window_days}d",
            closed="left",
        ).over(by_trainer).alias("_trainer_n_30d"),
    ])

    df = df.sort(["trainer_id", "track", "race_datetime"]).with_columns([
        pl.col("_win").cast(pl.Float64).rolling_sum_by(
            by="race_datetime",
            window_size=f"{ccfg.trainer_track_window_days}d",
            closed="left",
        ).over(by_trainer_track).alias("_trainer_w_180d"),
        pl.col("race_datetime").is_not_null().cast(pl.Float64).rolling_sum_by(
            by="race_datetime",
            window_size=f"{ccfg.trainer_track_window_days}d",
            closed="left",
        ).over(by_trainer_track).alias("_trainer_n_180d"),
    ])

    sh = ccfg.shrinkage_prior_runs
    df = df.with_columns([
        (
            (pl.col("_trainer_w_30d").fill_null(0) + sh * prior_p)
            / (pl.col("_trainer_n_30d").fill_null(0) + sh)
        ).alias("trainer_sr_30d"),
        (
            (pl.col("_trainer_w_180d").fill_null(0) + sh * prior_p)
            / (pl.col("_trainer_n_180d").fill_null(0) + sh)
        ).alias("trainer_sr_track_180d"),
    ])

    # Pace conflict closure used after the join (defined here so the
    # closure can be referenced by name below).
    def _conflict_for_race(traps: list[int | None], is_early: list[int]) -> float:
        e = [t for t, m in zip(traps, is_early) if t is not None and m]
        s = 0.0
        for i, a in enumerate(e):
            for b in e[i + 1:]:
                d = abs(a - b)
                if d > 0:
                    s += 1.0 / d
        return s

    # The trap-winrate table is fit on the training slice; default to
    # self-fitting for one-shot calls (training-set generation only).
    if trap_winrate_table is None:
        log.warning(
            "build_features_fast called without an externally-fit trap_winrate_table; "
            "fitting on the same frame. Acceptable only for training-set generation.",
        )
        trap_winrate_table = fit_trap_winrate(
            runs, min_sample=cfg.features.trap.trap_winrate_min_sample,
        )

    # -------------------------------------------------- final shape

    # The per-dog history features were computed on the deduplicated frame.
    # Join them back onto the full frame keyed on (dog_id, race_datetime)
    # so duplicate-time peers share their canonical row's history features.
    history_cols = [
        "n_prior_runs", "days_since_last_run", "runs_28d",
        *[f"calc_time_last_{n}" for n in fcfg.last_n_windows],
        "calc_time_best_90d", "calc_time_trend_6", "calc_time_std_6",
        "wins_at_track_dist", "wins_at_track_dist_rate",
        "early_pace_score",
        "trainer_sr_30d", "trainer_sr_track_180d",
    ]
    history = df.select(["dog_id", "race_datetime", *history_cols])
    out = full.join(history, on=["dog_id", "race_datetime"], how="left")

    # Race-shape features (per-runner inside a race) are computed on the
    # full frame because they depend on the field, not history.
    out = out.with_columns([
        (pl.col("early_pace_score") < scfg.early_pace_threshold)
            .cast(pl.Int8).alias("_is_early"),
    ])
    out = out.with_columns([
        (pl.col("_is_early").sum().over("race_id") - pl.col("_is_early"))
            .alias("n_other_early_pace_dogs"),
    ])
    conflicts_full = (
        out.group_by("race_id", maintain_order=True)
           .agg([pl.col("trap").alias("_traps"), pl.col("_is_early").alias("_isE")])
           .with_columns(
               pl.struct(["_traps", "_isE"])
                 .map_elements(
                     lambda s: _conflict_for_race(s["_traps"], s["_isE"]),
                     return_dtype=pl.Float64,
                 )
                 .alias("pace_conflict_score")
           )
           .select(["race_id", "pace_conflict_score"])
    )
    out = out.join(conflicts_full, on="race_id", how="left")
    out = apply_trap_winrate(out, trap_winrate_table)

    keep = [
        "race_id", "race_datetime", "track", "distance_m", "grade",
        "dog_id", "trap", "weight_kg",
        *history_cols,
        "n_other_early_pace_dogs", "pace_conflict_score",
        "track_dist_trap_winrate",
        "won",
    ]
    out = out.with_columns(pl.col("_win").alias("won"))
    return out.select([c for c in keep if c in out.columns]).sort(
        ["race_datetime", "race_id", "trap"]
    )
