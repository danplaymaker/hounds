"""Feature pipeline — orchestrates the per-race feature build.

The whole pipeline goes through `get_form_snapshot`, by construction. We
build features race-by-race so the as_of cutoff is enforced for every row.
The slow path (Python loop) is fine for Phase A; if it becomes the
bottleneck we can vectorise per-track.

Inputs: a single Polars frame conforming to RUN_SCHEMA covering ≥ training
window + look-back depth (60 days minimum, ideally 12+ months).

Output: one row per (race_id, dog_id) with all feature columns + `won` and
group key columns.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl
from tqdm import tqdm

from greyhound.data.schemas import Config, load_config
from greyhound.features.connections import trainer_strike_rate
from greyhound.features.form import FormCfg, get_form_snapshot
from greyhound.features.shape import (
    early_pace_score,
    race_shape_features,
    running_style_for_dog,
)
from greyhound.features.trap import apply_trap_winrate, fit_trap_winrate

log = logging.getLogger(__name__)


def build_features(
    runs: pl.DataFrame,
    cfg: Config,
    *,
    trap_winrate_table: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build a features frame from a runs frame.

    Caller is responsible for passing a `trap_winrate_table` fit on a
    train-only slice when this output will be used for evaluation. For
    one-shot training-set feature generation, pass None and we fit on the
    same frame (the typical use during model training).
    """
    fcfg = FormCfg(
        last_n_windows=tuple(cfg.features.form.last_n_windows),
        best_window_days=cfg.features.form.best_window_days,
        trend_window=cfg.features.form.trend_window,
        recent_run_window_days=cfg.features.form.recent_run_window_days,
        shrinkage_prior_runs=cfg.features.form.bayesian_shrinkage_prior_runs,
    )

    if trap_winrate_table is None:
        log.warning(
            "build_features called without an externally-fit trap_winrate_table; "
            "fitting on the same frame. Acceptable only for training-set generation."
        )
        trap_winrate_table = fit_trap_winrate(
            runs, min_sample=cfg.features.trap.trap_winrate_min_sample
        )

    race_groups = runs.group_by("race_id", maintain_order=True)
    out_rows: list[dict] = []

    for (race_id_t,), race in tqdm(race_groups, desc="races"):
        race_id = race_id_t
        race_dt = race["race_datetime"][0]
        track = race["track"][0]
        distance_m = race["distance_m"][0]

        per_runner: list[dict] = []
        for r in race.iter_rows(named=True):
            snap = get_form_snapshot(
                r["dog_id"], race_dt, runs,
                track=track, distance_m=distance_m, cfg=fcfg,
            )
            eps = early_pace_score(r["dog_id"], race_dt, runs)
            style = running_style_for_dog(r["dog_id"], race_dt, runs)
            trainer = trainer_strike_rate(
                r["trainer_id"] or "",
                race_dt,
                runs,
                window_days=cfg.features.connections.trainer_window_days,
                prior_runs=cfg.features.connections.shrinkage_prior_runs,
            )
            trainer_track = trainer_strike_rate(
                r["trainer_id"] or "",
                race_dt,
                runs,
                window_days=cfg.features.connections.trainer_track_window_days,
                track=track,
                prior_runs=cfg.features.connections.shrinkage_prior_runs,
            )
            per_runner.append({
                "race_id": race_id,
                "race_datetime": race_dt,
                "track": track,
                "distance_m": distance_m,
                "grade": r.get("grade"),
                "dog_id": r["dog_id"],
                "trap": r["trap"],
                "weight_kg": r.get("weight_kg"),
                "early_pace_score": eps,
                "running_style": style,
                "trainer_sr_30d": trainer["rate"],
                "trainer_sr_track_180d": trainer_track["rate"],
                **{k: v for k, v in snap.items()},
                "won": int((r.get("finish_position") or 0) == 1),
            })

        race_frame = pl.DataFrame(per_runner)
        race_frame = race_shape_features(
            race_frame,
            early_pace_threshold=cfg.features.shape.early_pace_threshold,
        )
        out_rows.extend(race_frame.to_dicts())

    features = pl.DataFrame(out_rows)
    features = apply_trap_winrate(features, trap_winrate_table)
    return features


def main() -> None:
    import argparse
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument(
        "--runs", default=None,
        help="Path to races_with_market.parquet (default: derived from config)"
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    runs_path = Path(args.runs) if args.runs else cfg.paths.processed_dir / "races_with_market.parquet"
    if not runs_path.exists():
        raise SystemExit(f"Runs file not found: {runs_path} — run `make ingest` first.")
    runs = pl.read_parquet(runs_path)
    features = build_features(runs, cfg)
    out = cfg.paths.processed_dir / "features.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    features.write_parquet(out)
    log.info("Wrote %d rows to %s", features.height, out)


if __name__ == "__main__":
    main()
