"""Form features — and the leakage-proof primitive that everything builds on.

BRIEF § A2.1: *the* foundation. `get_form_snapshot(dog_id, as_of, runs_df)`
returns features computed strictly from runs with `race_datetime < as_of`.
Any feature that touches a dog's history goes through this function. There
is no alternative path that reads `runs_df` directly for a given dog.

Going adjustment: a calculated time is `run_time - going_adjustment(going)`.
Until we have empirical going adjustments per track, we use a global linear
map (going is a published number; treat it as seconds-of-correction).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import polars as pl

# ----------------------------------------------------------------- sentinels

# Distinct sentinel for "no data". We use None (null in Polars), not 0 or -1,
# so models can learn that "missing" is its own state. Callers must handle.
_NA = None


@dataclass(frozen=True)
class FormCfg:
    last_n_windows: tuple[int, ...] = (1, 3, 6)
    best_window_days: int = 90
    trend_window: int = 6
    recent_run_window_days: int = 28
    shrinkage_prior_runs: int = 20


# ------------------------------------------------------- going adjustment

def calculated_time(run_time: float | None, going: float | None) -> float | None:
    """Return going-adjusted (a.k.a. "calculated") time.

    GBGB publishes `raceGoing` as hundredths of a second to ADD to the raw
    run time to produce a comparable adjusted time on a neutral track
    (positive going = slow track penalty). We follow that convention:

        calc_time = run_time + going_seconds

    GBGB's own `resultAdjustedTime` field equals this sum, so calculated
    times here are directly comparable to that.
    """
    if run_time is None or going is None:
        return None
    if np.isnan(run_time) or np.isnan(going):
        return None
    return float(run_time) + float(going)


# ----------------------------------------------------- the leakage primitive

def get_form_snapshot(
    dog_id: str,
    as_of: datetime,
    runs_df: pl.DataFrame,
    *,
    track: str | None = None,
    distance_m: int | None = None,
    cfg: FormCfg | None = None,
) -> dict[str, Any]:
    """Return form features for `dog_id` strictly using runs before `as_of`.

    This is the leakage-proof primitive. Every form feature anywhere in the
    codebase must come through here. The contract:

        For every row r in `runs_df` considered:
            r.race_datetime  <  as_of    (strict inequality)

    If `track` and `distance_m` are given, the at-track/at-distance variants
    are computed; otherwise those return None.

    The function tolerates a fully empty history and never raises on
    missing data — it returns nulls. Callers / models handle the nulls.
    """
    cfg = cfg or FormCfg()

    # Filter strictly: race_datetime < as_of. Equality means the row IS the
    # race we're trying to predict — must be excluded.
    history = runs_df.filter(
        (pl.col("dog_id") == dog_id) & (pl.col("race_datetime") < as_of)
    ).sort("race_datetime")

    out: dict[str, Any] = {
        "n_prior_runs": int(history.height),
        "days_since_last_run": _NA,
        "runs_28d": 0,
    }

    # No history → all features null. Returning early keeps the rest of the
    # function from needing null-guards on every line.
    if history.height == 0:
        for n in cfg.last_n_windows:
            out[f"calc_time_last_{n}"] = _NA
        out["calc_time_best_90d"] = _NA
        out["calc_time_trend_6"] = _NA
        out["calc_time_std_6"] = _NA
        out["wins_at_track_dist"] = _NA
        out["wins_at_track_dist_rate"] = _NA
        return out

    last_dt = history["race_datetime"][-1]
    out["days_since_last_run"] = (as_of - last_dt).total_seconds() / 86400.0

    # Runs in last 28 days (excluding the current race, naturally).
    cutoff_28 = as_of - timedelta(days=cfg.recent_run_window_days)
    out["runs_28d"] = int(history.filter(pl.col("race_datetime") >= cutoff_28).height)

    # Going-adjusted (calculated) times restricted to track/distance if given.
    same_td = history
    if track is not None:
        same_td = same_td.filter(pl.col("track") == track)
    if distance_m is not None:
        same_td = same_td.filter(pl.col("distance_m") == distance_m)

    if same_td.height > 0:
        calc_times = [
            calculated_time(rt, g)
            for rt, g in zip(same_td["run_time"].to_list(), same_td["going"].to_list())
        ]
        calc_times = [t for t in calc_times if t is not None]
    else:
        calc_times = []

    for n in cfg.last_n_windows:
        if len(calc_times) >= 1:
            tail = calc_times[-n:]
            out[f"calc_time_last_{n}"] = float(np.mean(tail))
        else:
            out[f"calc_time_last_{n}"] = _NA

    # Best in last 90 days (at track/distance if specified).
    cutoff_90 = as_of - timedelta(days=cfg.best_window_days)
    recent_td = same_td.filter(pl.col("race_datetime") >= cutoff_90)
    if recent_td.height > 0:
        recent_calc = [
            calculated_time(rt, g)
            for rt, g in zip(recent_td["run_time"].to_list(), recent_td["going"].to_list())
        ]
        recent_calc = [t for t in recent_calc if t is not None]
        out["calc_time_best_90d"] = float(min(recent_calc)) if recent_calc else _NA
    else:
        out["calc_time_best_90d"] = _NA

    # Trend (OLS slope of last N calc times) — negative = improving (faster).
    if len(calc_times) >= 2:
        tail = calc_times[-cfg.trend_window:]
        x = np.arange(len(tail), dtype=float)
        y = np.asarray(tail, dtype=float)
        slope = float(np.polyfit(x, y, 1)[0])
        out["calc_time_trend_6"] = slope
        out["calc_time_std_6"] = float(np.std(tail, ddof=0)) if len(tail) > 1 else _NA
    else:
        out["calc_time_trend_6"] = _NA
        out["calc_time_std_6"] = _NA

    # Wins at track/distance (Bayesian-shrunk rate). Shrinkage target is the
    # population mean for that field size; here we use 1/6 as a generic
    # baseline. The pipeline can override this when it knows the field size.
    if track is not None and distance_m is not None:
        n = int(same_td.height)
        w = int(same_td.filter(pl.col("finish_position") == 1).height)
        prior = cfg.shrinkage_prior_runs
        out["wins_at_track_dist"] = w
        out["wins_at_track_dist_rate"] = (w + prior * (1.0 / 6.0)) / (n + prior)
    else:
        out["wins_at_track_dist"] = _NA
        out["wins_at_track_dist_rate"] = _NA

    return out
