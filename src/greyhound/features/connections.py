"""Trainer / connections features.

Same leakage discipline as form: every aggregate is over runs with
race_datetime < as_of. Bayesian shrinkage to the population win rate so a
trainer with 3-from-4 doesn't dominate a trainer with 50-from-300.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import polars as pl


def trainer_strike_rate(
    trainer_id: str,
    as_of: datetime,
    runs_df: pl.DataFrame,
    *,
    window_days: int = 30,
    track: str | None = None,
    prior_runs: int = 50,
    prior_winrate: float = 1.0 / 6.0,
) -> dict[str, Any]:
    """Return shrunk win rate for a trainer over a recent window.

    If `track` is given, restricts to runs at that track (used for the
    180-day at-track variant).
    """
    cutoff = as_of - timedelta(days=window_days)
    f = (
        runs_df
        .filter(pl.col("trainer_id") == trainer_id)
        .filter(pl.col("race_datetime") < as_of)
        .filter(pl.col("race_datetime") >= cutoff)
    )
    if track is not None:
        f = f.filter(pl.col("track") == track)

    n = int(f.height)
    w = int(f.filter(pl.col("finish_position") == 1).height)
    rate = (w + prior_runs * prior_winrate) / (n + prior_runs)
    return {"n": n, "wins": w, "rate": float(rate)}
