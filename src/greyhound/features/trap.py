"""Trap / track-distance features.

BRIEF § A2.2: `track_dist_trap_winrate` must be computed from the training
set only, never from the test set. The contract here:

  - `fit_trap_winrate(train_runs)` returns a table.
  - `apply_trap_winrate(features, table)` joins it on (track, distance, trap).

The split-aware wrapper lives in `features/pipeline.py`; this module just
exposes the two pure functions.
"""

from __future__ import annotations

import polars as pl


def fit_trap_winrate(
    train_runs: pl.DataFrame,
    *,
    min_sample: int = 50,
    prior_winrate: float = 1.0 / 6.0,
    prior_strength: int = 20,
) -> pl.DataFrame:
    """Bayesian-shrunk win rate per (track, distance, trap) on a training slice.

    Sparse cells fall back to the global prior; cells with `< min_sample`
    receive heavy shrinkage but are not dropped (the model can still use
    them, they'll just sit near the prior).
    """
    if train_runs.height == 0:
        return pl.DataFrame(
            schema={
                "track": pl.Utf8,
                "distance_m": pl.Int32,
                "trap": pl.Int8,
                "track_dist_trap_winrate": pl.Float64,
                "track_dist_trap_n": pl.Int64,
            }
        )

    grouped = (
        train_runs
        .with_columns(pl.col("finish_position").eq(1).cast(pl.Int8).alias("_win"))
        .group_by(["track", "distance_m", "trap"])
        .agg(
            pl.len().alias("track_dist_trap_n"),
            pl.col("_win").sum().alias("_wins"),
        )
        .with_columns(
            (
                (pl.col("_wins") + prior_strength * prior_winrate)
                / (pl.col("track_dist_trap_n") + prior_strength)
            ).alias("track_dist_trap_winrate")
        )
        .drop("_wins")
    )

    return grouped


def apply_trap_winrate(features: pl.DataFrame, table: pl.DataFrame) -> pl.DataFrame:
    """Left-join the trap win-rate table onto a features frame.

    Missing cells get null and the model can learn that "unknown" means
    "treat as prior". An alternative is to fill with the prior here, but
    null-aware tree models handle it natively.
    """
    return features.join(
        table.select(["track", "distance_m", "trap", "track_dist_trap_winrate"]),
        on=["track", "distance_m", "trap"],
        how="left",
    )
