"""Race-shape / pace features.

`early_pace_score` for one dog is computable from the dog's own history
(its mean sectional-1 rank). Race-level shape (number of early-pace dogs,
pace-conflict score) is a function of the field, computed after each
runner's own pace score is set.
"""

from __future__ import annotations

import re
from datetime import datetime

import numpy as np
import polars as pl

# Common running-line comment patterns. Order matters — first match wins.
_STYLE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("early",   re.compile(r"\b(EP|Ep|early|Led [12]|Led)\b", re.IGNORECASE)),
    ("mid",     re.compile(r"\b(MP|mid|midd?ie?s?)\b", re.IGNORECASE)),
    ("rails",   re.compile(r"\bRail(s|ed)?\b", re.IGNORECASE)),
    ("wide",    re.compile(r"\bW(ide)?\b", re.IGNORECASE)),
    ("late",    re.compile(r"\b(LP|FF|late|finished fast)\b", re.IGNORECASE)),
]


def infer_running_style(comments: list[str | None]) -> str:
    """Vote across recent comments to pick a dominant running style.

    Falls back to 'unknown' when no patterns match. Crude but a useful
    seed feature; refine empirically once sectional data is in.
    """
    if not comments:
        return "unknown"
    votes: dict[str, int] = {}
    for c in comments:
        if not c:
            continue
        for label, pat in _STYLE_PATTERNS:
            if pat.search(c):
                votes[label] = votes.get(label, 0) + 1
                break
    if not votes:
        return "unknown"
    return max(votes.items(), key=lambda kv: kv[1])[0]


def early_pace_score(
    dog_id: str,
    as_of: datetime,
    runs_df: pl.DataFrame,
    *,
    window: int = 6,
) -> float | None:
    """Mean sectional-1 rank in the dog's last `window` races (strict < as_of).

    Sectional-1 rank is the dog's split-time rank within its race; 1 = first
    to first split. We compute the rank inside the function so callers
    don't have to pre-rank.
    """
    history = (
        runs_df
        .filter((pl.col("dog_id") == dog_id) & (pl.col("race_datetime") < as_of))
        .filter(pl.col("sectional_1").is_not_null())
        .sort("race_datetime")
        .tail(window)
    )
    if history.height == 0:
        return None

    # For each race in history, rank that race's sectional_1 and pull this
    # dog's row. We need the full race context — assume runs_df contains
    # all runners (it must, to compute rank correctly).
    race_ids = history["race_id"].to_list()
    field = (
        runs_df
        .filter(pl.col("race_id").is_in(race_ids))
        .filter(pl.col("sectional_1").is_not_null())
        .with_columns(
            pl.col("sectional_1")
            .rank(method="min")
            .over("race_id")
            .alias("_sec1_rank")
        )
        .filter(pl.col("dog_id") == dog_id)
    )
    if field.height == 0:
        return None
    return float(field["_sec1_rank"].mean())


def race_shape_features(
    race_runners: pl.DataFrame,
    *,
    early_pace_threshold: float = 2.5,
) -> pl.DataFrame:
    """Add race-level pace context to a frame of runners for one race.

    Requires an `early_pace_score` column already present. Produces:
      - n_other_early_pace_dogs (per runner: count of OTHER runners with
        early_pace_score < threshold)
      - pace_conflict_score (heuristic: log-sum of inverse trap distances
        between early-pace dogs)
    """
    if "early_pace_score" not in race_runners.columns:
        raise ValueError("race_shape_features expects an early_pace_score column")

    eps = race_runners["early_pace_score"].to_list()
    traps = race_runners["trap"].to_list()
    early_mask = [
        (e is not None) and (e < early_pace_threshold) for e in eps
    ]
    n_early_total = sum(early_mask)

    n_other_early: list[int] = []
    for is_early in early_mask:
        n_other_early.append(n_early_total - (1 if is_early else 0))

    # Pace conflict: inverse trap-distance between early-pace dogs, summed
    # then attributed equally to each runner. Higher = more crowding.
    conflict = 0.0
    early_traps = [t for t, m in zip(traps, early_mask) if m and t is not None]
    for i, a in enumerate(early_traps):
        for b in early_traps[i + 1:]:
            d = abs(a - b)
            if d > 0:
                conflict += 1.0 / d
    pace_conflict = [float(conflict)] * race_runners.height

    return race_runners.with_columns(
        pl.Series("n_other_early_pace_dogs", n_other_early, dtype=pl.Int8),
        pl.Series("pace_conflict_score", pace_conflict, dtype=pl.Float64),
    )


def running_style_for_dog(
    dog_id: str,
    as_of: datetime,
    runs_df: pl.DataFrame,
    *,
    window: int = 6,
) -> str:
    """Infer running style from this dog's recent comments (strict < as_of)."""
    comments = (
        runs_df
        .filter((pl.col("dog_id") == dog_id) & (pl.col("race_datetime") < as_of))
        .sort("race_datetime")
        .tail(window)["comment"]
        .to_list()
    )
    return infer_running_style(comments)


# Silence unused-import for numpy on minimal installs; we keep it imported
# because pipeline.py expects this module to expose it transitively.
_ = np
