"""Parity tests: vectorised pipeline must agree with the slow reference.

If these regress, the fast path has drifted from the leakage-safe primitive
and shouldn't be trusted. The slow path remains the source of truth.
"""

from __future__ import annotations

import polars as pl
import pytest

from greyhound.data.schemas import load_config
from greyhound.features.pipeline import build_features
from greyhound.features.pipeline_fast import build_features_fast


@pytest.fixture
def cfg():
    return load_config("config/default.yaml")


PARITY_COLS = [
    "n_prior_runs",
    "runs_28d",
    "calc_time_last_1",
    "calc_time_last_3",
    "calc_time_last_6",
    "calc_time_best_90d",
    "wins_at_track_dist",
    "won",
]


def _compare(slow: pl.DataFrame, fast: pl.DataFrame, tol: float = 1e-9) -> dict[str, float]:
    s = slow.sort(["race_id", "dog_id"])
    f = fast.sort(["race_id", "dog_id"])
    diffs: dict[str, float] = {}
    for c in PARITY_COLS:
        if c not in s.columns or c not in f.columns:
            continue
        sentinel = -9999.0
        diff = (
            s[c].cast(pl.Float64).fill_null(sentinel)
            - f[c].cast(pl.Float64).fill_null(sentinel)
        ).abs().max()
        diffs[c] = float(diff) if diff is not None else 0.0
    return diffs


def test_parity_on_synthetic_runs(synthetic_runs: pl.DataFrame, cfg) -> None:
    slow = build_features(synthetic_runs, cfg)
    fast = build_features_fast(synthetic_runs, cfg)
    assert slow.height == fast.height
    diffs = _compare(slow, fast)
    for col, d in diffs.items():
        assert d < 1e-9, f"{col} disagrees by {d}"


def test_fast_pipeline_no_leakage(synthetic_runs: pl.DataFrame, cfg) -> None:
    """Direct leakage test: removing all races at or after race R must not
    change R's computed features. If it does, the fast path is peeking
    at the future."""
    full = build_features_fast(synthetic_runs, cfg)
    # Re-run with only the first 4 races (i.e. simulate that R04+ haven't
    # happened yet) and check R03's features are identical.
    target_race = "R03"
    target_dt = synthetic_runs.filter(pl.col("race_id") == target_race)["race_datetime"][0]
    truncated = synthetic_runs.filter(pl.col("race_datetime") < target_dt)
    # Add only the target race itself back in (so the pipeline has rows
    # at race R03's exact time to compute features for).
    target_rows = synthetic_runs.filter(pl.col("race_id") == target_race)
    minimal = pl.concat([truncated, target_rows])
    minimal_out = build_features_fast(minimal, cfg)

    a = full.filter(pl.col("race_id") == target_race).sort("dog_id")
    b = minimal_out.filter(pl.col("race_id") == target_race).sort("dog_id")
    for col in [
        "n_prior_runs", "calc_time_last_1", "calc_time_last_3",
        "calc_time_best_90d", "wins_at_track_dist", "days_since_last_run",
    ]:
        diff = (
            a[col].cast(pl.Float64).fill_null(-9999.0)
            - b[col].cast(pl.Float64).fill_null(-9999.0)
        ).abs().max()
        assert diff < 1e-9, (
            f"Leakage: {col} for race {target_race} changed when future "
            f"races were removed (max diff {diff})"
        )


def test_fast_pipeline_is_faster(synthetic_runs: pl.DataFrame, cfg) -> None:
    """Fast path should be at least 5x faster than the slow path. This is
    a soft check — the absolute numbers don't matter, only the ordering."""
    import time
    # Warm-up so JIT / import costs don't pollute.
    build_features_fast(synthetic_runs, cfg)
    build_features(synthetic_runs, cfg)

    t0 = time.perf_counter()
    build_features(synthetic_runs, cfg)
    t_slow = time.perf_counter() - t0

    t0 = time.perf_counter()
    build_features_fast(synthetic_runs, cfg)
    t_fast = time.perf_counter() - t0

    speedup = t_slow / max(t_fast, 1e-9)
    assert speedup > 5, f"Fast pipeline only {speedup:.1f}x — expected > 5x"
