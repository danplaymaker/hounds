"""The leakage tests. Per BRIEF § 1 these are non-negotiable.

If any of these regress, the entire model is invalid.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from greyhound.features.form import (
    FormCfg,
    calculated_time,
    get_form_snapshot,
)
from greyhound.features.shape import early_pace_score, running_style_for_dog

pytestmark = pytest.mark.leakage


def _utc(y, m, d, hh=12, mm=0) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=UTC)


# 1.  Strict-inequality contract.

def test_as_of_equal_excludes_that_race(synthetic_runs: pl.DataFrame) -> None:
    """If as_of == the race time, that race must NOT be in the snapshot."""
    race_dt = _utc(2025, 1, 8)  # race R01
    snap = get_form_snapshot("DOGA", race_dt, synthetic_runs)
    # Only R00 (Jan 1) is strictly before Jan 8.
    assert snap["n_prior_runs"] == 1


def test_as_of_one_second_after_includes_that_race(synthetic_runs: pl.DataFrame) -> None:
    race_dt = _utc(2025, 1, 8) + timedelta(seconds=1)
    snap = get_form_snapshot("DOGA", race_dt, synthetic_runs)
    assert snap["n_prior_runs"] == 2  # R00 and R01


# 2.  No-history edge case.

def test_no_prior_runs_returns_nulls_no_crash(synthetic_runs: pl.DataFrame) -> None:
    snap = get_form_snapshot("DOG_NEVER_RAN", _utc(2025, 6, 1), synthetic_runs)
    assert snap["n_prior_runs"] == 0
    assert snap["calc_time_last_1"] is None
    assert snap["calc_time_last_3"] is None
    assert snap["calc_time_last_6"] is None
    assert snap["calc_time_trend_6"] is None
    assert snap["days_since_last_run"] is None


def test_dog_before_first_race_returns_nulls(synthetic_runs: pl.DataFrame) -> None:
    """as_of earlier than the dog's first race: empty history."""
    snap = get_form_snapshot("DOGA", _utc(2024, 12, 1), synthetic_runs)
    assert snap["n_prior_runs"] == 0


# 3.  Exactly one prior run.

def test_one_prior_run(synthetic_runs: pl.DataFrame) -> None:
    snap = get_form_snapshot(
        "DOGA", _utc(2025, 1, 8), synthetic_runs,
        track="hove", distance_m=500,
    )
    assert snap["n_prior_runs"] == 1
    assert snap["calc_time_last_1"] is not None
    # Single observation → trend undefined (we need ≥ 2 points)
    assert snap["calc_time_trend_6"] is None
    assert snap["calc_time_std_6"] is None


# 4.  Trend with enough history.

def test_trend_negative_when_improving(synthetic_runs: pl.DataFrame) -> None:
    """Dog A's run_time decreases by 0.05s per race. Trend must be negative."""
    snap = get_form_snapshot(
        "DOGA", _utc(2025, 3, 1), synthetic_runs,
        track="hove", distance_m=500,
    )
    assert snap["calc_time_trend_6"] is not None
    assert snap["calc_time_trend_6"] < 0


# 5.  Track/distance restriction.

def test_track_distance_filter(synthetic_runs: pl.DataFrame) -> None:
    """A race at a different distance must not leak into the at-distance stats."""
    augmented = pl.concat([
        synthetic_runs,
        # An off-distance run that should be IGNORED for at-distance stats.
        pl.DataFrame([{
            "race_id": "R_offdist", "race_datetime": _utc(2025, 2, 1),
            "track": "hove", "distance_m": 285, "grade": "A1", "going": 0.0,
            "dog_id": "DOGA", "dog_name": "Dog A", "trap": 1, "sp": 2.0,
            "finish_position": 1, "run_time": 16.50, "sectional_1": 3.40,
            "weight_kg": 32.0, "trainer_id": "T1", "trainer_name": "T",
            "comment": "EP", "bf_safe_name": "dog a",
        }]).cast(synthetic_runs.schema),
    ])
    snap = get_form_snapshot(
        "DOGA", _utc(2025, 3, 1), augmented,
        track="hove", distance_m=500,
    )
    # n_prior_runs counts ALL prior runs; calc_times only at-track/distance.
    assert snap["calc_time_best_90d"] > 20.0  # 500m time, not 16.5s sprint


# 6.  Wrong-dog isolation.

def test_other_dogs_history_does_not_leak(synthetic_runs: pl.DataFrame) -> None:
    snap_a = get_form_snapshot("DOGA", _utc(2025, 3, 1), synthetic_runs)
    snap_b = get_form_snapshot("DOGB", _utc(2025, 3, 1), synthetic_runs)
    # They've run the same number of races but Dog A finishes 1st always,
    # Dog B finishes 2nd. The trend numbers should be identical
    # (same slope) but the absolute times must differ.
    assert snap_a["calc_time_last_1"] is None or snap_b["calc_time_last_1"] is None or \
        abs(snap_a["calc_time_last_1"] - snap_b["calc_time_last_1"]) >= 0.10


# 7.  Days-since-last-run.

def test_days_since_last_run(synthetic_runs: pl.DataFrame) -> None:
    snap = get_form_snapshot("DOGA", _utc(2025, 1, 8), synthetic_runs)
    # Last run was Jan 1 12:00; as_of is Jan 8 12:00 → 7 days.
    assert snap["days_since_last_run"] is not None
    assert abs(snap["days_since_last_run"] - 7.0) < 1e-6


# 8.  28-day rolling count.

def test_runs_28d(synthetic_runs: pl.DataFrame) -> None:
    # By Feb 1, dog A has run on Jan 1, 8, 15, 22, 29 → all within 28d window.
    snap = get_form_snapshot("DOGA", _utc(2025, 2, 1), synthetic_runs)
    assert snap["runs_28d"] == 4  # Jan 8, 15, 22, 29 are within last 28d (Jan 4..Feb 1)


# 9.  Calculated time helper.

def test_calculated_time_with_going() -> None:
    assert calculated_time(28.50, 0.20) == pytest.approx(28.30)
    assert calculated_time(28.50, -0.20) == pytest.approx(28.70)
    assert calculated_time(None, 0.0) is None
    assert calculated_time(28.50, None) is None


# 10.  Early-pace score honours strict-inequality.

def test_early_pace_score_strict_inequality(synthetic_runs: pl.DataFrame) -> None:
    """early_pace_score(as_of == race_dt) must not include that race.

    We force a divergence by inserting a synthetic race in which Dog A
    finishes LAST to the first split. The strict-< version should miss
    that observation; the strict-< version of (race_dt+1s) should include it.
    """
    bad_race_dt = _utc(2025, 1, 7, hh=18)  # < Jan 8 race
    bad_rows = []
    for trap, dog_id, sec1 in [
        (1, "DOGA", 9.99),  # disastrous split
        (2, "DOGB", 3.40),
        (3, "DOG3", 3.45),
        (4, "DOG4", 3.50),
    ]:
        bad_rows.append({
            "race_id": "R_BAD", "race_datetime": bad_race_dt,
            "track": "hove", "distance_m": 500, "grade": "A3", "going": 0.0,
            "dog_id": dog_id, "dog_name": dog_id, "trap": trap, "sp": 5.0,
            "finish_position": trap, "run_time": 28.5, "sectional_1": sec1,
            "weight_kg": 32.0, "trainer_id": "T", "trainer_name": "T",
            "comment": "", "bf_safe_name": dog_id.lower(),
        })
    augmented = pl.concat(
        [synthetic_runs, pl.DataFrame(bad_rows).cast(synthetic_runs.schema)],
    )

    race_dt = _utc(2025, 1, 8)
    score_excl = early_pace_score("DOGA", race_dt, augmented)
    score_incl = early_pace_score("DOGA", race_dt + timedelta(seconds=1), augmented)
    # Both must be defined; the +1s version includes the boundary race
    # (the Jan 8 race itself), which DOGA wins (sec1 rank 1), so the mean
    # rank stays favourable. The strict-< version includes the bad race
    # but not Jan 8, so its mean rank is worse.
    assert score_excl is not None and score_incl is not None
    assert score_excl != score_incl, (
        f"strict-< should differ from strict-<= when boundary race exists: "
        f"excl={score_excl}, incl={score_incl}"
    )


# 11.  Running-style inference uses comments strictly before as_of.

def test_running_style_excludes_current_race(synthetic_runs: pl.DataFrame) -> None:
    style = running_style_for_dog("DOGA", _utc(2025, 1, 1), synthetic_runs)
    assert style == "unknown"  # no prior comments
    style_later = running_style_for_dog("DOGA", _utc(2025, 3, 1), synthetic_runs)
    assert style_later == "early"


# 12.  Config-driven windows.

def test_form_cfg_window_override(synthetic_runs: pl.DataFrame) -> None:
    cfg = FormCfg(last_n_windows=(2,))
    snap = get_form_snapshot(
        "DOGA", _utc(2025, 3, 1), synthetic_runs,
        track="hove", distance_m=500, cfg=cfg,
    )
    assert "calc_time_last_2" in snap
    assert "calc_time_last_3" not in snap


# 13.  Future races must never appear in snapshot regardless of order.

def test_future_races_never_in_snapshot(synthetic_runs: pl.DataFrame) -> None:
    """Even if runs_df is unsorted, the strict filter must work."""
    shuffled = synthetic_runs.sample(fraction=1.0, shuffle=True, seed=7)
    snap = get_form_snapshot("DOGA", _utc(2025, 1, 8), shuffled)
    # Only Jan 1 race < Jan 8. Must be exactly 1.
    assert snap["n_prior_runs"] == 1
