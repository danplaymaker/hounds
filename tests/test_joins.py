"""Tests for the GBGB ↔ BSP join.

Synthetic frames keep the test isolated from real-data quirks. The
key contract: non-runners (null run_time) must not be counted against
the match-rate gate.
"""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from greyhound.data.joins import join_runs_to_bsp, report_match_rates


def _dt(hh: int, mm: int) -> datetime:
    return datetime(2024, 5, 31, hh, mm, tzinfo=UTC)


def _runs_frame() -> pl.DataFrame:
    """Two races, six runners each. Last runner of race 2 didn't run."""
    rows = []
    for race_id, race_dt in [("RA", _dt(18, 0)), ("RB", _dt(19, 0))]:
        for trap in range(1, 7):
            rows.append({
                "race_id": race_id, "race_datetime": race_dt,
                "track": "hove", "distance_m": 500, "grade": "A3",
                "going": 0.0, "dog_id": f"D{race_id}{trap}",
                "dog_name": f"Dog {race_id}{trap}", "trap": trap, "sp": 5.0,
                "finish_position": trap if not (race_id == "RB" and trap == 6) else None,
                "run_time": 28.5 if not (race_id == "RB" and trap == 6) else None,
                "sectional_1": 3.5, "weight_kg": 30.0,
                "trainer_id": "t", "trainer_name": "T",
                "comment": "EP", "bf_safe_name": f"dog {race_id}{trap}",
            })
    return pl.DataFrame(rows)


def _bsp_frame(*, drop_nonrunner: bool = True) -> pl.DataFrame:
    """BSP rows for every runner — except the non-runner trap 6 in race RB,
    which Betfair correctly never opens a market on."""
    rows = []
    for race_id, race_dt in [("MA", _dt(18, 0)), ("MB", _dt(19, 0))]:
        traps = range(1, 7) if race_id == "MA" else range(1, 6) if drop_nonrunner else range(1, 7)
        for trap in traps:
            rows.append({
                "market_id": f"M_{race_id}_{trap}",
                "event_date": _dt(0, 0).date(),
                "track": "hove",
                "race_time": race_dt,
                "selection_id": 1000 + trap,
                "selection_name": f"Dog {race_id}{trap}",
                "bf_safe_name": f"dog {race_id}{trap}",
                "trap": trap,
                "bsp": 5.0,
                "won": trap == 1,
                "matched_volume": 1000.0,
            })
    return pl.DataFrame(rows)


def test_join_returns_one_row_per_run() -> None:
    runs = _runs_frame()
    bsp = _bsp_frame()
    out = join_runs_to_bsp(runs, bsp)
    assert out.height == runs.height  # 12 rows


def test_join_matches_within_tolerance() -> None:
    runs = _runs_frame()
    bsp = _bsp_frame()
    out = join_runs_to_bsp(runs, bsp)
    # 11 runners actually ran; all 11 should match.
    assert out["bsp"].is_not_null().sum() == 11


def test_join_skips_non_runner() -> None:
    runs = _runs_frame()
    bsp = _bsp_frame()
    out = join_runs_to_bsp(runs, bsp)
    nonrun = out.filter(pl.col("run_time").is_null())
    assert nonrun.height == 1
    assert nonrun["bsp"][0] is None


def test_match_rate_gate_passes_when_only_nonrunners_unmatched() -> None:
    runs = _runs_frame()
    bsp = _bsp_frame()
    out = join_runs_to_bsp(runs, bsp)
    # The gate only applies to actual runners. Should pass at 95%.
    assert report_match_rates(out, min_rate=0.95) is True


def test_match_rate_gate_fails_when_real_runners_missed() -> None:
    """If a runner's BSP is genuinely missing, the gate should reject."""
    runs = _runs_frame()
    bsp = _bsp_frame().filter(
        ~((pl.col("race_time") == _dt(18, 0)) & (pl.col("trap") == 1))
    )
    out = join_runs_to_bsp(runs, bsp)
    # 11 ran, only 10 matched -> rate = 10/11 ≈ 0.909, below 0.95.
    assert report_match_rates(out, min_rate=0.95) is False


def test_join_respects_tolerance() -> None:
    """Race times >2min apart on the same track+trap must not cross-match."""
    runs = _runs_frame()
    bsp = _bsp_frame().with_columns(
        # Shift every BSP row by +10 min — outside the 120s tolerance.
        (pl.col("race_time") + pl.duration(minutes=10)).alias("race_time")
    )
    out = join_runs_to_bsp(runs, bsp, time_tolerance_seconds=120)
    assert out["bsp"].is_not_null().sum() == 0
