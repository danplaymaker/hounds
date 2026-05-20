"""Shared fixtures.

The synthetic runs frame mimics the canonical Run schema so we can test
leakage discipline without needing real GBGB data.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest


def _utc(y, m, d, hh=12, mm=0) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=UTC)


@pytest.fixture
def synthetic_runs() -> pl.DataFrame:
    """Two dogs across 6 races at one track, plus opponents to fill fields."""
    rows: list[dict] = []
    base_dt = _utc(2025, 1, 1)
    # Six weekly races. Dog A wins all; dog B finishes second; trial fodder
    # at traps 3..6.
    for i in range(6):
        race_id = f"R{i:02d}"
        race_dt = base_dt + timedelta(days=7 * i)
        for trap, dog_id, dog_name, finish, run_time, sec1 in [
            (1, "DOGA", "Dog A", 1, 28.50 - i * 0.05, 3.40),
            (2, "DOGB", "Dog B", 2, 28.70 - i * 0.05, 3.50),
            (3, "DOG3", "Dog 3", 3, 28.85, 3.55),
            (4, "DOG4", "Dog 4", 4, 28.95, 3.60),
            (5, "DOG5", "Dog 5", 5, 29.10, 3.70),
            (6, "DOG6", "Dog 6", 6, 29.30, 3.80),
        ]:
            rows.append({
                "race_id": race_id,
                "race_datetime": race_dt,
                "track": "hove",
                "distance_m": 500,
                "grade": "A3",
                "going": 0.0,
                "dog_id": dog_id,
                "dog_name": dog_name,
                "trap": trap,
                "sp": 2.0 if dog_id == "DOGA" else 5.0,
                "finish_position": finish,
                "run_time": run_time,
                "sectional_1": sec1,
                "weight_kg": 32.0,
                "trainer_id": "T1" if dog_id == "DOGA" else f"T{trap}",
                "trainer_name": "Trainer One",
                "comment": "EP, Led 1" if dog_id == "DOGA" else "RB",
                "bf_safe_name": dog_name.lower().replace(" ", " "),
            })
    return pl.DataFrame(rows)
