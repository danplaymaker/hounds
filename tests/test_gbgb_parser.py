"""Parser tests using a real GBGB API response (Crayford, 1 June 2024,
meeting 411583) captured into tests/fixtures/. If GBGB changes the API
shape, these tests fail loud — which is the point.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from greyhound.ingest.gbgb_parser import (
    _combine_uk_dt,
    _decimal_sp,
    _going_to_seconds,
    _slug_trainer,
    parse_meeting_json,
)

FIXTURE = Path(__file__).parent / "fixtures" / "meeting_411583.json"


@pytest.fixture
def crayford_meeting() -> list[dict]:
    return json.loads(FIXTURE.read_text())


def test_parses_real_meeting(crayford_meeting: list[dict]) -> None:
    rows = parse_meeting_json(crayford_meeting)
    # 12 races times 6 traps typical = ~72 rows (some races may be 5-runner)
    assert len(rows) >= 60
    assert len(rows) <= 100


def test_track_resolved(crayford_meeting: list[dict]) -> None:
    rows = parse_meeting_json(crayford_meeting)
    assert all(r["track"] == "crayford" for r in rows)


def test_dog_id_is_string_and_present(crayford_meeting: list[dict]) -> None:
    rows = parse_meeting_json(crayford_meeting)
    for r in rows:
        assert isinstance(r["dog_id"], str)
        assert r["dog_id"]  # non-empty per BRIEF §6


def test_race_datetime_is_utc(crayford_meeting: list[dict]) -> None:
    rows = parse_meeting_json(crayford_meeting)
    for r in rows:
        # tzinfo could be datetime.timezone.utc or zoneinfo.ZoneInfo("UTC")
        # — compare via utcoffset() to be representation-agnostic.
        assert r["race_datetime"].utcoffset().total_seconds() == 0


def test_first_race_known_values(crayford_meeting: list[dict]) -> None:
    """Cross-check the first race against the inspected fixture."""
    rows = parse_meeting_json(crayford_meeting)
    # The first race in 411583 is 1042506 at 14:23 local → 13:23 UTC (BST).
    race_rows = [r for r in rows if r["race_id"] == "1042506"]
    assert len(race_rows) == 6
    expected_dt = datetime(2024, 6, 1, 13, 23, 0, tzinfo=UTC)
    assert race_rows[0]["race_datetime"] == expected_dt
    assert race_rows[0]["distance_m"] == 380
    # raceGoing was "10" → 0.10s
    assert race_rows[0]["going"] == pytest.approx(0.10)
    # Winner has resultPosition == 1 — trap 6 from the inspected fixture
    winner = next(r for r in race_rows if r["finish_position"] == 1)
    assert winner["dog_id"] == "622878"
    assert winner["trap"] == 6
    assert winner["sp"] == pytest.approx(13 / 8 + 1.0)  # "13/8" = 2.625
    assert winner["run_time"] == pytest.approx(23.58)
    assert winner["sectional_1"] == pytest.approx(3.63)
    assert winner["weight_kg"] == pytest.approx(23.4)


def test_trainer_slug() -> None:
    assert _slug_trainer("J M Liles") == "j_m_liles"
    assert _slug_trainer("O'Connor-Smith") == "o_connor_smith"
    assert _slug_trainer("") == ""


def test_going_parsing() -> None:
    assert _going_to_seconds("10") == pytest.approx(0.10)
    assert _going_to_seconds("0") == 0.0
    assert _going_to_seconds("-5") == pytest.approx(-0.05)
    assert _going_to_seconds("+8") == pytest.approx(0.08)
    assert _going_to_seconds("") is None
    assert _going_to_seconds(None) is None
    assert _going_to_seconds("???") is None


def test_decimal_sp_from_numerator_denominator() -> None:
    assert _decimal_sp({"resultPriceNumerator": 13, "resultPriceDenominator": 8}) == pytest.approx(2.625)
    assert _decimal_sp({"resultPriceNumerator": 1, "resultPriceDenominator": 1}) == 2.0


def test_decimal_sp_from_string() -> None:
    assert _decimal_sp({"SP": "5/2"}) == pytest.approx(3.5)
    assert _decimal_sp({"SP": "Evs"}) == 2.0
    assert _decimal_sp({"SP": "9/4F"}) == pytest.approx(3.25)  # 'F' = favourite tag
    assert _decimal_sp({"SP": ""}) is None
    assert _decimal_sp({}) is None


def test_combine_uk_dt_bst() -> None:
    # 1 June is BST (UTC+1)
    dt = _combine_uk_dt("01/06/2024", "14:23:00")
    assert dt == datetime(2024, 6, 1, 13, 23, 0, tzinfo=UTC)


def test_combine_uk_dt_gmt() -> None:
    # 1 February is GMT (UTC+0)
    dt = _combine_uk_dt("01/02/2024", "14:23:00")
    assert dt == datetime(2024, 2, 1, 14, 23, 0, tzinfo=UTC)


def test_combine_uk_dt_handles_missing() -> None:
    assert _combine_uk_dt("", "14:23:00") is None
    assert _combine_uk_dt("01/06/2024", "garbage") is None


def test_unknown_track_skipped() -> None:
    """If the meeting's track isn't in our canonical list, drop it."""
    payload = [{
        "trackName": "Some Imaginary Track",
        "meetingId": 999, "meetingDate": "01/06/2024",
        "races": [{"raceId": 1, "raceDate": "01/06/2024", "raceTime": "12:00:00",
                   "raceDistance": 400, "raceGoing": "0",
                   "traps": [{"trapNumber": "1", "dogId": 123, "dogName": "X",
                              "trainerName": "T", "SP": "2/1",
                              "resultPriceNumerator": 2, "resultPriceDenominator": 1,
                              "resultPosition": 1, "resultRunTime": "23.5",
                              "resultSectionalTime": "3.5", "resultDogWeight": "30.0",
                              "resultComment": "EP"}]}],
    }]
    assert parse_meeting_json(payload) == []


def test_missing_dog_id_row_dropped() -> None:
    """Per BRIEF §6: no GBGB ID, no row."""
    payload = [{
        "trackName": "Crayford", "meetingId": 1, "meetingDate": "01/06/2024",
        "races": [{"raceId": 1, "raceDate": "01/06/2024", "raceTime": "12:00:00",
                   "raceDistance": 380, "raceGoing": "0",
                   "traps": [
                       {"trapNumber": "1", "dogId": 100, "dogName": "Has ID",
                        "trainerName": "T", "resultPosition": 1},
                       {"trapNumber": "2", "dogId": None, "dogName": "No ID",
                        "trainerName": "T", "resultPosition": 2},
                   ]}],
    }]
    rows = parse_meeting_json(payload)
    assert len(rows) == 1
    assert rows[0]["dog_id"] == "100"
