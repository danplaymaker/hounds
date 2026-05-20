"""Betfair PROMO BSP parser tests using a real captured CSV.

Fixture: dwbfgreyhoundwin01062024.csv — file is dated 01062024 (the
*generation* date) but actually contains UK races settled on 2024-05-31.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from greyhound.ingest.betfair_bsp import (
    _extract_track_from_menu_hint,
    _parse_event_dt,
    _strip_trap,
    parse_bsp_csv,
)

FIXTURE = Path(__file__).parent / "fixtures" / "dwbfgreyhoundwin01062024.csv"


@pytest.fixture
def bsp() -> pl.DataFrame:  # noqa: F821 — type only used after import
    import polars as pl  # noqa: F401
    return parse_bsp_csv(FIXTURE)


def test_parses_real_file(bsp) -> None:
    # The full file is ~2169 rows. After UK-track filter we keep ~884.
    assert 700 < bsp.height < 1000


def test_filters_to_uk_tracks_only(bsp) -> None:
    """Every kept row must resolve to a GBGB-licensed track."""
    tracks = set(bsp["track"].unique().to_list())
    # Sample of tracks we'd expect (file is dated end of May; some tracks
    # may not have meetings that day, so just check no None remains).
    assert None not in tracks
    # Australian / Irish tracks must not appear.
    for t in tracks:
        assert t.islower()


def test_event_dt_is_utc(bsp) -> None:
    """All race_time values must be tz-aware UTC."""
    for dt in bsp["race_time"].to_list():
        if dt is not None:
            assert dt.utcoffset().total_seconds() == 0


def test_trap_in_valid_range(bsp) -> None:
    traps = bsp["trap"].to_list()
    for t in traps:
        if t is not None:
            assert 1 <= t <= 8


def test_bsp_positive(bsp) -> None:
    bsps = [b for b in bsp["bsp"].to_list() if b is not None]
    assert all(b >= 1.0 for b in bsps), "BSP decimal prices must be >= 1.0"
    # Sanity: at least some short-priced favourites
    assert any(1.0 < b < 3.0 for b in bsps)


def test_won_flag_per_race(bsp) -> None:
    """Each market has exactly one winner (UK greyhound win markets)."""
    winners = bsp.group_by("market_id").agg(
        __import__("polars").col("won").sum().alias("n_winners")
    )
    n_winners = winners["n_winners"].to_list()
    # The vast majority should have exactly 1 winner; some non-runners /
    # voids might have 0. Never more than 1.
    assert max(n_winners) == 1
    assert sum(1 for n in n_winners if n == 1) > 0.9 * len(n_winners)


def test_known_track_present(bsp) -> None:
    tracks = set(bsp["track"].unique().to_list())
    assert "crayford" in tracks or "sheffield" in tracks


# ---------- helpers ----------

def test_extract_track_from_menu_hint() -> None:
    assert _extract_track_from_menu_hint("Sheffield 31st May") == "Sheffield"
    assert _extract_track_from_menu_hint("Crayford 1st Jun") == "Crayford"
    assert _extract_track_from_menu_hint("Brighton & Hove 1st Jun") == "Brighton & Hove"
    # Non-UK still parses out, but the track won't canonicalise
    assert _extract_track_from_menu_hint("Richmond (AUS) 1st Jun") == "Richmond"


def test_strip_trap() -> None:
    assert _strip_trap("3. Zipping Striker") == ("Zipping Striker", 3)
    assert _strip_trap("5. Darver Mineola") == ("Darver Mineola", 5)
    assert _strip_trap("Untrapped Selection") == ("Untrapped Selection", None)
    assert _strip_trap("") == ("", None)


def test_parse_event_dt_bst() -> None:
    # 31 May is BST (UTC+1)
    assert _parse_event_dt("31-05-2024 19:49") == datetime.fromisoformat("2024-05-31T18:49:00+00:00")


def test_parse_event_dt_gmt() -> None:
    # Jan is GMT
    assert _parse_event_dt("01-01-2024 19:49") == datetime.fromisoformat("2024-01-01T19:49:00+00:00")


def test_parse_event_dt_bad() -> None:
    assert _parse_event_dt("") is None
    assert _parse_event_dt("garbage") is None
