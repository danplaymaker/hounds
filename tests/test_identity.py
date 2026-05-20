from __future__ import annotations

import pytest

from greyhound.data.identity import bf_safe_name, canonical_track, resolve_dog_id


def test_bf_safe_name_strips_apostrophe() -> None:
    assert bf_safe_name("Droopy's Joy") == "droopys joy"
    assert bf_safe_name("DROOPY’S JOY") == "droopys joy"  # curly apostrophe


def test_bf_safe_name_strips_dots() -> None:
    assert bf_safe_name("J.J. Special") == "jj special"


def test_bf_safe_name_handles_unicode() -> None:
    assert bf_safe_name("Café Au Lait") == "cafe au lait"


def test_bf_safe_name_handles_empty() -> None:
    assert bf_safe_name("") == ""


def test_canonical_track_resolves_alias(tmp_path) -> None:
    # Use the shipped tracks.yaml
    assert canonical_track("Crayford") == "crayford"
    assert canonical_track("Crayford & Bexleyheath") == "crayford"
    assert canonical_track("Brighton & Hove") == "hove"


def test_canonical_track_unknown_returns_none() -> None:
    assert canonical_track("Not A Real Track") is None


def test_resolve_dog_id_requires_id() -> None:
    assert resolve_dog_id(dog_id="GBGB-12345") == "GBGB-12345"
    with pytest.raises(ValueError):
        resolve_dog_id(dog_id=None, fallback_name="Droopy")
    with pytest.raises(ValueError):
        resolve_dog_id(dog_id="   ")
