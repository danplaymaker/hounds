"""Identity: name normalisation and dog/track resolution.

Per BRIEF § 6: dogs are identified by GBGB ID. `bf_safe_name` exists only
to *verify* a join, never to be the primary key. Track names get a canonical
form so GBGB ("Crayford & Bexleyheath") and Betfair ("Crayford") resolve to
the same key.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import yaml

_APOSTROPHE_RE = re.compile(r"['’`]")
_DOT_RE = re.compile(r"\.")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_MULTISPACE_RE = re.compile(r"\s+")


def bf_safe_name(raw: str) -> str:
    """Normalise a dog name for cross-source matching.

    Lowercases, strips apostrophes and dots, collapses non-alphanumerics
    to single spaces. Does NOT remove diacritics aggressively (Betfair and
    GBGB both publish ASCII), but normalises Unicode first to keep parity
    if a non-ASCII char ever slips through.
    """
    if raw is None:
        return ""
    s = unicodedata.normalize("NFKD", raw)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = _APOSTROPHE_RE.sub("", s)
    s = _DOT_RE.sub("", s)
    s = _NON_ALNUM_RE.sub(" ", s)
    s = _MULTISPACE_RE.sub(" ", s).strip()
    return s


@lru_cache(maxsize=1)
def _load_track_table(tracks_yaml: str) -> dict[str, str]:
    """Read tracks.yaml once, return alias -> canonical map."""
    with open(tracks_yaml) as f:
        data = yaml.safe_load(f)
    alias_to_canonical: dict[str, str] = {}
    for entry in data.get("tracks", []):
        canonical = entry["canonical"]
        for alias in entry.get("aliases", []):
            alias_to_canonical[_track_key(alias)] = canonical
        # The canonical form maps to itself.
        alias_to_canonical[_track_key(canonical)] = canonical
        alias_to_canonical[_track_key(entry["display"])] = canonical
    return alias_to_canonical


def _track_key(raw: str) -> str:
    """Aggressive normalisation used only for the track alias lookup."""
    s = unicodedata.normalize("NFKD", raw or "").encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def canonical_track(raw: str, tracks_yaml: str | Path = "config/tracks.yaml") -> str | None:
    """Resolve a track string from any source to its canonical key.

    Returns None if the track is unknown — callers must decide whether to
    drop the row or fail loudly.
    """
    table = _load_track_table(str(tracks_yaml))
    return table.get(_track_key(raw))


def resolve_dog_id(
    *,
    dog_id: str | None,
    fallback_name: str | None = None,
) -> str:
    """Return the dog's canonical ID.

    Per the brief, names are never primary keys. If `dog_id` is missing we
    raise — silently falling back to a name would defeat the constraint.
    `fallback_name` is accepted only so callers can give a useful error.
    """
    if dog_id is None or not dog_id.strip():
        raise ValueError(
            f"Missing GBGB dog_id (fallback name: {fallback_name!r}). "
            "Names are never primary keys — see BRIEF § 6."
        )
    return dog_id.strip()
