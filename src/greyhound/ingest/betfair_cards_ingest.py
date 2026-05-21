"""Ingest Betfair-fetched meeting JSONs into the local GBGB cache.

The standalone scripts/fetch_cards_betfair.py runs on the user's UK
machine and writes meeting JSONs with Betfair selection IDs (dogId='bf:N')
and raw Betfair track names ('Sheffield 21st May'). This module:

  1. Strips the date suffix from trackName ('Sheffield 21st May' -> 'Sheffield')
  2. Resolves each Betfair runner to a GBGB dog_id by matching on
     bf_safe_name against the historical cache
  3. Writes the cleaned JSON into data/raw/gbgb/meeting/ where the
     existing parser can pick it up

Run:
    python -m greyhound.ingest.betfair_cards_ingest --in /path/to/cards_today
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import polars as pl

from greyhound.data.identity import bf_safe_name
from greyhound.data.schemas import load_config

log = logging.getLogger(__name__)


# Betfair event names look like:
#   "Sheffield 21st May"
#   "Brighton & Hove 21st May"
#   "Crayford 1st Jun"
_BF_TRACK_RE = re.compile(
    r"^(.*?)\s+\d+(?:st|nd|rd|th)?\s+[A-Za-z]+\s*$",
)


def clean_track_name(raw: str) -> str:
    """Strip Betfair's date suffix from a track name."""
    if not raw:
        return ""
    m = _BF_TRACK_RE.match(raw.strip())
    return (m.group(1).strip() if m else raw.strip())


def build_dog_lookup(runs_path: Path) -> dict[str, str]:
    """Map bf_safe_name -> most-recent dog_id from history."""
    if not runs_path.exists():
        log.warning("Runs cache missing: %s — every Betfair dog will be a debutant",
                    runs_path)
        return {}
    runs = pl.read_parquet(runs_path).select(["dog_id", "bf_safe_name", "race_datetime"])
    runs = (
        runs
        .filter(pl.col("bf_safe_name") != "")
        .sort("race_datetime", descending=True)
        .unique(subset=["bf_safe_name"], keep="first", maintain_order=True)
    )
    return dict(zip(runs["bf_safe_name"].to_list(), runs["dog_id"].to_list()))


def ingest_one(payload: list[dict], dog_lookup: dict[str, str]) -> tuple[list[dict], int, int]:
    """Returns (cleaned_payload, n_matched, n_total)."""
    if not isinstance(payload, list):
        payload = [payload]
    cleaned: list[dict] = []
    n_matched = n_total = 0
    for meeting in payload:
        if not isinstance(meeting, dict):
            continue
        m2 = dict(meeting)
        m2["trackName"] = clean_track_name(meeting.get("trackName") or "")
        new_races = []
        for race in meeting.get("races") or []:
            r2 = dict(race)
            new_traps = []
            for trap in race.get("traps") or []:
                t2 = dict(trap)
                name = (t2.get("dogName") or "")
                safe = bf_safe_name(name)
                resolved = dog_lookup.get(safe)
                if resolved:
                    t2["dogId"] = resolved
                    n_matched += 1
                # else: keep the bf:... placeholder (debutant)
                n_total += 1
                new_traps.append(t2)
            r2["traps"] = new_traps
            new_races.append(r2)
        m2["races"] = new_races
        cleaned.append(m2)
    return cleaned, n_matched, n_total


def main() -> None:
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--in", dest="in_dir", required=True,
                    help="Directory containing Betfair-fetched meeting JSONs (bf_*.json).")
    args = ap.parse_args()
    cfg = load_config(args.config)
    src = Path(args.in_dir)
    if not src.exists():
        raise SystemExit(f"Input dir not found: {src}")

    runs_path = cfg.paths.interim_dir / "gbgb_runs.parquet"
    dog_lookup = build_dog_lookup(runs_path)
    log.info("Dog name lookup: %d unique safe-names", len(dog_lookup))

    out_dir = Path(cfg.ingest.gbgb.cache_dir) / "meeting"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(src.glob("bf_*.json"))
    log.info("Ingesting %d card JSONs from %s", len(files), src)
    total_matched = total_runners = 0
    for f in files:
        try:
            payload = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            log.warning("Skip %s: %s", f, e)
            continue
        cleaned, n_m, n_t = ingest_one(payload, dog_lookup)
        total_matched += n_m
        total_runners += n_t
        # Use the same filename so re-ingests are idempotent.
        (out_dir / f.name).write_text(
            json.dumps(cleaned, indent=2, default=str), encoding="utf-8",
        )
    log.info(
        "Done. Wrote %d meeting files. Name match: %d/%d (%.1f%%) runners "
        "resolved to GBGB dog_ids.",
        len(files), total_matched, total_runners,
        100 * total_matched / max(total_runners, 1),
    )


if __name__ == "__main__":
    main()
