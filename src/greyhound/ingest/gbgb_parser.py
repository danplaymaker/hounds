"""GBGB JSON → canonical Run rows (A1.2).

Pure functions. No I/O inside `parse_meeting_json`. The CLI wraps it with
file reads to convert a cache directory of meeting JSON into a Parquet
frame conforming to RUN_SCHEMA.

GBGB shape (observed from /api/results/meeting/{id}):

  list[ Meeting ]                 # always length 1 in practice
    Meeting:
      meetingDate     "DD/MM/YYYY"
      meetingId       int
      trackName       "Crayford"
      races: list[ Race ]
        Race:
          raceId          int
          raceDate        "DD/MM/YYYY"   # duplicates meetingDate
          raceTime        "HH:MM:SS"     # local time (Europe/London)
          raceClass       "A4"
          raceDistance    float (metres)
          raceHandicap    bool
          raceGoing       str (hundredths of a second to add → going_s = float/100)
          traps: list[ Trap ]
            Trap:
              trapNumber             str  (cast to int)
              trapHandicap           int|None
              dogId                  int  (primary key per BRIEF §6)
              dogName                str
              trainerName            str  (no trainerId in payload — slugify)
              SP                     str  "13/8" or "Evs" or empty
              resultPriceNumerator   int|None
              resultPriceDenominator int|None
              resultPosition         int|None   (null = withdrew/DNF)
              resultRunTime          str|None   (seconds, e.g. "23.58")
              resultSectionalTime    str|None   (seconds, e.g. "03.63")
              resultDogWeight        str|None   (kg, e.g. "23.4")
              resultAdjustedTime     str|None   (run + going/100)
              resultComment          str|None

Race times are local (Europe/London). We convert to UTC using zoneinfo
since BSP timestamps will be UTC too.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl

from greyhound.data.identity import bf_safe_name, canonical_track
from greyhound.data.schemas import RUN_SCHEMA, load_config

log = logging.getLogger(__name__)

_UK_TZ = ZoneInfo("Europe/London")
_UTC = ZoneInfo("UTC")

# Tracks that GBGB reports but Betfair has no UK greyhound markets for.
# These get dropped silently rather than logged as "unknown" — we know
# about them and have no use for them in Phase A.
_SKIP_QUIETLY: set[str] = {"Yarmouth", "Valley", "Henlow", "Star Pelaw"}


def parse_meeting_json(
    payload: list[dict] | dict,
    *,
    tracks_yaml: str | Path = "config/tracks.yaml",
) -> list[dict[str, Any]]:
    """Convert a /api/results/meeting/{id} payload into RUN_SCHEMA rows.

    Tolerates the (theoretical) case where GBGB returns a bare dict
    instead of a list. Returns [] on malformed / empty input.
    """
    meetings = payload if isinstance(payload, list) else [payload]
    rows: list[dict[str, Any]] = []
    for meeting in meetings:
        if not isinstance(meeting, dict):
            continue
        track_raw = meeting.get("trackName") or ""
        track = canonical_track(track_raw, tracks_yaml=tracks_yaml)
        if track is None:
            if track_raw not in _SKIP_QUIETLY:
                log.warning("Unknown track %r — skipping meeting %s",
                            track_raw, meeting.get("meetingId"))
            continue
        meeting_date_str = meeting.get("meetingDate") or ""

        for race in meeting.get("races", []) or []:
            race_id = race.get("raceId")
            if race_id is None:
                continue
            race_dt = _combine_uk_dt(
                race.get("raceDate") or meeting_date_str,
                race.get("raceTime") or "",
            )
            if race_dt is None:
                log.warning("Bad datetime on race %s — skipping", race_id)
                continue

            distance_m = int(race.get("raceDistance") or 0)
            grade = race.get("raceClass")
            going_s = _going_to_seconds(race.get("raceGoing"))

            for trap in race.get("traps", []) or []:
                dog_id_raw = trap.get("dogId")
                if dog_id_raw is None:
                    continue  # No dog ID, no row — BRIEF §6.
                dog_id = str(dog_id_raw)
                dog_name = trap.get("dogName") or ""
                trap_num = _to_int(trap.get("trapNumber"))
                sp = _decimal_sp(trap)
                finish = trap.get("resultPosition")
                run_time = _to_float(trap.get("resultRunTime"))
                sec1 = _to_float(trap.get("resultSectionalTime"))
                weight = _to_float(trap.get("resultDogWeight"))
                trainer_name = (trap.get("trainerName") or "").strip()
                trainer_id = _slug_trainer(trainer_name)
                comment = trap.get("resultComment") or ""

                rows.append({
                    "race_id": str(race_id),
                    "race_datetime": race_dt,
                    "track": track,
                    "distance_m": distance_m,
                    "grade": grade,
                    "going": going_s,
                    "dog_id": dog_id,
                    "dog_name": dog_name,
                    "trap": trap_num,
                    "sp": sp,
                    "finish_position": finish,
                    "run_time": run_time,
                    "sectional_1": sec1,
                    "weight_kg": weight,
                    "trainer_id": trainer_id,
                    "trainer_name": trainer_name,
                    "comment": comment,
                    "bf_safe_name": bf_safe_name(dog_name),
                })

    return rows


# -------------------------------------------------------------- helpers

_TRAINER_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug_trainer(name: str) -> str:
    """Slug a trainer name. GBGB has no trainer ID; this is the best we can do.
    Two trainers with identical normalised names would collide, but in
    practice the licensed-trainer namespace is small and conflict-free.
    """
    if not name:
        return ""
    return _TRAINER_SLUG_RE.sub("_", name.lower()).strip("_")


def _to_int(v: Any) -> int | None:
    if v is None:
        return None
    try:
        s = str(v).strip()
        if not s:
            return None
        return int(re.sub(r"[^0-9-]", "", s)) if re.search(r"[^0-9-]", s) else int(s)
    except (ValueError, TypeError):
        return None


def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _going_to_seconds(raw: Any) -> float | None:
    """GBGB publishes going in hundredths of a second to ADD to the raw
    run time to produce a comparable adjusted time. So `going_s` here is
    a small positive number on a slow track.

    Empty / non-numeric → None.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    # Strip any '+' sign GBGB might prepend.
    s = s.lstrip("+")
    try:
        return float(s) / 100.0
    except ValueError:
        return None


def _decimal_sp(trap: dict) -> float | None:
    """Prefer the numerator/denominator pair (always present when SP is set);
    fall back to parsing the SP string ('Evs', '5/2', '11/10F', etc.).
    """
    num = trap.get("resultPriceNumerator")
    den = trap.get("resultPriceDenominator")
    if num is not None and den is not None and den != 0:
        return float(num) / float(den) + 1.0

    s = (trap.get("SP") or "").strip()
    if not s:
        return None
    # "Evs" = even money = 2.0 decimal
    if s.lower().startswith("ev"):
        return 2.0
    # Strip favourite markers like 'F', 'JF', 'CF'
    s = re.sub(r"[A-Za-z]+$", "", s).strip()
    m = re.match(r"^(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)$", s)
    if m:
        n, d = float(m.group(1)), float(m.group(2))
        return n / d + 1.0 if d else None
    try:
        return float(s)
    except ValueError:
        return None


def _combine_uk_dt(date_str: str, time_str: str) -> datetime | None:
    """Combine GBGB's "DD/MM/YYYY" + "HH:MM:SS" (local UK time) into UTC."""
    if not date_str:
        return None
    try:
        d_part = datetime.strptime(date_str.strip(), "%d/%m/%Y").date()
    except ValueError:
        return None
    if not time_str:
        time_str = "00:00:00"
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            t_part = datetime.strptime(time_str.strip(), fmt).time()
            break
        except ValueError:
            t_part = None
    if t_part is None:
        return None
    local = datetime.combine(d_part, t_part).replace(tzinfo=_UK_TZ)
    return local.astimezone(_UTC)


# ------------------------------------------------------- batch ingestion

def parse_all_cached(
    cache_dir: Path,
    *,
    tracks_yaml: str | Path = "config/tracks.yaml",
) -> pl.DataFrame:
    """Walk `<cache_dir>/meeting/*.json` and concatenate parsed rows."""
    meeting_dir = cache_dir / "meeting"
    if not meeting_dir.exists():
        log.warning("No meetings cache at %s", meeting_dir)
        return pl.DataFrame(schema=RUN_SCHEMA)

    rows: list[dict[str, Any]] = []
    for path in sorted(meeting_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            log.warning("Skip %s: %s", path, e)
            continue
        rows.extend(parse_meeting_json(payload, tracks_yaml=tracks_yaml))

    if not rows:
        return pl.DataFrame(schema=RUN_SCHEMA)

    df = pl.DataFrame(rows)
    # Cast to canonical schema (subset of cols present).
    keep = [c for c in RUN_SCHEMA if c in df.columns]
    return df.select(keep).cast({c: RUN_SCHEMA[c] for c in keep})


def main() -> None:
    import argparse
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    df = parse_all_cached(Path(cfg.ingest.gbgb.cache_dir))
    out = cfg.paths.interim_dir / "gbgb_runs.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)
    log.info("Parsed %d runner rows → %s", df.height, out)


if __name__ == "__main__":
    main()
