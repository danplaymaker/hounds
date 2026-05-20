"""Betfair PROMO BSP CSV ingestion (A1.3).

Free daily files from https://promo.betfair.com/betfairsp/prices/ — naming
convention `dwbfgreyhoundwin{DDMMYYYY}.csv`. Each file contains every
selection from every greyhound win market settled that day (UK *and*
overseas — Australian, Irish, etc.). We filter to UK GBGB tracks.

Observed columns (lowercase):
  event_id, menu_hint, event_name, event_dt, selection_id, selection_name,
  win_lose, bsp, ppwap, morningwap, ppmax, ppmin, ipmax, ipmin,
  morningtradedvol, pptradedvol, iptradedvol

event_dt is in UK local time ("DD-MM-YYYY HH:MM"); we convert to UTC.
selection_name carries the trap number as a prefix ("3. Some Dog").
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from greyhound.data.identity import bf_safe_name, canonical_track
from greyhound.data.schemas import BSP_SCHEMA, load_config

log = logging.getLogger(__name__)

_UK_TZ = ZoneInfo("Europe/London")
_UTC = ZoneInfo("UTC")

# Selection names look like "3. Some Dog" (trap. name). Capture both.
_TRAP_RE = re.compile(r"^\s*(\d)\s*[\.\-:)]\s*(.+)$")

# menu_hint for non-UK meetings carries a country code in parens, e.g.
# "Richmond (AUS) 1st Jun" / "Cork (IRE) 1st Jun". UK meetings have no
# country tag. We canonicalise the track via tracks.yaml; rows whose track
# doesn't resolve are dropped.
_MENU_TRACK_RE = re.compile(r"^\s*([A-Za-z &'\-]+?)\s+(?:\([A-Z]{2,4}\)\s+)?\d")


def _extract_track_from_menu_hint(menu_hint: str) -> str:
    """Pull the track name out of a menu_hint string.

    Examples:
      "Sheffield 31st May"              -> "Sheffield"
      "Crayford 1st Jun"                -> "Crayford"
      "Brighton & Hove 1st Jun"         -> "Brighton & Hove"
      "Richmond (AUS) 1st Jun"          -> "Richmond"  (will fail to canon)
    """
    if not menu_hint:
        return ""
    m = _MENU_TRACK_RE.match(menu_hint)
    if m:
        return m.group(1).strip()
    # Fallback: everything before the first numeric.
    parts = re.split(r"\s\d", menu_hint, maxsplit=1)
    return parts[0].strip()


def _strip_trap(selection_name: str) -> tuple[str, int | None]:
    if not selection_name:
        return "", None
    m = _TRAP_RE.match(selection_name)
    if m:
        return m.group(2).strip(), int(m.group(1))
    return selection_name.strip(), None


def _parse_event_dt(s: str) -> datetime | None:
    """UK-local "DD-MM-YYYY HH:MM" → aware UTC."""
    if not s:
        return None
    for fmt in ("%d-%m-%Y %H:%M", "%d-%m-%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            local = datetime.strptime(s.strip(), fmt).replace(tzinfo=_UK_TZ)
            return local.astimezone(_UTC)
        except ValueError:
            continue
    return None


def parse_bsp_csv(
    path: Path,
    *,
    tracks_yaml: str | Path = "config/tracks.yaml",
) -> pl.DataFrame:
    """Parse one PROMO BSP CSV → rows conforming to BSP_SCHEMA.

    Filters to UK GBGB tracks; non-UK rows produce track=None and are dropped.
    """
    try:
        raw = pl.read_csv(path, infer_schema_length=2000)
    except Exception as e:
        log.warning("Could not read %s: %s", path, e)
        return pl.DataFrame(schema=BSP_SCHEMA)

    needed = {"event_dt", "menu_hint", "selection_name", "bsp", "win_lose"}
    missing = needed - set(raw.columns)
    if missing:
        log.warning("BSP CSV %s missing columns: %s", path.name, missing)
        return pl.DataFrame(schema=BSP_SCHEMA)

    # Extract derived fields in Python (small enough; clearer than chained
    # expressions, and the typed helpers above already do the work).
    menu_hints = raw["menu_hint"].to_list()
    selection_names = raw["selection_name"].to_list()
    event_dts = raw["event_dt"].to_list()

    tracks: list[str | None] = []
    race_times: list[datetime | None] = []
    traps: list[int | None] = []
    safe_names: list[str] = []
    bare_names: list[str] = []
    for mh, sn, edt in zip(menu_hints, selection_names, event_dts):
        track_raw = _extract_track_from_menu_hint(mh or "")
        tracks.append(canonical_track(track_raw, tracks_yaml=tracks_yaml))
        race_times.append(_parse_event_dt(edt or ""))
        bare, trap = _strip_trap(sn or "")
        traps.append(trap)
        bare_names.append(bare)
        safe_names.append(bf_safe_name(bare))

    df = pl.DataFrame({
        "market_id":      raw["event_id"].cast(pl.Utf8),
        "track":          pl.Series(tracks, dtype=pl.Utf8),
        "race_time":      pl.Series(race_times, dtype=pl.Datetime(time_zone="UTC")),
        "selection_id":   raw["selection_id"].cast(pl.Int64),
        "selection_name": pl.Series(bare_names, dtype=pl.Utf8),
        "bf_safe_name":   pl.Series(safe_names, dtype=pl.Utf8),
        "trap":           pl.Series(traps, dtype=pl.Int8),
        "bsp":            raw["bsp"].cast(pl.Float64),
        "won":            (raw["win_lose"].cast(pl.Int64) == 1).cast(pl.Boolean),
        "matched_volume": raw["pptradedvol"].cast(pl.Float64)
                          if "pptradedvol" in raw.columns
                          else pl.Series([None] * raw.height, dtype=pl.Float64),
    })
    df = df.with_columns(pl.col("race_time").dt.date().alias("event_date"))
    df = df.select(list(BSP_SCHEMA.keys()))

    # Drop non-UK rows (unresolved track).
    before = df.height
    df = df.filter(pl.col("track").is_not_null())
    log.info("%s: kept %d/%d rows after UK-track filter",
             path.name, df.height, before)
    return df


def parse_all_cached(
    cache_dir: Path,
    *,
    tracks_yaml: str | Path = "config/tracks.yaml",
) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for csv in sorted(cache_dir.rglob("*.csv")):
        try:
            frames.append(parse_bsp_csv(csv, tracks_yaml=tracks_yaml))
        except Exception as e:
            log.warning("Failed to parse %s: %s", csv, e)
    if not frames:
        return pl.DataFrame(schema=BSP_SCHEMA)
    return pl.concat(frames, how="vertical_relaxed")


def main() -> None:
    import argparse
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    df = parse_all_cached(Path(cfg.ingest.betfair_bsp.cache_dir))
    out = cfg.paths.interim_dir / "betfair_bsp.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)
    log.info("Parsed %d BSP rows → %s", df.height, out)


if __name__ == "__main__":
    main()
