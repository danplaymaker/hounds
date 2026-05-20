"""Betfair PROMO BSP CSV ingestion (A1.3).

Free BSP CSVs come from https://promo.betfair.com/betfairsp/prices/ as
daily files. This module parses any cached CSVs into a typed parquet
frame. Downloading is left as an offline step (cache the CSVs into the
configured cache dir), to keep this layer free of network logic that
already exists in scraper.

CSV columns (PROMO, as observed): SP, EVENT_DT, EVENT_NAME, EVENT_ID,
SELECTION_NAME, SELECTION_ID, WIN_LOSE, BSP, PPWAP, MORNINGWAP,
PPMAX, PPMIN, IPMAX, IPMIN, MORNINGTRADEDVOL, PPTRADEDVOL, IPTRADEDVOL.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path

import polars as pl

from greyhound.data.identity import bf_safe_name, canonical_track
from greyhound.data.schemas import BSP_SCHEMA, load_config

log = logging.getLogger(__name__)


def parse_bsp_csv(path: Path, tracks_yaml: str | Path = "config/tracks.yaml") -> pl.DataFrame:
    """Parse one PROMO BSP CSV. Returns rows conforming to BSP_SCHEMA."""
    df = pl.read_csv(path, infer_schema_length=1000)

    needed = {"EVENT_DT", "EVENT_NAME", "SELECTION_NAME", "BSP", "WIN_LOSE"}
    missing = needed - set(df.columns)
    if missing:
        log.warning("BSP CSV %s missing columns: %s", path, missing)
        return pl.DataFrame(schema=BSP_SCHEMA)

    track_raw = df["EVENT_NAME"].map_elements(_extract_track, return_dtype=pl.Utf8)
    canonical = track_raw.map_elements(
        lambda t: canonical_track(t, tracks_yaml=tracks_yaml), return_dtype=pl.Utf8
    )
    race_time = df["EVENT_DT"].map_elements(_parse_dt, return_dtype=pl.Datetime(time_zone="UTC"))
    safe = df["SELECTION_NAME"].map_elements(_strip_trap, return_dtype=pl.Struct({
        "name": pl.Utf8, "trap": pl.Int8,
    }))

    out = pl.DataFrame({
        "market_id":      df.get_column("EVENT_ID").cast(pl.Utf8) if "EVENT_ID" in df.columns else pl.Series([None] * df.height, dtype=pl.Utf8),
        "event_date":     race_time.dt.date(),
        "track":          canonical,
        "race_time":      race_time,
        "selection_id":   df["SELECTION_ID"].cast(pl.Int64) if "SELECTION_ID" in df.columns else pl.Series([None] * df.height, dtype=pl.Int64),
        "selection_name": df["SELECTION_NAME"].cast(pl.Utf8),
        "bf_safe_name":   safe.struct.field("name").map_elements(bf_safe_name, return_dtype=pl.Utf8),
        "trap":           safe.struct.field("trap"),
        "bsp":            df["BSP"].cast(pl.Float64),
        "won":            (df["WIN_LOSE"].cast(pl.Utf8) == "1") if df["WIN_LOSE"].dtype == pl.Utf8 else (df["WIN_LOSE"].cast(pl.Int8) == 1),
        "matched_volume": df["PPTRADEDVOL"].cast(pl.Float64) if "PPTRADEDVOL" in df.columns else pl.Series([None] * df.height, dtype=pl.Float64),
    })
    return out


_TRAP_RE = re.compile(r"^\s*(\d)\s*[\.\-:)]\s*(.+)$")


def _strip_trap(name: str) -> dict:
    """Selection names from Betfair greyhound markets are like '1. Rapid Ranger'.
    Pulls trap out. If no leading trap digit, returns trap=None.
    """
    if name is None:
        return {"name": "", "trap": None}
    m = _TRAP_RE.match(name)
    if m:
        return {"name": m.group(2).strip(), "trap": int(m.group(1))}
    return {"name": name.strip(), "trap": None}


def _extract_track(event_name: str) -> str | None:
    """Event names look like 'Crayford 19:43 R5 480m'. Track is the first token."""
    if not event_name:
        return None
    # Take everything before the first time-like substring or 'R<digit>'.
    m = re.match(r"^([A-Za-z &'\-]+?)\s+\d", event_name)
    return (m.group(1).strip() if m else event_name.split(maxsplit=1)[0]).strip()


def _parse_dt(s: str) -> datetime | None:
    if not s:
        return None
    for fmt in ("%d-%m-%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=__import__("datetime").timezone.utc)
        except ValueError:
            continue
    return None


def parse_all_cached(cache_dir: Path, tracks_yaml: str | Path = "config/tracks.yaml") -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for csv in cache_dir.rglob("*.csv"):
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
