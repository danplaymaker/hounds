"""GBGB HTML → typed rows (A1.2).

Pure function. No I/O inside `parse_race_html`. The CLI wraps it with file
reads for batch runs.

The selectors below are placeholders shaped to GBGB's typical results
layout — finalise once a sample page is captured.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
from selectolax.parser import HTMLParser

from greyhound.data.identity import bf_safe_name, canonical_track
from greyhound.data.schemas import RUN_SCHEMA, load_config

log = logging.getLogger(__name__)


def parse_race_html(html: str, *, tracks_yaml: str | Path = "config/tracks.yaml") -> list[dict[str, Any]]:
    """Parse one race results HTML page → list of runner dicts.

    Returns rows conforming to RUN_SCHEMA (ish — Polars cast happens later).
    Returns empty list on a malformed/empty page, with a warning logged.
    """
    tree = HTMLParser(html)

    race_id = _first_attr(tree, "meta[name=race-id]", "content") or _first_text(tree, ".race-id")
    if not race_id:
        log.warning("No race_id in HTML — skipping page")
        return []

    track_raw = _first_text(tree, ".race-track") or ""
    track = canonical_track(track_raw, tracks_yaml=tracks_yaml)
    distance_m_s = _first_text(tree, ".race-distance") or ""
    distance_m = int(re.sub(r"[^0-9]", "", distance_m_s)) if distance_m_s else 0
    grade = _first_text(tree, ".race-grade")
    going_s = _first_text(tree, ".race-going") or ""
    going = float(re.sub(r"[^0-9.+\-]", "", going_s)) if going_s else None
    race_dt = _parse_dt(_first_text(tree, ".race-datetime") or "")

    rows: list[dict[str, Any]] = []
    for row in tree.css(".runner-row"):
        dog_id = (row.attributes.get("data-dog-id") or "").strip()
        dog_name = _first_text(row, ".dog-name") or ""
        trap_s = _first_text(row, ".trap") or "0"
        try:
            trap = int(re.sub(r"[^0-9]", "", trap_s))
        except ValueError:
            trap = 0
        finish_s = _first_text(row, ".finish-position") or ""
        finish = int(re.sub(r"[^0-9]", "", finish_s)) if finish_s.strip() else None
        sp_s = _first_text(row, ".sp") or ""
        sp = _parse_decimal_odds(sp_s)
        run_time = _to_float(_first_text(row, ".run-time"))
        sectional_1 = _to_float(_first_text(row, ".sectional-1"))
        weight_kg = _to_float(_first_text(row, ".weight-kg"))
        trainer_id = (row.attributes.get("data-trainer-id") or "").strip()
        trainer_name = _first_text(row, ".trainer-name") or ""
        comment = _first_text(row, ".comment") or ""

        rows.append({
            "race_id": race_id,
            "race_datetime": race_dt,
            "track": track,
            "distance_m": distance_m,
            "grade": grade,
            "going": going,
            "dog_id": dog_id,
            "dog_name": dog_name,
            "trap": trap,
            "sp": sp,
            "finish_position": finish,
            "run_time": run_time,
            "sectional_1": sectional_1,
            "weight_kg": weight_kg,
            "trainer_id": trainer_id,
            "trainer_name": trainer_name,
            "comment": comment,
            "bf_safe_name": bf_safe_name(dog_name),
        })

    return rows


def _first_text(tree, selector: str) -> str | None:
    node = tree.css_first(selector)
    if node is None:
        return None
    return (node.text() or "").strip() or None


def _first_attr(tree, selector: str, attr: str) -> str | None:
    node = tree.css_first(selector)
    if node is None:
        return None
    return node.attributes.get(attr)


def _to_float(s: str | None) -> float | None:
    if s is None:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_decimal_odds(s: str) -> float | None:
    """Accept '5/2', '5-2', or '3.50'. Returns decimal-odds float or None."""
    s = (s or "").strip()
    if not s:
        return None
    m = re.match(r"^(\d+(?:\.\d+)?)[/\-](\d+(?:\.\d+)?)$", s)
    if m:
        num, den = float(m.group(1)), float(m.group(2))
        return num / den + 1.0 if den else None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_dt(s: str) -> datetime | None:
    if not s:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M", "%d/%m/%Y %H:%M"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    return None


def parse_all_cached(cache_dir: Path, tracks_yaml: str | Path = "config/tracks.yaml") -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for html_path in cache_dir.rglob("*.html"):
        if html_path.name.startswith("_"):
            continue
        try:
            html = html_path.read_text(encoding="utf-8")
        except OSError:
            continue
        rows.extend(parse_race_html(html, tracks_yaml=tracks_yaml))

    if not rows:
        return pl.DataFrame(schema=RUN_SCHEMA)
    return pl.DataFrame(rows).cast({k: v for k, v in RUN_SCHEMA.items() if k in rows[0]})


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
