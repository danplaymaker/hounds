#!/usr/bin/env python3
"""Bulk-download Betfair PROMO BSP CSVs for UK greyhound win markets.

Standalone — uses only the Python 3.8+ standard library. Runs on your UK
machine, downloads daily files for a date range, caches to disk, resumable
on interrupt. Once done, zip the output dir and upload.

USAGE
-----
    python download_bsp.py --from 2024-05-21 --to 2026-05-20 --out ./bsp_csvs
    # then: zip -r bsp.zip ./bsp_csvs   and upload bsp.zip here

The filename Betfair uses is dwbfgreyhoundwin{DDMMYYYY}.csv — the date in
the filename is the *generation* date, so the file dwbfgreyhoundwin{D}.csv
contains UK races that ran on D - 1 day. To cover races on R, fetch the
file dated R + 1 day. The --from/--to flags refer to the GENERATION date
(the date in the filename), so use --from = first-race-date + 1 day.

Polite by default: 1.5 s between requests. Backs off exponentially on
5xx/HTTPError. Skips files already in --out.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

URL_TMPL = "https://promo.betfair.com/betfairsp/prices/dwbfgreyhoundwin{ddmmyyyy}.csv"
USER_AGENT = "greyhound-research/1.0 (personal research)"


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def fetch_one(d: date, out_dir: Path, *, timeout: int = 30) -> str:
    """Fetch a single day's CSV. Returns one of: 'cached', 'ok', 'missing', 'error'."""
    ddmmyyyy = d.strftime("%d%m%Y")
    url = URL_TMPL.format(ddmmyyyy=ddmmyyyy)
    out = out_dir / f"dwbfgreyhoundwin{ddmmyyyy}.csv"
    if out.exists() and out.stat().st_size > 0:
        return "cached"

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            if not data:
                return "missing"
            if b"<html" in data[:200].lower() or b"<!DOCTYPE" in data[:200]:
                # HTML response (geo-block / error page) — not a CSV.
                return "missing"
            out.write_bytes(data)
            return "ok"
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return "missing"
        raise


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="from_", type=parse_date, required=True,
                    help="First date to fetch (YYYY-MM-DD). This is the filename date.")
    ap.add_argument("--to", dest="to_", type=parse_date, required=True,
                    help="Last date to fetch (YYYY-MM-DD). Inclusive.")
    ap.add_argument("--out", type=Path, default=Path("./bsp_csvs"),
                    help="Output directory (default: ./bsp_csvs).")
    ap.add_argument("--delay", type=float, default=1.5,
                    help="Seconds between requests (default: 1.5).")
    ap.add_argument("--max-retries", type=int, default=5,
                    help="Per-file retry count on 5xx (default: 5).")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    if args.from_ > args.to_:
        print("--from must be <= --to", file=sys.stderr)
        return 2

    total_days = (args.to_ - args.from_).days + 1
    counts = {"cached": 0, "ok": 0, "missing": 0, "error": 0}
    d = args.from_
    i = 0
    while d <= args.to_:
        i += 1
        attempts = 0
        backoff = 2.0
        status = "error"
        while attempts < args.max_retries:
            attempts += 1
            try:
                status = fetch_one(d, args.out)
                break
            except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError) as e:
                print(f"[{i}/{total_days}] {d}: retry {attempts}/{args.max_retries} after {e}",
                      file=sys.stderr)
                time.sleep(backoff)
                backoff *= 2
        counts[status] = counts.get(status, 0) + 1
        marker = {"cached": "·", "ok": "+", "missing": "?", "error": "x"}.get(status, "?")
        print(f"[{i}/{total_days}] {d}  {marker}  {status}")
        if status not in {"cached"}:
            time.sleep(args.delay)
        d = d + timedelta(days=1)

    print()
    print(f"Done. Output in: {args.out.resolve()}")
    print(f"  Downloaded: {counts.get('ok', 0)}")
    print(f"  Already cached: {counts.get('cached', 0)}")
    print(f"  Missing (no file for that date): {counts.get('missing', 0)}")
    print(f"  Errors: {counts.get('error', 0)}")
    print()
    print("Next: zip the directory and upload it:")
    print(f"  cd {args.out.parent}  &&  zip -r bsp.zip {args.out.name}")
    return 0 if counts.get("error", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
