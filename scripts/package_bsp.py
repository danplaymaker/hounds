#!/usr/bin/env python3
"""Filter downloaded Betfair BSP CSVs to UK greyhound rows and chunk into
small monthly zips for upload.

After running download_bsp.py, run this against the same output directory.
It produces zips at most a few MB each, easy to upload one at a time.

USAGE
-----
    python package_bsp.py --in ./bsp_csvs --out ./bsp_chunks

Each chunk contains:
  - All CSVs whose filename date falls in one calendar month
  - Rows pre-filtered to UK GBGB-licensed tracks
  - Same CSV layout as the originals (header preserved)

Stdlib only. UK track list is embedded so this script stands alone.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import sys
import unicodedata
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------- track list
# Aliases observed in Betfair menu_hint for the 18 GBGB-licensed UK tracks.
# Lowercased + non-alphanumerics stripped for matching.

UK_TRACKS_RAW = [
    "Central Park", "Crayford", "Crayford & Bexleyheath", "Doncaster",
    "Harlow", "Henlow", "Hove", "Brighton & Hove", "Kinsley", "Monmore",
    "Monmore Green", "Newcastle", "Nottingham", "Colwick Park", "Oxford",
    "Pelaw Grange", "Perry Barr", "Birmingham", "Romford", "Sheffield",
    "Owlerton", "Suffolk Downs", "Mildenhall", "Sunderland", "Swindon",
    "Towcester",
]


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", s.lower())


UK_KEYS = {_norm(t) for t in UK_TRACKS_RAW}


_MENU_TRACK_RE = re.compile(r"^\s*([A-Za-z &'\-]+?)\s+(?:\([A-Z]{2,4}\)\s+)?\d")


def is_uk_row(menu_hint: str) -> bool:
    """Return True if menu_hint resolves to a UK GBGB-licensed track."""
    if not menu_hint:
        return False
    # If a non-UK country code is in parens, drop fast.
    if re.search(r"\((?!UK\)|GB\))[A-Z]{2,4}\)", menu_hint):
        return False
    m = _MENU_TRACK_RE.match(menu_hint)
    track = m.group(1).strip() if m else menu_hint.split()[0]
    return _norm(track) in UK_KEYS


def filename_to_month(name: str) -> str | None:
    """`dwbfgreyhoundwin01062024.csv` -> "2024-06". None if unparseable."""
    m = re.search(r"(\d{2})(\d{2})(\d{4})", name)
    if not m:
        return None
    try:
        d = datetime.strptime(m.group(0), "%d%m%Y").date()
    except ValueError:
        return None
    return f"{d.year:04d}-{d.month:02d}"


def filter_csv_to_uk(src: Path) -> bytes:
    """Read src, drop rows whose menu_hint isn't a UK GBGB track. Returns
    the filtered CSV bytes (header preserved)."""
    with src.open("r", newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        if "menu_hint" not in header:
            return src.read_bytes()  # Pass through if we can't filter

        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=header)
        writer.writeheader()
        kept = 0
        for row in reader:
            if is_uk_row(row.get("menu_hint", "")):
                writer.writerow(row)
                kept += 1
    return buf.getvalue().encode("utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_", type=Path, required=True,
                    help="Directory containing downloaded dwbfgreyhoundwin*.csv files.")
    ap.add_argument("--out", type=Path, default=Path("./bsp_chunks"),
                    help="Output directory for monthly zips (default: ./bsp_chunks).")
    ap.add_argument("--no-filter", action="store_true",
                    help="Skip UK filtering, include all rows. Larger output.")
    args = ap.parse_args()

    if not args.in_.exists():
        print(f"--in path does not exist: {args.in_}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)

    by_month: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(args.in_.glob("dwbfgreyhoundwin*.csv")):
        m = filename_to_month(path.name)
        if m is None:
            print(f"Skipping unrecognised filename: {path.name}", file=sys.stderr)
            continue
        by_month[m].append(path)

    if not by_month:
        print(f"No dwbfgreyhoundwin*.csv files found in {args.in_}", file=sys.stderr)
        return 1

    total_in = total_out = 0
    for month in sorted(by_month):
        files = by_month[month]
        zip_path = args.out / f"bsp_{month}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for path in files:
                if args.no_filter:
                    data = path.read_bytes()
                else:
                    data = filter_csv_to_uk(path)
                z.writestr(path.name, data)
                total_in += path.stat().st_size
                total_out += len(data)
        size_mb = zip_path.stat().st_size / (1024 * 1024)
        print(f"  {zip_path.name}  {len(files):3d} files  {size_mb:6.2f} MB")

    print()
    print(f"Done. {len(by_month)} monthly zips written to {args.out.resolve()}")
    if not args.no_filter:
        pct = 100.0 * total_out / total_in if total_in else 0
        print(f"UK filter kept {total_out / 1024 / 1024:.1f} MB of {total_in / 1024 / 1024:.1f} MB ({pct:.1f}%)")
    print()
    print("Upload the bsp_*.zip files one at a time, or in batches.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
