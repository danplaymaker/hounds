#!/usr/bin/env python3
"""Fetch current bookmaker prices from Oddschecker for the model's stand-outs.

Standalone Mac/UK script — Oddschecker uses Cloudflare protection so it
must run from a real (residential) IP, not a cloud container.

Install once:
    pip3 install cloudscraper selectolax pyyaml

Run against today's standouts CSV (already produced by predict_card):
    python3 fetch_oddschecker.py \\
        --standouts ~/Downloads/standouts_2026-05-22.csv \\
        --out       ~/Downloads/bookmaker_prices_2026-05-22.csv \\
        --books     bet365,skybet,paddypower,williamhill,coral,ladbrokes

Output CSV columns:
    race_datetime, track, distance_m, trap, dog_name,
    model_prob, model_fair_odds,
    best_book, best_price, best_edge_vs_fair,
    bet365, skybet, paddypower, ...

A 'best_edge_vs_fair' > 0 means some bookmaker is offering above the
model's fair odds — a candidate value bet.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# Canonical track -> Oddschecker URL slug. If a slug is wrong, edit it
# here and re-run; Oddschecker's URLs occasionally rename.
TRACK_SLUG = {
    "central_park":  "central-park",
    "crayford":      "crayford",
    "doncaster":     "doncaster",
    "harlow":        "harlow",
    "henlow":        "henlow",
    "hove":          "hove",
    "kinsley":       "kinsley",
    "monmore":       "monmore",
    "newcastle":     "newcastle",
    "nottingham":    "nottingham",
    "oxford":        "oxford",
    "pelaw_grange":  "pelaw-grange",
    "perry_barr":    "perry-barr",
    "romford":       "romford",
    "sheffield":     "sheffield",
    "sunderland":    "sunderland",
    "suffolk_downs": "suffolk-downs",
    "swindon":       "swindon",
    "towcester":     "towcester",
}

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7_2) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.6 Safari/605.1.15"
)

# Friendly name -> Oddschecker bookmaker short-code. Extend as you confirm
# more codes from the page. The script also accepts the short codes directly
# in --books, so missing entries here don't break anything; they just mean
# you have to type the short code instead of the brand name.
BOOKMAKER_CODES: dict[str, str] = {
    "bet365":      "B3",
    "williamhill": "WH",
    "paddypower":  "PP",
    "skybet":      "SX",
    "ladbrokes":   "LD",
    "coral":       "CE",
    "betfred":     "FR",
    "boylesports": "BR",
    "unibet":      "UN",
    "betvictor":   "BE",
    "betfair":     "BF",
    "888sport":    "EE",
    "betway":      "BY",
}


def resolve_book_codes(books_arg: str) -> list[str]:
    """Accept either friendly names or short codes in the --books flag."""
    out: list[str] = []
    for b in books_arg.split(","):
        b = b.strip()
        if not b:
            continue
        out.append(BOOKMAKER_CODES.get(b.lower(), b))   # pass through unknown
    return out


def fractional_to_decimal(s: str) -> float | None:
    """'9/4' -> 3.25, '5/2F' -> 3.50, 'EVS' -> 2.0, '2.85' -> 2.85."""
    if not s:
        return None
    s = s.strip()
    if not s:
        return None
    if s.lower() in ("evs", "ev", "evens"):
        return 2.0
    # Trim trailing favourite markers
    s = re.sub(r"[A-Za-z]+$", "", s).strip()
    m = re.match(r"^(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)$", s)
    if m:
        n, d = float(m.group(1)), float(m.group(2))
        return n / d + 1.0 if d else None
    try:
        return float(s)
    except ValueError:
        return None


def build_url(track_slug: str, race_dt: datetime) -> str:
    """Oddschecker race-card URL pattern (as of 2026):
        https://www.oddschecker.com/greyhounds/<YYYY-MM-DD>-<track>/<HH:MM>/winner
    """
    date_part = race_dt.strftime("%Y-%m-%d")
    time_part = race_dt.strftime("%H:%M")
    return f"https://www.oddschecker.com/greyhounds/{date_part}-{track_slug}/{time_part}/winner"


def make_session():
    try:
        import cloudscraper
        return cloudscraper.create_scraper(browser={"custom": UA})
    except ImportError:
        import requests
        s = requests.Session()
        s.headers.update({"User-Agent": UA})
        return s


def parse_oddschecker_html(html: str, want_books: list[str]) -> list[dict]:
    """Parse Oddschecker's bookmaker odds table.

    Returns list of {'dog_name': str, 'prices': {book_code: decimal_odds, ...}}.

    Book codes are upper-case short codes ('B3', 'WH', ...). Cells with
    data-odig='0' or inner text 'SP' are skipped — those bookmakers
    haven't priced the race yet and only offer Starting Price.
    """
    try:
        from selectolax.parser import HTMLParser
    except ImportError:
        raise SystemExit("Install selectolax: pip3 install selectolax")

    tree = HTMLParser(html)
    runners: list[dict] = []
    want_set = {b.upper() for b in want_books}

    rows = tree.css("tr[data-bname]")
    for row in rows:
        name = (row.attributes.get("data-bname") or "").strip()
        name_clean = re.sub(r"^\d+\.\s*", "", name).strip()
        prices: dict[str, float] = {}
        for cell in row.css("td[data-bk]"):
            bk_raw = cell.attributes.get("data-bk") or ""
            bk = bk_raw.strip().upper()
            if not bk:
                continue
            if want_set and bk not in want_set:
                continue
            odig = cell.attributes.get("data-odig") or ""
            price: float | None = None
            try:
                v = float(odig)
                if v > 1.0:
                    price = v
            except ValueError:
                pass
            if price is None:
                text = cell.text(strip=True)
                if text and text.upper() != "SP":
                    price = fractional_to_decimal(text)
            if price and price > 1.0:
                prices[bk] = price
        if name_clean and prices:
            runners.append({"dog_name": name_clean, "prices": prices})
    return runners


def fetch_race(session, url: str, *, timeout: int = 20, retries: int = 3,
               backoff: float = 2.0) -> str | None:
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, timeout=timeout)
            if resp.status_code == 200:
                return resp.text
            if resp.status_code == 404:
                return None
            print(f"  {url}: HTTP {resp.status_code}", file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            print(f"  {url}: {e}", file=sys.stderr)
        time.sleep(backoff * attempt)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--standouts", required=True,
                    help="Path to standouts_<date>.csv (from predict_card).")
    ap.add_argument("--out", required=True, help="Output CSV path.")
    ap.add_argument(
        "--books",
        default="bet365,skybet,paddypower,williamhill,coral,ladbrokes,betfred,boylesports,unibet,betvictor",
        help=("Comma-separated bookmaker names (bet365, skybet, ...) or "
              "Oddschecker short codes (B3, WH, ...). Unknown names are "
              "passed through as-is so you can add codes ad-hoc."),
    )
    ap.add_argument("--delay", type=float, default=2.0,
                    help="Polite delay between Oddschecker requests (default 2s).")
    args = ap.parse_args()

    want_books = resolve_book_codes(args.books)
    print(f"Looking for bookmakers: {want_books}")

    rows = list(csv.DictReader(open(args.standouts)))
    if not rows:
        print("No standouts in input.", file=sys.stderr)
        return 1
    print(f"Loaded {len(rows)} stand-outs from {args.standouts}")

    session = make_session()

    # Group rows by race (track + race_datetime). Each race fetched once.
    races: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        key = (r["track"], r["race_datetime"])
        races.setdefault(key, []).append(r)
    print(f"Distinct races to fetch: {len(races)}")

    out_fields = [
        "race_datetime", "track", "distance_m", "trap", "dog_name",
        "model_prob", "model_fair_odds",
        "best_book", "best_price", "best_edge_vs_fair",
        *want_books,
    ]
    out_f = open(args.out, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_f, fieldnames=out_fields)
    writer.writeheader()

    n_fetched = n_matched = 0
    for (track, race_dt_str), race_rows in sorted(races.items(), key=lambda x: x[0][1]):
        slug = TRACK_SLUG.get(track)
        if not slug:
            print(f"Skip — unknown track slug: {track}", file=sys.stderr)
            continue
        # Parse various ISO formats. Python 3.9 needs '+00:00', not '+0000'.
        s = race_dt_str.replace("Z", "+00:00")
        m = re.search(r"([+\-])(\d{2})(\d{2})$", s)
        if m:
            s = s[:m.start()] + f"{m.group(1)}{m.group(2)}:{m.group(3)}"
        race_dt = datetime.fromisoformat(s)
        # Convert to UK local for the URL
        import zoneinfo
        race_dt_uk = race_dt.astimezone(zoneinfo.ZoneInfo("Europe/London"))
        url = build_url(slug, race_dt_uk)

        html = fetch_race(session, url)
        n_fetched += 1
        if not html:
            print(f"  no HTML for {url}", file=sys.stderr)
            for r in race_rows:
                out_row = {k: r.get(k, "") for k in out_fields if k in r}
                writer.writerow(out_row)
            time.sleep(args.delay)
            continue

        runners = parse_oddschecker_html(html, want_books)
        runner_by_name = {
            re.sub(r"[^a-z0-9]+", " ", r["dog_name"].lower()).strip(): r
            for r in runners
        }

        for r in race_rows:
            dog_norm = re.sub(r"[^a-z0-9]+", " ", r["dog_name"].lower()).strip()
            match = runner_by_name.get(dog_norm)
            prices = match["prices"] if match else {}
            if match:
                n_matched += 1
            fair = float(r["model_fair_odds"]) if r.get("model_fair_odds") else None
            best_book = best_price = None
            for bk, pr in prices.items():
                if best_price is None or pr > best_price:
                    best_book, best_price = bk, pr
            edge = (best_price / fair - 1.0) if (best_price and fair) else None
            row_out = {
                "race_datetime":    r["race_datetime"],
                "track":            r["track"],
                "distance_m":       r.get("distance_m", ""),
                "trap":             r.get("trap", ""),
                "dog_name":         r["dog_name"],
                "model_prob":       r.get("model_prob", ""),
                "model_fair_odds":  r.get("model_fair_odds", ""),
                "best_book":        best_book or "",
                "best_price":       f"{best_price:.2f}" if best_price else "",
                "best_edge_vs_fair": f"{edge:.4f}" if edge is not None else "",
            }
            for bk in want_books:
                row_out[bk] = f"{prices[bk]:.2f}" if bk in prices else ""
            writer.writerow(row_out)

        time.sleep(args.delay)

    out_f.close()
    print(f"\nFetched {n_fetched} races, matched {n_matched}/{len(rows)} runners.")
    print(f"Output written to: {args.out}")
    print(
        "\nSort by best_edge_vs_fair in your spreadsheet — positive values are "
        "candidate value bets (a bookmaker is offering above the model's fair price)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
