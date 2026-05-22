#!/usr/bin/env python3
"""monitor_late_steamers.py — follow sharp money, back the soft books.

Watches UK greyhound markets only in the last N minutes before the off
(default 20). Tracks each runner's best price at a 'sharp' bookmaker
(default Bet365). When the sharp price drops significantly within a
short window AND a 'soft' bookmaker is still offering above the sharp's
new level, fires an alert with the specific soft-book opportunity.

The logic:
    1. Sharp books move first because their books are well-funded and
       traders watch the exchange / each other closely.
    2. Soft books are slower to react — there's an arbitrage window
       between sharp tightening and soft following.
    3. If you're fast you can back the dog at the soft's stale price
       while the sharp price has already proven the market moved.

USAGE
-----
    python3 monitor_late_steamers.py \\
        --pre-off       20m          # only watch within 20m of each race off
        --interval      30           # poll every 30s
        --sharp         bet365       # which book is 'sharp' (can list multiple)
        --soft          coral,ladbrokes,boylesports,betfred,williamhill,unibet,betvictor,skybet,paddypower
        --sharp-pct     0.10         # alert when sharp drops >= 10%
        --sharp-window  5m           # over a 5-minute window
        --soft-edge     0.10         # require soft >= sharp * (1 + 0.10)
        --min-price     2.0
        --log           ~/Downloads/late_steamers.csv

Requires: cloudscraper, selectolax
"""

from __future__ import annotations

import argparse
import csv
import re
import signal
import subprocess
import sys
import time
import zoneinfo
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

UK = zoneinfo.ZoneInfo("Europe/London")
UTC = timezone.utc

UK_TRACK_SLUGS = {
    "central-park", "crayford", "doncaster", "harlow", "henlow", "hove",
    "kinsley", "monmore", "newcastle", "nottingham", "oxford",
    "pelaw-grange", "perry-barr", "romford", "sheffield", "sunderland",
    "suffolk-downs", "swindon", "towcester",
}

BOOKMAKER_CODES = {
    "bet365": "B3", "williamhill": "WH", "paddypower": "PP", "skybet": "SX",
    "ladbrokes": "LD", "coral": "CE", "betfred": "FR", "boylesports": "BR",
    "unibet": "UN", "betvictor": "BE", "betfair": "BF", "888sport": "EE",
    "betway": "BY",
}
CODE_TO_NAME = {v: k for k, v in BOOKMAKER_CODES.items()}

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7_2) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.6 Safari/605.1.15"
)

RESET = "\033[0m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"


def play_sound(name: str = "Submarine") -> None:
    try:
        subprocess.run(
            ["afplay", f"/System/Library/Sounds/{name}.aiff"],
            check=False, timeout=5,
        )
    except Exception:
        pass


def notify(title: str, message: str, sound: str = "Submarine") -> None:
    play_sound(sound)
    try:
        subprocess.run(
            ["terminal-notifier", "-title", title, "-message", message,
             "-sound", sound],
            check=False, timeout=5,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return
    except FileNotFoundError:
        pass
    try:
        t = title.replace('"', '\\"')
        m = message.replace('"', '\\"')
        subprocess.run(
            ["osascript", "-e",
             f'display notification "{m}" with title "{t}" sound name "{sound}"'],
            check=False, timeout=5,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


def fractional_to_decimal(s: str) -> float | None:
    if not s:
        return None
    s = s.strip()
    if s.lower() in ("evs", "ev", "evens"):
        return 2.0
    s = re.sub(r"[A-Za-z]+$", "", s).strip()
    m = re.match(r"^(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)$", s)
    if m:
        n, d = float(m.group(1)), float(m.group(2))
        return n / d + 1.0 if d else None
    try:
        return float(s)
    except ValueError:
        return None


@dataclass
class RaceRef:
    url: str
    slug: str
    dt_uk: datetime
    dt_utc: datetime


_RACE_URL_RE = re.compile(
    r"/greyhounds/(\d{4}-\d{2}-\d{2})-([a-z][a-z0-9-]+)/(\d{2}:\d{2})/winner"
)


def discover_races(session) -> list[RaceRef]:
    r = session.get("https://www.oddschecker.com/greyhounds", timeout=15)
    if r.status_code != 200:
        print(f"  landing page: HTTP {r.status_code}", file=sys.stderr)
        return []
    seen: set[str] = set()
    out: list[RaceRef] = []
    for m in _RACE_URL_RE.finditer(r.text):
        if m.group(0) in seen:
            continue
        seen.add(m.group(0))
        date_str, slug, time_str = m.group(1), m.group(2), m.group(3)
        if slug not in UK_TRACK_SLUGS:
            continue
        try:
            dt_uk = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M").replace(tzinfo=UK)
        except ValueError:
            continue
        out.append(RaceRef(
            url=f"https://www.oddschecker.com{m.group(0)}",
            slug=slug, dt_uk=dt_uk, dt_utc=dt_uk.astimezone(UTC),
        ))
    return out


def parse_prices(html: str, want_books: set[str]) -> dict[str, dict[str, float]]:
    """Returns {dog_name_lower: {book_code: decimal_odds}}."""
    from selectolax.parser import HTMLParser
    tree = HTMLParser(html)
    out: dict[str, dict[str, float]] = {}
    for row in tree.css("tr[data-bname]"):
        name_raw = (row.attributes.get("data-bname") or "").strip()
        name = re.sub(r"^\d+\.\s*", "", name_raw).strip()
        if not name:
            continue
        prices: dict[str, float] = {}
        for cell in row.css("td[data-bk]"):
            bk = (cell.attributes.get("data-bk") or "").strip().upper()
            if not bk or (want_books and bk not in want_books):
                continue
            odig = cell.attributes.get("data-odig") or ""
            price = None
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
        if prices:
            out[name.lower()] = prices
    return out


@dataclass
class Sample:
    t: datetime
    prices: dict[str, float]   # book_code -> price


class LateSteamerMonitor:
    def __init__(self, args: argparse.Namespace):
        import cloudscraper
        self.session = cloudscraper.create_scraper(browser={"custom": UA})
        self.args = args

        # Resolve sharp / soft book codes
        self.sharp_codes: set[str] = self._codes(args.sharp)
        self.soft_codes: set[str] = self._codes(args.soft)
        self.all_codes: set[str] = self.sharp_codes | self.soft_codes
        print(f"Sharp books: {sorted(self.sharp_codes)} ({sorted(CODE_TO_NAME.get(c, c) for c in self.sharp_codes)})", flush=True)
        print(f"Soft books:  {sorted(self.soft_codes)} ({sorted(CODE_TO_NAME.get(c, c) for c in self.soft_codes)})", flush=True)

        self.pre_off = timedelta(seconds=args.pre_off_seconds)
        self.sharp_window = timedelta(seconds=args.sharp_window_seconds)

        # Per-(race, dog) history of price snapshots
        self._history: dict[tuple[str, str], deque[Sample]] = defaultdict(
            lambda: deque(maxlen=200)
        )
        self._last_alert: dict[tuple[str, str], datetime] = {}
        self._known_races: dict[str, RaceRef] = {}
        self._last_discover: datetime | None = None

        # Log
        self.log_f = None
        self.log_writer = None
        if args.log:
            new = not Path(args.log).exists()
            self.log_f = open(args.log, "a", newline="", encoding="utf-8")
            self.log_writer = csv.DictWriter(
                self.log_f,
                fieldnames=[
                    "alerted_at_utc", "race_time_uk", "track", "dog_name",
                    "sharp_book", "sharp_from", "sharp_to", "sharp_pct",
                    "soft_book", "soft_price", "soft_vs_sharp_pct",
                ],
            )
            if new:
                self.log_writer.writeheader()
                self.log_f.flush()
        self._stop = False
        signal.signal(signal.SIGINT, self._sigint)

    def _codes(self, names: str) -> set[str]:
        out: set[str] = set()
        for n in names.split(","):
            n = n.strip()
            if not n:
                continue
            out.add(BOOKMAKER_CODES.get(n.lower(), n.upper()))
        return out

    def _sigint(self, *_):
        print("\nStopping…", flush=True)
        self._stop = True

    def _maybe_refresh(self, now: datetime) -> None:
        if (self._last_discover is None
                or (now - self._last_discover).total_seconds() >= 180):
            races = discover_races(self.session)
            for r in races:
                self._known_races[r.url] = r
            self._last_discover = now
            in_window = sum(
                1 for r in self._known_races.values()
                if now <= r.dt_utc <= now + self.pre_off
            )
            print(
                f"[{now.strftime('%H:%M:%S')}] {len(self._known_races)} UK races known; "
                f"{in_window} within {int(self.pre_off.total_seconds()/60)}m of off",
                flush=True,
            )

    def _active_races(self, now: datetime) -> list[RaceRef]:
        out: list[RaceRef] = []
        for r in self._known_races.values():
            if not (now <= r.dt_utc + timedelta(minutes=2)):
                continue
            if r.dt_utc - now > self.pre_off:
                continue
            out.append(r)
        return sorted(out, key=lambda r: r.dt_utc)

    def _best(self, prices: dict[str, float], codes: set[str]) -> tuple[str | None, float | None]:
        sub = {b: p for b, p in prices.items() if b in codes}
        if not sub:
            return None, None
        book, price = max(sub.items(), key=lambda kv: kv[1])
        return book, price

    def _sharp_window_low(self, history: deque[Sample], now: datetime) -> tuple[str | None, float | None]:
        """The MAX sharp price seen in the window (i.e. how high it was before
        the steam). Returns (best_sharp_book_in_window, max_sharp_price)."""
        cutoff = now - self.sharp_window
        best_book: str | None = None
        best_price: float = -1
        for s in history:
            if s.t < cutoff:
                continue
            for b, p in s.prices.items():
                if b in self.sharp_codes and p > best_price:
                    best_book, best_price = b, p
        if best_book is None:
            return None, None
        return best_book, best_price

    def _poll_race(self, race: RaceRef, now: datetime) -> None:
        try:
            resp = self.session.get(race.url, timeout=15)
        except Exception as e:
            print(f"  {race.url}: {e}", file=sys.stderr)
            return
        if resp.status_code != 200:
            return
        prices_by_dog = parse_prices(resp.text, self.all_codes)
        if not prices_by_dog:
            return

        for dog_name, prices in prices_by_dog.items():
            sharp_book, sharp_now = self._best(prices, self.sharp_codes)
            soft_book, soft_now = self._best(prices, self.soft_codes)
            if sharp_now is None or sharp_now < self.args.min_price:
                continue
            key = (race.url, dog_name)
            self._history[key].append(Sample(now, prices.copy()))

            # Look at how high sharp WAS in the window
            from_book, from_price = self._sharp_window_low(self._history[key], now)
            if from_price is None or from_price <= sharp_now:
                continue  # no movement, or sharp moved UP

            sharp_change = (sharp_now - from_price) / from_price  # negative = shortened
            if sharp_change > -self.args.sharp_pct:
                continue

            if soft_now is None:
                continue
            soft_edge = soft_now / sharp_now - 1.0
            if soft_edge < self.args.soft_edge:
                continue

            # Cooldown
            last = self._last_alert.get(key)
            if last is not None and (now - last).total_seconds() < 180:
                continue
            self._last_alert[key] = now

            track = race.slug.replace("-", " ")
            sharp_label = CODE_TO_NAME.get(sharp_book or "", sharp_book or "?").title()
            soft_label = CODE_TO_NAME.get(soft_book or "", soft_book or "?").title()
            from_label = CODE_TO_NAME.get(from_book or "", from_book or "?").title()

            title = f"LATE STEAMER — {race.dt_uk.strftime('%H:%M')} {track}"
            msg = (
                f"{dog_name.title()}: SHARP {from_label} {from_price:.2f} "
                f"→ {sharp_label} {sharp_now:.2f} ({sharp_change*100:+.1f}%); "
                f"SOFT {soft_label} STILL {soft_now:.2f} ({soft_edge*100:+.1f}% above sharp)"
            )
            sys.stdout.write("\a")
            print(f"{BOLD}{RED}🔥 {title}{RESET}", flush=True)
            print(
                f"   {CYAN}{dog_name.title()}{RESET}: "
                f"SHARP moved {from_label} {YELLOW}{from_price:.2f}{RESET} → "
                f"{sharp_label} {RED}{sharp_now:.2f}{RESET} "
                f"({sharp_change*100:+.1f}%)",
                flush=True,
            )
            print(
                f"   {GREEN}BACK {soft_label} @ {soft_now:.2f}{RESET} "
                f"(still {soft_edge*100:+.1f}% above the sharp's new level)",
                flush=True,
            )
            notify(title, msg)

            if self.log_writer is not None:
                self.log_writer.writerow({
                    "alerted_at_utc": now.strftime("%Y-%m-%d %H:%M:%S"),
                    "race_time_uk": race.dt_uk.strftime("%Y-%m-%d %H:%M"),
                    "track": track,
                    "dog_name": dog_name.title(),
                    "sharp_book": sharp_label,
                    "sharp_from": f"{from_price:.2f}",
                    "sharp_to": f"{sharp_now:.2f}",
                    "sharp_pct": f"{sharp_change:.4f}",
                    "soft_book": soft_label,
                    "soft_price": f"{soft_now:.2f}",
                    "soft_vs_sharp_pct": f"{soft_edge:.4f}",
                })
                self.log_f.flush()

    def run(self) -> None:
        while not self._stop:
            now = datetime.now(UTC)
            self._maybe_refresh(now)
            active = self._active_races(now)
            if active:
                print(
                    f"\n[{now.strftime('%H:%M:%S')}] polling {len(active)} race(s) "
                    f"within {int(self.pre_off.total_seconds()/60)}m of off",
                    flush=True,
                )
                for race in active:
                    if self._stop:
                        break
                    self._poll_race(race, datetime.now(UTC))
            else:
                upcoming = sorted(
                    (r for r in self._known_races.values() if r.dt_utc > now),
                    key=lambda r: r.dt_utc,
                )
                if upcoming:
                    next_r = upcoming[0]
                    delta = next_r.dt_utc - now
                    print(
                        f"[{now.strftime('%H:%M:%S')}] no race within window; "
                        f"next at {next_r.dt_uk.strftime('%H:%M UK')} "
                        f"({int(delta.total_seconds()/60)}m)",
                        flush=True,
                    )
            slept = 0
            while slept < self.args.interval and not self._stop:
                time.sleep(min(3, self.args.interval - slept))
                slept += 3
        if self.log_f:
            self.log_f.close()


def parse_duration(s: str) -> int:
    m = re.match(r"^(\d+)([hms]?)$", s.strip().lower())
    if not m:
        raise argparse.ArgumentTypeError(f"Bad duration: {s}")
    n = int(m.group(1))
    return n * {"": 1, "s": 1, "m": 60, "h": 3600}[m.group(2)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pre-off", dest="pre_off_seconds", type=parse_duration,
                    default="20m",
                    help="Only watch races within this window of off (default 20m).")
    ap.add_argument("--interval", type=int, default=30,
                    help="Seconds between full poll cycles (default 30).")
    ap.add_argument("--sharp", default="bet365",
                    help="Comma-separated sharp bookmaker(s) (default: bet365).")
    ap.add_argument("--soft",
                    default="coral,ladbrokes,boylesports,betfred,williamhill,unibet,betvictor,skybet,paddypower",
                    help="Comma-separated soft bookmaker(s).")
    ap.add_argument("--sharp-pct", type=float, default=0.10,
                    help="Alert when sharp price shortens by >= this (default 0.10 = 10%%).")
    ap.add_argument("--sharp-window", dest="sharp_window_seconds",
                    type=parse_duration, default="5m",
                    help="Time window for the sharp drop calculation (default 5m).")
    ap.add_argument("--soft-edge", type=float, default=0.10,
                    help="Require soft best price >= sharp * (1 + this) "
                         "(default 0.10 = soft still 10%% above sharp).")
    ap.add_argument("--min-price", type=float, default=2.0,
                    help="Ignore steams on prices below this (default 2.0).")
    ap.add_argument("--log", default=None,
                    help="Optional CSV log of alerts.")
    args = ap.parse_args()
    LateSteamerMonitor(args).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
