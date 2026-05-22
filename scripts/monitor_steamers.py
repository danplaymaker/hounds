#!/usr/bin/env python3
"""monitor_steamers.py — watch all UK greyhound markets for price movers.

Polls Oddschecker every --interval seconds and tracks the best bookmaker
price for every runner. When a dog's best price shortens by ≥ --pct over
the last --window minutes, fire an alert (sound + terminal + CSV log).

That's a 'steamer' — money is being backed on the dog and books are
moving in response. It's smart-money signal completely independent of
our model. Combine it with the model's standouts list and the value
monitor for a triangulated picture: model likes the dog AND market is
moving on it AND a bookie is still offering above fair = strong play.

USAGE
-----
    python3 monitor_steamers.py \\
        --interval     60                   # poll every 60s
        --pct          0.15                 # alert when price drops 15%
        --window       10m                  # over a 10-minute window
        --min-price    2.0                  # ignore moves under 2.0 (noise)
        --watch-before 3h                   # only races within next 3h
        --books        bet365,williamhill,paddypower,skybet,ladbrokes,coral
        --log          ~/Downloads/steamers.csv

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

# UK GBGB-licensed tracks. Any race URL we see at a non-UK track gets ignored.
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

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7_2) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.6 Safari/605.1.15"
)

# ANSI colours
RESET = "\033[0m"
RED = "\033[91m"      # steamer = shortening = "going red hot"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"


def play_sound(name: str = "Hero") -> None:
    try:
        subprocess.run(
            ["afplay", f"/System/Library/Sounds/{name}.aiff"],
            check=False, timeout=5,
        )
    except Exception:
        pass


def notify(title: str, message: str, sound: str = "Hero") -> None:
    play_sound(sound)
    try:
        subprocess.run(
            ["terminal-notifier", "-title", title, "-message", message,
             "-sound", sound],
            check=False, timeout=5,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
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


# -------------------------------------------------------- parsing helpers

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
    slug: str        # central-park, hove, ...
    dt_uk: datetime
    dt_utc: datetime


_RACE_URL_RE = re.compile(
    r"/greyhounds/(\d{4}-\d{2}-\d{2})-([a-z][a-z0-9-]+)/(\d{2}:\d{2})/winner"
)


def discover_races(session) -> list[RaceRef]:
    """Pull /greyhounds and extract upcoming UK race refs."""
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
            dt_uk = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
            dt_uk = dt_uk.replace(tzinfo=UK)
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


# -------------------------------------------------------- monitor

@dataclass
class PriceSample:
    t: datetime
    best_book: str
    best_price: float


class SteamerMonitor:
    def __init__(self, args: argparse.Namespace):
        import cloudscraper
        self.session = cloudscraper.create_scraper(browser={"custom": UA})
        self.args = args
        self.want_books = {
            BOOKMAKER_CODES.get(b.strip().lower(), b.strip().upper())
            for b in args.books.split(",") if b.strip()
        }
        self.window = timedelta(seconds=args.window_seconds)
        self.watch_before = timedelta(seconds=args.watch_before_seconds)
        # In-memory price history: {(race_url, dog_name): deque[PriceSample]}
        self._history: dict[tuple[str, str], deque[PriceSample]] = defaultdict(
            lambda: deque(maxlen=200)
        )
        # Per-(race,dog) cooldown so we don't re-alert too often.
        self._last_alert: dict[tuple[str, str], datetime] = {}
        self._known_races: dict[str, RaceRef] = {}
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
                    "best_book", "from_price", "to_price",
                    "pct_change", "window_seconds",
                ],
            )
            if new:
                self.log_writer.writeheader()
                self.log_f.flush()
        self._stop = False
        signal.signal(signal.SIGINT, self._sigint)
        print(f"Tracking bookmakers: {sorted(self.want_books)}", flush=True)
        print(
            f"Steamer rule: best price drops ≥ {args.pct*100:.0f}% over "
            f"{args.window_seconds//60}m (min start price {args.min_price})",
            flush=True,
        )

    def _sigint(self, *_):
        print("\nStopping…", flush=True)
        self._stop = True

    def _maybe_refresh_races(self, now: datetime) -> None:
        """Re-discover the race list every 5 min."""
        if not hasattr(self, "_last_discover"):
            self._last_discover = None
        if (self._last_discover is None
                or (now - self._last_discover).total_seconds() >= 300):
            races = discover_races(self.session)
            for r in races:
                self._known_races[r.url] = r
            self._last_discover = now
            print(f"[{now.strftime('%H:%M:%S')}] discovered {len(self._known_races)} UK races", flush=True)

    def _active_races(self, now: datetime) -> list[RaceRef]:
        out: list[RaceRef] = []
        for r in self._known_races.values():
            if now + self.watch_before < r.dt_utc:
                continue
            if now > r.dt_utc + timedelta(minutes=5):
                continue
            out.append(r)
        return sorted(out, key=lambda r: r.dt_utc)

    def _poll_race(self, race: RaceRef, now: datetime) -> None:
        try:
            resp = self.session.get(race.url, timeout=15)
        except Exception as e:
            print(f"  {race.url}: {e}", file=sys.stderr)
            return
        if resp.status_code != 200:
            return
        prices_by_dog = parse_prices(resp.text, self.want_books)
        if not prices_by_dog:
            return

        for dog_name, prices in prices_by_dog.items():
            if not prices:
                continue
            best_book, best_price = max(prices.items(), key=lambda kv: kv[1])
            if best_price < self.args.min_price:
                continue
            key = (race.url, dog_name)
            history = self._history[key]
            history.append(PriceSample(now, best_book, best_price))

            # Find the oldest sample within window
            cutoff = now - self.window
            old: PriceSample | None = None
            for s in history:
                if s.t >= cutoff:
                    old = s
                    break
            if old is None or old is history[-1]:
                continue
            # Steamer = price has SHORTENED, so to_price < from_price
            change = (best_price - old.best_price) / old.best_price
            if change > -self.args.pct:
                continue

            # Cooldown — at least 5 minutes between alerts on same dog
            last = self._last_alert.get(key)
            if last is not None and (now - last).total_seconds() < 300:
                continue
            self._last_alert[key] = now

            track = race.slug.replace("-", " ")
            title = (
                f"STEAMER — {race.dt_uk.strftime('%H:%M')} {track}"
            )
            msg = (
                f"{dog_name.title()}: {old.best_book} {old.best_price:.2f} → "
                f"{best_book} {best_price:.2f}  "
                f"({change*100:+.1f}% over "
                f"{int((now - old.t).total_seconds()/60)}m)"
            )
            sys.stdout.write("\a")
            print(f"{BOLD}{RED}🔥 {title}{RESET}  {msg}", flush=True)
            notify(title, msg)

            if self.log_writer is not None:
                self.log_writer.writerow({
                    "alerted_at_utc": now.strftime("%Y-%m-%d %H:%M:%S"),
                    "race_time_uk": race.dt_uk.strftime("%Y-%m-%d %H:%M"),
                    "track": track,
                    "dog_name": dog_name.title(),
                    "best_book": best_book,
                    "from_price": f"{old.best_price:.2f}",
                    "to_price": f"{best_price:.2f}",
                    "pct_change": f"{change:.4f}",
                    "window_seconds": int((now - old.t).total_seconds()),
                })
                self.log_f.flush()

    def run(self) -> None:
        while not self._stop:
            now = datetime.now(UTC)
            self._maybe_refresh_races(now)
            active = self._active_races(now)
            if not active:
                upcoming = sorted(self._known_races.values(),
                                  key=lambda r: r.dt_utc)
                next_r = next((r for r in upcoming if r.dt_utc > now), None)
                if next_r:
                    delta = next_r.dt_utc - now
                    print(
                        f"[{now.strftime('%H:%M:%S')}] no races in window; "
                        f"next at {next_r.dt_uk.strftime('%H:%M UK')} "
                        f"({int(delta.total_seconds()/60)}m)",
                        flush=True,
                    )
                else:
                    print("No upcoming UK races known; sleeping.", flush=True)
            else:
                print(
                    f"\n[{now.strftime('%H:%M:%S UTC')}] polling {len(active)} race(s)…",
                    flush=True,
                )
                for race in active:
                    if self._stop:
                        break
                    self._poll_race(race, datetime.now(UTC))
            slept = 0
            while slept < self.args.interval and not self._stop:
                time.sleep(min(5, self.args.interval - slept))
                slept += 5
        if self.log_f:
            self.log_f.close()


def parse_duration(s: str) -> int:
    m = re.match(r"^(\d+)([hms]?)$", s.strip().lower())
    if not m:
        raise argparse.ArgumentTypeError(f"Bad duration: {s}")
    n = int(m.group(1))
    unit = m.group(2)
    return n * {"": 1, "s": 1, "m": 60, "h": 3600}[unit]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interval", type=int, default=60,
                    help="Seconds between full poll cycles (default 60).")
    ap.add_argument("--pct", type=float, default=0.15,
                    help="Alert when best price drops by ≥ this fraction "
                         "(default 0.15 = 15%%).")
    ap.add_argument("--window", dest="window_seconds", type=parse_duration,
                    default="10m",
                    help="Time window for the drop calculation (default 10m).")
    ap.add_argument("--min-price", type=float, default=2.0,
                    help="Ignore moves on prices below this (default 2.0). "
                         "Stops noisy alerts on heavy odds-on favourites.")
    ap.add_argument("--watch-before", dest="watch_before_seconds",
                    type=parse_duration, default="3h",
                    help="How long before each race to start tracking it "
                         "(default 3h).")
    ap.add_argument("--books",
                    default="bet365,williamhill,paddypower,skybet,ladbrokes,coral,betfred,boylesports,unibet,betvictor")
    ap.add_argument("--log", default=None,
                    help="Optional CSV log of steamer alerts.")
    args = ap.parse_args()
    SteamerMonitor(args).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
