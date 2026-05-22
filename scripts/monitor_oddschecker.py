#!/usr/bin/env python3
"""Monitor Oddschecker for value bets on the model's daily standouts.

Polls each upcoming race every N minutes and compares the best bookmaker
price to the model's fair_odds. Fires a macOS notification (+ terminal
bell + log entry) when a bookmaker is offering above the model's price.

USAGE
-----
    python3 monitor_oddschecker.py \\
        --standouts ~/Downloads/standouts_20260522.csv \\
        --interval  300            # poll every 5 minutes
        --edge      0.10           # alert when offered >= fair * 1.10
        --watch-before 6h          # start watching each race 6h before off
        --books     bet365,skybet,paddypower,williamhill,coral,ladbrokes \\
        --log       ~/Downloads/value_alerts.csv

Press Ctrl-C to stop. The script exits cleanly when every race has gone off.

ALERT MECHANISMS (all enabled by default; toggle with --notify)
  - macOS notification (sound + banner)
  - Terminal bell + colored print
  - CSV log row written immediately (so you can grep / refresh in Numbers)

DESIGN
  - Polls Oddschecker once per upcoming race, every --interval seconds.
  - Per-race cool-down: a price improvement of < --reraise-tick triggers
    no new alert. Default reraise tick is 0.20 (decimal odds points).
  - Stops polling a race 5 min after its off time.

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
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Match the canonical track -> slug mapping in fetch_oddschecker.py
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

BOOKMAKER_CODES = {
    "bet365": "B3", "williamhill": "WH", "paddypower": "PP", "skybet": "SX",
    "ladbrokes": "LD", "coral": "CE", "betfred": "FR", "boylesports": "BR",
    "unibet": "UN", "betvictor": "BE", "betfair": "BF", "888sport": "EE",
    "betway": "BY",
}

UK = zoneinfo.ZoneInfo("Europe/London")
UTC = timezone.utc
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7_2) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.6 Safari/605.1.15"
)


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


def make_session():
    import cloudscraper
    return cloudscraper.create_scraper(browser={"custom": UA})


def parse_iso(s: str) -> datetime:
    s = s.replace("Z", "+00:00")
    m = re.search(r"([+\-])(\d{2})(\d{2})$", s)
    if m:
        s = s[:m.start()] + f"{m.group(1)}{m.group(2)}:{m.group(3)}"
    return datetime.fromisoformat(s)


def build_url(track_slug: str, race_dt_uk: datetime) -> str:
    date_part = race_dt_uk.strftime("%Y-%m-%d")
    time_part = race_dt_uk.strftime("%H:%M")
    return f"https://www.oddschecker.com/greyhounds/{date_part}-{track_slug}/{time_part}/winner"


def parse_prices(html: str, want_books: set[str]) -> dict[str, dict[str, float]]:
    """Returns {dog_name_normalized: {book_code: price}}."""
    from selectolax.parser import HTMLParser
    tree = HTMLParser(html)
    out: dict[str, dict[str, float]] = {}
    for row in tree.css("tr[data-bname]"):
        raw = (row.attributes.get("data-bname") or "").strip()
        name = re.sub(r"^\d+\.\s*", "", raw).strip()
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


def mac_notify(title: str, message: str, *, sound: str = "Glass") -> None:
    try:
        # Escape any double quotes
        title_e = title.replace('"', '\\"')
        msg_e = message.replace('"', '\\"')
        subprocess.run(
            ["osascript", "-e",
             f'display notification "{msg_e}" with title "{title_e}" sound name "{sound}"'],
            check=False, timeout=5,
        )
    except Exception:
        pass


# ANSI colours for the terminal
RESET = "\033[0m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"


def fmt_pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


class Monitor:
    def __init__(self, standouts_rows: list[dict], cfg: argparse.Namespace):
        self.cfg = cfg
        self.session = make_session()
        # Bookmaker codes we care about
        self.want_books = {
            BOOKMAKER_CODES.get(b.strip().lower(), b.strip().upper())
            for b in cfg.books.split(",") if b.strip()
        }
        print(f"Watching bookmakers: {sorted(self.want_books)}", flush=True)

        # Build a list of races to monitor and the dogs in each
        # Each race: {url, race_dt_utc, race_dt_uk, track, dogs: [{name, fair, prob, trap}], best_seen: {dog: (bk, price)}}
        races: dict[tuple[str, str], dict] = {}
        for r in standouts_rows:
            slug = TRACK_SLUG.get(r["track"])
            if not slug:
                continue
            try:
                dt_utc = parse_iso(r["race_datetime"])
            except Exception:
                continue
            dt_uk = dt_utc.astimezone(UK)
            key = (r["track"], r["race_datetime"])
            race = races.setdefault(key, {
                "track": r["track"],
                "race_dt_utc": dt_utc,
                "race_dt_uk": dt_uk,
                "url": build_url(slug, dt_uk),
                "dogs": [],
                "last_alert": {},
            })
            try:
                fair = float(r["model_fair_odds"]) if r.get("model_fair_odds") else None
                prob = float(r["model_prob"]) if r.get("model_prob") else None
            except ValueError:
                fair = prob = None
            race["dogs"].append({
                "name": r["dog_name"],
                "name_norm": (r["dog_name"] or "").lower(),
                "trap": r.get("trap", ""),
                "fair": fair,
                "prob": prob,
            })
        self.races = list(races.values())
        self.races.sort(key=lambda r: r["race_dt_utc"])
        print(f"Loaded {len(self.races)} races to monitor.", flush=True)

        # Open log
        self.log_f = None
        self.log_writer = None
        if cfg.log:
            new_file = not Path(cfg.log).exists()
            self.log_f = open(cfg.log, "a", newline="", encoding="utf-8")
            self.log_writer = csv.DictWriter(
                self.log_f,
                fieldnames=[
                    "logged_at_utc", "race_time_uk", "track", "dog_name",
                    "trap", "fair_odds", "best_book", "best_price",
                    "edge_vs_fair", "alerted",
                ],
            )
            if new_file:
                self.log_writer.writeheader()
                self.log_f.flush()

        self._stop = False
        signal.signal(signal.SIGINT, self._sigint)

    def _sigint(self, *_) -> None:
        print("\nStopping… (Ctrl-C)", flush=True)
        self._stop = True

    def _active_races(self, now_utc: datetime) -> list[dict]:
        out: list[dict] = []
        watch_before = timedelta(seconds=self.cfg.watch_before_seconds)
        # Stop polling 5 min after off
        watch_after = timedelta(minutes=5)
        for r in self.races:
            if now_utc + watch_before < r["race_dt_utc"]:
                continue  # too early
            if now_utc > r["race_dt_utc"] + watch_after:
                continue  # raced
            out.append(r)
        return out

    def _poll_race(self, race: dict) -> None:
        now = datetime.now(UTC)
        url = race["url"]
        try:
            resp = self.session.get(url, timeout=20)
        except Exception as e:
            print(f"  {url}: {e}", file=sys.stderr, flush=True)
            return
        if resp.status_code != 200:
            if resp.status_code != 404:  # 404 is normal pre-publication
                print(f"  {url}: HTTP {resp.status_code}", file=sys.stderr, flush=True)
            return

        prices_by_dog = parse_prices(resp.text, self.want_books)
        if not prices_by_dog:
            return  # not yet priced

        for dog in race["dogs"]:
            prices = prices_by_dog.get(dog["name_norm"])
            if not prices or not dog["fair"]:
                continue
            best_book, best_price = max(prices.items(), key=lambda kv: kv[1])
            edge = best_price / dog["fair"] - 1.0

            # Log every observation (whether we alert or not)
            if self.log_writer is not None:
                self.log_writer.writerow({
                    "logged_at_utc": now.strftime("%Y-%m-%d %H:%M:%S"),
                    "race_time_uk": race["race_dt_uk"].strftime("%Y-%m-%d %H:%M"),
                    "track": race["track"],
                    "dog_name": dog["name"],
                    "trap": dog["trap"],
                    "fair_odds": f"{dog['fair']:.2f}",
                    "best_book": best_book,
                    "best_price": f"{best_price:.2f}",
                    "edge_vs_fair": f"{edge:.4f}",
                    "alerted": "yes" if edge >= self.cfg.edge else "no",
                })
                self.log_f.flush()

            if edge < self.cfg.edge:
                continue

            # Don't re-alert unless the price has improved by --reraise-tick
            last = race["last_alert"].get(dog["name_norm"])
            if last is not None and best_price < last + self.cfg.reraise_tick:
                continue
            race["last_alert"][dog["name_norm"]] = best_price

            title = f"VALUE — {race['race_dt_uk'].strftime('%H:%M')} {race['track']}"
            msg = (
                f"{dog['name']} (T{dog['trap']}): {best_book} "
                f"@ {best_price:.2f}  vs fair {dog['fair']:.2f}  ({fmt_pct(edge)})"
            )
            # Terminal alert
            sys.stdout.write("\a")  # bell
            print(f"{BOLD}{GREEN}🎯 {title}{RESET}  {msg}", flush=True)
            mac_notify(title, msg)

    def run(self) -> None:
        while not self._stop:
            now = datetime.now(UTC)
            active = self._active_races(now)
            if not active and now > max(r["race_dt_utc"] for r in self.races):
                print(f"\nAll races have gone off. Exiting.", flush=True)
                break
            if active:
                next_race = min(r["race_dt_utc"] for r in active)
                print(
                    f"\n[{now.strftime('%H:%M:%S UTC')}] {len(active)} race(s) active; "
                    f"next off at {next_race.astimezone(UK).strftime('%H:%M UK')}",
                    flush=True,
                )
                for race in active:
                    if self._stop:
                        break
                    self._poll_race(race)
            else:
                upcoming = [r for r in self.races if r["race_dt_utc"] > now]
                if upcoming:
                    next_dt = min(r["race_dt_utc"] for r in upcoming).astimezone(UK)
                    print(
                        f"[{now.strftime('%H:%M:%S UTC')}] no races in window; "
                        f"next at {next_dt.strftime('%H:%M UK')}",
                        flush=True,
                    )
            # Sleep in 5-second chunks so Ctrl-C is responsive
            slept = 0
            while slept < self.cfg.interval and not self._stop:
                time.sleep(min(5, self.cfg.interval - slept))
                slept += 5

        if self.log_f:
            self.log_f.close()


def parse_duration(s: str) -> int:
    """'6h' -> 21600 (seconds); '30m' -> 1800; '120' -> 120 (seconds)."""
    m = re.match(r"^(\d+)([hms]?)$", s.strip().lower())
    if not m:
        raise argparse.ArgumentTypeError(f"Bad duration: {s}")
    n = int(m.group(1))
    unit = m.group(2)
    return n * {"": 1, "s": 1, "m": 60, "h": 3600}[unit]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--standouts", required=True)
    ap.add_argument("--books",
                    default="bet365,skybet,paddypower,williamhill,coral,ladbrokes,betfred,boylesports,unibet,betvictor")
    ap.add_argument("--interval", type=int, default=300, help="Seconds between polls (default 300 = 5 min).")
    ap.add_argument("--watch-before", dest="watch_before_seconds", type=parse_duration,
                    default="6h",
                    help="How long before each race's off to start polling it (default 6h).")
    ap.add_argument("--edge", type=float, default=0.10,
                    help="Edge threshold to alert (default 0.10 = bookmaker offers >= 10%% above fair).")
    ap.add_argument("--reraise-tick", type=float, default=0.20,
                    help="Min decimal-odds increase to re-alert on the same dog (default 0.20).")
    ap.add_argument("--log", default=None,
                    help="Optional CSV log of every observation (defaults to no log).")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.standouts)))
    if not rows:
        print("No rows in standouts CSV.", file=sys.stderr)
        return 1
    Monitor(rows, args).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
