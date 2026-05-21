#!/usr/bin/env python3
"""Fetch UK greyhound race cards from Betfair on a UK-IP machine.

Standalone — install one dependency and run. Output is a directory of
JSON meeting files plus a zip you can upload back to the project.

    pip3 install betfairlightweight

    python3 fetch_cards_betfair.py \\
        --username BrassMonki \\
        --password 'Gerrard.08' \\
        --app-key  w0gkke7i7bcJ2Ysc \\
        --date     tomorrow \\
        --out      cards_tomorrow

Then zip and upload:
    zip -r cards_tomorrow.zip cards_tomorrow

Args
----
  --date YYYY-MM-DD | today | tomorrow   (UK local race date)
  --with-prices                          also fetch best-back live prices
  --out DIR                              output directory (created)
  --secrets PATH                         alternative: a YAML file with
                                         {betfair: {username, password, app_key}}
                                         (so you don't put creds on the cmdline)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zoneinfo
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

UK = zoneinfo.ZoneInfo("Europe/London")
UTC = timezone.utc
GREYHOUND_EVENT_TYPE_ID = "4339"
TRAP_RE = re.compile(r"^\s*(\d)\s*[\.\-:)]\s*(.+)$")
MARKET_NAME_RE = re.compile(r"^[RAS]?(\d+)\s+(\d+)m\s*(.*)$")


def parse_market_meta(name: str):
    if not name:
        return None, None, None
    m = MARKET_NAME_RE.match(name.strip())
    if m:
        return int(m.group(1)), int(m.group(2)), (m.group(3) or None) or None
    d = re.search(r"(\d+)m", name)
    n = re.search(r"R(?:ace)?\s*(\d+)", name, re.IGNORECASE)
    return (int(n.group(1)) if n else None,
            int(d.group(1)) if d else None,
            None)


def strip_trap(name: str):
    if not name:
        return "", None
    m = TRAP_RE.match(name)
    if m:
        return m.group(2).strip(), int(m.group(1))
    return name.strip(), None


def parse_target_date(raw: str) -> date:
    today_uk = datetime.now(UK).date()
    if raw == "today":
        return today_uk
    if raw == "tomorrow":
        return today_uk + timedelta(days=1)
    return datetime.strptime(raw, "%Y-%m-%d").date()


def load_secrets_yaml(path: Path) -> tuple[str, str, str]:
    """Minimal YAML parser — only needs username/password/app_key fields."""
    text = path.read_text(encoding="utf-8")
    u = p = k = None
    section = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        stripped = line.lstrip()
        if not line.startswith(" ") and stripped.endswith(":"):
            section = stripped[:-1].strip()
            continue
        if ":" in stripped and section == "betfair":
            k_name, _, v = stripped.partition(":")
            v = v.strip().strip('"').strip("'")
            if k_name.strip() == "username": u = v
            elif k_name.strip() == "password": p = v
            elif k_name.strip() == "app_key":  k = v
    if not (u and p and k):
        raise SystemExit(f"{path}: missing username/password/app_key")
    return u, p, k


def login(username: str, password: str, app_key: str):
    import betfairlightweight
    client = betfairlightweight.APIClient(
        username=username, password=password, app_key=app_key,
    )
    client.login_interactive()
    return client


def market_to_payload(market) -> dict:
    race_num, distance_m, _ = parse_market_meta(market.market_name)
    track_raw = market.event.name if market.event else ""
    start_dt = market.market_start_time  # UTC
    local = start_dt.astimezone(UK)
    race_date_str = local.strftime("%d/%m/%Y")
    race_time_str = local.strftime("%H:%M:%S")

    traps = []
    for runner in market.runners:
        bare, trap_num = strip_trap(runner.runner_name)
        traps.append({
            "trapNumber": str(trap_num) if trap_num is not None else "",
            "trapHandicap": None,
            # Real GBGB dog_id will be resolved on the server side by
            # matching the dog NAME via bf_safe_name. Use the Betfair
            # selection_id as a placeholder so the meeting JSON is valid.
            "dogId": f"bf:{runner.selection_id}",
            "dogName": bare,
            "trainerName": "",
            "SP": "",
            "resultPriceNumerator": None,
            "resultPriceDenominator": None,
            "resultPosition": None,
            "resultRunTime": None,
            "resultSectionalTime": None,
            "resultDogWeight": None,
            "resultComment": "",
            "_betfair_selection_id": runner.selection_id,
        })

    return {
        "meetingId": market.market_id,
        "meetingDate": race_date_str,
        "trackName": track_raw,
        "_betfair_market_id": market.market_id,
        "races": [{
            "raceId": str(market.market_id),
            "raceNumber": str(race_num) if race_num else "",
            "raceDate": race_date_str,
            "raceTime": race_time_str,
            "raceClass": "",
            "raceDistance": distance_m or 0,
            "raceHandicap": False,
            "raceGoing": "0",
            "traps": traps,
        }],
    }


def fetch_markets(client, target: date) -> list:
    from betfairlightweight import filters
    start_uk = datetime.combine(target, time(0, 0), tzinfo=UK).astimezone(UTC)
    end_uk = start_uk + timedelta(days=1)
    mfilter = filters.market_filter(
        event_type_ids=[GREYHOUND_EVENT_TYPE_ID],
        market_type_codes=["WIN"],
        market_countries=["GB"],
        market_start_time={
            "from": start_uk.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "to":   end_uk.strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    )
    return client.betting.list_market_catalogue(
        filter=mfilter,
        max_results=1000,
        market_projection=[
            "MARKET_START_TIME", "RUNNER_DESCRIPTION", "EVENT", "MARKET_DESCRIPTION",
        ],
    )


def fetch_prices(client, market_ids: list[str]) -> dict:
    from betfairlightweight import filters
    out = {}
    for i in range(0, len(market_ids), 25):
        batch = market_ids[i:i + 25]
        books = client.betting.list_market_book(
            market_ids=batch,
            price_projection=filters.price_projection(
                price_data=["EX_BEST_OFFERS"],
                ex_best_offers_overrides=filters.ex_best_offers_overrides(
                    best_prices_depth=1,
                ),
            ),
        )
        for b in books:
            d = {}
            for r in b.runners:
                if r.ex and r.ex.available_to_back:
                    d[r.selection_id] = r.ex.available_to_back[0].price
            out[b.market_id] = d
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--username")
    ap.add_argument("--password")
    ap.add_argument("--app-key")
    ap.add_argument("--secrets", help="YAML file with betfair.username / .password / .app_key")
    ap.add_argument("--date", default="tomorrow", help="YYYY-MM-DD or today/tomorrow")
    ap.add_argument("--out", default="cards_out", help="Output directory")
    ap.add_argument("--with-prices", action="store_true")
    args = ap.parse_args()

    if args.secrets:
        u, p, k = load_secrets_yaml(Path(args.secrets))
    else:
        u, p, k = args.username, args.password, args.app_key
        if not (u and p and k):
            print("Need --username + --password + --app-key, or --secrets path.", file=sys.stderr)
            return 2

    target = parse_target_date(args.date)
    print(f"Fetching UK greyhound win markets for {target} (UK local)…")
    client = login(u, p, k)
    try:
        markets = fetch_markets(client, target)
        print(f"Got {len(markets)} markets")
        if not markets:
            print("No markets returned. Betfair may not have published "
                  "tomorrow's cards yet — try again later in the day.")
            return 0

        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        for m in markets:
            payload = market_to_payload(m)
            fname = f"bf_{m.market_id.replace('.', '_')}.json"
            (out_dir / fname).write_text(json.dumps([payload], indent=2, default=str), encoding="utf-8")
        print(f"Wrote {len(markets)} meeting JSONs to {out_dir}/")

        if args.with_prices:
            prices = fetch_prices(client, [m.market_id for m in markets])
            (out_dir / "live_prices.json").write_text(
                json.dumps(prices, indent=2), encoding="utf-8",
            )
            print(f"Wrote live_prices.json")

        print()
        print(f"Next: zip the directory and upload it.")
        print(f"    cd {out_dir.parent}  &&  zip -r {out_dir.name}.zip {out_dir.name}")
        return 0
    finally:
        try:
            client.logout()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
