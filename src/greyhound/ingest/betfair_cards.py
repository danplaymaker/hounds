"""Pull tomorrow's (or any given day's) UK greyhound race cards from the
Betfair Exchange API and emit them in our canonical meeting-JSON shape.

Output goes to data/raw/gbgb/meeting/<betfair_event_id>.json so the
existing parser/feature pipeline can consume it without changes. Names
get matched back to GBGB dog_ids via bf_safe_name when we have history
for the dog; otherwise the dog gets a synthetic ID prefixed `bf:` and
the model will treat it as a debutant (no prior history).

Run:
    python -m greyhound.ingest.betfair_cards --date 2026-05-22
    python -m greyhound.ingest.betfair_cards --date today
    python -m greyhound.ingest.betfair_cards --date tomorrow

Requires config/secrets.yaml with Betfair username, password, app_key.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import yaml

from greyhound.data.identity import bf_safe_name, canonical_track
from greyhound.data.schemas import load_config

log = logging.getLogger(__name__)

_GREYHOUND_EVENT_TYPE_ID = "4339"   # Betfair greyhound racing
_UK_TZ = ZoneInfo("Europe/London")


def load_secrets(path: str | Path = "config/secrets.yaml") -> dict:
    """Read the gitignored secrets file. Fails loud if missing keys."""
    p = Path(path)
    if not p.exists():
        raise SystemExit(
            f"Missing {p}. Copy config/secrets.example.yaml to config/secrets.yaml "
            "and fill in your Betfair credentials."
        )
    with open(p) as f:
        secrets = yaml.safe_load(f) or {}
    bf = (secrets.get("betfair") or {})
    for k in ("username", "password", "app_key"):
        if not bf.get(k):
            raise SystemExit(f"config/secrets.yaml: betfair.{k} is empty")
    return bf


def parse_target_date(raw: str) -> date:
    today_uk = datetime.now(_UK_TZ).date()
    if raw == "today":
        return today_uk
    if raw == "tomorrow":
        return today_uk + timedelta(days=1)
    return datetime.strptime(raw, "%Y-%m-%d").date()


# -------------------------------------------------------- Betfair client

def login(bf_secrets: dict):
    """Authenticate and return a betfairlightweight APIClient."""
    import betfairlightweight  # local import — heavyweight at module level

    method = bf_secrets.get("login_method", "interactive")
    client = betfairlightweight.APIClient(
        username=bf_secrets["username"],
        password=bf_secrets["password"],
        app_key=bf_secrets["app_key"],
    )
    if method == "interactive":
        client.login_interactive()
    else:
        client.login()
    return client


# -------------------------------------------------- market-name parsing

# Market names look like "R5 380m" or "A4 380m" or "R6 380m H/Cap".
_MARKET_NAME_RE = re.compile(r"^[RAS]?(\d+)\s+(\d+)m\s*(.*)$")


def parse_market_meta(market_name: str) -> tuple[int | None, int | None, str | None]:
    """(race_number, distance_m, suffix) — best-effort."""
    if not market_name:
        return None, None, None
    m = _MARKET_NAME_RE.match(market_name.strip())
    if not m:
        # Sometimes the name is just "Race 5 380m" or similar
        d = re.search(r"(\d+)m", market_name)
        n = re.search(r"R(?:ace)?\s*(\d+)", market_name, re.IGNORECASE)
        return (int(n.group(1)) if n else None,
                int(d.group(1)) if d else None,
                None)
    return int(m.group(1)), int(m.group(2)), (m.group(3) or None) or None


# Selection names look like "1. Some Dog" — same as PROMO BSP files.
_TRAP_RE = re.compile(r"^\s*(\d)\s*[\.\-:)]\s*(.+)$")


def strip_trap(selection_name: str) -> tuple[str, int | None]:
    if not selection_name:
        return "", None
    m = _TRAP_RE.match(selection_name)
    if m:
        return m.group(2).strip(), int(m.group(1))
    return selection_name.strip(), None


# ----------------------------------------------------- dog-ID resolution

def build_dog_lookup(runs_path: Path) -> dict[str, str]:
    """Map bf_safe_name -> dog_id from the most recent appearance.

    A safe_name can collide across dogs over time (rare). We take the
    MOST RECENT runner with that name, which gives us the right dog when
    a previously-retired name has been reused.
    """
    if not runs_path.exists():
        log.warning("Runs file not found: %s — every Betfair dog will be a debutant", runs_path)
        return {}
    runs = pl.read_parquet(runs_path).select(["dog_id", "bf_safe_name", "race_datetime"])
    runs = (
        runs
        .filter(pl.col("bf_safe_name") != "")
        .sort("race_datetime", descending=True)
        .unique(subset=["bf_safe_name"], keep="first", maintain_order=True)
    )
    return dict(zip(runs["bf_safe_name"].to_list(), runs["dog_id"].to_list()))


# ------------------------------------------------------ pull tomorrow

def fetch_markets(client, target: date) -> list:
    """List all UK greyhound win markets running on `target` (UK local date)."""
    import betfairlightweight
    from betfairlightweight import filters

    start_uk = datetime.combine(target, time(0, 0), tzinfo=_UK_TZ).astimezone(UTC)
    end_uk = start_uk + timedelta(days=1)

    mfilter = filters.market_filter(
        event_type_ids=[_GREYHOUND_EVENT_TYPE_ID],
        market_type_codes=["WIN"],
        market_countries=["GB"],
        market_start_time={
            "from": start_uk.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "to":   end_uk.strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    )
    markets = client.betting.list_market_catalogue(
        filter=mfilter,
        max_results=1000,
        market_projection=[
            "MARKET_START_TIME",
            "RUNNER_DESCRIPTION",
            "EVENT",
            "MARKET_DESCRIPTION",
        ],
    )
    return markets


def fetch_live_prices(client, market_ids: list[str]) -> dict[str, dict[int, float]]:
    """Best-back price for each (market_id, selection_id). Optional — gives
    the manual workflow something to compare to immediately."""
    from betfairlightweight import filters
    out: dict[str, dict[int, float]] = {}
    # Betfair allows up to 25 market_ids per request
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
            sel_to_price: dict[int, float] = {}
            for r in b.runners:
                if r.ex and r.ex.available_to_back:
                    sel_to_price[r.selection_id] = r.ex.available_to_back[0].price
            out[b.market_id] = sel_to_price
    return out


# ---------------------------------- shape Betfair output into GBGB meeting JSON

def market_to_meeting_payload(market, dog_lookup: dict[str, str]) -> dict:
    """Convert one Betfair market into a one-race 'meeting' payload that
    parse_meeting_json can consume. We use the market_id as the synthetic
    meeting/race ID."""
    race_num, distance_m, _ = parse_market_meta(market.market_name)
    track_raw = market.event.name if market.event else ""
    track = canonical_track(track_raw) or track_raw
    start_dt = market.market_start_time  # already UTC
    # Reformat the UTC datetime to match the GBGB parser's expectations
    # (it expects local UK time as "HH:MM:SS" and a "DD/MM/YYYY" date).
    local = start_dt.astimezone(_UK_TZ)
    race_date_str = local.strftime("%d/%m/%Y")
    race_time_str = local.strftime("%H:%M:%S")

    traps = []
    for runner in market.runners:
        name, trap_num = strip_trap(runner.runner_name)
        safe = bf_safe_name(name)
        dog_id = dog_lookup.get(safe) or f"bf:{runner.selection_id}"
        traps.append({
            "trapNumber": str(trap_num) if trap_num is not None else "",
            "trapHandicap": None,
            "dogId": dog_id,
            "dogName": name,
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
        "meetingId": int(market.market_id.replace(".", "").replace("1", "", 1)) if market.market_id.startswith("1.") else market.market_id,
        "meetingDate": race_date_str,
        "trackName": track_raw,
        "_canonical_track": track,
        "_betfair_market_id": market.market_id,
        "races": [{
            "raceId": str(market.market_id),
            "raceNumber": str(race_num) if race_num else "",
            "raceDate": race_date_str,
            "raceTime": race_time_str,
            "raceClass": "",          # Betfair doesn't expose grade reliably
            "raceDistance": distance_m or 0,
            "raceHandicap": False,
            "raceGoing": "0",         # not known pre-race; assume neutral
            "traps": traps,
        }],
    }


def write_meeting_jsons(cfg, payloads: list[dict], target: date) -> list[Path]:
    out_dir = Path(cfg.ingest.gbgb.cache_dir) / "meeting"
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for p in payloads:
        mid = p["_betfair_market_id"].replace(".", "_")
        out = out_dir / f"bf_{mid}.json"
        out.write_text(json.dumps([p], indent=2, default=str), encoding="utf-8")
        written.append(out)
    return written


# --------------------------------------------------------------- main

def main() -> None:
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--secrets", default="config/secrets.yaml")
    ap.add_argument(
        "--date", default="tomorrow",
        help="UK-local race date (YYYY-MM-DD, or 'today' / 'tomorrow').",
    )
    ap.add_argument(
        "--with-prices", action="store_true",
        help="Also fetch current best-back prices and dump to data/processed/live_prices.csv",
    )
    args = ap.parse_args()
    cfg = load_config(args.config)
    secrets = load_secrets(args.secrets)
    target = parse_target_date(args.date)
    log.info("Fetching UK greyhound win markets for %s (UK local)", target)

    client = login(secrets)
    try:
        markets = fetch_markets(client, target)
        log.info("Got %d markets from Betfair", len(markets))
        if not markets:
            log.warning("No markets returned. Betfair may not have published "
                        "tomorrow's cards yet — try again later.")
            return

        runs_path = cfg.paths.interim_dir / "gbgb_runs.parquet"
        dog_lookup = build_dog_lookup(runs_path)
        log.info("Dog name lookup loaded: %d unique safe_names", len(dog_lookup))

        payloads = [market_to_meeting_payload(m, dog_lookup) for m in markets]
        unknown_track = [p for p in payloads if p["_canonical_track"] in (None, "")]
        if unknown_track:
            log.warning("%d markets had an unresolved track name", len(unknown_track))

        written = write_meeting_jsons(cfg, payloads, target)
        log.info("Wrote %d meeting JSONs", len(written))

        if args.with_prices:
            mids = [m.market_id for m in markets]
            prices = fetch_live_prices(client, mids)
            rows = []
            for p in payloads:
                mid = p["_betfair_market_id"]
                for trap in p["races"][0]["traps"]:
                    rows.append({
                        "market_id": mid,
                        "track": p["_canonical_track"],
                        "race_time_utc": p["races"][0]["raceTime"],
                        "trap": trap["trapNumber"],
                        "dog_name": trap["dogName"],
                        "dog_id": trap["dogId"],
                        "live_back_price": prices.get(mid, {}).get(trap["_betfair_selection_id"]),
                    })
            df = pl.DataFrame(rows)
            out = cfg.paths.processed_dir / f"live_prices_{target.isoformat()}.csv"
            df.write_csv(out)
            log.info("Wrote %d live-price rows -> %s", df.height, out)

    finally:
        try:
            client.logout()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
