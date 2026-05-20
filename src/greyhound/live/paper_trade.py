"""Paper-trading harness (A5).

Stub for daily paper trading. Pulls today's cards, snapshots Betfair prices
at scheduled times before each race, runs the trained model, logs would-be
bets. Reconciliation against BSP runs after the meeting.

The Betfair API portion requires credentials and live access. Left as a
clear extension point — the orchestration shape is in place.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime

import polars as pl

from greyhound.data.schemas import Config, load_config

log = logging.getLogger(__name__)


def fetch_today_cards(today: date) -> pl.DataFrame:
    """Pull today's race cards from GBGB. Stub returns an empty frame.

    Real implementation: call gbgb scraper with today's date and parse
    the *cards* (not results) page — different URL pattern.
    """
    log.info("fetch_today_cards: stub for %s", today)
    return pl.DataFrame()


def snapshot_betfair_prices(race_market_id: str, when: datetime) -> dict[str, float]:
    """Return {selection_id: best_back_price} at `when`. Stub."""
    log.info("snapshot_betfair_prices: stub %s @ %s", race_market_id, when)
    return {}


def run_once(cfg: Config) -> None:
    today = datetime.now(tz=UTC).date()
    cards = fetch_today_cards(today)
    log.info("Cards for %s: %d rows", today, cards.height)
    # For each race and each scheduled offset, snapshot prices and score.
    # Detailed implementation deferred — depends on the trained-model
    # artifact loader (CalibratedModel.load) and a live cards parser.


def main() -> None:
    import argparse
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_once(cfg)


if __name__ == "__main__":
    main()
