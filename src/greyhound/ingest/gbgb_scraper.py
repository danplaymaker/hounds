"""GBGB results scraper (A1.1).

NOTE: the exact GBGB results endpoints/HTML structure must be verified
before production runs — `base_url` in the config is a placeholder.
This module implements the *shape* described in the brief: date iteration,
cache-aware fetches, polite delay, exponential backoff, resumability.

Run: `python -m greyhound.ingest.gbgb_scraper --config config/default.yaml`
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, timedelta
from pathlib import Path

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from greyhound.data.schemas import Config, load_config

log = logging.getLogger(__name__)


class GbgbScraper:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.gbgb = cfg.ingest.gbgb
        self.cache = Path(self.gbgb.cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)

    async def fetch_one(self, client: httpx.AsyncClient, url: str, cache_path: Path) -> str:
        """Fetch a URL, caching by file path. Cached hits are O(disk read)."""
        if cache_path.exists():
            return cache_path.read_text(encoding="utf-8")

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.gbgb.backoff.max_attempts),
            wait=wait_exponential(
                multiplier=self.gbgb.backoff.initial_seconds,
                exp_base=self.gbgb.backoff.multiplier,
            ),
            retry=retry_if_exception_type((httpx.HTTPError,)),
            reraise=True,
        ):
            with attempt:
                resp = await client.get(url, timeout=30.0)
                resp.raise_for_status()
                html = resp.text

        cache_path.write_text(html, encoding="utf-8")
        await asyncio.sleep(self.gbgb.request_delay_seconds)
        return html

    async def fetch_date(self, client: httpx.AsyncClient, d: date) -> None:
        """Fetch the results index for date `d` and every race page beneath it.

        Index URL pattern, race URL pattern, and HTML structure are
        placeholders — refine once we've inspected the real site.
        """
        index_url = f"{self.gbgb.base_url}/results/{d.isoformat()}"
        index_path = self.cache / f"{d.year:04d}/{d.month:02d}/{d.day:02d}/_index.html"
        index_html = await self.fetch_one(client, index_url, index_path)

        # Real implementation: parse index_html for race links (track, race_id),
        # fetch each. Left as a tracked TODO until we have a sample page.
        log.info("Fetched %s (%d bytes)", index_url, len(index_html))

    async def run(self) -> None:
        start = self.gbgb.start_date
        end = self.gbgb.end_date or (date.today() - timedelta(days=1))
        days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
        log.info("Scraping %d dates: %s → %s", len(days), start, end)

        limits = httpx.Limits(max_connections=self.gbgb.max_concurrency)
        headers = {"User-Agent": self.gbgb.user_agent}
        async with httpx.AsyncClient(limits=limits, headers=headers) as client:
            sem = asyncio.Semaphore(self.gbgb.max_concurrency)

            async def bound(d: date) -> None:
                async with sem:
                    try:
                        await self.fetch_date(client, d)
                    except Exception as e:
                        log.exception("Failed %s: %s", d, e)

            await asyncio.gather(*(bound(d) for d in days))


def main() -> None:
    import argparse
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    asyncio.run(GbgbScraper(cfg).run())


if __name__ == "__main__":
    main()
