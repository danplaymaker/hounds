"""GBGB results scraper (A1.1).

GBGB exposes a JSON API at https://api.gbgb.org.uk/api. Endpoints used:

  GET /api/results?date=YYYY-MM-DD&page=N
       Paginated runner search for a date. We use it ONLY to enumerate
       distinct meetingIds for the date (the rich per-meeting data comes
       from the meeting endpoint below).

  GET /api/results/meeting/{meetingId}
       Returns the meeting including every race and every runner — the
       canonical source. Note: a `raceId` query param is accepted but
       ignored for our purposes; the response contains every race.

Strategy:
  1. For each date in the configured window, page the runner search to
     collect distinct meetingIds. Cache page responses.
  2. For each meetingId not yet cached, fetch the meeting JSON. Cache it.

Both steps are idempotent — re-runs read from cache and only hit the
network for missing pages. 2-second polite delay between requests is
configurable. Backoff is via tenacity on httpx.HTTPError.
"""

from __future__ import annotations

import asyncio
import json
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
        (self.cache / "by_date").mkdir(parents=True, exist_ok=True)
        (self.cache / "meeting").mkdir(parents=True, exist_ok=True)

    async def fetch_json(
        self,
        client: httpx.AsyncClient,
        url: str,
        cache_path: Path,
    ) -> dict | list | None:
        """Fetch URL with cache + backoff. Returns parsed JSON or None on 404."""
        if cache_path.exists():
            try:
                return json.loads(cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                log.warning("Cache file corrupt, re-fetching: %s", cache_path)
                cache_path.unlink()

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
                if resp.status_code == 404:
                    return None
                resp.raise_for_status()
                text = resp.text

        cache_path.write_text(text, encoding="utf-8")
        await asyncio.sleep(self.gbgb.request_delay_seconds)
        return json.loads(text)

    # --------------------------------------------------------- date paging

    async def enumerate_meetings_for_date(
        self,
        client: httpx.AsyncClient,
        d: date,
    ) -> set[int]:
        """Page through /api/results?date=... collecting distinct meetingIds.

        The endpoint always returns 20 items per page; we read page 1 to
        learn pageCount, then fetch the rest. Each page is cached on disk.
        """
        meeting_ids: set[int] = set()
        first_url = f"{self.gbgb.base_url}/api/results?date={d.isoformat()}&page=1"
        first_path = (
            self.cache / "by_date" / f"{d.year:04d}/{d.month:02d}/{d.day:02d}/page_1.json"
        )
        first = await self.fetch_json(client, first_url, first_path)
        if first is None or not isinstance(first, dict):
            log.info("No results for %s", d)
            return meeting_ids

        page_count = int(first.get("meta", {}).get("pageCount", 1) or 1)
        for item in first.get("items", []):
            mid = item.get("meetingId")
            if mid is not None:
                meeting_ids.add(int(mid))

        for page in range(2, page_count + 1):
            url = f"{self.gbgb.base_url}/api/results?date={d.isoformat()}&page={page}"
            path = (
                self.cache / "by_date" / f"{d.year:04d}/{d.month:02d}/{d.day:02d}/page_{page}.json"
            )
            payload = await self.fetch_json(client, url, path)
            if payload is None:
                break
            for item in payload.get("items", []):
                mid = item.get("meetingId")
                if mid is not None:
                    meeting_ids.add(int(mid))

        log.info("Date %s: %d distinct meetings across %d pages", d, len(meeting_ids), page_count)
        return meeting_ids

    # --------------------------------------------------------- meetings

    async def fetch_meeting(
        self,
        client: httpx.AsyncClient,
        meeting_id: int,
    ) -> None:
        url = f"{self.gbgb.base_url}/api/results/meeting/{meeting_id}"
        path = self.cache / "meeting" / f"{meeting_id}.json"
        await self.fetch_json(client, url, path)

    # --------------------------------------------------------- driver

    async def run(self) -> None:
        start = self.gbgb.start_date
        end = self.gbgb.end_date or (date.today() - timedelta(days=1))
        days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
        log.info("Scraping %d dates: %s → %s", len(days), start, end)

        limits = httpx.Limits(max_connections=self.gbgb.max_concurrency)
        headers = {
            "User-Agent": self.gbgb.user_agent,
            "Accept": "application/json",
        }
        async with httpx.AsyncClient(
            base_url="", limits=limits, headers=headers, follow_redirects=True,
        ) as client:
            for d in days:
                try:
                    meeting_ids = await self.enumerate_meetings_for_date(client, d)
                except Exception as e:
                    log.exception("Failed to enumerate %s: %s", d, e)
                    continue
                for mid in sorted(meeting_ids):
                    try:
                        await self.fetch_meeting(client, mid)
                    except Exception as e:
                        log.exception("Failed meeting %s: %s", mid, e)


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
