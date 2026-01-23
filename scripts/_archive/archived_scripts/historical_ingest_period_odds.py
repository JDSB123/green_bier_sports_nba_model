#!/usr/bin/env python3
"""
Ingest Historical Period Odds (1H, Quarters) via Per-Event API.

The bulk historical endpoint doesn't support period markets, so we need to
query each event individually. This script reads the already-ingested events
and fetches 1H/quarter odds for each one.

Usage:
    # Fetch 1st half markets for both seasons
    python scripts/historical_ingest_period_odds.py --markets 1h

    # Fetch all quarter markets
    python scripts/historical_ingest_period_odds.py --markets quarters

    # Fetch everything (1H + quarters)
    python scripts/historical_ingest_period_odds.py --markets all

    # Specific season only
    python scripts/historical_ingest_period_odds.py --season 2023-2024 --markets 1h

    # Dry run
    python scripts/historical_ingest_period_odds.py --markets 1h --dry-run
"""
from __future__ import annotations
from src.utils.logging import get_logger
from src.utils.historical_guard import resolve_historical_output_root, require_historical_mode, ensure_historical_path
from src.config import settings

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

import httpx

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


logger = get_logger(__name__)


# Market definitions
FIRST_HALF_MARKETS = ["h2h_h1", "spreads_h1", "totals_h1"]
QUARTER_MARKETS = [
    "h2h_q1", "spreads_q1", "totals_q1",
    "h2h_q2", "spreads_q2", "totals_q2",
    "h2h_q3", "spreads_q3", "totals_q3",
    "h2h_q4", "spreads_q4", "totals_q4",
]
ALL_PERIOD_MARKETS = FIRST_HALF_MARKETS + QUARTER_MARKETS


@dataclass
class IngestionStats:
    """Track ingestion statistics."""
    events_processed: int = 0
    events_with_odds: int = 0
    events_failed: int = 0
    api_calls: int = 0
    credits_used: int = 0


class SimpleRateLimiter:
    """Simple async rate limiter."""

    def __init__(self, min_interval: float = 1.0):
        self.min_interval = min_interval
        self._last_request: Optional[float] = None
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        import time
        async with self._lock:
            now = time.monotonic()
            if self._last_request is not None:
                elapsed = now - self._last_request
                if elapsed < self.min_interval:
                    await asyncio.sleep(self.min_interval - elapsed)
            self._last_request = time.monotonic()


class HistoricalPeriodOddsFetcher:
    """Fetches period odds for individual events."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.the-odds-api.com/v4",
        rate_limit: float = 0.5,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.rate_limiter = SimpleRateLimiter(min_interval=rate_limit)
        self.stats = IngestionStats()

    async def fetch_event_odds(
        self,
        event_id: str,
        commence_time: str,
        markets: List[str],
        sport: str = "basketball_nba",
        regions: str = "us",
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch historical odds for a specific event.

        Args:
            event_id: The Odds API event ID
            commence_time: ISO datetime for the snapshot
            markets: List of market keys to fetch
            sport: Sport key
            regions: Regions to query

        Returns:
            Event odds data or None if failed
        """
        await self.rate_limiter.acquire()
        self.stats.api_calls += 1

        # Cost: 10 credits per region per market
        num_regions = len(regions.split(","))
        self.stats.credits_used += 10 * num_regions * len(markets)

        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": ",".join(markets),
            "date": commence_time,
            "oddsFormat": "american",
            "dateFormat": "iso",
        }

        url = f"{self.base_url}/historical/sports/{sport}/events/{event_id}/odds"

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(url, params=params)

                if resp.status_code == 404:
                    # Event not found in historical data
                    logger.debug(
                        f"Event {event_id} not found in historical data")
                    return None

                if resp.status_code == 422:
                    # Invalid request (possibly markets not available)
                    logger.debug(f"Markets not available for event {event_id}")
                    return None

                resp.raise_for_status()
                return resp.json()

        except Exception as e:
            logger.warning(f"Failed to fetch event {event_id}: {e}")
            self.stats.events_failed += 1
            return None


class HistoricalPeriodIngestor:
    """Orchestrates period odds ingestion."""

    def __init__(
        self,
        api_key: str,
        data_dir: str | None = None,
        rate_limit: float = 0.5,
    ):
        self.fetcher = HistoricalPeriodOddsFetcher(
            api_key, rate_limit=rate_limit)
        resolved_dir = Path(
            data_dir) if data_dir else resolve_historical_output_root("the_odds")
        self.data_dir = resolved_dir
        self.events_dir = self.data_dir / "events"
        self.period_odds_dir = self.data_dir / "period_odds"
        self.period_odds_dir.mkdir(parents=True, exist_ok=True)

    def _load_events(self, season: str) -> List[Dict[str, Any]]:
        """Load all events for a season."""
        events = []
        season_dir = self.events_dir / season

        if not season_dir.exists():
            logger.warning(f"No events directory for {season}")
            return events

        for f in sorted(season_dir.glob("events_*.json")):
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                    event_list = data.get("data", []) if isinstance(
                        data, dict) else data
                    events.extend(event_list)
            except Exception as e:
                logger.warning(f"Failed to load {f}: {e}")

        # Deduplicate by event_id
        seen = set()
        unique_events = []
        for event in events:
            eid = event.get("id")
            if eid and eid not in seen:
                seen.add(eid)
                unique_events.append(event)

        return unique_events

    def _get_season_for_date(self, commence_time: str) -> str:
        """Determine season from commence time."""
        dt = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
        if dt.month >= 10:
            return f"{dt.year}-{dt.year + 1}"
        else:
            return f"{dt.year - 1}-{dt.year}"

    async def ingest_season(
        self,
        season: str,
        markets: List[str],
        dry_run: bool = False,
    ) -> IngestionStats:
        """
        Ingest period odds for all events in a season.

        Args:
            season: Season string (e.g., "2023-2024")
            markets: List of market keys to fetch
            dry_run: If True, only estimate costs

        Returns:
            IngestionStats with results
        """
        events = self._load_events(season)

        if not events:
            logger.warning(f"No events found for {season}")
            return self.fetcher.stats

        logger.info(f"Found {len(events)} events for {season}")
        logger.info(f"Markets to fetch: {markets}")

        # Cost estimate
        num_regions = 1  # us
        estimated_credits = len(events) * 10 * num_regions * len(markets)
        logger.info(f"Estimated credits: {estimated_credits:,}")

        if dry_run:
            logger.info("DRY RUN - no data will be fetched")
            self.fetcher.stats.credits_used = estimated_credits
            return self.fetcher.stats

        # Create output directory for this season
        season_dir = self.period_odds_dir / season
        season_dir.mkdir(parents=True, exist_ok=True)

        # Process events
        all_odds = []

        for i, event in enumerate(events, 1):
            event_id = event.get("id")
            commence_time = event.get("commence_time")

            if not event_id or not commence_time:
                continue

            self.fetcher.stats.events_processed += 1

            if i % 50 == 0:
                logger.info(
                    f"Progress: {i}/{len(events)} events "
                    f"({self.fetcher.stats.events_with_odds} with odds)"
                )

            # Fetch odds
            result = await self.fetcher.fetch_event_odds(
                event_id=event_id,
                commence_time=commence_time,
                markets=markets,
            )

            if result and result.get("data"):
                self.fetcher.stats.events_with_odds += 1

                # Add metadata
                result["_event_id"] = event_id
                result["_commence_time"] = commence_time
                result["_markets_requested"] = markets
                all_odds.append(result)

        # Save all odds to single file per season
        if all_odds:
            market_type = "1h" if markets == FIRST_HALF_MARKETS else "periods"
            output_file = season_dir / f"period_odds_{market_type}.json"

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump({
                    "season": season,
                    "markets": markets,
                    "events_count": len(all_odds),
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "data": all_odds,
                }, f, indent=2)

            logger.info(
                f"Saved {len(all_odds)} events with odds to {output_file}")

        return self.fetcher.stats

    async def ingest_all_seasons(
        self,
        markets: List[str],
        seasons: Optional[List[str]] = None,
        dry_run: bool = False,
    ) -> IngestionStats:
        """Ingest period odds for all available seasons."""

        # Discover seasons
        available_seasons = []
        if self.events_dir.exists():
            for d in self.events_dir.iterdir():
                if d.is_dir():
                    available_seasons.append(d.name)

        if seasons:
            available_seasons = [s for s in available_seasons if s in seasons]

        if not available_seasons:
            logger.warning("No seasons found")
            return self.fetcher.stats

        logger.info(f"Processing seasons: {sorted(available_seasons)}")

        for season in sorted(available_seasons):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {season}")
            logger.info(f"{'='*60}")
            await self.ingest_season(season, markets, dry_run)

        return self.fetcher.stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest historical period odds (1H, quarters) per event"
    )

    parser.add_argument(
        "--season",
        type=str,
        help="Specific season to process (e.g., 2023-2024)",
    )

    parser.add_argument(
        "--markets",
        type=str,
        required=True,
        choices=["1h", "quarters", "all"],
        help="Which period markets to fetch",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Estimate costs without fetching",
    )

    parser.add_argument(
        "--rate-limit",
        type=float,
        default=0.5,
        help="Seconds between API requests (default: 0.5)",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Historical data directory (defaults to HISTORICAL_OUTPUT_ROOT/the_odds)",
    )

    return parser.parse_args()


async def main_async() -> int:
    require_historical_mode()
    args = parse_args()
    if args.data_dir:
        ensure_historical_path(Path(args.data_dir), "data-dir")

    # Validate API key
    api_key = settings.the_odds_api_key
    if not api_key:
        logger.error("THE_ODDS_API_KEY not set")
        return 1

    # Select markets
    if args.markets == "1h":
        markets = FIRST_HALF_MARKETS
    elif args.markets == "quarters":
        markets = QUARTER_MARKETS
    else:
        markets = ALL_PERIOD_MARKETS

    # Create ingestor
    ingestor = HistoricalPeriodIngestor(
        api_key=api_key,
        data_dir=args.data_dir,
        rate_limit=args.rate_limit,
    )

    # Run ingestion
    try:
        seasons = [args.season] if args.season else None
        stats = await ingestor.ingest_all_seasons(
            markets=markets,
            seasons=seasons,
            dry_run=args.dry_run,
        )

        print("\n" + "="*60)
        print("=== Period Odds Ingestion Summary ===")
        print("="*60)
        print(f"Events processed: {stats.events_processed}")
        print(f"Events with odds: {stats.events_with_odds}")
        print(f"Events failed: {stats.events_failed}")
        print(f"API calls: {stats.api_calls}")
        print(f"Credits used: {stats.credits_used:,}")

        return 0

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        return 1


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())
