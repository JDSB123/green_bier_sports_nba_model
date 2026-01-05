#!/usr/bin/env python3
"""
Parallel Historical Odds Ingestion for 2021-2022 and 2022-2023 Seasons.

This script pulls historical NBA data from The Odds API for the 2021-2022 and
2022-2023 seasons using parallel workers. Focuses on FIRST HALF and FULL GAME markets.

Markets:
    - FULL GAME: h2h, spreads, totals
    - FIRST HALF: h2h_h1, spreads_h1, totals_h1

Storage Structure:
    data/historical/the_odds/
    ├── events/           # Historical events by date
    │   └── {season}/
    │       └── events_{YYYY-MM-DD}.json
    ├── odds/             # Historical odds snapshots by date
    │   └── {season}/
    │       └── odds_{YYYY-MM-DD}_{market_group}.json
    └── metadata/         # Ingestion tracking
        └── progress_{season}.json

Usage:
    # Pull data for both seasons in parallel
    python scripts/ingest_historical_odds_2021_2023.py

    # Dry run (estimate costs)
    python scripts/ingest_historical_odds_2021_2023.py --dry-run

    # Resume interrupted ingestion
    python scripts/ingest_historical_odds_2021_2023.py --resume

NOTE: Historical API endpoints require a paid plan.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ==============================================================================
# MARKET DEFINITIONS
# ==============================================================================

# FULL GAME markets
FULL_GAME_MARKETS = ["h2h", "spreads", "totals"]

# FIRST HALF markets
FIRST_HALF_MARKETS = ["h2h_h1", "spreads_h1", "totals_h1"]

# Combined markets for this script
TARGET_MARKETS = FULL_GAME_MARKETS + FIRST_HALF_MARKETS

# NBA season date ranges
NBA_SEASONS = {
    "2021-2022": {
        "start": date(2021, 10, 19),
        "end": date(2022, 4, 10),
        "playoffs_end": date(2022, 6, 16),  # Approximate
    },
    "2022-2023": {
        "start": date(2022, 10, 18),
        "end": date(2023, 4, 9),
        "playoffs_end": date(2023, 6, 12),  # Approximate
    },
}


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class IngestionProgress:
    """Track ingestion progress for resumption."""
    season: str
    last_date_processed: Optional[str] = None
    dates_completed: List[str] = field(default_factory=list)
    dates_failed: List[str] = field(default_factory=list)
    total_events_fetched: int = 0
    total_api_calls: int = 0
    estimated_credits_used: int = 0
    started_at: Optional[str] = None
    last_updated_at: Optional[str] = None


@dataclass
class IngestionStats:
    """Statistics for current ingestion run."""
    api_calls: int = 0
    events_fetched: int = 0
    markets_fetched: int = 0
    errors: int = 0
    credits_estimated: int = 0


# ==============================================================================
# RATE LIMITER
# ==============================================================================

class SimpleRateLimiter:
    """Simple async rate limiter for API requests."""
    
    def __init__(self, min_interval: float = 1.0):
        """
        Args:
            min_interval: Minimum seconds between requests
        """
        self.min_interval = min_interval
        self._last_request: Optional[float] = None
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        """Wait until it's safe to make the next request."""
        async with self._lock:
            import time
            now = time.monotonic()
            
            if self._last_request is not None:
                elapsed = now - self._last_request
                if elapsed < self.min_interval:
                    wait_time = self.min_interval - elapsed
                    await asyncio.sleep(wait_time)
            
            self._last_request = time.monotonic()


# ==============================================================================
# HISTORICAL DATA FETCHER
# ==============================================================================

class HistoricalOddsFetcher:
    """
    Fetches historical NBA odds from The Odds API.
    
    Uses the historical endpoints which require a paid plan:
    - /historical/sports/{sport}/events
    - /historical/sports/{sport}/odds
    """
    
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
        
    async def _request(
        self,
        endpoint: str,
        params: Dict[str, Any],
        timeout: int = 60,
    ) -> Dict[str, Any]:
        """Make rate-limited API request."""
        import httpx
        
        await self.rate_limiter.acquire()
        self.stats.api_calls += 1
        
        params["apiKey"] = self.api_key
        params["dateFormat"] = "iso"
        params["oddsFormat"] = "american"
        
        url = f"{self.base_url}{endpoint}"
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url, params=params)
            
            # Log remaining quota from headers
            remaining = resp.headers.get("x-requests-remaining")
            used = resp.headers.get("x-requests-used")
            if remaining:
                logger.debug(f"API quota: {remaining} remaining, {used} used")
            
            resp.raise_for_status()
            return resp.json()
    
    async def fetch_historical_events(
        self,
        date_iso: str,
        sport: str = "basketball_nba",
    ) -> Dict[str, Any]:
        """
        Fetch historical events for a given date.
        
        Args:
            date_iso: ISO format date string (e.g., "2022-01-15T12:00:00Z")
            sport: Sport key
            
        Returns:
            Dict with 'data' containing list of events
        """
        logger.info(f"Fetching historical events for {date_iso}")
        self.stats.credits_estimated += 1  # Events endpoint costs 1 credit
        
        try:
            data = await self._request(
                f"/historical/sports/{sport}/events",
                {"date": date_iso}
            )
            events = data.get("data", [])
            self.stats.events_fetched += len(events)
            logger.info(f"Found {len(events)} events for {date_iso}")
            return data
        except Exception as e:
            logger.error(f"Failed to fetch events for {date_iso}: {e}")
            self.stats.errors += 1
            raise
    
    async def fetch_historical_odds(
        self,
        date_iso: str,
        markets: List[str],
        regions: str = "us",
        sport: str = "basketball_nba",
    ) -> Dict[str, Any]:
        """
        Fetch historical odds snapshot for all events on a date.
        
        Args:
            date_iso: ISO format date string
            markets: List of market keys to fetch
            regions: Regions (default: us)
            sport: Sport key
            
        Returns:
            Dict with 'data' containing list of events with odds
        """
        markets_str = ",".join(markets)
        logger.info(f"Fetching historical odds for {date_iso} ({len(markets)} markets)")
        
        # Cost: 10 credits per region per market
        num_regions = len(regions.split(","))
        self.stats.credits_estimated += 10 * num_regions * len(markets)
        self.stats.markets_fetched += len(markets)
        
        try:
            data = await self._request(
                f"/historical/sports/{sport}/odds",
                {
                    "date": date_iso,
                    "regions": regions,
                    "markets": markets_str,
                }
            )
            events = data.get("data", [])
            logger.info(f"Fetched odds for {len(events)} events")
            return data
        except Exception as e:
            logger.error(f"Failed to fetch odds for {date_iso}: {e}")
            self.stats.errors += 1
            raise
    
    async def fetch_historical_event_odds(
        self,
        event_id: str,
        date_iso: str,
        markets: List[str],
        regions: str = "us",
        sport: str = "basketball_nba",
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch historical odds for a specific event.
        
        This is used for period markets that aren't available via bulk endpoint.
        
        Args:
            event_id: The event ID
            date_iso: ISO format date string
            markets: List of market keys to fetch
            regions: Regions
            sport: Sport key
            
        Returns:
            Dict with event odds data or None if failed
        """
        markets_str = ",".join(markets)
        logger.debug(f"Fetching event {event_id} odds for {date_iso}")
        
        # Cost: 10 credits per region per market
        num_regions = len(regions.split(","))
        self.stats.credits_estimated += 10 * num_regions * len(markets)
        
        try:
            data = await self._request(
                f"/historical/sports/{sport}/events/{event_id}/odds",
                {
                    "date": date_iso,
                    "regions": regions,
                    "markets": markets_str,
                }
            )
            return data
        except Exception as e:
            # 422 means markets not available for this event/date
            if "422" in str(e):
                logger.debug(f"Markets not available for event {event_id}")
            else:
                logger.warning(f"Failed to fetch event odds for {event_id}: {e}")
            self.stats.errors += 1
            return None


# ==============================================================================
# STORAGE MANAGER
# ==============================================================================

class HistoricalDataStorage:
    """
    Manages storage of historical odds data.
    
    Data is stored in a separate directory from live data to maintain
    model integrity.
    """
    
    def __init__(self, base_dir: str = "data/historical/the_odds"):
        self.base_dir = Path(base_dir)
        self.events_dir = self.base_dir / "events"
        self.odds_dir = self.base_dir / "odds"
        self.metadata_dir = self.base_dir / "metadata"
        
        # Ensure directories exist
        for d in [self.events_dir, self.odds_dir, self.metadata_dir]:
            d.mkdir(parents=True, exist_ok=True)
    
    def _get_season_for_date(self, d: date) -> str:
        """Determine NBA season for a given date."""
        # NBA season spans Oct-Jun
        if d.month >= 10:  # Oct-Dec
            return f"{d.year}-{d.year + 1}"
        else:  # Jan-Sep
            return f"{d.year - 1}-{d.year}"
    
    def _get_date_str(self, d: date) -> str:
        """Get date string for filename."""
        return d.strftime("%Y-%m-%d")
    
    def save_events(self, d: date, data: Dict[str, Any]) -> Path:
        """Save historical events data."""
        season = self._get_season_for_date(d)
        season_dir = self.events_dir / season
        season_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"events_{self._get_date_str(d)}.json"
        path = season_dir / filename
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.debug(f"Saved events to {path}")
        return path
    
    def save_odds(
        self,
        d: date,
        data: Dict[str, Any],
        market_group: str = "full_game_first_half",
    ) -> Path:
        """Save historical odds data."""
        season = self._get_season_for_date(d)
        season_dir = self.odds_dir / season
        season_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"odds_{self._get_date_str(d)}_{market_group}.json"
        path = season_dir / filename
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.debug(f"Saved odds to {path}")
        return path
    
    def load_progress(self, season: str) -> Optional[IngestionProgress]:
        """Load ingestion progress for resumption."""
        path = self.metadata_dir / f"progress_{season}.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return IngestionProgress(**data)
        return None
    
    def save_progress(self, progress: IngestionProgress) -> None:
        """Save ingestion progress."""
        progress.last_updated_at = datetime.now(timezone.utc).isoformat()
        path = self.metadata_dir / f"progress_{progress.season}.json"
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(progress), f, ensure_ascii=False, indent=2)
        
        logger.debug(f"Saved progress to {path}")
    
    def has_data_for_date(self, d: date, data_type: str = "odds") -> bool:
        """Check if data already exists for a date."""
        season = self._get_season_for_date(d)
        date_str = self._get_date_str(d)
        
        if data_type == "events":
            path = self.events_dir / season / f"events_{date_str}.json"
            return path.exists()
        elif data_type == "odds":
            # Check if any odds file exists for this date
            season_dir = self.odds_dir / season
            if season_dir.exists():
                return any(f.name.startswith(f"odds_{date_str}") for f in season_dir.glob("*.json"))
            return False
        else:
            return False


# ==============================================================================
# INGESTION ORCHESTRATOR
# ==============================================================================

class HistoricalIngestionOrchestrator:
    """
    Orchestrates the historical data ingestion process for a single season.
    
    Features:
    - Rate limiting to respect API limits
    - Progress tracking for resumption
    - Cost estimation
    """
    
    def __init__(
        self,
        api_key: str,
        storage: HistoricalDataStorage,
        rate_limit: float = 1.0,
    ):
        self.fetcher = HistoricalOddsFetcher(api_key, rate_limit=rate_limit)
        self.storage = storage
    
    def _get_snapshot_datetime(self, d: date, hour: int = 12) -> str:
        """Convert date to ISO datetime for API query."""
        dt = datetime(d.year, d.month, d.day, hour, 0, 0, tzinfo=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    def _daterange(self, start: date, end: date) -> Iterable[date]:
        """Yield each date in range."""
        current = start
        while current <= end:
            yield current
            current += timedelta(days=1)
    
    async def ingest_date(
        self,
        d: date,
        markets: List[str],
        skip_existing: bool = True,
    ) -> Tuple[bool, List[str]]:
        """
        Ingest all data for a single date.
        
        Args:
            d: Date to ingest
            markets: List of market keys to fetch
            skip_existing: Skip if data already exists
            
        Returns:
            Tuple of (success, list of files saved)
        """
        date_str = d.strftime("%Y-%m-%d")
        saved_files = []
        
        if skip_existing and self.storage.has_data_for_date(d, "odds"):
            logger.info(f"Skipping {date_str} - data already exists")
            return True, []
        
        date_iso = self._get_snapshot_datetime(d)
        logger.info(f"Ingesting data for {date_str}")
        
        try:
            # 1. Fetch events first
            events_data = await self.fetcher.fetch_historical_events(date_iso)
            events = events_data.get("data", [])
            
            if not events:
                logger.info(f"No events found for {date_str}")
                # Still save empty events file to mark as processed
                path = self.storage.save_events(d, events_data)
                saved_files.append(str(path))
                return True, saved_files
            
            # Save events
            path = self.storage.save_events(d, events_data)
            saved_files.append(str(path))
            
            # 2. Fetch FULL GAME markets via bulk endpoint
            full_game_data = None
            try:
                full_game_data = await self.fetcher.fetch_historical_odds(
                    date_iso, FULL_GAME_MARKETS
                )
                logger.info(f"Fetched full game odds for {date_str}")
            except Exception as e:
                logger.warning(f"Failed to fetch full game odds for {date_str}: {e}")
            
            # 3. Fetch FIRST HALF markets via per-event endpoint
            # (bulk endpoint doesn't support period markets for older seasons)
            first_half_events = []
            if events:
                logger.info(f"Fetching first half odds for {len(events)} events")
                tasks = [
                    self.fetcher.fetch_historical_event_odds(
                        event.get("id"),
                        event.get("commence_time", date_iso),
                        FIRST_HALF_MARKETS,
                    )
                    for event in events
                    if event.get("id") and event.get("commence_time")
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, dict) and result.get("data"):
                        first_half_events.append(result["data"])
                
                logger.info(f"Fetched first half odds for {len(first_half_events)} events")
            
            # 4. Combine and save odds data
            if full_game_data or first_half_events:
                # Merge full game and first half data
                combined_events = {}
                
                # Add full game data
                if full_game_data:
                    for event in full_game_data.get("data", []):
                        event_id = event.get("id")
                        if event_id:
                            combined_events[event_id] = event
                
                # Merge first half markets into existing events
                for event in first_half_events:
                    event_id = event.get("id")
                    if event_id:
                        if event_id not in combined_events:
                            combined_events[event_id] = event
                        else:
                            # Merge bookmakers from first half into full game event
                            existing_bookmakers = {
                                bm.get("key"): bm
                                for bm in combined_events[event_id].get("bookmakers", [])
                            }
                            
                            for bm in event.get("bookmakers", []):
                                bm_key = bm.get("key")
                                if bm_key in existing_bookmakers:
                                    # Merge markets
                                    existing_markets = {
                                        m.get("key"): m
                                        for m in existing_bookmakers[bm_key].get("markets", [])
                                    }
                                    for market in bm.get("markets", []):
                                        existing_markets[market.get("key")] = market
                                    existing_bookmakers[bm_key]["markets"] = list(existing_markets.values())
                                else:
                                    existing_bookmakers[bm_key] = bm
                            
                            combined_events[event_id]["bookmakers"] = list(existing_bookmakers.values())
                
                # Create combined response
                combined_data = {
                    "timestamp": full_game_data.get("timestamp") if full_game_data else date_iso,
                    "data": list(combined_events.values()),
                }
                
                path = self.storage.save_odds(d, combined_data, "full_game_first_half")
                saved_files.append(str(path))
            else:
                logger.warning(f"No odds data fetched for {date_str}")
            
            return True, saved_files
            
        except Exception as e:
            logger.error(f"Failed to ingest {date_str}: {e}")
            return False, saved_files
    
    async def ingest_season(
        self,
        season: str,
        markets: List[str],
        include_playoffs: bool = True,
        resume: bool = False,
        dry_run: bool = False,
    ) -> IngestionProgress:
        """
        Ingest a full NBA season.
        
        Args:
            season: Season string (e.g., "2021-2022")
            markets: List of market keys to fetch
            include_playoffs: Include playoff dates
            resume: Resume from last progress checkpoint
            dry_run: Only estimate costs, don't fetch
            
        Returns:
            IngestionProgress with results
        """
        if season not in NBA_SEASONS:
            raise ValueError(f"Unknown season: {season}. Available: {list(NBA_SEASONS.keys())}")
        
        season_info = NBA_SEASONS[season]
        start_date = season_info["start"]
        end_date = season_info["playoffs_end"] if include_playoffs else season_info["end"]
        
        # Load or create progress
        progress = None
        if resume:
            progress = self.storage.load_progress(season)
        
        if progress is None:
            progress = IngestionProgress(
                season=season,
                started_at=datetime.now(timezone.utc).isoformat(),
            )
        
        # Calculate dates to process
        all_dates = list(self._daterange(start_date, end_date))
        completed_dates = set(progress.dates_completed)
        dates_to_process = [d for d in all_dates if d.strftime("%Y-%m-%d") not in completed_dates]
        
        total_days = len(dates_to_process)
        
        logger.info(f"Season {season}: {len(all_dates)} total days, {total_days} to process")
        logger.info(f"Markets: {markets}")
        
        # Estimate costs
        num_regions = 1  # us
        estimated_credits = total_days * (1 + 10 * num_regions * len(markets))  # events + odds
        logger.info(f"Estimated API credits: {estimated_credits:,}")
        
        if dry_run:
            logger.info("DRY RUN - no data will be fetched")
            progress.estimated_credits_used = estimated_credits
            return progress
        
        # Process each date
        for i, d in enumerate(dates_to_process, 1):
            date_str = d.strftime("%Y-%m-%d")
            logger.info(f"[{season}] Processing {date_str} ({i}/{total_days})")
            
            success, files = await self.ingest_date(d, markets)
            
            if success:
                progress.dates_completed.append(date_str)
                progress.last_date_processed = date_str
            else:
                progress.dates_failed.append(date_str)
            
            # Update progress
            progress.total_events_fetched = self.fetcher.stats.events_fetched
            progress.total_api_calls = self.fetcher.stats.api_calls
            progress.estimated_credits_used = self.fetcher.stats.credits_estimated
            
            # Save progress periodically
            if i % 10 == 0:
                self.storage.save_progress(progress)
                logger.info(f"[{season}] Progress saved: {len(progress.dates_completed)} dates completed")
        
        # Final save
        self.storage.save_progress(progress)
        
        logger.info(f"[{season}] Season ingestion complete: {len(progress.dates_completed)} dates, "
                   f"{progress.total_events_fetched} events, "
                   f"~{progress.estimated_credits_used:,} credits used")
        
        return progress


# ==============================================================================
# PARALLEL WORKER
# ==============================================================================

async def ingest_season_worker(
    season: str,
    api_key: str,
    storage: HistoricalDataStorage,
    markets: List[str],
    include_playoffs: bool = True,
    resume: bool = False,
    dry_run: bool = False,
    rate_limit: float = 1.0,
) -> IngestionProgress:
    """
    Worker function to ingest a single season.
    
    This is designed to be run in parallel with other seasons.
    """
    orchestrator = HistoricalIngestionOrchestrator(
        api_key=api_key,
        storage=storage,
        rate_limit=rate_limit,
    )
    
    return await orchestrator.ingest_season(
        season=season,
        markets=markets,
        include_playoffs=include_playoffs,
        resume=resume,
        dry_run=dry_run,
    )


# ==============================================================================
# CLI
# ==============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pull historical NBA odds for 2021-2022 and 2022-2023 seasons (parallel)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Pull data for both seasons in parallel
    python scripts/ingest_historical_odds_2021_2023.py

    # Dry run (estimate costs)
    python scripts/ingest_historical_odds_2021_2023.py --dry-run

    # Resume interrupted ingestion
    python scripts/ingest_historical_odds_2021_2023.py --resume

Markets fetched:
    - FULL GAME: h2h, spreads, totals
    - FIRST HALF: h2h_h1, spreads_h1, totals_h1
        """
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint for each season",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Estimate costs without fetching data",
    )
    
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=1.0,
        help="Seconds between API requests per worker (default: 1.0)",
    )
    
    parser.add_argument(
        "--include-playoffs",
        action="store_true",
        default=True,
        help="Include playoff dates (default: True)",
    )
    
    parser.add_argument(
        "--no-playoffs",
        action="store_true",
        help="Exclude playoff dates (regular season only)",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/historical/the_odds",
        help="Output directory for historical data",
    )
    
    return parser.parse_args()


async def main_async() -> int:
    args = parse_args()
    
    storage = HistoricalDataStorage(args.output_dir)
    
    # Validate API key
    api_key = settings.the_odds_api_key
    if not api_key:
        logger.error("THE_ODDS_API_KEY not set. Cannot proceed.")
        return 1
    
    # Markets to fetch
    markets = TARGET_MARKETS  # FULL GAME + FIRST HALF
    
    # Seasons to process
    seasons = ["2021-2022", "2022-2023"]
    
    include_playoffs = args.include_playoffs and not args.no_playoffs
    
    logger.info("=" * 60)
    logger.info("PARALLEL HISTORICAL ODDS INGESTION")
    logger.info("=" * 60)
    logger.info(f"Seasons: {', '.join(seasons)}")
    logger.info(f"Markets: {', '.join(markets)}")
    logger.info(f"Include playoffs: {include_playoffs}")
    logger.info(f"Rate limit: {args.rate_limit}s per worker")
    logger.info("=" * 60)
    
    # Create parallel tasks for each season
    tasks = [
        ingest_season_worker(
            season=season,
            api_key=api_key,
            storage=storage,
            markets=markets,
            include_playoffs=include_playoffs,
            resume=args.resume,
            dry_run=args.dry_run,
            rate_limit=args.rate_limit,
        )
        for season in seasons
    ]
    
    # Run all seasons in parallel
    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Print summary
        print("\n" + "=" * 60)
        print("=== INGESTION SUMMARY ===")
        print("=" * 60)
        
        for i, (season, result) in enumerate(zip(seasons, results)):
            print(f"\n{season}:")
            if isinstance(result, Exception):
                print(f"  ERROR: {result}")
            else:
                print(f"  Dates completed: {len(result.dates_completed)}")
                print(f"  Dates failed: {len(result.dates_failed)}")
                print(f"  Total events: {result.total_events_fetched}")
                print(f"  API calls: {result.total_api_calls}")
                print(f"  Estimated credits: {result.estimated_credits_used:,}")
        
        # Check for errors
        errors = [r for r in results if isinstance(r, Exception)]
        if errors:
            print(f"\n{len(errors)} season(s) had errors")
            return 1
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Interrupted by user - progress saved")
        return 1
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        return 1


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())
