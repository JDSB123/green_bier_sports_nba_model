#!/usr/bin/env python3
"""
Comprehensive Historical Odds Ingestion from The Odds API.

This script ingests historical NBA data from The Odds API for the 2023-2024 and
2024-2025 seasons. Data is stored in a SEPARATE directory (data/historical/)
to maintain model integrity and avoid disrupting the live prediction pipeline.

Storage Structure:
    data/historical/the_odds/
    ├── events/           # Historical events by date
    │   └── {season}/
    │       └── events_{YYYY-MM-DD}.json
    ├── odds/             # Historical odds snapshots by date
    │   └── {season}/
    │       └── odds_{YYYY-MM-DD}_{market_group}.json
    ├── player_props/     # Player props by date (if available)
    │   └── {season}/
    │       └── props_{YYYY-MM-DD}.json
    └── metadata/         # Ingestion tracking and API usage
        └── ingestion_log.json

Supported Markets:
    - Featured: h2h, spreads, totals
    - Game periods: h2h_h1, spreads_h1, totals_h1, h2h_q1-q4, spreads_q1-q4, totals_q1-q4
    - Alternates: alternate_spreads, alternate_totals
    - Player props: player_points, player_rebounds, player_assists, etc.

Usage:
    # Ingest full 2023-2024 season
    python scripts/ingest_historical_odds.py --season 2023-2024

    # Ingest specific date range
    python scripts/ingest_historical_odds.py --start-date 2023-10-24 --end-date 2024-04-14

    # Ingest with specific markets only
    python scripts/ingest_historical_odds.py --season 2023-2024 --markets featured

    # Resume interrupted ingestion
    python scripts/ingest_historical_odds.py --season 2023-2024 --resume

    # Dry run (show what would be fetched)
    python scripts/ingest_historical_odds.py --season 2023-2024 --dry-run

NOTE: Historical API endpoints require a paid plan. The script will track API
usage and costs for billing awareness.

Cost estimation (per The Odds API pricing):
- Historical odds: 10 credits per region per market
- Historical events: 1 credit per request
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


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
# MARKET DEFINITIONS
# ==============================================================================

# Featured markets (most common, always available)
FEATURED_MARKETS = ["h2h", "spreads", "totals"]

# Game period markets for NBA
GAME_PERIOD_MARKETS = [
    # First Half
    "h2h_h1", "spreads_h1", "totals_h1",
    # Second Half
    "h2h_h2", "spreads_h2", "totals_h2",
    # Quarters
    "h2h_q1", "spreads_q1", "totals_q1",
    "h2h_q2", "spreads_q2", "totals_q2",
    "h2h_q3", "spreads_q3", "totals_q3",
    "h2h_q4", "spreads_q4", "totals_q4",
    # 3-way moneyline variants
    "h2h_3_way", "h2h_3_way_h1", "h2h_3_way_h2",
    "h2h_3_way_q1", "h2h_3_way_q2", "h2h_3_way_q3", "h2h_3_way_q4",
]

# Alternate markets
ALTERNATE_MARKETS = [
    "alternate_spreads", "alternate_totals",
    "alternate_spreads_h1", "alternate_totals_h1",
    "alternate_spreads_h2", "alternate_totals_h2",
    "alternate_spreads_q1", "alternate_totals_q1",
    "alternate_spreads_q2", "alternate_totals_q2",
    "alternate_spreads_q3", "alternate_totals_q3",
    "alternate_spreads_q4", "alternate_totals_q4",
]

# Team totals markets
TEAM_TOTALS_MARKETS = [
    "team_totals",
    "team_totals_h1", "team_totals_h2",
    "team_totals_q1", "team_totals_q2", "team_totals_q3", "team_totals_q4",
    "alternate_team_totals",
    "alternate_team_totals_h1", "alternate_team_totals_h2",
    "alternate_team_totals_q1", "alternate_team_totals_q2",
    "alternate_team_totals_q3", "alternate_team_totals_q4",
]

# NBA Player Props (per Odds API documentation)
PLAYER_PROPS_MARKETS = [
    # Core stats
    "player_points", "player_rebounds", "player_assists",
    "player_threes", "player_blocks", "player_steals",
    "player_turnovers", "player_blocks_steals",
    # Combined stats
    "player_points_rebounds_assists",
    "player_points_rebounds", "player_points_assists",
    "player_rebounds_assists",
    # Other
    "player_field_goals", "player_frees_made", "player_frees_attempts",
    # Quarter stats
    "player_points_q1", "player_rebounds_q1", "player_assists_q1",
    # Special
    "player_first_basket", "player_first_team_basket",
    "player_double_double", "player_triple_double",
    "player_method_of_first_basket",
]

# Alternate player props
ALTERNATE_PLAYER_PROPS = [
    "player_points_alternate", "player_rebounds_alternate",
    "player_assists_alternate", "player_blocks_alternate",
    "player_steals_alternate", "player_turnovers_alternate",
    "player_threes_alternate",
    "player_points_assists_alternate",
    "player_points_rebounds_alternate",
    "player_rebounds_assists_alternate",
    "player_points_rebounds_assists_alternate",
]

# Market groups for organized fetching
MARKET_GROUPS = {
    "featured": FEATURED_MARKETS,
    "periods": GAME_PERIOD_MARKETS,
    "alternates": ALTERNATE_MARKETS,
    "team_totals": TEAM_TOTALS_MARKETS,
    "player_props": PLAYER_PROPS_MARKETS,
    "alternate_props": ALTERNATE_PLAYER_PROPS,
}

# NBA season date ranges (regular season only)
NBA_SEASONS = {
    "2023-2024": {
        "start": date(2023, 10, 24),
        "end": date(2024, 4, 14),
        "playoffs_end": date(2024, 6, 17),
    },
    "2024-2025": {
        "start": date(2024, 10, 22),
        "end": date(2025, 4, 13),
        "playoffs_end": date(2025, 6, 22),
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
# HISTORICAL DATA FETCHER
# ==============================================================================

class HistoricalOddsFetcher:
    """
    Fetches historical NBA odds from The Odds API.
    
    Uses the historical endpoints which require a paid plan:
    - /historical/sports/{sport}/events
    - /historical/sports/{sport}/odds
    - /historical/sports/{sport}/events/{eventId}/odds
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.the-odds-api.com/v4",
        rate_limit: float = 0.5,  # seconds between requests
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
            date_iso: ISO format date string (e.g., "2024-01-15T12:00:00Z")
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
    ) -> Dict[str, Any]:
        """
        Fetch historical odds for a specific event.
        
        This is useful for fetching player props and additional markets
        that require per-event queries.
        
        Args:
            event_id: The event ID
            date_iso: ISO format date string
            markets: List of market keys to fetch
            regions: Regions
            sport: Sport key
            
        Returns:
            Dict with event odds data
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
            logger.error(f"Failed to fetch event odds for {event_id}: {e}")
            self.stats.errors += 1
            raise


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
        self.props_dir = self.base_dir / "player_props"
        self.metadata_dir = self.base_dir / "metadata"
        
        # Ensure directories exist
        for d in [self.events_dir, self.odds_dir, self.props_dir, self.metadata_dir]:
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
        market_group: str = "all",
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
    
    def save_player_props(self, d: date, data: Dict[str, Any]) -> Path:
        """Save player props data."""
        season = self._get_season_for_date(d)
        season_dir = self.props_dir / season
        season_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"props_{self._get_date_str(d)}.json"
        path = season_dir / filename
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.debug(f"Saved player props to {path}")
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
        elif data_type == "odds":
            # Check if any odds file exists for this date
            season_dir = self.odds_dir / season
            if season_dir.exists():
                return any(f.name.startswith(f"odds_{date_str}") for f in season_dir.glob("*.json"))
            return False
        elif data_type == "props":
            path = self.props_dir / season / f"props_{date_str}.json"
        else:
            return False
        
        return path.exists() if 'path' in locals() else False
    
    def get_ingestion_summary(self) -> Dict[str, Any]:
        """Get summary of all ingested data."""
        summary = {
            "seasons": {},
            "total_files": 0,
            "total_size_mb": 0,
        }
        
        for data_dir in [self.events_dir, self.odds_dir, self.props_dir]:
            for season_dir in data_dir.glob("*"):
                if season_dir.is_dir():
                    season = season_dir.name
                    if season not in summary["seasons"]:
                        summary["seasons"][season] = {
                            "events_files": 0,
                            "odds_files": 0,
                            "props_files": 0,
                            "total_size_mb": 0,
                        }
                    
                    files = list(season_dir.glob("*.json"))
                    size_mb = sum(f.stat().st_size for f in files) / (1024 * 1024)
                    
                    if data_dir == self.events_dir:
                        summary["seasons"][season]["events_files"] = len(files)
                    elif data_dir == self.odds_dir:
                        summary["seasons"][season]["odds_files"] = len(files)
                    elif data_dir == self.props_dir:
                        summary["seasons"][season]["props_files"] = len(files)
                    
                    summary["seasons"][season]["total_size_mb"] += size_mb
                    summary["total_files"] += len(files)
                    summary["total_size_mb"] += size_mb
        
        return summary


# ==============================================================================
# INGESTION ORCHESTRATOR
# ==============================================================================

class HistoricalIngestionOrchestrator:
    """
    Orchestrates the historical data ingestion process.
    
    Features:
    - Rate limiting to respect API limits
    - Progress tracking for resumption
    - Market grouping for efficient fetching
    - Cost estimation
    """
    
    def __init__(
        self,
        api_key: str,
        storage: HistoricalDataStorage,
        include_player_props: bool = False,
        rate_limit: float = 1.0,
    ):
        self.fetcher = HistoricalOddsFetcher(api_key, rate_limit=rate_limit)
        self.storage = storage
        self.include_player_props = include_player_props
    
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
        market_groups: List[str],
        skip_existing: bool = True,
    ) -> Tuple[bool, List[str]]:
        """
        Ingest all data for a single date.
        
        Args:
            d: Date to ingest
            market_groups: List of market group names to fetch
            skip_existing: Skip if data already exists
            
        Returns:
            Tuple of (success, list of files saved)
        """
        date_str = d.strftime("%Y-%m-%d")
        saved_files = []
        
        # Skip weekends (typically no games) - optional optimization
        # if d.weekday() in [0, 1, 2, 3, 4, 5, 6]:  # All days for safety
        
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
            
            # 2. Fetch odds for each market group
            for group_name in market_groups:
                markets = MARKET_GROUPS.get(group_name, [])
                if not markets:
                    continue
                
                # Skip player props if not explicitly requested
                if group_name in ["player_props", "alternate_props"]:
                    if not self.include_player_props:
                        continue
                
                try:
                    odds_data = await self.fetcher.fetch_historical_odds(
                        date_iso, markets
                    )
                    
                    if group_name in ["player_props", "alternate_props"]:
                        path = self.storage.save_player_props(d, odds_data)
                    else:
                        path = self.storage.save_odds(d, odds_data, group_name)
                    
                    saved_files.append(str(path))
                    
                except Exception as e:
                    logger.error(f"Failed to fetch {group_name} for {date_str}: {e}")
                    # Continue with other groups
            
            return True, saved_files
            
        except Exception as e:
            logger.error(f"Failed to ingest {date_str}: {e}")
            return False, saved_files
    
    async def ingest_season(
        self,
        season: str,
        market_groups: Optional[List[str]] = None,
        include_playoffs: bool = True,
        resume: bool = False,
        dry_run: bool = False,
    ) -> IngestionProgress:
        """
        Ingest a full NBA season.
        
        Args:
            season: Season string (e.g., "2023-2024")
            market_groups: Market groups to fetch (default: featured + periods)
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
        
        # Default market groups
        if market_groups is None:
            market_groups = ["featured", "periods"]
        
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
        logger.info(f"Market groups: {market_groups}")
        
        # Estimate costs
        total_markets = sum(len(MARKET_GROUPS.get(g, [])) for g in market_groups)
        estimated_credits = total_days * (1 + 10 * total_markets)  # events + odds
        logger.info(f"Estimated API credits: {estimated_credits:,}")
        
        if dry_run:
            logger.info("DRY RUN - no data will be fetched")
            progress.estimated_credits_used = estimated_credits
            return progress
        
        # Process each date
        for i, d in enumerate(dates_to_process, 1):
            date_str = d.strftime("%Y-%m-%d")
            logger.info(f"Processing {date_str} ({i}/{total_days})")
            
            success, files = await self.ingest_date(d, market_groups)
            
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
                logger.info(f"Progress saved: {len(progress.dates_completed)} dates completed")
        
        # Final save
        self.storage.save_progress(progress)
        
        logger.info(f"Season ingestion complete: {len(progress.dates_completed)} dates, "
                   f"{progress.total_events_fetched} events, "
                   f"~{progress.estimated_credits_used:,} credits used")
        
        return progress
    
    async def ingest_date_range(
        self,
        start_date: date,
        end_date: date,
        market_groups: Optional[List[str]] = None,
        dry_run: bool = False,
    ) -> IngestionProgress:
        """
        Ingest a custom date range.
        
        Args:
            start_date: Start date
            end_date: End date
            market_groups: Market groups to fetch
            dry_run: Only estimate costs
            
        Returns:
            IngestionProgress with results
        """
        if market_groups is None:
            market_groups = ["featured", "periods"]
        
        season = self.storage._get_season_for_date(start_date)
        progress = IngestionProgress(
            season=season,
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        
        dates = list(self._daterange(start_date, end_date))
        
        logger.info(f"Date range: {start_date} to {end_date} ({len(dates)} days)")
        
        # Estimate costs
        total_markets = sum(len(MARKET_GROUPS.get(g, [])) for g in market_groups)
        estimated_credits = len(dates) * (1 + 10 * total_markets)
        logger.info(f"Estimated API credits: {estimated_credits:,}")
        
        if dry_run:
            progress.estimated_credits_used = estimated_credits
            return progress
        
        for i, d in enumerate(dates, 1):
            date_str = d.strftime("%Y-%m-%d")
            logger.info(f"Processing {date_str} ({i}/{len(dates)})")
            
            success, files = await self.ingest_date(d, market_groups)
            
            if success:
                progress.dates_completed.append(date_str)
                progress.last_date_processed = date_str
            else:
                progress.dates_failed.append(date_str)
            
            progress.total_events_fetched = self.fetcher.stats.events_fetched
            progress.total_api_calls = self.fetcher.stats.api_calls
            progress.estimated_credits_used = self.fetcher.stats.credits_estimated
        
        self.storage.save_progress(progress)
        return progress


# ==============================================================================
# CLI
# ==============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest historical NBA odds from The Odds API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full 2023-2024 season (featured + period markets)
    python scripts/ingest_historical_odds.py --season 2023-2024

    # Specific date range
    python scripts/ingest_historical_odds.py --start-date 2024-01-01 --end-date 2024-01-31

    # Include all markets including player props
    python scripts/ingest_historical_odds.py --season 2023-2024 --markets all --include-props

    # Resume interrupted ingestion
    python scripts/ingest_historical_odds.py --season 2023-2024 --resume

    # Estimate costs without fetching
    python scripts/ingest_historical_odds.py --season 2023-2024 --dry-run

Market groups:
    featured    - h2h, spreads, totals (most common)
    periods     - half and quarter markets
    alternates  - alternate spreads/totals
    team_totals - team total markets
    player_props - player prop markets (requires --include-props)
    all         - all of the above
        """
    )
    
    # Date selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--season",
        type=str,
        choices=list(NBA_SEASONS.keys()),
        help="NBA season to ingest (e.g., 2023-2024)",
    )
    group.add_argument(
        "--start-date",
        type=str,
        help="Start date (YYYY-MM-DD) - requires --end-date",
    )
    
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date (YYYY-MM-DD) - required with --start-date",
    )
    
    # Market selection
    parser.add_argument(
        "--markets",
        type=str,
        nargs="+",
        default=["featured", "periods"],
        choices=["featured", "periods", "alternates", "team_totals", "player_props", "all"],
        help="Market groups to fetch (default: featured periods)",
    )
    
    parser.add_argument(
        "--include-props",
        action="store_true",
        help="Include player props (expensive, many API calls)",
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
    
    # Execution options
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint",
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
        help="Seconds between API requests (default: 1.0)",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/historical/the_odds",
        help="Output directory for historical data",
    )
    
    # Info commands
    parser.add_argument(
        "--show-summary",
        action="store_true",
        help="Show summary of existing data and exit",
    )
    
    return parser.parse_args()


def _parse_date(s: str) -> date:
    """Parse date string to date object."""
    return datetime.strptime(s, "%Y-%m-%d").date()


async def main_async() -> int:
    args = parse_args()
    
    storage = HistoricalDataStorage(args.output_dir)
    
    # Show summary and exit
    if args.show_summary:
        summary = storage.get_ingestion_summary()
        print("\n=== Historical Data Summary ===")
        print(f"Total files: {summary['total_files']}")
        print(f"Total size: {summary['total_size_mb']:.2f} MB")
        print("\nBy season:")
        for season, data in summary["seasons"].items():
            print(f"  {season}:")
            print(f"    Events files: {data['events_files']}")
            print(f"    Odds files: {data['odds_files']}")
            print(f"    Props files: {data['props_files']}")
            print(f"    Size: {data['total_size_mb']:.2f} MB")
        return 0
    
    # Validate API key
    api_key = settings.the_odds_api_key
    if not api_key:
        logger.error("THE_ODDS_API_KEY not set. Cannot proceed.")
        return 1
    
    # Expand 'all' markets
    market_groups = args.markets
    if "all" in market_groups:
        market_groups = ["featured", "periods", "alternates", "team_totals"]
        if args.include_props:
            market_groups.extend(["player_props", "alternate_props"])
    
    # Create orchestrator
    orchestrator = HistoricalIngestionOrchestrator(
        api_key=api_key,
        storage=storage,
        include_player_props=args.include_props,
        rate_limit=args.rate_limit,
    )
    
    # Run ingestion
    try:
        if args.season:
            progress = await orchestrator.ingest_season(
                season=args.season,
                market_groups=market_groups,
                include_playoffs=not args.no_playoffs,
                resume=args.resume,
                dry_run=args.dry_run,
            )
        else:
            if not args.end_date:
                logger.error("--end-date required when using --start-date")
                return 1
            
            start = _parse_date(args.start_date)
            end = _parse_date(args.end_date)
            
            if end < start:
                logger.error("End date must be after start date")
                return 1
            
            progress = await orchestrator.ingest_date_range(
                start_date=start,
                end_date=end,
                market_groups=market_groups,
                dry_run=args.dry_run,
            )
        
        # Print summary
        print("\n=== Ingestion Summary ===")
        print(f"Season: {progress.season}")
        print(f"Dates completed: {len(progress.dates_completed)}")
        print(f"Dates failed: {len(progress.dates_failed)}")
        print(f"Total events: {progress.total_events_fetched}")
        print(f"API calls: {progress.total_api_calls}")
        print(f"Estimated credits: {progress.estimated_credits_used:,}")
        
        if progress.dates_failed:
            print(f"\nFailed dates: {', '.join(progress.dates_failed[:10])}")
            if len(progress.dates_failed) > 10:
                print(f"  ... and {len(progress.dates_failed) - 10} more")
        
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
