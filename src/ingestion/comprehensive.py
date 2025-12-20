"""
Comprehensive Data Ingestion Module.

Uses ALL available API endpoints with intelligent caching to minimize API calls
while maximizing data richness.

Data Sources:
1. The Odds API - Betting odds (FG, 1H, Q1 markets)
2. API-Basketball - Game stats, team stats, standings
3. Action Network - Betting splits (requires login)
4. ESPN - Schedule, injuries (FREE)

Caching Strategy:
- Static data (teams): 7 days
- Semi-static (standings, stats): 24 hours
- Dynamic (odds, injuries): 2 hours
- Real-time (live odds): 15 minutes

Usage:
    from src.ingestion.comprehensive import ComprehensiveIngestion

    ingestion = ComprehensiveIngestion()
    await ingestion.ingest_all()  # Full ingestion with caching
    await ingestion.ingest_for_slate(date="2024-01-15")  # Slate-specific

    # Check what data we have
    status = ingestion.get_status()
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Any, Dict, List, Optional
from pathlib import Path

from src.config import settings
from src.utils.logging import get_logger
from src.utils.api_cache import api_cache, APICache

logger = get_logger(__name__)


@dataclass
class IngestionResult:
    """Result from an ingestion operation."""
    source: str
    endpoint: str
    success: bool
    record_count: int = 0
    cached: bool = False
    error: Optional[str] = None
    duration_ms: int = 0


@dataclass
class IngestionStatus:
    """Overall ingestion status."""
    results: List[IngestionResult] = field(default_factory=list)
    total_api_calls: int = 0
    total_cache_hits: int = 0
    errors: List[str] = field(default_factory=list)


class ComprehensiveIngestion:
    """
    Comprehensive data ingestion with intelligent caching.

    Uses ALL available endpoints while minimizing API usage through caching.
    """

    def __init__(self, force_refresh: bool = False):
        """Initialize ingestion.

        Args:
            force_refresh: If True, bypass cache for all calls
        """
        self.force_refresh = force_refresh
        self.status = IngestionStatus()

    # =========================================================================
    # THE ODDS API - Full Integration
    # =========================================================================

    async def fetch_the_odds_events(self, sport: str = "basketball_nba") -> List[Dict]:
        """Fetch event list from The Odds API (needed for event-specific odds).

        TTL: 2 hours (events change with schedule)
        """
        from src.ingestion import the_odds

        key = f"the_odds_events_{sport}_{date.today().isoformat()}"

        async def fetch():
            return await the_odds.fetch_events(sport=sport)

        data = await api_cache.get_or_fetch(
            key=key,
            fetch_fn=fetch,
            ttl_hours=APICache.TTL_FREQUENT,
            source="the_odds",
            endpoint="/events",
            force_refresh=self.force_refresh,
        )

        self._record_result("the_odds", "/events", True, len(data), cached=not self.force_refresh)
        return data

    async def fetch_the_odds_full_game(self, sport: str = "basketball_nba") -> List[Dict]:
        """Fetch full game odds (spreads, totals, moneyline).

        TTL: 15 minutes (odds are live)
        """
        from src.ingestion import the_odds

        key = f"the_odds_fg_{sport}_{datetime.now().strftime('%Y%m%d_%H')}"

        async def fetch():
            return await the_odds.fetch_odds(
                sport=sport,
                markets="h2h,spreads,totals",
            )

        data = await api_cache.get_or_fetch(
            key=key,
            fetch_fn=fetch,
            ttl_hours=APICache.TTL_LIVE,
            source="the_odds",
            endpoint="/odds",
            force_refresh=self.force_refresh,
        )

        self._record_result("the_odds", "/odds (FG)", True, len(data))
        return data

    async def fetch_the_odds_first_half(
        self,
        event_ids: Optional[List[str]] = None,
        sport: str = "basketball_nba",
    ) -> List[Dict]:
        """Fetch first half odds for events.

        TTL: 15 minutes (odds are live)

        Args:
            event_ids: List of event IDs to fetch. If None, fetches events first.
            sport: Sport identifier
        """
        from src.ingestion import the_odds

        # Get event IDs if not provided
        if event_ids is None:
            events = await self.fetch_the_odds_events(sport)
            event_ids = [e.get("id") for e in events if e.get("id")]

        if not event_ids:
            logger.warning("No event IDs available for first half odds")
            return []

        results = []
        for event_id in event_ids[:10]:  # Limit to 10 to conserve API calls
            key = f"the_odds_1h_{event_id}_{datetime.now().strftime('%Y%m%d_%H')}"

            async def fetch(eid=event_id):
                return await the_odds.fetch_event_odds(
                    event_id=eid,
                    sport=sport,
                    markets="spreads_h1,totals_h1,h2h_h1",
                )

            try:
                data = await api_cache.get_or_fetch(
                    key=key,
                    fetch_fn=fetch,
                    ttl_hours=APICache.TTL_LIVE,
                    source="the_odds",
                    endpoint=f"/events/{event_id}/odds",
                    force_refresh=self.force_refresh,
                )
                if data:
                    results.append(data)
            except Exception as e:
                logger.warning(f"Failed to fetch 1H odds for {event_id}: {e}")

        self._record_result("the_odds", "/events/*/odds (1H)", True, len(results))
        return results

    async def fetch_the_odds_scores(self, sport: str = "basketball_nba") -> List[Dict]:
        """Fetch recent scores for reconciliation.

        TTL: 2 hours
        """
        from src.ingestion import the_odds

        key = f"the_odds_scores_{sport}_{date.today().isoformat()}"

        async def fetch():
            return await the_odds.fetch_scores(sport=sport)

        data = await api_cache.get_or_fetch(
            key=key,
            fetch_fn=fetch,
            ttl_hours=APICache.TTL_FREQUENT,
            source="the_odds",
            endpoint="/scores",
            force_refresh=self.force_refresh,
        )

        self._record_result("the_odds", "/scores", True, len(data))
        return data

    async def fetch_the_odds_betting_splits(
        self,
        sport: str = "basketball_nba",
    ) -> List[Dict]:
        """Fetch betting splits from The Odds API.

        TTL: 2 hours

        Note: Requires paid plan (Group 2+). Returns empty on free tier.
        """
        from src.ingestion import the_odds

        key = f"the_odds_splits_{sport}_{date.today().isoformat()}"

        async def fetch():
            try:
                return await the_odds.fetch_betting_splits(sport=sport)
            except Exception as e:
                if "403" in str(e) or "Forbidden" in str(e):
                    logger.info("Betting splits requires paid plan - skipping")
                    return []
                raise

        data = await api_cache.get_or_fetch(
            key=key,
            fetch_fn=fetch,
            ttl_hours=APICache.TTL_FREQUENT,
            source="the_odds",
            endpoint="/betting-splits",
            force_refresh=self.force_refresh,
        )

        self._record_result("the_odds", "/betting-splits", True, len(data))
        return data

    # =========================================================================
    # API-BASKETBALL - Full Integration
    # =========================================================================

    async def fetch_api_basketball_essential(self) -> Dict[str, Any]:
        """Fetch essential API-Basketball data with caching.

        Tier 1 endpoints:
        - /teams (TTL: 7 days)
        - /games (TTL: 24 hours)
        - /statistics (TTL: 24 hours)
        - /games/statistics/teams (TTL: 24 hours)
        """
        from src.ingestion.api_basketball import APIBasketballClient

        client = APIBasketballClient()
        results = {}

        # Teams - very static, cache 7 days
        key_teams = f"api_bball_teams_{settings.current_season}"
        results["teams"] = await api_cache.get_or_fetch(
            key=key_teams,
            fetch_fn=client.fetch_teams,
            ttl_hours=APICache.TTL_STATIC,
            source="api_basketball",
            endpoint="/teams",
            force_refresh=self.force_refresh,
        )

        # Games - changes daily
        key_games = f"api_bball_games_{settings.current_season}_{date.today().isoformat()}"
        results["games"] = await api_cache.get_or_fetch(
            key=key_games,
            fetch_fn=client.fetch_games,
            ttl_hours=APICache.TTL_DAILY,
            source="api_basketball",
            endpoint="/games",
            force_refresh=self.force_refresh,
        )

        # Statistics - changes daily
        key_stats = f"api_bball_stats_{settings.current_season}_{date.today().isoformat()}"
        results["statistics"] = await api_cache.get_or_fetch(
            key=key_stats,
            fetch_fn=client.fetch_statistics,
            ttl_hours=APICache.TTL_DAILY,
            source="api_basketball",
            endpoint="/statistics",
            force_refresh=self.force_refresh,
        )

        # Game stats - changes daily
        key_game_stats = f"api_bball_game_stats_{settings.current_season}_{date.today().isoformat()}"
        results["game_stats_teams"] = await api_cache.get_or_fetch(
            key=key_game_stats,
            fetch_fn=client.fetch_game_stats_teams,
            ttl_hours=APICache.TTL_DAILY,
            source="api_basketball",
            endpoint="/games/statistics/teams",
            force_refresh=self.force_refresh,
        )

        for endpoint, data in results.items():
            count = data.count if hasattr(data, "count") else len(data) if isinstance(data, list) else 0
            self._record_result("api_basketball", f"/{endpoint}", True, count)

        return results

    async def fetch_api_basketball_standings(self) -> Any:
        """Fetch standings with caching.

        TTL: 24 hours
        """
        from src.ingestion.api_basketball import APIBasketballClient

        client = APIBasketballClient()
        key = f"api_bball_standings_{settings.current_season}_{date.today().isoformat()}"

        result = await api_cache.get_or_fetch(
            key=key,
            fetch_fn=client.fetch_standings,
            ttl_hours=APICache.TTL_DAILY,
            source="api_basketball",
            endpoint="/standings",
            force_refresh=self.force_refresh,
        )

        count = result.count if hasattr(result, "count") else 0
        self._record_result("api_basketball", "/standings", True, count)
        return result

    # =========================================================================
    # ACTION NETWORK - Betting Splits
    # =========================================================================

    async def fetch_action_network_splits(self, target_date: Optional[str] = None) -> List[Dict]:
        """Fetch betting splits from Action Network.

        Requires ACTION_NETWORK_USERNAME and ACTION_NETWORK_PASSWORD.

        TTL: 2 hours

        Args:
            target_date: Date string (YYYY-MM-DD). Defaults to today.
        """
        from src.ingestion.betting_splits import fetch_splits_action_network

        # Check if credentials are configured
        if not settings.action_network_username or not settings.action_network_password:
            logger.info("Action Network credentials not configured - skipping")
            self._record_result("action_network", "/games/nba", False, 0, error="No credentials")
            return []

        target = target_date or date.today().isoformat()
        key = f"action_network_splits_{target}"

        async def fetch():
            return await fetch_splits_action_network(date=target)

        try:
            data = await api_cache.get_or_fetch(
                key=key,
                fetch_fn=fetch,
                ttl_hours=APICache.TTL_FREQUENT,
                source="action_network",
                endpoint="/games/nba",
                force_refresh=self.force_refresh,
            )

            self._record_result("action_network", "/games/nba", True, len(data))
            return data
        except Exception as e:
            logger.warning(f"Action Network fetch failed: {e}")
            self._record_result("action_network", "/games/nba", False, 0, error=str(e))
            return []

    # =========================================================================
    # ESPN - FREE Data (Schedule, Injuries)
    # =========================================================================

    async def fetch_espn_schedule(self, target_date: Optional[str] = None) -> Dict[str, Any]:
        """Fetch schedule from ESPN (FREE, unlimited).

        TTL: 24 hours

        Args:
            target_date: Date string (YYYYMMDD format). Defaults to today.
        """
        from src.ingestion.espn import fetch_espn_schedule

        target = target_date or datetime.now().strftime("%Y%m%d")
        key = f"espn_schedule_{target}"

        data = await api_cache.get_or_fetch(
            key=key,
            fetch_fn=lambda: fetch_espn_schedule(target),
            ttl_hours=APICache.TTL_DAILY,
            source="espn",
            endpoint="/scoreboard",
            force_refresh=self.force_refresh,
        )

        event_count = len(data.get("events", [])) if isinstance(data, dict) else 0
        self._record_result("espn", "/scoreboard", True, event_count)
        return data

    async def fetch_espn_injuries(self) -> List[Dict]:
        """Fetch injuries from ESPN (FREE, unlimited).

        TTL: 2 hours
        """
        from src.ingestion.injuries import fetch_injuries_espn

        key = f"espn_injuries_{date.today().isoformat()}"

        data = await api_cache.get_or_fetch(
            key=key,
            fetch_fn=fetch_injuries_espn,
            ttl_hours=APICache.TTL_FREQUENT,
            source="espn",
            endpoint="/injuries",
            force_refresh=self.force_refresh,
        )

        self._record_result("espn", "/injuries", True, len(data))
        return data

    # =========================================================================
    # COMPOSITE OPERATIONS
    # =========================================================================

    async def ingest_all(self) -> IngestionStatus:
        """Run full ingestion using ALL available endpoints with caching.

        Returns:
            IngestionStatus with results from all sources
        """
        logger.info("Starting comprehensive ingestion...")

        # Run all ingestions in parallel where possible
        tasks = [
            # The Odds API - Core
            self.fetch_the_odds_full_game(),
            self.fetch_the_odds_events(),
            self.fetch_the_odds_scores(),
            self.fetch_the_odds_betting_splits(),

            # API-Basketball - Essential
            self.fetch_api_basketball_essential(),
            self.fetch_api_basketball_standings(),

            # Action Network - Betting splits
            self.fetch_action_network_splits(),

            # ESPN - FREE
            self.fetch_espn_schedule(),
            self.fetch_espn_injuries(),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for result in results:
            if isinstance(result, Exception):
                self.status.errors.append(str(result))
                logger.error(f"Ingestion error: {result}")

        # Fetch first-half odds (depends on events)
        try:
            await self.fetch_the_odds_first_half()
        except Exception as e:
            logger.warning(f"First half odds fetch failed: {e}")
            self.status.errors.append(f"1H odds: {e}")

        logger.info(f"Ingestion complete: {len(self.status.results)} endpoints, {len(self.status.errors)} errors")
        return self.status

    async def ingest_for_slate(self, target_date: Optional[str] = None) -> IngestionStatus:
        """Ingest data needed for a specific slate.

        Optimized for daily picks generation.

        Args:
            target_date: Date string (YYYY-MM-DD). Defaults to today.
        """
        target = target_date or date.today().isoformat()
        logger.info(f"Ingesting for slate: {target}")

        # Essential for slate
        tasks = [
            self.fetch_the_odds_full_game(),
            self.fetch_the_odds_events(),
            self.fetch_espn_schedule(target.replace("-", "")),
            self.fetch_espn_injuries(),
            self.fetch_api_basketball_essential(),
        ]

        # Optional enhancements
        optional_tasks = [
            self.fetch_action_network_splits(target),
            self.fetch_the_odds_betting_splits(),
        ]

        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.gather(*optional_tasks, return_exceptions=True)

        # Fetch first-half odds
        try:
            await self.fetch_the_odds_first_half()
        except Exception as e:
            logger.warning(f"First half odds: {e}")

        return self.status

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _record_result(
        self,
        source: str,
        endpoint: str,
        success: bool,
        count: int,
        cached: bool = False,
        error: Optional[str] = None,
    ) -> None:
        """Record an ingestion result."""
        result = IngestionResult(
            source=source,
            endpoint=endpoint,
            success=success,
            record_count=count,
            cached=cached,
            error=error,
        )
        self.status.results.append(result)

        if cached:
            self.status.total_cache_hits += 1
        else:
            self.status.total_api_calls += 1

    def get_status(self) -> Dict[str, Any]:
        """Get ingestion status summary."""
        successful = sum(1 for r in self.status.results if r.success)
        failed = sum(1 for r in self.status.results if not r.success)
        total_records = sum(r.record_count for r in self.status.results)

        return {
            "endpoints_called": len(self.status.results),
            "successful": successful,
            "failed": failed,
            "total_records": total_records,
            "api_calls": self.status.total_api_calls,
            "cache_hits": self.status.total_cache_hits,
            "errors": self.status.errors,
            "cache_stats": api_cache.get_stats(),
        }


# Convenience function
async def run_comprehensive_ingestion(force_refresh: bool = False) -> IngestionStatus:
    """Run full comprehensive ingestion.

    Args:
        force_refresh: If True, bypass all caches

    Returns:
        IngestionStatus with results
    """
    ingestion = ComprehensiveIngestion(force_refresh=force_refresh)
    return await ingestion.ingest_all()
