"""
BetsAPI Client for NBA Data Ingestion.

BetsAPI provides:
- Live/in-play odds
- Pre-match odds
- Player props
- All quarters (Q1, Q2, Q3, Q4)
- Multiple bookmakers

API Base URLs:
- Primary: https://api.b365api.com/v3
- Fallback: https://api.betsapi.com/v3

Rate Limits:
- Default: 3,600 requests/hour
- Can purchase volume packages for higher limits

Usage:
    from src.ingestion.betsapi import BetsAPIClient

    client = BetsAPIClient()
    events = await client.fetch_upcoming_events()
    odds = await client.fetch_odds(event_id="12345")
    live = await client.fetch_inplay_events()
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, date, timezone
from typing import Any, Dict, List, Optional
import json
from pathlib import Path

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.utils.logging import get_logger
from src.utils.api_cache import api_cache, APICache
from src.ingestion.standardize import normalize_team_to_espn

logger = get_logger(__name__)

# BetsAPI Constants
PRIMARY_BASE_URL = "https://api.b365api.com/v3"
FALLBACK_BASE_URL = "https://api.betsapi.com/v3"
NBA_SPORT_ID = 18  # Basketball
NBA_LEAGUE_ID = 12733  # NBA

TIMEOUT = 30


@dataclass
class BetsAPIEvent:
    """Parsed BetsAPI event."""
    event_id: str
    home_team: str
    away_team: str
    home_team_espn: str  # Standardized
    away_team_espn: str  # Standardized
    commence_time: datetime
    league: str
    is_inplay: bool = False
    score_home: Optional[int] = None
    score_away: Optional[int] = None
    quarter: Optional[int] = None
    time_remaining: Optional[str] = None


@dataclass
class BetsAPIOdds:
    """Parsed BetsAPI odds."""
    event_id: str
    bookmaker: str
    market_type: str  # spreads, totals, h2h, props
    period: str  # full_game, 1h, q1, q2, q3, q4
    home_odds: Optional[float] = None
    away_odds: Optional[float] = None
    spread_line: Optional[float] = None
    total_line: Optional[float] = None
    over_odds: Optional[float] = None
    under_odds: Optional[float] = None
    updated_at: Optional[datetime] = None


class BetsAPIClient:
    """Client for BetsAPI with NBA-focused ingestion."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize BetsAPI client.

        Args:
            api_key: BetsAPI key. Defaults to settings.betsapi_key.
        """
        self.api_key = api_key or settings.betsapi_key
        if not self.api_key:
            logger.warning("BETSAPI_KEY not configured - BetsAPI calls will fail")

        self.base_url = PRIMARY_BASE_URL
        self.output_dir = Path(settings.data_raw_dir) / "betsapi"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
    )
    async def _fetch(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make HTTP request to BetsAPI.

        Args:
            endpoint: API endpoint path
            params: Query parameters

        Returns:
            JSON response as dict
        """
        if not self.api_key:
            raise ValueError("BETSAPI_KEY not configured")

        url = f"{self.base_url}/{endpoint}"
        all_params = {"token": self.api_key}
        if params:
            all_params.update(params)

        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            try:
                resp = await client.get(url, params=all_params)
                resp.raise_for_status()
                data = resp.json()

                if data.get("success") == 0:
                    error_msg = data.get("error", "Unknown BetsAPI error")
                    raise ValueError(f"BetsAPI error: {error_msg}")

                return data

            except httpx.HTTPStatusError as e:
                # Try fallback URL on failure
                if self.base_url == PRIMARY_BASE_URL:
                    logger.warning(f"Primary BetsAPI failed, trying fallback: {e}")
                    self.base_url = FALLBACK_BASE_URL
                    return await self._fetch(endpoint, params)
                raise

    def _save(self, name: str, data: Dict[str, Any]) -> str:
        """Save response to JSON file."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = self.output_dir / f"{name}_{ts}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return str(path)

    # =========================================================================
    # EVENT ENDPOINTS
    # =========================================================================

    async def fetch_upcoming_events(
        self,
        sport_id: int = NBA_SPORT_ID,
        league_id: int = NBA_LEAGUE_ID,
        day: Optional[str] = None,
    ) -> List[BetsAPIEvent]:
        """Fetch upcoming NBA events.

        Args:
            sport_id: Sport ID (18 = Basketball)
            league_id: League ID (12733 = NBA)
            day: Date filter (YYYYMMDD format). Defaults to today.

        Returns:
            List of parsed events
        """
        target_day = day or datetime.now().strftime("%Y%m%d")

        cache_key = f"betsapi_upcoming_{league_id}_{target_day}"

        async def fetch():
            data = await self._fetch("events/upcoming", {
                "sport_id": sport_id,
                "league_id": league_id,
                "day": target_day,
            })
            self._save(f"upcoming_{target_day}", data)
            return data

        data = await api_cache.get_or_fetch(
            key=cache_key,
            fetch_fn=fetch,
            ttl_hours=APICache.TTL_FREQUENT,
            source="betsapi",
            endpoint="/events/upcoming",
        )

        return self._parse_events(data.get("results", []))

    async def fetch_inplay_events(
        self,
        sport_id: int = NBA_SPORT_ID,
    ) -> List[BetsAPIEvent]:
        """Fetch live/in-play NBA events.

        Args:
            sport_id: Sport ID (18 = Basketball)

        Returns:
            List of live events with scores
        """
        # Live data - very short cache (5 minutes)
        cache_key = f"betsapi_inplay_{sport_id}_{datetime.now().strftime('%Y%m%d_%H%M')}"

        async def fetch():
            data = await self._fetch("events/inplay", {
                "sport_id": sport_id,
            })
            self._save("inplay", data)
            return data

        data = await api_cache.get_or_fetch(
            key=cache_key,
            fetch_fn=fetch,
            ttl_hours=0.08,  # ~5 minutes
            source="betsapi",
            endpoint="/events/inplay",
        )

        # Filter to NBA only
        nba_events = [
            e for e in data.get("results", [])
            if e.get("league", {}).get("id") == NBA_LEAGUE_ID
        ]

        return self._parse_events(nba_events, is_inplay=True)

    async def fetch_ended_events(
        self,
        sport_id: int = NBA_SPORT_ID,
        league_id: int = NBA_LEAGUE_ID,
        day: Optional[str] = None,
    ) -> List[BetsAPIEvent]:
        """Fetch ended/completed NBA events.

        Args:
            sport_id: Sport ID
            league_id: League ID
            day: Date filter (YYYYMMDD)

        Returns:
            List of completed events with final scores
        """
        target_day = day or datetime.now().strftime("%Y%m%d")

        cache_key = f"betsapi_ended_{league_id}_{target_day}"

        async def fetch():
            data = await self._fetch("events/ended", {
                "sport_id": sport_id,
                "league_id": league_id,
                "day": target_day,
            })
            self._save(f"ended_{target_day}", data)
            return data

        data = await api_cache.get_or_fetch(
            key=cache_key,
            fetch_fn=fetch,
            ttl_hours=APICache.TTL_DAILY,
            source="betsapi",
            endpoint="/events/ended",
        )

        return self._parse_events(data.get("results", []))

    # =========================================================================
    # ODDS ENDPOINTS
    # =========================================================================

    async def fetch_odds(
        self,
        event_id: str,
        source: str = "bet365",
    ) -> List[BetsAPIOdds]:
        """Fetch odds for a specific event.

        Args:
            event_id: BetsAPI event ID
            source: Bookmaker source (bet365, pinnacle, etc.)

        Returns:
            List of parsed odds across all markets
        """
        cache_key = f"betsapi_odds_{event_id}_{source}_{datetime.now().strftime('%Y%m%d_%H')}"

        async def fetch():
            data = await self._fetch(f"event/odds", {
                "event_id": event_id,
                "source": source,
            })
            return data

        data = await api_cache.get_or_fetch(
            key=cache_key,
            fetch_fn=fetch,
            ttl_hours=APICache.TTL_LIVE,
            source="betsapi",
            endpoint="/event/odds",
        )

        return self._parse_odds(event_id, data.get("results", {}))

    async def fetch_odds_summary(
        self,
        event_id: str,
    ) -> Dict[str, Any]:
        """Fetch odds summary for an event (all bookmakers).

        Args:
            event_id: BetsAPI event ID

        Returns:
            Dict with odds from multiple bookmakers
        """
        cache_key = f"betsapi_odds_summary_{event_id}_{datetime.now().strftime('%Y%m%d_%H')}"

        async def fetch():
            data = await self._fetch("event/odds/summary", {
                "event_id": event_id,
            })
            return data

        return await api_cache.get_or_fetch(
            key=cache_key,
            fetch_fn=fetch,
            ttl_hours=APICache.TTL_LIVE,
            source="betsapi",
            endpoint="/event/odds/summary",
        )

    # =========================================================================
    # BATCH OPERATIONS
    # =========================================================================

    async def fetch_all_nba_odds(
        self,
        target_date: Optional[str] = None,
    ) -> Dict[str, List[BetsAPIOdds]]:
        """Fetch odds for all NBA events on a date.

        Args:
            target_date: Date (YYYYMMDD). Defaults to today.

        Returns:
            Dict mapping event_id to list of odds
        """
        # Get upcoming events
        events = await self.fetch_upcoming_events(day=target_date)

        if not events:
            logger.info("No NBA events found for odds fetch")
            return {}

        # Fetch odds for each event (with rate limiting)
        all_odds = {}
        for event in events[:15]:  # Limit to 15 to conserve API calls
            try:
                odds = await self.fetch_odds(event.event_id)
                all_odds[event.event_id] = odds
                await asyncio.sleep(0.2)  # Rate limit
            except Exception as e:
                logger.warning(f"Failed to fetch odds for {event.event_id}: {e}")

        logger.info(f"Fetched odds for {len(all_odds)} NBA events")
        return all_odds

    async def fetch_live_nba(self) -> Dict[str, Any]:
        """Fetch all live NBA data (events + odds).

        Returns:
            Dict with live events and their current odds
        """
        events = await self.fetch_inplay_events()

        result = {
            "events": events,
            "odds": {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Fetch live odds for each event
        for event in events[:10]:  # Limit for rate limiting
            try:
                odds = await self.fetch_odds(event.event_id)
                result["odds"][event.event_id] = odds
            except Exception as e:
                logger.warning(f"Failed to fetch live odds for {event.event_id}: {e}")

        return result

    # =========================================================================
    # PARSING HELPERS
    # =========================================================================

    def _parse_events(
        self,
        events: List[Dict[str, Any]],
        is_inplay: bool = False,
    ) -> List[BetsAPIEvent]:
        """Parse raw BetsAPI events to dataclass."""
        parsed = []

        for event in events:
            try:
                home_raw = event.get("home", {}).get("name", "")
                away_raw = event.get("away", {}).get("name", "")

                # Standardize team names
                home_espn, _ = normalize_team_to_espn(home_raw, source="betsapi")
                away_espn, _ = normalize_team_to_espn(away_raw, source="betsapi")

                # Parse timestamp
                time_val = event.get("time")
                if time_val:
                    commence = datetime.fromtimestamp(int(time_val), tz=timezone.utc)
                else:
                    commence = datetime.now(timezone.utc)

                # Parse scores for live events
                scores = event.get("scores", {})
                score_home = None
                score_away = None
                if scores:
                    score_home = int(scores.get("home", {}).get("score", 0))
                    score_away = int(scores.get("away", {}).get("score", 0))

                parsed.append(BetsAPIEvent(
                    event_id=str(event.get("id", "")),
                    home_team=home_raw,
                    away_team=away_raw,
                    home_team_espn=home_espn,
                    away_team_espn=away_espn,
                    commence_time=commence,
                    league=event.get("league", {}).get("name", "NBA"),
                    is_inplay=is_inplay,
                    score_home=score_home,
                    score_away=score_away,
                    quarter=event.get("timer", {}).get("q"),
                    time_remaining=event.get("timer", {}).get("tm"),
                ))
            except Exception as e:
                logger.warning(f"Failed to parse BetsAPI event: {e}")

        return parsed

    def _parse_odds(
        self,
        event_id: str,
        odds_data: Dict[str, Any],
    ) -> List[BetsAPIOdds]:
        """Parse raw BetsAPI odds to dataclass."""
        parsed = []

        # Process different market types
        for market_key, market_data in odds_data.items():
            try:
                # Determine period and market type from key
                period = "full_game"
                market_type = market_key

                if "_1h" in market_key or "_h1" in market_key:
                    period = "1h"
                    market_type = market_key.replace("_1h", "").replace("_h1", "")
                elif "_q1" in market_key:
                    period = "q1"
                    market_type = market_key.replace("_q1", "")
                elif "_q2" in market_key:
                    period = "q2"
                    market_type = market_key.replace("_q2", "")
                elif "_q3" in market_key:
                    period = "q3"
                    market_type = market_key.replace("_q3", "")
                elif "_q4" in market_key:
                    period = "q4"
                    market_type = market_key.replace("_q4", "")

                # Parse based on market structure
                if isinstance(market_data, dict):
                    for bookmaker, bookie_odds in market_data.items():
                        if isinstance(bookie_odds, dict):
                            parsed.append(BetsAPIOdds(
                                event_id=event_id,
                                bookmaker=bookmaker,
                                market_type=market_type,
                                period=period,
                                home_odds=self._safe_float(bookie_odds.get("home")),
                                away_odds=self._safe_float(bookie_odds.get("away")),
                                spread_line=self._safe_float(bookie_odds.get("handicap")),
                                total_line=self._safe_float(bookie_odds.get("total")),
                                over_odds=self._safe_float(bookie_odds.get("over")),
                                under_odds=self._safe_float(bookie_odds.get("under")),
                            ))

            except Exception as e:
                logger.warning(f"Failed to parse odds for {market_key}: {e}")

        return parsed

    def _safe_float(self, val: Any) -> Optional[float]:
        """Safely convert value to float."""
        if val is None:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def fetch_betsapi_events(day: Optional[str] = None) -> List[BetsAPIEvent]:
    """Fetch upcoming NBA events from BetsAPI."""
    client = BetsAPIClient()
    return await client.fetch_upcoming_events(day=day)


async def fetch_betsapi_live() -> Dict[str, Any]:
    """Fetch live NBA data from BetsAPI."""
    client = BetsAPIClient()
    return await client.fetch_live_nba()


async def fetch_betsapi_odds(event_id: str) -> List[BetsAPIOdds]:
    """Fetch odds for a specific event."""
    client = BetsAPIClient()
    return await client.fetch_odds(event_id)
