from __future__ import annotations
import os
import datetime as dt
from typing import Any, Dict, List
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.utils.logging import get_logger
from src.utils.circuit_breaker import get_odds_api_breaker
from src.utils.security import mask_api_key
from src.ingestion.standardize import standardize_game_data

logger = get_logger(__name__)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
)
async def fetch_odds(
    sport: str = "basketball_nba",
    regions: str = "us",
    markets: str = "h2h,spreads,totals",
    standardize: bool = True,  # Always True - standardization is mandatory
) -> list[Dict[str, Any]]:
    """
    Fetch odds from The Odds API and standardize to ESPN format.
    
    Team name standardization is MANDATORY and always performed to ensure
    data consistency across sources.
    
    Args:
        sport: Sport identifier
        regions: Regions to fetch odds for
        markets: Markets to fetch
        standardize: Always True - team names are always standardized (kept for API compatibility)
    
    Returns:
        List of game dictionaries with standardized team names in ESPN format
    """
    logger.info(f"Fetching odds for {sport} with markets: {markets}")
    
    # Validate API key
    if not settings.the_odds_api_key:
        raise ValueError("THE_ODDS_API_KEY is not set")
    
    params = {
        "apiKey": settings.the_odds_api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    
    # Use circuit breaker to prevent cascading failures
    breaker = get_odds_api_breaker()
    
    async def _fetch_with_client():
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{settings.the_odds_base_url}/sports/{sport}/odds", params=params
            )
            resp.raise_for_status()
            return resp.json()
    
    try:
        data = await breaker.call_async(_fetch_with_client)
        logger.info(f"Successfully fetched {len(data)} games with odds")
    except Exception as e:
        logger.error(f"Failed to fetch odds from The Odds API: {e}")
        raise

    # ALWAYS standardize team names to ESPN format (mandatory)
    standardized_data = []
    invalid_count = 0
    for game in data:
        try:
            standardized = standardize_game_data(game, source="the_odds")
            # Only include games with valid team names (prevent fake data)
            if standardized.get("_data_valid", False):
                standardized_data.append(standardized)
            else:
                invalid_count += 1
                logger.warning(
                    "Skipping game with invalid team names: "
                    f"home='{standardized.get('home_team', 'N/A')}', "
                    f"away='{standardized.get('away_team', 'N/A')}'"
                )
        except Exception as e:
            logger.error(
                "Error standardizing game data: "
                f"{e}. Game: {game.get('home_team', 'N/A')} vs {game.get('away_team', 'N/A')}"
            )
            # Do NOT add invalid data - skip it entirely
            invalid_count += 1

    logger.info(
        f"Standardized {len(standardized_data)} valid games (skipped {invalid_count} invalid games)"
    )
    if invalid_count > 0:
        logger.warning(
            f"⚠️  {invalid_count} games were skipped due to invalid/unstandardized team names"
        )
    return standardized_data


# All available period/alt markets for NBA
ALL_PERIOD_MARKETS = (
    # First Half
    "h2h_h1,spreads_h1,totals_h1,"
    # First Quarter
    "h2h_q1,spreads_q1,totals_q1,"
    # Second Quarter
    "h2h_q2,spreads_q2,totals_q2,"
    # Third Quarter
    "h2h_q3,spreads_q3,totals_q3,"
    # Fourth Quarter
    "h2h_q4,spreads_q4,totals_q4,"
    # Alternate lines
    "alternate_spreads,alternate_totals,"
    # Player props (if available)
    "player_points,player_rebounds,player_assists"
)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
)
async def fetch_event_odds(
    event_id: str,
    sport: str = "basketball_nba",
    regions: str = "us",
    markets: str = ALL_PERIOD_MARKETS,
    standardize: bool = True,
) -> Dict[str, Any]:
    """
    Fetch odds for a specific event from The Odds API.

    Fetches ALL available markets including:
    - First half (1H): spreads, totals, moneyline
    - All quarters (Q1, Q2, Q3, Q4): spreads, totals, moneyline
    - Alternate lines: spreads, totals
    - Player props: points, rebounds, assists

    Args:
        event_id: The event ID from The Odds API
        sport: Sport identifier
        regions: Regions to fetch odds for
        markets: Markets to fetch (default: ALL available markets)
        standardize: If True, standardize team names to ESPN format

    Returns:
        Event dictionary with odds data including all period markets
    """
    logger.info(f"Fetching event odds for {event_id} with markets: {markets}")
    params = {
        "apiKey": settings.the_odds_api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{settings.the_odds_base_url}/sports/{sport}/events/{event_id}/odds",
            params=params
        )
        resp.raise_for_status()
        data = resp.json()
        logger.info(f"Successfully fetched event odds for {event_id}")
        
        # Standardize team names to ESPN format
        if standardize:
            try:
                data = standardize_game_data(data, source="the_odds")
            except Exception as e:
                logger.warning(f"Error standardizing event odds: {e}")
        
        return data


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
)
async def fetch_events(
    sport: str = "basketball_nba",
    standardize: bool = True,
) -> List[Dict[str, Any]]:
    """
    Fetch all upcoming events (games) with their IDs.
    
    Use this to get event IDs for fetching event-specific odds
    (e.g., first half markets, alternate lines).
    
    Args:
        sport: Sport identifier
        standardize: If True, standardize team names to ESPN format
    
    Returns:
        List of event dictionaries with IDs, teams, and commence times
    """
    logger.info(f"Fetching events for {sport}")
    params = {
        "apiKey": settings.the_odds_api_key,
        "dateFormat": "iso",
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{settings.the_odds_base_url}/sports/{sport}/events",
            params=params
        )
        resp.raise_for_status()
        data = resp.json()
        logger.info(f"Successfully fetched {len(data)} events")
        
        # Standardize team names to ESPN format
        if standardize:
            standardized_data = []
            for event in data:
                try:
                    standardized = standardize_game_data(event, source="the_odds")
                    standardized_data.append(standardized)
                except Exception as e:
                    logger.warning(f"Error standardizing event data: {e}")
                    standardized_data.append(event)
            return standardized_data
        
        return data


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
)
async def fetch_scores(
    sport: str = "basketball_nba",
    days_from: int = 1,
    standardize: bool = True,
) -> List[Dict[str, Any]]:
    """
    Fetch recent game scores for result verification.
    
    Args:
        sport: Sport identifier
        days_from: Number of days in the past to fetch scores for (1-3)
        standardize: If True, standardize team names to ESPN format
    
    Returns:
        List of games with scores (completed and in-progress)
    """
    logger.info(f"Fetching scores for {sport}, days_from={days_from}")
    params = {
        "apiKey": settings.the_odds_api_key,
        "daysFrom": str(days_from),
        "dateFormat": "iso",
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{settings.the_odds_base_url}/sports/{sport}/scores",
            params=params
        )
        resp.raise_for_status()
        data = resp.json()
        logger.info(f"Successfully fetched scores for {len(data)} games")
        
        # Standardize team names to ESPN format
        if standardize:
            standardized_data = []
            for game in data:
                try:
                    standardized = standardize_game_data(game, source="the_odds")
                    standardized_data.append(standardized)
                except Exception as e:
                    logger.warning(f"Error standardizing score data: {e}")
                    standardized_data.append(game)
            return standardized_data
        
        return data


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
)
async def fetch_historical_odds(
    date: str,
    sport: str = "basketball_nba",
    regions: str = "us",
    markets: str = "h2h,spreads,totals",
    standardize: bool = True,
) -> Dict[str, Any]:
    """
    Fetch historical odds snapshot for backtesting.
    
    NOTE: This endpoint may require a paid plan. Returns 403 if not enabled.
    
    Args:
        date: ISO format date string (e.g., "2025-12-01T12:00:00Z")
        sport: Sport identifier
        regions: Regions to fetch odds for
        markets: Markets to fetch
        standardize: If True, standardize team names to ESPN format
    
    Returns:
        Dictionary with historical odds data including:
        - timestamp: The snapshot timestamp
        - previous_timestamp: Previous available snapshot
        - next_timestamp: Next available snapshot
        - data: List of events with odds
    
    Raises:
        httpx.HTTPStatusError: If the request fails (403 = not enabled)
    """
    logger.info(f"Fetching historical odds for {sport} at {date}")
    params = {
        "apiKey": settings.the_odds_api_key,
        "regions": regions,
        "markets": markets,
        "date": date,
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{settings.the_odds_base_url}/historical/sports/{sport}/odds",
            params=params
        )
        resp.raise_for_status()
        data = resp.json()
        
        events = data.get("data", [])
        logger.info(f"Successfully fetched historical odds: {len(events)} events")
        
        # Standardize team names to ESPN format
        if standardize and events:
            standardized_events = []
            for event in events:
                try:
                    standardized = standardize_game_data(event, source="the_odds")
                    standardized_events.append(standardized)
                except Exception as e:
                    logger.warning(f"Error standardizing historical event: {e}")
                    standardized_events.append(event)
            data["data"] = standardized_events
        
        return data


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
)
async def fetch_historical_events(
    date: str,
    sport: str = "basketball_nba",
    standardize: bool = True,
) -> Dict[str, Any]:
    """
    Fetch historical events list for backtesting.
    
    NOTE: This endpoint may require a paid plan. Returns 403 if not enabled.
    
    Args:
        date: ISO format date string (e.g., "2025-12-01T12:00:00Z")
        sport: Sport identifier
        standardize: If True, standardize team names to ESPN format
    
    Returns:
        Dictionary with historical events data
    """
    logger.info(f"Fetching historical events for {sport} at {date}")
    params = {
        "apiKey": settings.the_odds_api_key,
        "date": date,
        "dateFormat": "iso",
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{settings.the_odds_base_url}/historical/sports/{sport}/events",
            params=params
        )
        resp.raise_for_status()
        data = resp.json()
        
        events = data.get("data", [])
        logger.info(f"Successfully fetched historical events: {len(events)} events")
        
        # Standardize team names
        if standardize and events:
            standardized_events = []
            for event in events:
                try:
                    standardized = standardize_game_data(event, source="the_odds")
                    standardized_events.append(standardized)
                except Exception as e:
                    logger.warning(f"Error standardizing historical event: {e}")
                    standardized_events.append(event)
            data["data"] = standardized_events
        
        return data


async def save_odds(
    data: list[Dict[str, Any]],
    out_dir: str | None = None,
    prefix: str = "odds",
) -> str:
    """Save odds data to a timestamped JSON file.
    
    Args:
        data: Data to save
        out_dir: Output directory (defaults to data/raw/the_odds)
        prefix: Filename prefix (e.g., "odds", "events", "scores")
    
    Returns:
        Path to saved file
    """
    out_dir = out_dir or os.path.join(settings.data_raw_dir, "the_odds")
    os.makedirs(out_dir, exist_ok=True)
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(out_dir, f"{prefix}_{ts}.json")
    # Write minimal JSON without external deps
    import json
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {prefix} data to {out_path}")
    return out_path


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
)
async def fetch_betting_splits(
    sport: str = "basketball_nba",
    regions: str = "us",
) -> list[Dict[str, Any]]:
    """
    Fetch betting splits (public percentages) from The Odds API.
    
    NOTE: This endpoint requires a paid plan (Group 2 or higher).
    Returns 403 if not enabled for your API key.
    
    Returns:
        List of game dictionaries with betting splits data
    """
    logger.info(f"Fetching betting splits for {sport}")
    params = {
        "apiKey": settings.the_odds_api_key,
        "regions": regions,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        url = f"{settings.the_odds_base_url}/sports/{sport}/betting-splits"
        resp = await client.get(url, params=params)
        
        if resp.status_code == 403:
            logger.warning("Betting splits endpoint not enabled for this API key.")
            return []
            
        resp.raise_for_status()
        data = resp.json()
        
        # Structure is usually {"data": [...]}
        splits = data.get("data", [])
        logger.info(f"Successfully fetched betting splits for {len(splits)} games")
        return splits


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
)
async def fetch_participants(
    sport: str = "basketball_nba",
    standardize: bool = True,
) -> List[Dict[str, Any]]:
    """
    Fetch participants (teams) from The Odds API.
    
    This endpoint provides a reference list of all participants (teams) 
    in the sport, useful for team name standardization and validation.
    
    Args:
        sport: Sport identifier
        standardize: If True, standardize team names to ESPN format
    
    Returns:
        List of participant dictionaries with team information
    """
    logger.info(f"Fetching participants for {sport}")
    params = {
        "apiKey": settings.the_odds_api_key,
        "dateFormat": "iso",
    }
    async with httpx.AsyncClient(timeout=30) as client:
        url = f"{settings.the_odds_base_url}/sports/{sport}/participants"
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        
        participants = data if isinstance(data, list) else data.get("data", [])
        logger.info(f"Successfully fetched {len(participants)} participants")
        
        # Standardize team names if requested
        if standardize:
            standardized_participants = []
            for participant in participants:
                try:
                    name = participant.get("name") or participant.get("team") or participant.get("id")
                    if name:
                        from src.ingestion.standardize import normalize_team_to_espn
                        standardized_name, is_valid = normalize_team_to_espn(str(name), source="the_odds")
                        if is_valid:
                            participant["name"] = standardized_name
                            participant["name_standardized"] = True
                        else:
                            participant["name_standardized"] = False
                    standardized_participants.append(participant)
                except Exception as e:
                    logger.warning(f"Error standardizing participant {participant.get('name', 'N/A')}: {e}")
                    standardized_participants.append(participant)
            return standardized_participants
        
        return participants
