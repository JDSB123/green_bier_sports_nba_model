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
) -> list[Dict[str, Any]]:
    """
    Fetch odds from The Odds API and standardize to ESPN format.

    Team name standardization is MANDATORY and always performed to ensure
    data consistency across sources.

    Args:
        sport: Sport identifier
        regions: Regions to fetch odds for
        markets: Markets to fetch

    Returns:
        List of game dictionaries with standardized team names in ESPN format

    Raises:
        ValueError: If THE_ODDS_API_KEY is not set
        httpx.HTTPStatusError: If the API request fails
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
    # NO FALLBACK to unstandardized data - skip invalid entries
    standardized_data = []
    invalid_count = 0
    for game in data:
        try:
            standardized = standardize_game_data(game, source="the_odds")
            if standardized.get("_data_valid", False):
                standardized_data.append(standardized)
            else:
                invalid_count += 1
                logger.warning(
                    f"Skipping game with invalid team names: "
                    f"home='{standardized.get('home_team', 'N/A')}', "
                    f"away='{standardized.get('away_team', 'N/A')}'"
                )
        except Exception as e:
            logger.error(
                f"Error standardizing game data: {e}. "
                f"Game: {game.get('home_team', 'N/A')} vs {game.get('away_team', 'N/A')}"
            )
            invalid_count += 1

    if invalid_count > 0:
        logger.info(
            f"Standardized {len(standardized_data)} valid games "
            f"(skipped {invalid_count} invalid)"
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
) -> Dict[str, Any]:
    """
    Fetch odds for a specific event from The Odds API.

    Fetches ALL available markets including:
    - First half (1H): spreads, totals, moneyline
    - All quarters (Q1, Q2, Q3, Q4): spreads, totals, moneyline
    - Alternate lines: spreads, totals
    - Player props: points, rebounds, assists

    Team name standardization is MANDATORY and always performed.

    Args:
        event_id: The event ID from The Odds API
        sport: Sport identifier
        regions: Regions to fetch odds for
        markets: Markets to fetch (default: ALL available markets)

    Returns:
        Event dictionary with odds data including all period markets (standardized)

    Raises:
        ValueError: If THE_ODDS_API_KEY is not set or event has invalid team names
        httpx.HTTPStatusError: If the API request fails
    """
    logger.info(f"Fetching event odds for {event_id} with markets: {markets}")

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
                f"{settings.the_odds_base_url}/sports/{sport}/events/{event_id}/odds",
                params=params
            )
            resp.raise_for_status()
            return resp.json()

    try:
        data = await breaker.call_async(_fetch_with_client)
        logger.info(f"Successfully fetched event odds for {event_id}")
    except Exception as e:
        logger.error(f"Failed to fetch event odds from The Odds API: {e}")
        raise

    # ALWAYS standardize team names to ESPN format (mandatory)
    try:
        standardized = standardize_game_data(data, source="the_odds")
        if not standardized.get("_data_valid", False):
            raise ValueError(
                f"Event {event_id} has invalid team names: "
                f"home='{standardized.get('home_team', 'N/A')}', "
                f"away='{standardized.get('away_team', 'N/A')}'"
            )
        return standardized
    except Exception as e:
        logger.error(f"Error standardizing event odds for {event_id}: {e}")
        raise


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
)
async def fetch_events(
    sport: str = "basketball_nba",
) -> List[Dict[str, Any]]:
    """
    Fetch all upcoming events (games) with their IDs.

    Use this to get event IDs for fetching event-specific odds
    (e.g., first half markets, alternate lines).

    Team name standardization is MANDATORY and always performed.

    Args:
        sport: Sport identifier

    Returns:
        List of event dictionaries with IDs, teams, and commence times (standardized)

    Raises:
        ValueError: If THE_ODDS_API_KEY is not set
        httpx.HTTPStatusError: If the API request fails
    """
    logger.info(f"Fetching events for {sport}")

    # Validate API key
    if not settings.the_odds_api_key:
        raise ValueError("THE_ODDS_API_KEY is not set")

    params = {
        "apiKey": settings.the_odds_api_key,
        "dateFormat": "iso",
    }

    # Use circuit breaker to prevent cascading failures
    breaker = get_odds_api_breaker()

    async def _fetch_with_client():
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{settings.the_odds_base_url}/sports/{sport}/events",
                params=params
            )
            resp.raise_for_status()
            return resp.json()

    try:
        data = await breaker.call_async(_fetch_with_client)
        logger.info(f"Successfully fetched {len(data)} events")
    except Exception as e:
        logger.error(f"Failed to fetch events from The Odds API: {e}")
        raise

    # ALWAYS standardize team names to ESPN format (mandatory)
    # NO FALLBACK to unstandardized data - skip invalid entries
    standardized_data = []
    invalid_count = 0
    for event in data:
        try:
            standardized = standardize_game_data(event, source="the_odds")
            if standardized.get("_data_valid", False):
                standardized_data.append(standardized)
            else:
                invalid_count += 1
                logger.warning(
                    f"Skipping event with invalid team names: "
                    f"home='{standardized.get('home_team', 'N/A')}', "
                    f"away='{standardized.get('away_team', 'N/A')}'"
                )
        except Exception as e:
            logger.error(
                f"Error standardizing event data: {e}. "
                f"Event: {event.get('home_team', 'N/A')} vs {event.get('away_team', 'N/A')}"
            )
            invalid_count += 1

    if invalid_count > 0:
        logger.info(
            f"Standardized {len(standardized_data)} valid events "
            f"(skipped {invalid_count} invalid)"
        )

    return standardized_data


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
)
async def fetch_scores(
    sport: str = "basketball_nba",
    days_from: int = 1,
) -> List[Dict[str, Any]]:
    """
    Fetch recent game scores for result verification.

    Team name standardization is MANDATORY and always performed.

    Args:
        sport: Sport identifier
        days_from: Number of days in the past to fetch scores for (1-3)

    Returns:
        List of games with scores (completed and in-progress), standardized

    Raises:
        ValueError: If THE_ODDS_API_KEY is not set
        httpx.HTTPStatusError: If the API request fails
    """
    logger.info(f"Fetching scores for {sport}, days_from={days_from}")

    # Validate API key
    if not settings.the_odds_api_key:
        raise ValueError("THE_ODDS_API_KEY is not set")

    params = {
        "apiKey": settings.the_odds_api_key,
        "daysFrom": str(days_from),
        "dateFormat": "iso",
    }

    # Use circuit breaker to prevent cascading failures
    breaker = get_odds_api_breaker()

    async def _fetch_with_client():
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{settings.the_odds_base_url}/sports/{sport}/scores",
                params=params
            )
            resp.raise_for_status()
            return resp.json()

    try:
        data = await breaker.call_async(_fetch_with_client)
        logger.info(f"Successfully fetched scores for {len(data)} games")
    except Exception as e:
        logger.error(f"Failed to fetch scores from The Odds API: {e}")
        raise

    # ALWAYS standardize team names to ESPN format (mandatory)
    # NO FALLBACK to unstandardized data - skip invalid entries
    standardized_data = []
    invalid_count = 0
    for game in data:
        try:
            standardized = standardize_game_data(game, source="the_odds")
            if standardized.get("_data_valid", False):
                standardized_data.append(standardized)
            else:
                invalid_count += 1
                logger.warning(
                    f"Skipping score with invalid team names: "
                    f"home='{standardized.get('home_team', 'N/A')}', "
                    f"away='{standardized.get('away_team', 'N/A')}'"
                )
        except Exception as e:
            logger.error(
                f"Error standardizing score data: {e}. "
                f"Game: {game.get('home_team', 'N/A')} vs {game.get('away_team', 'N/A')}"
            )
            invalid_count += 1

    if invalid_count > 0:
        logger.info(
            f"Standardized {len(standardized_data)} valid scores "
            f"(skipped {invalid_count} invalid)"
        )

    return standardized_data


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
)
async def fetch_historical_odds(
    date: str,
    sport: str = "basketball_nba",
    regions: str = "us",
    markets: str = "h2h,spreads,totals",
) -> Dict[str, Any]:
    """
    Fetch historical odds snapshot for backtesting.

    NOTE: This endpoint may require a paid plan. Returns 403 if not enabled.

    Team name standardization is MANDATORY and always performed.

    Args:
        date: ISO format date string (e.g., "2025-12-01T12:00:00Z")
        sport: Sport identifier
        regions: Regions to fetch odds for
        markets: Markets to fetch

    Returns:
        Dictionary with historical odds data including:
        - timestamp: The snapshot timestamp
        - previous_timestamp: Previous available snapshot
        - next_timestamp: Next available snapshot
        - data: List of events with odds (only those with valid team names)

    Raises:
        ValueError: If THE_ODDS_API_KEY is not set
        httpx.HTTPStatusError: If the request fails (403 = not enabled)
    """
    logger.info(f"Fetching historical odds for {sport} at {date}")

    # Validate API key
    if not settings.the_odds_api_key:
        raise ValueError("THE_ODDS_API_KEY is not set")

    params = {
        "apiKey": settings.the_odds_api_key,
        "regions": regions,
        "markets": markets,
        "date": date,
        "oddsFormat": "american",
        "dateFormat": "iso",
    }

    # Use circuit breaker to prevent cascading failures
    breaker = get_odds_api_breaker()

    async def _fetch_with_client():
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{settings.the_odds_base_url}/historical/sports/{sport}/odds",
                params=params
            )
            resp.raise_for_status()
            return resp.json()

    try:
        data = await breaker.call_async(_fetch_with_client)
        events = data.get("data", [])
        logger.info(f"Successfully fetched historical odds: {len(events)} events")
    except Exception as e:
        logger.error(f"Failed to fetch historical odds from The Odds API: {e}")
        raise

    # ALWAYS standardize team names to ESPN format (mandatory)
    # NO FALLBACK to unstandardized data - skip invalid entries
    if events:
        standardized_events = []
        invalid_count = 0
        for event in events:
            try:
                standardized = standardize_game_data(event, source="the_odds")
                if standardized.get("_data_valid", False):
                    standardized_events.append(standardized)
                else:
                    invalid_count += 1
                    logger.warning(
                        f"Skipping historical event with invalid team names: "
                        f"home='{standardized.get('home_team', 'N/A')}', "
                        f"away='{standardized.get('away_team', 'N/A')}'"
                    )
            except Exception as e:
                logger.error(f"Error standardizing historical event: {e}")
                invalid_count += 1

        data["data"] = standardized_events
        if invalid_count > 0:
            logger.info(
                f"Standardized {len(standardized_events)} valid events "
                f"(skipped {invalid_count} invalid)"
            )

    return data


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
)
async def fetch_historical_events(
    date: str,
    sport: str = "basketball_nba",
) -> Dict[str, Any]:
    """
    Fetch historical events list for backtesting.

    NOTE: This endpoint may require a paid plan. Returns 403 if not enabled.

    Team name standardization is MANDATORY and always performed.

    Args:
        date: ISO format date string (e.g., "2025-12-01T12:00:00Z")
        sport: Sport identifier

    Returns:
        Dictionary with historical events data (only events with valid team names)

    Raises:
        ValueError: If THE_ODDS_API_KEY is not set
        httpx.HTTPStatusError: If the request fails (403 = not enabled)
    """
    logger.info(f"Fetching historical events for {sport} at {date}")

    # Validate API key
    if not settings.the_odds_api_key:
        raise ValueError("THE_ODDS_API_KEY is not set")

    params = {
        "apiKey": settings.the_odds_api_key,
        "date": date,
        "dateFormat": "iso",
    }

    # Use circuit breaker to prevent cascading failures
    breaker = get_odds_api_breaker()

    async def _fetch_with_client():
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{settings.the_odds_base_url}/historical/sports/{sport}/events",
                params=params
            )
            resp.raise_for_status()
            return resp.json()

    try:
        data = await breaker.call_async(_fetch_with_client)
        events = data.get("data", [])
        logger.info(f"Successfully fetched historical events: {len(events)} events")
    except Exception as e:
        logger.error(f"Failed to fetch historical events from The Odds API: {e}")
        raise

    # ALWAYS standardize team names to ESPN format (mandatory)
    # NO FALLBACK to unstandardized data - skip invalid entries
    if events:
        standardized_events = []
        invalid_count = 0
        for event in events:
            try:
                standardized = standardize_game_data(event, source="the_odds")
                if standardized.get("_data_valid", False):
                    standardized_events.append(standardized)
                else:
                    invalid_count += 1
                    logger.warning(
                        f"Skipping historical event with invalid team names: "
                        f"home='{standardized.get('home_team', 'N/A')}', "
                        f"away='{standardized.get('away_team', 'N/A')}'"
                    )
            except Exception as e:
                logger.error(f"Error standardizing historical event: {e}")
                invalid_count += 1

        data["data"] = standardized_events
        if invalid_count > 0:
            logger.info(
                f"Standardized {len(standardized_events)} valid events "
                f"(skipped {invalid_count} invalid)"
            )

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
    Returns empty list if 403 (not enabled for your API key).

    Team name standardization is MANDATORY and always performed.

    Args:
        sport: Sport identifier
        regions: Regions to fetch splits for

    Returns:
        List of game dictionaries with betting splits data (only valid team names)

    Raises:
        ValueError: If THE_ODDS_API_KEY is not set
        httpx.HTTPStatusError: If the request fails (except 403)
    """
    logger.info(f"Fetching betting splits for {sport}")

    # Validate API key
    if not settings.the_odds_api_key:
        raise ValueError("THE_ODDS_API_KEY is not set")

    params = {
        "apiKey": settings.the_odds_api_key,
        "regions": regions,
    }

    # Use circuit breaker to prevent cascading failures
    breaker = get_odds_api_breaker()

    async def _fetch_with_client():
        async with httpx.AsyncClient(timeout=30) as client:
            url = f"{settings.the_odds_base_url}/sports/{sport}/betting-splits"
            resp = await client.get(url, params=params)

            # 403 = paid plan not enabled - graceful degradation
            if resp.status_code == 403:
                logger.warning("Betting splits endpoint not enabled for this API key.")
                return {"data": [], "_not_enabled": True}

            resp.raise_for_status()
            return resp.json()

    try:
        data = await breaker.call_async(_fetch_with_client)
        if data.get("_not_enabled"):
            return []
        splits = data.get("data", [])
        logger.info(f"Successfully fetched betting splits for {len(splits)} games")
    except Exception as e:
        logger.error(f"Failed to fetch betting splits from The Odds API: {e}")
        raise

    # ALWAYS standardize team names to ESPN format (mandatory)
    # NO FALLBACK to unstandardized data - skip invalid entries
    standardized_splits = []
    invalid_count = 0
    for split in splits:
        try:
            standardized = standardize_game_data(split, source="the_odds")
            if standardized.get("_data_valid", False):
                standardized_splits.append(standardized)
            else:
                invalid_count += 1
                logger.warning(
                    f"Skipping betting split with invalid team names: "
                    f"home='{standardized.get('home_team', 'N/A')}', "
                    f"away='{standardized.get('away_team', 'N/A')}'"
                )
        except Exception as e:
            logger.error(f"Error standardizing betting split: {e}")
            invalid_count += 1

    if invalid_count > 0:
        logger.info(
            f"Standardized {len(standardized_splits)} valid splits "
            f"(skipped {invalid_count} invalid)"
        )

    return standardized_splits


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
)
async def fetch_participants(
    sport: str = "basketball_nba",
) -> List[Dict[str, Any]]:
    """
    Fetch participants (teams) from The Odds API.

    This endpoint provides a reference list of all participants (teams)
    in the sport, useful for team name standardization and validation.

    Team name standardization is MANDATORY and always performed.
    Only participants with valid NBA team names are returned.

    Args:
        sport: Sport identifier

    Returns:
        List of participant dictionaries with standardized team names

    Raises:
        ValueError: If THE_ODDS_API_KEY is not set
        httpx.HTTPStatusError: If the request fails
    """
    logger.info(f"Fetching participants for {sport}")

    # Validate API key
    if not settings.the_odds_api_key:
        raise ValueError("THE_ODDS_API_KEY is not set")

    params = {
        "apiKey": settings.the_odds_api_key,
        "dateFormat": "iso",
    }

    # Use circuit breaker to prevent cascading failures
    breaker = get_odds_api_breaker()

    async def _fetch_with_client():
        async with httpx.AsyncClient(timeout=30) as client:
            url = f"{settings.the_odds_base_url}/sports/{sport}/participants"
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            return resp.json()

    try:
        data = await breaker.call_async(_fetch_with_client)
        participants = data if isinstance(data, list) else data.get("data", [])
        logger.info(f"Successfully fetched {len(participants)} participants")
    except Exception as e:
        logger.error(f"Failed to fetch participants from The Odds API: {e}")
        raise

    # Import here to avoid circular imports
    from src.ingestion.standardize import normalize_team_to_espn

    # ALWAYS standardize team names to ESPN format (mandatory)
    # Only include participants with valid NBA team names
    standardized_participants = []
    invalid_count = 0
    for participant in participants:
        try:
            name = participant.get("name") or participant.get("team") or participant.get("id")
            if name:
                standardized_name, is_valid = normalize_team_to_espn(str(name), source="the_odds")
                if is_valid:
                    participant["name"] = standardized_name
                    participant["name_standardized"] = True
                    participant["_data_valid"] = True
                    standardized_participants.append(participant)
                else:
                    invalid_count += 1
                    logger.warning(f"Skipping participant with invalid team name: '{name}'")
            else:
                invalid_count += 1
                logger.warning("Skipping participant with no name field")
        except Exception as e:
            logger.error(f"Error standardizing participant {participant.get('name', 'N/A')}: {e}")
            invalid_count += 1

    if invalid_count > 0:
        logger.info(
            f"Standardized {len(standardized_participants)} valid participants "
            f"(skipped {invalid_count} invalid)"
        )

    return standardized_participants
