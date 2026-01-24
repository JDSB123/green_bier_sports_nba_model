"""
Injury data ingestion for NBA predictions.

Sources:
- NBA official injury reports (via API-Basketball or ESPN)
- Rotowire injury news
- Fantasy sports injury feeds

This module provides infrastructure for fetching and standardizing
injury data that can be used by the FeatureEngineer for impact estimation.
"""
from __future__ import annotations
import os
import datetime as dt
from typing import Any, Dict, List, Optional
import httpx
from dataclasses import dataclass

from src.config import settings


@dataclass
class InjuryReport:
    """
    Standardized injury report.

    All fields are required except where explicitly Optional.
    No placeholder values - source must be explicitly set.
    """
    player_id: str
    player_name: str
    team: str
    # Required - must be explicitly set (e.g., "espn", "api_basketball")
    source: str
    # out, doubtful, questionable, probable, available (this is a valid status, not placeholder)
    status: str = "questionable"
    team_id: Optional[str] = None
    injury_type: Optional[str] = None
    injury_location: Optional[str] = None  # knee, ankle, back, etc.
    report_date: Optional[dt.datetime] = None
    expected_return: Optional[dt.datetime] = None
    # Player stats for impact calculation
    ppg: float = 0.0
    minutes_per_game: float = 0.0
    usage_rate: float = 0.0


# Team name normalization for matching
TEAM_ABBREV_MAP = {
    "ATL": "Hawks", "BOS": "Celtics", "BKN": "Nets", "CHA": "Hornets",
    "CHI": "Bulls", "CLE": "Cavaliers", "DAL": "Mavericks", "DEN": "Nuggets",
    "DET": "Pistons", "GSW": "Warriors", "HOU": "Rockets", "IND": "Pacers",
    "LAC": "Clippers", "LAL": "Lakers", "MEM": "Grizzlies", "MIA": "Heat",
    "MIL": "Bucks", "MIN": "Timberwolves", "NOP": "Pelicans", "NYK": "Knicks",
    "OKC": "Thunder", "ORL": "Magic", "PHI": "76ers", "PHX": "Suns",
    "POR": "Trail Blazers", "SAC": "Kings", "SAS": "Spurs", "TOR": "Raptors",
    "UTA": "Jazz", "WAS": "Wizards",
}

TEAM_FULL_NAMES = {
    "Atlanta Hawks": "Hawks", "Boston Celtics": "Celtics",
    "Brooklyn Nets": "Nets", "Charlotte Hornets": "Hornets",
    "Chicago Bulls": "Bulls", "Cleveland Cavaliers": "Cavaliers",
    "Dallas Mavericks": "Mavericks", "Denver Nuggets": "Nuggets",
    "Detroit Pistons": "Pistons", "Golden State Warriors": "Warriors",
    "Houston Rockets": "Rockets", "Indiana Pacers": "Pacers",
    "Los Angeles Clippers": "Clippers", "LA Clippers": "Clippers",
    "Los Angeles Lakers": "Lakers", "LA Lakers": "Lakers",
    "Memphis Grizzlies": "Grizzlies", "Miami Heat": "Heat",
    "Milwaukee Bucks": "Bucks", "Minnesota Timberwolves": "Timberwolves",
    "New Orleans Pelicans": "Pelicans", "New York Knicks": "Knicks",
    "Oklahoma City Thunder": "Thunder", "Orlando Magic": "Magic",
    "Philadelphia 76ers": "76ers", "Phoenix Suns": "Suns",
    "Portland Trail Blazers": "Trail Blazers", "Sacramento Kings": "Kings",
    "San Antonio Spurs": "Spurs", "Toronto Raptors": "Raptors",
    "Utah Jazz": "Jazz", "Washington Wizards": "Wizards",
}

# Status normalization
STATUS_MAP = {
    "out": "out",
    "o": "out",
    "injured": "out",
    "doubtful": "doubtful",
    "d": "doubtful",
    "questionable": "questionable",
    "q": "questionable",
    "gtd": "questionable",  # Game-time decision
    "game time decision": "questionable",
    "probable": "probable",
    "p": "probable",
    "available": "available",
    "active": "available",
    "healthy": "available",
    "day-to-day": "questionable",
}


def normalize_team(team_name: str) -> str:
    """Normalize team name to standard short form."""
    team_name = team_name.strip()

    # Check abbreviation
    if team_name.upper() in TEAM_ABBREV_MAP:
        return TEAM_ABBREV_MAP[team_name.upper()]

    # Check full name
    if team_name in TEAM_FULL_NAMES:
        return TEAM_FULL_NAMES[team_name]

    # Try partial match
    for full, short in TEAM_FULL_NAMES.items():
        if short.lower() in team_name.lower():
            return short

    return team_name


def normalize_status(status: str) -> str:
    """Normalize injury status."""
    status = status.lower().strip()
    return STATUS_MAP.get(status, "questionable")


async def fetch_injuries_espn() -> List[Dict[str, Any]]:
    """
    Fetch injury data from ESPN's unofficial API.

    Note: ESPN doesn't have an official public API, but their
    internal endpoints can be accessed. This may break if they change.

    Returns empty list on failure - no mock data in production.
    Use fetch_all_injuries() as the single source of truth which aggregates
    from multiple sources (ESPN + API-Basketball).
    """
    from src.utils.logging import get_logger

    logger = get_logger(__name__)
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()

            injuries = []
            for team_data in data.get("injuries", []):
                # ESPN format: team_data has 'displayName' directly (not nested under 'team')
                team_name = team_data.get("displayName")

                # Skip if team name is missing - no placeholder values
                if not team_name:
                    logger.warning(
                        f"Skipping injury data with missing team name from ESPN")
                    continue

                for player in team_data.get("injuries", []):
                    player_name = player.get("athlete", {}).get("displayName")
                    # Skip if player name is missing - no placeholder values
                    if not player_name:
                        logger.warning(
                            f"Skipping injury with missing player name for team {team_name} from ESPN")
                        continue

                    # Extract injury details from nested structure
                    details = player.get("details", {})
                    injury_type = details.get("type")  # e.g., "Knee", "Ankle"
                    injury_location = details.get(
                        "location")  # e.g., "Leg", "Arm"
                    # e.g., "Bruise", "Strain"
                    injury_detail = details.get("detail")
                    injury_side = details.get("side")  # e.g., "Left", "Right"

                    # Format injury description
                    injury_desc = None
                    if injury_type:
                        parts = [p for p in [injury_side, injury_type,
                                             injury_detail] if p and p != "Not Specified"]
                        if parts:
                            injury_desc = " ".join(parts)

                    injuries.append({
                        "player_name": player_name,
                        "team": normalize_team(team_name),
                        "status": normalize_status(player.get("status", "questionable")),
                        "injury_type": injury_desc,
                        "injury_location": injury_location,
                        "source": "espn",
                    })

            if not injuries:
                logger.warning(
                    "ESPN returned empty injury data - this may be normal for future dates or off-season")
                return []

            logger.info(f"Fetched {len(injuries)} injuries from ESPN")
            return injuries
    except httpx.HTTPStatusError as e:
        logger.error(
            f"HTTP error fetching ESPN injuries: {e.response.status_code} {e.response.reason_phrase}. URL: {url}")
        # Explicit failure - log and return empty, do not raise (other sources may succeed)
        return []
    except httpx.RequestError as e:
        logger.error(f"Request error fetching ESPN injuries: {e}. URL: {url}")
        # Explicit failure - log and return empty, do not raise (other sources may succeed)
        return []
    except Exception as e:
        logger.error(
            f"Unexpected error fetching ESPN injuries: {type(e).__name__}: {e}. URL: {url}", exc_info=True)
        # Explicit failure - log with full traceback and return empty, do not raise (other sources may succeed)
        return []


async def fetch_injuries_api_basketball(
    league: int = 12,  # NBA
    seasons: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch injury data from API-Basketball.

    Note: API-Basketball has an injuries endpoint (requires subscription).

    Returns empty list on failure - no mock data in production.
    Use fetch_all_injuries() as the single source of truth which aggregates
    from multiple sources (ESPN + API-Basketball).
    """
    from src.utils.logging import get_logger

    logger = get_logger(__name__)

    if seasons is None:
        # Try all configured seasons to capture historical injuries (2023+)
        seasons = settings.seasons_to_process or [settings.current_season]
        # Ensure unique ordering
        seasons = list(dict.fromkeys(seasons))
    if not settings.api_basketball_key:
        logger.debug("API-Basketball key not configured, skipping")
        return []

    headers = {"x-apisports-key": settings.api_basketball_key}
    url = f"{settings.api_basketball_base_url}/injuries"

    injuries: List[Dict[str, Any]] = []
    try:
        async with httpx.AsyncClient(timeout=30, headers=headers) as client:
            for season in seasons:
                params = {"league": league, "season": season}
                resp = await client.get(url, params=params)
                resp_text = resp.text[:500] if resp.text else ""
                try:
                    resp.raise_for_status()
                except httpx.HTTPStatusError as e:
                    logger.error(
                        f"HTTP error fetching API-Basketball injuries: {e.response.status_code} {e.response.reason_phrase}. URL: {url} season={season} body_snip={resp_text}")
                    continue

                data = resp.json()
                if not data.get("response"):
                    logger.info(
                        f"API-Basketball empty response for season {season}. body_snip={resp_text}")
                    continue

                for item in data.get("response", []):
                    player = item.get("player", {})
                    team = item.get("team", {})

                    player_name = player.get("name")
                    team_name = team.get("name")

                    # Skip if required fields are missing - no placeholder values
                    if not player_name:
                        logger.warning(
                            f"Skipping injury data with missing player name from API-Basketball")
                        continue
                    if not team_name:
                        logger.warning(
                            f"Skipping injury data with missing team name for player {player_name} from API-Basketball")
                        continue

                    player_id = player.get("id")
                    if not player_id:
                        logger.warning(
                            f"Skipping injury for player {player_name} with missing player_id from API-Basketball")
                        continue

                    injuries.append({
                        "player_id": str(player_id),
                        "player_name": player_name,
                        "team": normalize_team(team_name),
                        "team_id": str(team.get("id", "")) if team.get("id") else None,
                        "status": normalize_status(item.get("status", "questionable")),
                        "injury_type": item.get("reason"),
                        "report_date": item.get("date"),
                        "source": "api_basketball",
                    })

        if injuries:
            logger.info(
                f"Fetched {len(injuries)} injuries from API-Basketball across seasons {seasons}")
        else:
            logger.debug(
                f"API-Basketball returned empty injury data across seasons {seasons}")
        return injuries
    except httpx.RequestError as e:
        logger.error(
            f"Request error fetching API-Basketball injuries: {e}. URL: {url}")
        # Explicit failure - log and return empty, do not raise (other sources may succeed)
        return []
    except Exception as e:
        logger.error(
            f"Unexpected error fetching API-Basketball injuries: {type(e).__name__}: {e}. URL: {url}", exc_info=True)
        # Explicit failure - log with full traceback and return empty, do not raise (other sources may succeed)
        return []


async def fetch_all_injuries() -> List[InjuryReport]:
    """
    SINGLE SOURCE OF TRUTH for injury data.

    Fetches injuries from all available sources (ESPN + API-Basketball) and merges them.
    This is the ONLY function that should be called to get injury data in production.

    Returns:
        List of standardized InjuryReport objects. Empty list if no sources available.
        NEVER returns mock data - only real data from configured sources.

    Sources tried (in order):
        1. ESPN (free, no API key required)
        2. API-Basketball (if API key configured)

    If all sources fail, returns empty list. This is intentional - better to have
    no injury data than fake/mock data that could mislead predictions.
    """
    from src.utils.logging import get_logger

    logger = get_logger(__name__)
    all_injuries: Dict[str, InjuryReport] = {}
    sources_used = []

    # Try ESPN first (free)
    logger.info("Fetching injuries from ESPN...")
    espn_injuries = await fetch_injuries_espn()
    espn_count = 0
    for inj in espn_injuries:
        key = f"{inj['player_name']}_{inj['team']}"
        all_injuries[key] = InjuryReport(
            player_id=key,
            player_name=inj["player_name"],
            team=inj["team"],
            status=inj["status"],
            injury_type=inj.get("injury_type"),
            report_date=dt.datetime.now(),
            source=inj["source"],
        )
        espn_count += 1

    if espn_count > 0:
        sources_used.append(f"ESPN ({espn_count} injuries)")

    # Try API-Basketball (if key available)
    if settings.api_basketball_key:
        logger.info("Fetching injuries from API-Basketball...")
        api_injuries = await fetch_injuries_api_basketball()
        api_count = 0
        for inj in api_injuries:
            key = f"{inj['player_name']}_{inj['team']}"
            # Merge or add
            if key in all_injuries:
                # Update with API-Basketball data (may have more details)
                existing = all_injuries[key]
                if inj.get("player_id"):
                    existing.player_id = inj["player_id"]
                if inj.get("team_id"):
                    existing.team_id = inj["team_id"]
                # Prefer API-Basketball source if more detailed
                existing.source = f"{existing.source}+api_basketball"
                api_count += 1
            else:
                all_injuries[key] = InjuryReport(
                    player_id=inj.get("player_id", key),
                    player_name=inj["player_name"],
                    team=inj["team"],
                    team_id=inj.get("team_id"),
                    status=inj["status"],
                    injury_type=inj.get("injury_type"),
                    report_date=(
                        dt.datetime.fromisoformat(inj["report_date"])
                        if inj.get("report_date")
                        else dt.datetime.now()
                    ),
                    source=inj["source"],
                )
                api_count += 1

        if api_count > 0:
            sources_used.append(f"API-Basketball ({api_count} injuries)")
    else:
        logger.debug("API-Basketball key not configured, skipping")

    final_count = len(all_injuries)

    if final_count > 0:
        logger.info(
            f"Successfully fetched {final_count} unique injuries from: {', '.join(sources_used)}")
    else:
        # Explicit warning - no silent failures
        logger.warning(
            "No injury data available from any source. "
            "This may indicate API failures, missing API keys, or empty responses. "
            "Predictions will proceed without injury impact adjustments."
        )

    return list(all_injuries.values())


async def save_injuries(
    injuries: List[InjuryReport],
    out_dir: Optional[str] = None,
) -> str:
    """Save injury reports to CSV."""
    import pandas as pd

    out_dir = out_dir or os.path.join(settings.data_processed_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Convert to DataFrame
    rows = []
    for inj in injuries:
        rows.append({
            "player_id": inj.player_id,
            "player_name": inj.player_name,
            "team": inj.team,
            "team_id": inj.team_id,
            "status": inj.status,
            "injury_type": inj.injury_type,
            "injury_location": inj.injury_location,
            "report_date": inj.report_date,
            "expected_return": inj.expected_return,
            "ppg": inj.ppg,
            "minutes_per_game": inj.minutes_per_game,
            "usage_rate": inj.usage_rate,
            "source": inj.source,
        })

    df = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, "injuries.csv")
    df.to_csv(out_path, index=False)

    return out_path


async def enrich_injuries_with_stats(
    injuries: List[InjuryReport],
    player_stats_df: Optional[Any] = None,
) -> List[InjuryReport]:
    """
    Enrich injury reports with player statistics.

    This is crucial for estimating injury impact on team performance.
    """
    if player_stats_df is None:
        # Try to load from file
        stats_path = os.path.join(
            settings.data_processed_dir, "player_stats.csv")
        if os.path.exists(stats_path):
            import pandas as pd
            player_stats_df = pd.read_csv(stats_path)
        else:
            return injuries  # No stats available

    # Create lookup by player name
    stats_lookup = {}
    if player_stats_df is not None and len(player_stats_df) > 0:
        for _, row in player_stats_df.iterrows():
            name = row.get("player_name", row.get("name", ""))
            if name:
                stats_lookup[name.lower()] = {
                    "ppg": row.get("ppg", row.get("points_per_game", 0)) or 0,
                    "mpg": row.get("mpg", row.get("minutes_per_game", 0)) or 0,
                    "usg": row.get("usage_rate", row.get("usg_pct", 0)) or 0,
                }

    # Enrich injuries
    for inj in injuries:
        player_key = inj.player_name.lower() if inj.player_name else ""
        if player_key in stats_lookup:
            stats = stats_lookup[player_key]
            inj.ppg = stats["ppg"]
            inj.minutes_per_game = stats["mpg"]
            inj.usage_rate = stats["usg"]

    return injuries
