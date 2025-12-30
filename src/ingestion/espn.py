"""
ESPN Schedule Ingestion - Single Source of Truth for Game Schedules.

This module ingests NBA game schedules from ESPN's API, which serves as the
canonical source for:
- Game dates and times
- Home and away team designations
- Team names (ESPN format)

All other data sources (The Odds API, API-Basketball) should be normalized
to match ESPN's team names and game structure.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings, get_nba_season
from src.utils.logging import get_logger

logger = get_logger(__name__)

BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
TIMEOUT = 30


@dataclass
class ESPNGame:
    """
    Standardized game data from ESPN.
    
    Format: AWAY TEAM vs. HOME TEAM
    Teams are stored in this order for consistency across the pipeline.
    """
    
    game_id: str
    date: datetime
    away_team: str  # ESPN team name (AWAY)
    home_team: str  # ESPN team name (HOME)
    away_team_id: Optional[str] = None
    home_team_id: Optional[str] = None
    status: Optional[str] = None
    away_score: Optional[int] = None
    home_score: Optional[int] = None
    season: Optional[str] = None


class ESPNScheduleClient:
    """Client for ESPN schedule API."""
    
    def __init__(self, output_dir: str | None = None):
        self.output_dir = output_dir or os.path.join(
            settings.data_raw_dir, "espn"
        )
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
    )
    async def _fetch(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Make HTTP request to ESPN API."""
        url = f"{BASE_URL}/{endpoint}"
        logger.debug(f"Fetching ESPN endpoint: {url} with params: {params}")
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.get(url, params=params or {})
            resp.raise_for_status()
            data = resp.json()
            logger.info(f"Successfully fetched {endpoint}")
            return data
    
    def _save(self, name: str, data: dict[str, Any]) -> str:
        """Save response to JSON file with timestamp."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.json"
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        return path
    
    async def fetch_schedule(
        self,
        dates: Optional[List[str]] = None,
        season: Optional[str] = None,
    ) -> List[ESPNGame]:
        """
        Fetch NBA schedule from ESPN.
        
        Args:
            dates: List of dates in YYYYMMDD format (e.g., ["20251208"])
            season: Season string (e.g., "2025") - defaults to current season
        
        Returns:
            List of ESPNGame objects with standardized team names
        """
        if season is None:
            season = get_nba_season().split("-")[0]  # "2025-2026" -> "2025"
        
        games: List[ESPNGame] = []
        
        if dates:
            # Fetch specific dates
            for date_str in dates:
                try:
                    data = await self._fetch("scoreboard", params={"dates": date_str})
                    games.extend(self._parse_scoreboard(data, season))
                except Exception as e:
                    logger.error(f"Error fetching schedule for {date_str}: {e}")
        else:
            # Fetch current/upcoming games
            data = await self._fetch("scoreboard")
            games.extend(self._parse_scoreboard(data, season))
        
        # Save raw data
        if games:
            self._save("schedule", {"games": [self._game_to_dict(g) for g in games]})
        
        return games
    
    def _parse_scoreboard(self, data: dict[str, Any], season: str) -> List[ESPNGame]:
        """Parse ESPN scoreboard response into ESPNGame objects."""
        games: List[ESPNGame] = []
        
        events = data.get("events", [])
        for event in events:
            try:
                game_id = event.get("id")
                date_str = event.get("date")
                if not game_id or not date_str:
                    continue
                
                # Parse date
                game_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                
                # Get teams
                competitions = event.get("competitions", [])
                if not competitions:
                    continue
                
                competition = competitions[0]
                competitors = competition.get("competitors", [])
                if len(competitors) < 2:
                    continue
                
                # Find home and away teams
                # Standard format: AWAY TEAM vs. HOME TEAM
                home_team = None
                away_team = None
                home_team_id = None
                away_team_id = None
                home_score = None
                away_score = None
                
                for comp in competitors:
                    is_home = comp.get("homeAway") == "home"
                    team_name = comp.get("team", {}).get("displayName", "")
                    team_id = comp.get("team", {}).get("id", "")
                    score = comp.get("score")
                    
                    if is_home:
                        home_team = team_name
                        home_team_id = team_id
                        home_score = int(score) if score else None
                    else:
                        away_team = team_name
                        away_team_id = team_id
                        away_score = int(score) if score else None
                
                if not home_team or not away_team:
                    continue
                
                # Get status
                status_obj = competition.get("status", {})
                status = status_obj.get("type", {}).get("name", "")
                
                # Create game with standard format: AWAY vs. HOME
                games.append(ESPNGame(
                    game_id=game_id,
                    date=game_date,
                    away_team=away_team,  # AWAY first
                    home_team=home_team,  # HOME second
                    away_team_id=away_team_id,
                    home_team_id=home_team_id,
                    status=status,
                    away_score=away_score,
                    home_score=home_score,
                    season=season,
                ))
            except Exception as e:
                logger.warning(f"Error parsing game event: {e}")
                continue
        
        return games
    
    def _game_to_dict(self, game: ESPNGame) -> dict[str, Any]:
        """Convert ESPNGame to dictionary for JSON serialization.
        
        Format: AWAY TEAM vs. HOME TEAM (away_team field comes first)
        """
        return {
            "game_id": game.game_id,
            "date": game.date.isoformat(),
            "away_team": game.away_team,  # AWAY first
            "home_team": game.home_team,  # HOME second
            "away_team_id": game.away_team_id,
            "home_team_id": game.home_team_id,
            "status": game.status,
            "away_score": game.away_score,
            "home_score": game.home_score,
            "season": game.season,
        }


async def fetch_espn_schedule(
    dates: Optional[List[str]] = None,
    season: Optional[str] = None,
) -> List[ESPNGame]:
    """
    Convenience function to fetch ESPN schedule.

    Args:
        dates: List of dates in YYYYMMDD format (e.g., ["20251208"])
        season: Season string (e.g., "2025")

    Returns:
        List of ESPNGame objects
    """
    client = ESPNScheduleClient()
    return await client.fetch_schedule(dates=dates, season=season)


@dataclass
class ESPNTeamStanding:
    """Team standings data from ESPN."""
    team_name: str
    team_id: str
    wins: int
    losses: int
    win_pct: float
    games_behind: float
    streak: str
    home_record: str
    away_record: str
    last_10: str
    conference: str


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
)
async def fetch_espn_standings() -> Dict[str, ESPNTeamStanding]:
    """
    Fetch current NBA standings from ESPN.

    ESPN is the PRIMARY and ONLY source for team records.
    Raises error on failure - no silent fallbacks.

    ESPN's standings API is FREE and provides real-time accurate W-L records.

    Returns:
        Dictionary mapping team name to ESPNTeamStanding object

    Raises:
        RuntimeError: If ESPN API fails or returns no data
    """
    url = "https://site.api.espn.com/apis/v2/sports/basketball/nba/standings"
    standings: Dict[str, ESPNTeamStanding] = {}

    logger.info("[API] Fetching fresh ESPN standings (STRICT MODE)")

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()

        # Parse conferences (Eastern and Western)
        for conference in data.get("children", []):
            conf_name = conference.get("abbreviation", "")
            standings_data = conference.get("standings", {})
            entries = standings_data.get("entries", [])

            for entry in entries:
                try:
                    team_info = entry.get("team", {})
                    team_name = team_info.get("displayName", "")
                    team_id = str(team_info.get("id", ""))

                    if not team_name:
                        continue

                    # Parse stats array
                    stats = entry.get("stats", [])
                    stats_dict = {}
                    for stat in stats:
                        stat_name = stat.get("name", "")
                        stat_value = stat.get("displayValue", "")
                        stats_dict[stat_name] = stat_value

                    # Extract key values
                    wins = int(stats_dict.get("wins", "0"))
                    losses = int(stats_dict.get("losses", "0"))
                    win_pct_str = stats_dict.get("winPercent", "0")
                    try:
                        win_pct = float(win_pct_str)
                    except ValueError:
                        win_pct = wins / (wins + losses) if (wins + losses) > 0 else 0.0

                    games_behind_str = stats_dict.get("gamesBehind", "0")
                    try:
                        games_behind = float(games_behind_str) if games_behind_str != "-" else 0.0
                    except ValueError:
                        games_behind = 0.0

                    standings[team_name] = ESPNTeamStanding(
                        team_name=team_name,
                        team_id=team_id,
                        wins=wins,
                        losses=losses,
                        win_pct=win_pct,
                        games_behind=games_behind,
                        streak=stats_dict.get("streak", ""),
                        home_record=stats_dict.get("Home", ""),
                        away_record=stats_dict.get("Road", ""),
                        last_10=stats_dict.get("Last Ten Games", ""),
                        conference=conf_name,
                    )
                except Exception as e:
                    logger.warning(f"Error parsing team standing: {e}")
                    continue

        if not standings:
            raise RuntimeError("STRICT MODE: ESPN returned no standings data")

        logger.info(f"[API] Fetched ESPN standings for {len(standings)} teams")
        return standings

    except Exception as e:
        logger.error(f"STRICT MODE: ESPN standings fetch FAILED: {e}")
        raise RuntimeError(f"STRICT MODE: Cannot proceed without ESPN standings - {e}")

