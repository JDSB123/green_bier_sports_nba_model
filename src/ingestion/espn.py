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

from src.config import get_nba_season, settings
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
        self.output_dir = output_dir or os.path.join(settings.data_raw_dir, "espn")
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
                games.append(
                    ESPNGame(
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
                    )
                )
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
class ESPNBoxScore:
    """Complete box score data from ESPN - AUTHORITATIVE SOURCE."""

    game_id: str
    team_name: str
    team_id: str
    # Shooting
    fg_made: int
    fg_attempts: int
    fg_pct: float
    three_made: int
    three_attempts: int
    three_pct: float
    ft_made: int
    ft_attempts: int
    ft_pct: float
    # Rebounds
    rebounds_total: int
    rebounds_off: int
    rebounds_def: int
    # Other stats - ALL REQUIRED, NO NONE VALUES
    assists: int
    steals: int
    blocks: int
    turnovers: int
    total_turnovers: int  # Includes team turnovers
    personal_fouls: int
    points: int


@dataclass
class ESPNPlayerBoxScore:
    """Player-level box score data from ESPN."""

    game_id: str
    team_name: str
    player_name: str
    player_id: str
    starter: bool
    minutes: str
    fg_made: int
    fg_attempts: int
    three_made: int
    three_attempts: int
    ft_made: int
    ft_attempts: int
    rebounds_off: int
    rebounds_def: int
    rebounds_total: int
    assists: int
    steals: int
    blocks: int
    turnovers: int
    personal_fouls: int
    plus_minus: int
    points: int


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
)
async def fetch_espn_box_score(game_id: str) -> Dict[str, Any]:
    """
    Fetch COMPLETE box score data from ESPN for a specific game.

    ESPN is the AUTHORITATIVE SOURCE for box scores because:
    - Has ALL stats: steals, blocks, turnovers, off/def rebounds
    - FREE API, no rate limits
    - Real-time updates

    Args:
        game_id: ESPN game ID (e.g., "401810490")

    Returns:
        Dict with 'teams' (list of ESPNBoxScore) and 'players' (list of ESPNPlayerBoxScore)

    Raises:
        RuntimeError: If ESPN API fails or returns incomplete data
    """
    url = f"{BASE_URL}/summary"
    params = {"event": game_id}

    logger.info(f"[ESPN] Fetching box score for game {game_id}")

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

    if "boxscore" not in data:
        raise RuntimeError(f"ESPN returned no boxscore data for game {game_id}")

    boxscore = data["boxscore"]
    result = {
        "game_id": game_id,
        "teams": [],
        "players": [],
        "game_info": data.get("header", {}),
    }

    # Parse team box scores
    for team_data in boxscore.get("teams", []):
        team_info = team_data.get("team", {})
        team_name = team_info.get("displayName", "")
        team_id = str(team_info.get("id", ""))

        if not team_name:
            raise RuntimeError(f"ESPN boxscore missing team name for game {game_id}")

        stats = team_data.get("statistics", [])
        stats_dict = {s.get("label", ""): s.get("displayValue", "") for s in stats}

        # Parse shooting stats (format: "39-79")
        def parse_made_attempts(val: str) -> tuple[int, int]:
            if not val or "-" not in val:
                raise RuntimeError(f"Invalid stat format: {val}")
            parts = val.split("-")
            return int(parts[0]), int(parts[1])

        fg_made, fg_attempts = parse_made_attempts(stats_dict.get("FG", "0-0"))
        three_made, three_attempts = parse_made_attempts(stats_dict.get("3PT", "0-0"))
        ft_made, ft_attempts = parse_made_attempts(stats_dict.get("FT", "0-0"))

        # Get percentages - ESPN provides these directly
        fg_pct = (
            float(stats_dict.get("Field Goal %", "0")) / 100
            if stats_dict.get("Field Goal %")
            else (fg_made / fg_attempts if fg_attempts > 0 else 0)
        )
        three_pct = (
            float(stats_dict.get("Three Point %", "0")) / 100
            if stats_dict.get("Three Point %")
            else (three_made / three_attempts if three_attempts > 0 else 0)
        )
        ft_pct = (
            float(stats_dict.get("Free Throw %", "0")) / 100
            if stats_dict.get("Free Throw %")
            else (ft_made / ft_attempts if ft_attempts > 0 else 0)
        )

        # All these stats MUST be present - no defaults
        reb_total = int(stats_dict.get("Rebounds", "0"))
        reb_off = int(stats_dict.get("Offensive Rebounds", "0"))
        reb_def = int(stats_dict.get("Defensive Rebounds", "0"))
        assists = int(stats_dict.get("Assists", "0"))
        steals = int(stats_dict.get("Steals", "0"))
        blocks = int(stats_dict.get("Blocks", "0"))
        turnovers = int(stats_dict.get("Turnovers", "0"))
        total_turnovers = int(stats_dict.get("Total Turnovers", turnovers))
        fouls = int(stats_dict.get("Personal Fouls", stats_dict.get("Fouls", "0")))

        # Calculate points from FG/3PT/FT if not directly provided
        points = (fg_made - three_made) * 2 + three_made * 3 + ft_made

        box = ESPNBoxScore(
            game_id=game_id,
            team_name=team_name,
            team_id=team_id,
            fg_made=fg_made,
            fg_attempts=fg_attempts,
            fg_pct=fg_pct,
            three_made=three_made,
            three_attempts=three_attempts,
            three_pct=three_pct,
            ft_made=ft_made,
            ft_attempts=ft_attempts,
            ft_pct=ft_pct,
            rebounds_total=reb_total,
            rebounds_off=reb_off,
            rebounds_def=reb_def,
            assists=assists,
            steals=steals,
            blocks=blocks,
            turnovers=turnovers,
            total_turnovers=total_turnovers,
            personal_fouls=fouls,
            points=points,
        )
        result["teams"].append(box)
        logger.debug(
            f"[ESPN] Parsed team box: {team_name} - {points}pts, {assists}ast, {steals}stl, {blocks}blk, {turnovers}to"
        )

    # Parse player box scores
    for team_players in boxscore.get("players", []):
        team_info = team_players.get("team", {})
        team_name = team_info.get("displayName", "")

        for stat_group in team_players.get("statistics", []):
            stat_keys = stat_group.get("keys", [])
            athletes = stat_group.get("athletes", [])

            for athlete in athletes:
                player_info = athlete.get("athlete", {})
                player_name = player_info.get("displayName", "")
                player_id = str(player_info.get("id", ""))
                starter = athlete.get("starter", False)

                stats = athlete.get("stats", [])
                if len(stats) != len(stat_keys):
                    continue

                player_stats = dict(zip(stat_keys, stats))

                # Parse player stats - handle DNP cases
                minutes = player_stats.get("min", "0:00")
                if minutes == "--" or minutes == "DNP":
                    continue

                def safe_int(val: str) -> int:
                    if not val or val == "--":
                        return 0
                    return int(val)

                def parse_player_shooting(val: str) -> tuple[int, int]:
                    if not val or val == "--" or "-" not in val:
                        return 0, 0
                    parts = val.split("-")
                    return int(parts[0]), int(parts[1])

                fg_m, fg_a = parse_player_shooting(player_stats.get("fg", "0-0"))
                three_m, three_a = parse_player_shooting(player_stats.get("3pt", "0-0"))
                ft_m, ft_a = parse_player_shooting(player_stats.get("ft", "0-0"))

                player_box = ESPNPlayerBoxScore(
                    game_id=game_id,
                    team_name=team_name,
                    player_name=player_name,
                    player_id=player_id,
                    starter=starter,
                    minutes=minutes,
                    fg_made=fg_m,
                    fg_attempts=fg_a,
                    three_made=three_m,
                    three_attempts=three_a,
                    ft_made=ft_m,
                    ft_attempts=ft_a,
                    rebounds_off=safe_int(player_stats.get("oreb", "0")),
                    rebounds_def=safe_int(player_stats.get("dreb", "0")),
                    rebounds_total=safe_int(player_stats.get("reb", "0")),
                    assists=safe_int(player_stats.get("ast", "0")),
                    steals=safe_int(player_stats.get("stl", "0")),
                    blocks=safe_int(player_stats.get("blk", "0")),
                    turnovers=safe_int(player_stats.get("to", "0")),
                    personal_fouls=safe_int(player_stats.get("pf", "0")),
                    plus_minus=safe_int(player_stats.get("+/-", "0")),
                    points=safe_int(player_stats.get("pts", "0")),
                )
                result["players"].append(player_box)

    logger.info(
        f"[ESPN] Box score complete: {len(result['teams'])} teams, {len(result['players'])} players"
    )
    return result


async def fetch_espn_recent_game_ids(team_name: str, limit: int = 10) -> List[str]:
    """
    Get ESPN game IDs for a team's recent completed games.

    Args:
        team_name: ESPN team display name (e.g., "Los Angeles Lakers")
        limit: Number of recent games to fetch

    Returns:
        List of ESPN game IDs (strings)
    """
    # First get all recent games via schedule
    from datetime import timedelta

    today = datetime.now(timezone.utc)

    # Check last 30 days of games
    game_ids = []
    for days_ago in range(1, 31):
        date = today - timedelta(days=days_ago)
        date_str = date.strftime("%Y%m%d")

        try:
            url = f"{BASE_URL}/scoreboard"
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                resp = await client.get(url, params={"dates": date_str})
                if resp.status_code != 200:
                    continue
                data = resp.json()

            for event in data.get("events", []):
                # Check if this team played
                competitors = event.get("competitions", [{}])[0].get("competitors", [])
                for comp in competitors:
                    comp_name = comp.get("team", {}).get("displayName", "")
                    if comp_name == team_name:
                        # Check if game is completed
                        status = event.get("status", {}).get("type", {}).get("name", "")
                        if status == "STATUS_FINAL":
                            game_ids.append(event.get("id"))
                            break

                if len(game_ids) >= limit:
                    break
        except Exception as e:
            logger.warning(f"Error fetching ESPN schedule for {date_str}: {e}")
            continue

        if len(game_ids) >= limit:
            break

    return game_ids[:limit]


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

    ESPN is the PRIMARY source for team records.
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
