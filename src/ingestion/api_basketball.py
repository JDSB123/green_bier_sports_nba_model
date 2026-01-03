"""
API-Basketball Client for NBA Data Ingestion.

ROBUST ENDPOINTS (ranked by prediction value):
    TIER 1 - ESSENTIAL (always ingest):
        /teams              - 34 NBA teams with IDs (base reference)
        /games              - Season schedule with Q1-Q4 scores (core data)
        /statistics         - Team PPG, PAPG, W-L home/away (key features)
        /games/statistics/teams   - Full box scores per game (advanced stats)

    TIER 2 - VALUABLE (ingest for richer features):
        /standings          - Conference/Division standings & rankings
        /games?h2h          - Head-to-head history between teams
        /games/statistics/players - Player-level box scores

    TIER 3 - REFERENCE ONLY (static, ingest occasionally):
        /players            - Team rosters (rarely changes mid-season)
        /bookmakers         - Sportsbook names (static reference)
        /bets               - Bet type definitions (static reference)

    NOT AVAILABLE:
        /odds               - Returns 0 records for NBA (use The Odds API instead)

Usage:
    from src.ingestion.api_basketball import APIBasketballClient

    client = APIBasketballClient()
    await client.ingest_all()        # All robust endpoints
    await client.ingest_essential()  # Tier 1 only (faster, less API calls)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.utils.logging import get_logger
from src.utils.team_names import normalize_team_name, get_canonical_name

logger = get_logger(__name__)

BASE_URL = settings.api_basketball_base_url
HEADERS = {"x-apisports-key": settings.api_basketball_key}

# Constants
NBA_LEAGUE_ID = 12
BATCH_SIZE = 20
TIMEOUT = 30


@dataclass
class EndpointResult:
    """Result from an endpoint fetch."""

    name: str
    data: dict[str, Any]
    count: int
    path: str | None = None


class APIBasketballClient:
    """Client for API-Basketball with NBA-focused ingestion."""

    def __init__(
        self,
        season: str | None = None,
        league_id: int = NBA_LEAGUE_ID,
        output_dir: str | None = None,
    ):
        self.season = season or settings.current_season
        self.league_id = league_id
        self.output_dir = output_dir or os.path.join(
            settings.data_raw_dir, "api_basketball"
        )
        
        # Validate API key
        if not settings.api_basketball_key:
            raise ValueError("API_BASKETBALL_KEY is not set")
        
        self.headers = {"x-apisports-key": settings.api_basketball_key}
        self.base_url = settings.api_basketball_base_url

        # Cache for reuse across endpoints
        self._teams: list[dict] = []
        self._games: list[dict] = []

    # =========================================================================
    # CORE HTTP METHOD
    # =========================================================================

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
    )
    async def _fetch(
        self, endpoint: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Make HTTP request to API-Basketball with circuit breaker protection."""
        from src.utils.circuit_breaker import get_api_basketball_breaker
        
        url = f"{self.base_url}/{endpoint}"
        logger.debug(f"Fetching endpoint: {url}")
        
        async def _fetch_with_client():
            async with httpx.AsyncClient(timeout=TIMEOUT, headers=self.headers) as client:
                resp = await client.get(url, params=params or {})
                resp.raise_for_status()
                data = resp.json()
                logger.info(f"Successfully fetched {endpoint}: {len(data.get('response', []))} records")
                return data
        
        # Use circuit breaker to prevent cascading failures
        breaker = get_api_basketball_breaker()
        try:
            return await breaker.call_async(_fetch_with_client)
        except Exception as e:
            logger.error(f"Failed to fetch from API-Basketball ({endpoint}): {e}")
            raise

    def _save(self, name: str, data: dict[str, Any]) -> str:
        """Save response to JSON file with timestamp."""
        os.makedirs(self.output_dir, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = os.path.join(self.output_dir, f"{name}_{ts}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.debug(f"Saved {name} data to {path}")
        return path

    # =========================================================================
    # INDIVIDUAL ENDPOINT METHODS
    # =========================================================================

    async def fetch_teams(self) -> EndpointResult:
        """Fetch all NBA teams."""
        data = await self._fetch(
            "teams", {"league": self.league_id, "season": self.season}
        )
        self._teams = data.get("response", [])
        path = self._save("teams", data)
        return EndpointResult(
            name="teams", data=data, count=len(self._teams), path=path
        )

    async def fetch_games(self, standardize: bool = True) -> EndpointResult:
        """
        Fetch all games for the season with Q1-Q4 scores.
        
        Team name standardization is MANDATORY and always performed to ensure
        data consistency across sources.
        
        Args:
            standardize: Always True - team names are always standardized (kept for API compatibility)
        """
        data = await self._fetch(
            "games", {"league": self.league_id, "season": self.season}
        )
        games = data.get("response", [])
        
        # ALWAYS standardize team names to ESPN format (mandatory)
        from src.ingestion.standardize import standardize_game_data
        standardized_games = []
        invalid_count = 0
        for game in games:
            try:
                # Extract team names from API-Basketball format
                teams = game.get("teams", {})
                home_team = teams.get("home", {}).get("name", "")
                away_team = teams.get("away", {}).get("name", "")
                
                # Create game dict for standardization
                game_dict = {
                    "home_team": home_team,
                    "away_team": away_team,
                    "date": game.get("date"),
                    "teams": teams,  # Include full teams structure for better extraction
                }
                standardized = standardize_game_data(game_dict, source="api_basketball")
                
                # Only include games with valid team names (prevent fake data)
                if standardized.get("_data_valid", False):
                    # Update game with standardized team names
                    if standardized["home_team"]:
                        teams["home"]["name"] = standardized["home_team"]
                    if standardized["away_team"]:
                        teams["away"]["name"] = standardized["away_team"]
                    standardized_games.append(game)
                else:
                    invalid_count += 1
                    logger.warning(
                        f"Skipping game with invalid team names: "
                        f"home='{home_team}', away='{away_team}'"
                    )
            except Exception as e:
                logger.error(f"Error standardizing game data: {e}. Game: {game.get('teams', {}).get('home', {}).get('name', 'N/A')} vs {game.get('teams', {}).get('away', {}).get('name', 'N/A')}")
                # Do NOT add invalid data - skip it entirely
                invalid_count += 1
        
        games = standardized_games
        data["response"] = games
        logger.info(f"Standardized {len(standardized_games)} valid games from API-Basketball (skipped {invalid_count} invalid games)")
        if invalid_count > 0:
            logger.warning(f"⚠️  {invalid_count} games were skipped due to invalid/unstandardized team names")
        
        self._games = games
        path = self._save("games", data)
        return EndpointResult(
            name="games", data=data, count=len(self._games), path=path
        )

    async def fetch_games_by_date(
        self, game_date: str, timezone: str = "America/Chicago"
    ) -> EndpointResult:
        """Fetch games for a specific date in the given timezone.

        IMPORTANT: API-Basketball stores dates in UTC. A 7pm EST game on Dec 4
        appears as Dec 5 00:00 UTC in the API. This method queries by local
        timezone to get the correct games for that date.

        Args:
            game_date: Date string in YYYY-MM-DD format (e.g., "2025-12-04")
            timezone: IANA timezone string (default: America/Chicago for CST)

        Returns:
            EndpointResult with games for that date in the specified timezone
        """
        from src.config import get_nba_season
        from datetime import datetime as dt

        # Parse date and get correct season
        parsed_date = dt.strptime(game_date, "%Y-%m-%d").date()
        season = get_nba_season(parsed_date)

        # Query with timezone parameter to get correct local-date games
        data = await self._fetch(
            "games",
            {
                "league": self.league_id,
                "date": game_date,
                "timezone": timezone,
            },
        )
        games = data.get("response", [])
        path = self._save(f"games_{game_date}", data)
        return EndpointResult(
            name=f"games_{game_date}", data=data, count=len(games), path=path
        )

    async def fetch_players(self) -> EndpointResult:
        """Fetch rosters for all teams."""
        if not self._teams:
            await self.fetch_teams()

        all_players: list[dict] = []
        for team in self._teams:
            team_id = team["id"]
            try:
                data = await self._fetch(
                    "players", {"team": team_id, "season": self.season}
                )
                for player in data.get("response", []):
                    player["team_id"] = team_id
                all_players.extend(data.get("response", []))
            except Exception as e:
                logger.warning(f"Failed to fetch players for team {team_id}: {e}")

        payload = {"response": all_players, "results": len(all_players)}
        path = self._save("players", payload)
        return EndpointResult(
            name="players", data=payload, count=len(all_players), path=path
        )

    async def fetch_standings(self, stage: str | None = None) -> EndpointResult:
        """
        Fetch conference/division standings with full rankings.
        
        The standings include:
        - Conference rank (1-15)
        - Division rank (1-5)
        - Win/Loss record
        - Win percentage
        - Games behind leader
        - Streak (W/L)
        - Last 10 games record
        - Home/Away records
        
        Args:
            stage: Optional stage filter (e.g., "Regular Season")
        
        Returns:
            EndpointResult with standings data including:
            - position: Conference rank
            - group.name: Conference/Division name
            - games.win/lose: Record
            - streak: Current streak
            - form: Last 5 games
        """
        params = {"league": self.league_id, "season": self.season}
        if stage:
            params["stage"] = stage
            
        data = await self._fetch("standings", params)
        
        # Flatten nested structure and extract rankings
        raw_standings = data.get("response", [])
        standings = []
        
        for group in raw_standings:
            if isinstance(group, list):
                for team_standing in group:
                    standings.append(team_standing)
            else:
                standings.append(group)
        
        # Add computed rankings if not present
        for i, standing in enumerate(standings):
            if "conference_rank" not in standing:
                standing["conference_rank"] = standing.get("position", i + 1)
            if "games_behind" not in standing:
                # Calculate games behind from win/loss difference
                games = standing.get("games", {})
                win = games.get("win", {}).get("total", 0)
                lose = games.get("lose", {}).get("total", 0)
                standing["win_pct"] = win / (win + lose) if (win + lose) > 0 else 0.5
        
        # Sort by win percentage to determine proper rankings
        standings.sort(key=lambda x: x.get("win_pct", 0), reverse=True)
        for i, standing in enumerate(standings):
            standing["overall_rank"] = i + 1
        
        path = self._save("standings", data)
        return EndpointResult(
            name="standings", data={"response": standings}, count=len(standings), path=path
        )

    async def fetch_statistics(self) -> EndpointResult:
        """Fetch season statistics (PPG, PAPG, W-L) for all teams."""
        if not self._teams:
            await self.fetch_teams()

        all_stats: list[dict] = []
        for team in self._teams:
            team_id = team["id"]
            try:
                data = await self._fetch(
                    "statistics",
                    {"league": self.league_id, "season": self.season, "team": team_id},
                )
                stats = data.get("response", {})
                if stats:
                    all_stats.append(stats)
            except Exception as e:
                logger.warning(f"Failed to fetch statistics for team {team_id}: {e}")

        payload = {"response": all_stats, "results": len(all_stats)}
        path = self._save("statistics", payload)
        return EndpointResult(
            name="statistics", data=payload, count=len(all_stats), path=path
        )

    async def fetch_game_stats_teams(self) -> EndpointResult:
        """Fetch box scores for all finished games."""
        if not self._games:
            await self.fetch_games()

        finished_ids = [
            g["id"] for g in self._games if g.get("status", {}).get("short") == "FT"
        ]

        all_stats: list[dict] = []
        for i in range(0, len(finished_ids), BATCH_SIZE):
            batch = finished_ids[i : i + BATCH_SIZE]
            try:
                data = await self._fetch(
                    "games/statistics/teams", {"ids": "-".join(map(str, batch))}
                )
                all_stats.extend(data.get("response", []))
            except Exception as e:
                logger.warning(f"Failed to fetch game_stats_teams batch {i}: {e}")

        payload = {"response": all_stats, "results": len(all_stats)}
        path = self._save("game_stats_teams", payload)
        return EndpointResult(
            name="game_stats_teams",
            data=payload,
            count=len(all_stats),
            path=path,
        )

    async def fetch_game_stats_players(self) -> EndpointResult:
        """Fetch player box scores for all finished games."""
        if not self._games:
            await self.fetch_games()

        finished_ids = [
            g["id"] for g in self._games if g.get("status", {}).get("short") == "FT"
        ]

        all_stats: list[dict] = []
        for i in range(0, len(finished_ids), BATCH_SIZE):
            batch = finished_ids[i : i + BATCH_SIZE]
            try:
                data = await self._fetch(
                    "games/statistics/players", {"ids": "-".join(map(str, batch))}
                )
                all_stats.extend(data.get("response", []))
            except Exception as e:
                logger.warning(f"Failed to fetch game_stats_players batch {i}: {e}")

        payload = {"response": all_stats, "results": len(all_stats)}
        path = self._save("game_stats_players", payload)
        return EndpointResult(
            name="game_stats_players",
            data=payload,
            count=len(all_stats),
            path=path,
        )

    async def fetch_h2h(self) -> EndpointResult:
        """Fetch head-to-head history for upcoming matchups."""
        if not self._games:
            await self.fetch_games()

        upcoming = [
            g
            for g in self._games
            if g.get("status", {}).get("short") in ("NS", "TBD")
        ]

        all_h2h: list[dict] = []
        seen: set[str] = set()

        for game in upcoming:
            home_id = game.get("teams", {}).get("home", {}).get("id")
            away_id = game.get("teams", {}).get("away", {}).get("id")
            if not (home_id and away_id):
                continue

            # Normalize to avoid duplicate calls
            key = f"{min(home_id, away_id)}-{max(home_id, away_id)}"
            if key in seen:
                continue
            seen.add(key)

            try:
                data = await self._fetch("games", {"h2h": f"{home_id}-{away_id}"})
                for g in data.get("response", []):
                    g["matchup"] = key
                all_h2h.extend(data.get("response", []))
            except Exception as e:
                logger.warning(f"Failed to fetch h2h for {key}: {e}")

        payload = {"response": all_h2h, "results": len(all_h2h)}
        path = self._save("h2h", payload)
        return EndpointResult(name="h2h", data=payload, count=len(all_h2h), path=path)

    async def fetch_games_by_date_range(
        self,
        start_date: str,
        end_date: str,
        timezone: str = "America/Chicago",
    ) -> EndpointResult:
        """
        Fetch all games in a date range for backtesting.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            timezone: IANA timezone string
        
        Returns:
            EndpointResult with all games in the date range
        """
        from datetime import datetime as dt, timedelta
        
        all_games: list[dict] = []
        current_date = dt.strptime(start_date, "%Y-%m-%d")
        end = dt.strptime(end_date, "%Y-%m-%d")
        
        logger.info(f"Fetching games from {start_date} to {end_date}")
        
        while current_date <= end:
            date_str = current_date.strftime("%Y-%m-%d")
            try:
                data = await self._fetch(
                    "games",
                    {
                        "league": self.league_id,
                        "date": date_str,
                        "timezone": timezone,
                    },
                )
                games = data.get("response", [])
                for g in games:
                    g["query_date"] = date_str
                all_games.extend(games)
            except Exception as e:
                logger.warning(f"Error fetching games for {date_str}: {e}")
            
            current_date += timedelta(days=1)
        
        payload = {"response": all_games, "results": len(all_games)}
        path = self._save(f"games_{start_date}_to_{end_date}", payload)
        return EndpointResult(
            name=f"games_{start_date}_to_{end_date}",
            data=payload,
            count=len(all_games),
            path=path,
        )

    async def fetch_full_h2h(
        self,
        team1_id: int,
        team2_id: int,
        seasons: list[str] | None = None,
    ) -> EndpointResult:
        """
        Fetch complete head-to-head history between two teams.
        
        Args:
            team1_id: First team ID
            team2_id: Second team ID
            seasons: Optional list of seasons to fetch (default: current season)
        
        Returns:
            EndpointResult with all H2H games across specified seasons
        """
        seasons = seasons or [self.season]
        all_h2h: list[dict] = []
        h2h_key = f"{team1_id}-{team2_id}"
        
        for season in seasons:
            try:
                data = await self._fetch(
                    "games",
                    {
                        "h2h": h2h_key,
                        "league": self.league_id,
                        "season": season,
                    },
                )
                games = data.get("response", [])
                for g in games:
                    g["matchup"] = h2h_key
                    g["h2h_season"] = season
                all_h2h.extend(games)
            except Exception as e:
                logger.warning(f"Error fetching H2H for {h2h_key} season {season}: {e}")
        
        # Sort by date (most recent first)
        all_h2h.sort(key=lambda x: x.get("date", ""), reverse=True)
        
        payload = {"response": all_h2h, "results": len(all_h2h)}
        path = self._save(f"h2h_{team1_id}_{team2_id}", payload)
        return EndpointResult(
            name=f"h2h_{team1_id}_{team2_id}",
            data=payload,
            count=len(all_h2h),
            path=path,
        )

    async def fetch_team_recent_games(
        self,
        team_id: int,
        last_n: int = 10,
    ) -> EndpointResult:
        """
        Fetch a team's most recent games for form calculation.
        
        Args:
            team_id: Team ID
            last_n: Number of recent games to fetch
        
        Returns:
            EndpointResult with the team's last N games
        """
        # Fetch all games for the team this season
        data = await self._fetch(
            "games",
            {
                "league": self.league_id,
                "season": self.season,
                "team": team_id,
            },
        )
        
        games = data.get("response", [])
        
        # Filter to finished games only
        finished_games = [
            g for g in games if g.get("status", {}).get("short") == "FT"
        ]
        
        # Sort by date (most recent first)
        finished_games.sort(key=lambda x: x.get("date", ""), reverse=True)
        
        # Take last N games
        recent_games = finished_games[:last_n]
        
        # Add computed fields for form analysis
        for game in recent_games:
            teams = game.get("teams", {})
            scores = game.get("scores", {})
            
            home_team = teams.get("home", {})
            away_team = teams.get("away", {})
            home_score = scores.get("home", {}).get("total", 0) or 0
            away_score = scores.get("away", {}).get("total", 0) or 0
            
            # Determine if this team won
            is_home = home_team.get("id") == team_id
            if is_home:
                game["team_score"] = home_score
                game["opp_score"] = away_score
                game["is_home"] = True
            else:
                game["team_score"] = away_score
                game["opp_score"] = home_score
                game["is_home"] = False
            
            game["won"] = game["team_score"] > game["opp_score"]
            game["margin"] = game["team_score"] - game["opp_score"]
        
        payload = {"response": recent_games, "results": len(recent_games)}
        path = self._save(f"team_{team_id}_recent", payload)
        return EndpointResult(
            name=f"team_{team_id}_recent",
            data=payload,
            count=len(recent_games),
            path=path,
        )

    async def fetch_bookmakers(self) -> EndpointResult:
        """Fetch all bookmakers (sportsbook reference)."""
        data = await self._fetch("bookmakers")
        path = self._save("bookmakers", data)
        return EndpointResult(
            name="bookmakers",
            data=data,
            count=len(data.get("response", [])),
            path=path,
        )

    async def fetch_bets(self) -> EndpointResult:
        """Fetch all bet types (spreads, totals, props, etc.)."""
        data = await self._fetch("bets")
        path = self._save("bets", data)
        return EndpointResult(
            name="bets", data=data, count=len(data.get("response", [])), path=path
        )

    # =========================================================================
    # FULL INGESTION
    # =========================================================================

    async def ingest_essential(self) -> list[EndpointResult]:
        """Ingest TIER 1 endpoints only (fastest, fewest API calls).

        TIER 1 - ESSENTIAL:
            /teams              - Base team reference
            /games              - Core game data with Q1-Q4 scores
            /statistics         - Team PPG, PAPG, W-L (key features)
            /games/statistics/teams - Full box scores (advanced stats)
        """
        results: list[EndpointResult] = []

        # Order matters: teams/games first (cached for subsequent calls)
        endpoints = [
            ("teams", self.fetch_teams),
            ("games", self.fetch_games),
            ("statistics", self.fetch_statistics),
            ("game_stats_teams", self.fetch_game_stats_teams),
        ]

        for name, fetch_fn in endpoints:
            logger.info(f"Ingesting: {name}")
            try:
                result = await fetch_fn()
                results.append(result)
                logger.info(f"  [OK] {result.count} records -> {result.path}")
            except Exception as e:
                logger.error(f"  [ERROR] {name} failed: {e}")

        return results

    async def ingest_all(self) -> list[EndpointResult]:
        """Ingest ALL robust endpoints (TIER 1 + TIER 2 + TIER 3).

        TIER 1 - ESSENTIAL:
            /teams, /games, /statistics, /games/statistics/teams

        TIER 2 - VALUABLE:
            /standings, /games?h2h, /games/statistics/players

        TIER 3 - REFERENCE:
            /players, /bookmakers, /bets

        Order matters:
        1. teams    - Required for players, statistics
        2. games    - Required for game_stats_*, h2h
        """
        results: list[EndpointResult] = []

        endpoints = [
            # TIER 1 - ESSENTIAL
            ("teams", self.fetch_teams),
            ("games", self.fetch_games),
            ("statistics", self.fetch_statistics),
            ("game_stats_teams", self.fetch_game_stats_teams),
            # TIER 2 - VALUABLE
            ("standings", self.fetch_standings),
            ("h2h", self.fetch_h2h),
            ("game_stats_players", self.fetch_game_stats_players),
            # TIER 3 - REFERENCE
            ("players", self.fetch_players),
            ("bookmakers", self.fetch_bookmakers),
            ("bets", self.fetch_bets),
        ]

        for name, fetch_fn in endpoints:
            logger.info(f"Ingesting: {name}")
            try:
                result = await fetch_fn()
                results.append(result)
                logger.info(f"  [OK] {result.count} records -> {result.path}")
            except Exception as e:
                logger.error(f"  [ERROR] {name} failed: {e}")

        return results


# =============================================================================
# BACKWARDS COMPATIBILITY - Legacy function wrappers
# =============================================================================
# 
# ⚠️ DEPRECATED: These functions are maintained for backwards compatibility only.
# New code should use APIBasketballClient class methods directly.
# These may be removed in a future version.
#
# Keep these for any existing code that imports them directly

async def fetch_games(
    season: str | int | None = None, league: int | None = None
) -> dict[str, Any]:
    """Legacy wrapper for fetching games."""
    client = APIBasketballClient()
    params: dict[str, Any] = {}
    if season:
        params["season"] = season
    if league:
        params["league"] = league
    if not params:
        raise ValueError("fetch_games requires at least one parameter")
    return await client._fetch("games", params)


async def fetch_teams(
    *,
    id: int | None = None,
    league: int | None = None,
    season: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Legacy wrapper for fetching teams."""
    client = APIBasketballClient()
    params: dict[str, Any] = {}
    if id:
        params["id"] = id
    if league:
        params["league"] = league
    if season:
        params["season"] = season
    params.update({k: v for k, v in kwargs.items() if v is not None})
    if not params:
        raise ValueError("fetch_teams requires at least one parameter")
    return await client._fetch("teams", params)


async def fetch_statistics(
    *, league: int, season: str, team: int, date: str | None = None
) -> dict[str, Any]:
    """Legacy wrapper for fetching team statistics."""
    client = APIBasketballClient()
    params = {"league": league, "season": season, "team": team}
    if date:
        params["date"] = date
    return await client._fetch("statistics", params)


async def fetch_players(
    *,
    id: int | None = None,
    team: int | None = None,
    season: str | None = None,
    search: str | None = None,
) -> dict[str, Any]:
    """Legacy wrapper for fetching players."""
    client = APIBasketballClient()
    params: dict[str, Any] = {}
    if id:
        params["id"] = id
    if team:
        params["team"] = team
    if season:
        params["season"] = season
    if search:
        params["search"] = search
    if not params:
        raise ValueError("fetch_players requires at least one parameter")
    return await client._fetch("players", params)


async def fetch_standings(
    *, league: int, season: str, **kwargs: Any
) -> dict[str, Any]:
    """Legacy wrapper for fetching standings."""
    client = APIBasketballClient()
    params = {"league": league, "season": season}
    params.update({k: v for k, v in kwargs.items() if v is not None})
    return await client._fetch("standings", params)


def normalize_standings_response(data: dict[str, Any]) -> dict[str, Dict[str, Any]]:
    """
    Normalize API-Basketball standings into convenient lookups.

    Returns:
        {
            "by_id": {team_id: {...}},
            "by_team": {canonical_team_id_or_name: {...}},
        }
    """
    raw = data.get("response", [])
    entries: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, list):
            entries.extend(item)
        elif isinstance(item, dict):
            entries.append(item)

    by_id: Dict[Any, Dict[str, Any]] = {}
    by_team: Dict[str, Dict[str, Any]] = {}

    for entry in entries:
        team_info = entry.get("team", {}) or {}
        team_id = team_info.get("id")
        team_name = (
            team_info.get("name")
            or team_info.get("nickname")
            or team_info.get("code")
            or ""
        )

        games = entry.get("games", {}) or {}
        wins = games.get("win", {}).get("total", 0)
        losses = games.get("lose", {}).get("total", 0)
        played = games.get("played", wins + losses)
        win_pct = games.get("win", {}).get("percentage")
        if win_pct is None and (wins + losses) > 0:
            win_pct = wins / (wins + losses)

        record = {
            "wins": wins,
            "losses": losses,
            "games_played": played,
            "win_pct": float(win_pct) if win_pct is not None else 0.0,
            "position": entry.get("position"),
            "conference_rank": entry.get("conference", {}).get("rank")
            if isinstance(entry.get("conference"), dict)
            else entry.get("conference_rank"),
            "division_rank": entry.get("division", {}).get("rank")
            if isinstance(entry.get("division"), dict)
            else entry.get("division_rank"),
            "streak": entry.get("streak", {}).get("long")
            if isinstance(entry.get("streak"), dict)
            else entry.get("streak"),
            "form": entry.get("form"),
            "home_record": games.get("win", {}).get("home"),
            "away_record": games.get("win", {}).get("away"),
        }

        if team_id is not None:
            by_id[team_id] = record

        if team_name:
            canonical = normalize_team_name(team_name)
            canonical_full = get_canonical_name(canonical)
            by_team[canonical] = record
            by_team[canonical_full] = record

    return {"by_id": by_id, "by_team": by_team}


async def fetch_game_stats_teams(
    *, id: int | str | None = None, ids: list[int | str] | None = None
) -> dict[str, Any]:
    """Legacy wrapper for fetching game team stats."""
    client = APIBasketballClient()
    params: dict[str, Any] = {}
    if id:
        params["id"] = id
    if ids:
        params["ids"] = "-".join(str(i) for i in ids)
    if not params:
        raise ValueError("fetch_game_stats_teams requires id or ids")
    return await client._fetch("games/statistics/teams", params)


async def fetch_game_stats_players(
    *,
    id: int | str | None = None,
    ids: list[int | str] | None = None,
    player: int | None = None,
    season: str | None = None,
) -> dict[str, Any]:
    """Legacy wrapper for fetching game player stats."""
    client = APIBasketballClient()
    params: dict[str, Any] = {}
    if id:
        params["id"] = id
    if ids:
        params["ids"] = "-".join(str(i) for i in ids)
    if player:
        params["player"] = player
    if season:
        params["season"] = season
    if not params:
        raise ValueError("fetch_game_stats_players requires id/ids or player/season")
    return await client._fetch("games/statistics/players", params)


async def fetch_h2h(
    *,
    h2h: str,
    date: str | None = None,
    league: int | None = None,
    season: str | None = None,
    timezone: str | None = None,
) -> dict[str, Any]:
    """Legacy wrapper for fetching head-to-head."""
    client = APIBasketballClient()
    params: dict[str, Any] = {"h2h": h2h}
    if date:
        params["date"] = date
    if league:
        params["league"] = league
    if season:
        params["season"] = season
    if timezone:
        params["timezone"] = timezone
    return await client._fetch("games", params)


async def fetch_bookmakers(
    *, id: int | None = None, search: str | None = None
) -> dict[str, Any]:
    """Legacy wrapper for fetching bookmakers."""
    client = APIBasketballClient()
    params: dict[str, Any] = {}
    if id:
        params["id"] = id
    if search:
        params["search"] = search
    return await client._fetch("bookmakers", params)


async def fetch_bets(
    *, id: int | None = None, search: str | None = None
) -> dict[str, Any]:
    """Legacy wrapper for fetching bet types."""
    client = APIBasketballClient()
    params: dict[str, Any] = {}
    if id:
        params["id"] = id
    if search:
        params["search"] = search
    return await client._fetch("bets", params)


# Legacy save functions (sync wrappers around _save)
def _save_json(data: dict[str, Any], prefix: str, out_dir: str | None = None) -> str:
    """Legacy save helper."""
    client = APIBasketballClient(output_dir=out_dir)
    return client._save(prefix, data)


async def save_games(data: dict[str, Any], out_dir: str | None = None) -> str:
    return _save_json(data, "games", out_dir)


async def save_teams(data: dict[str, Any], out_dir: str | None = None) -> str:
    return _save_json(data, "teams", out_dir)


async def save_statistics(data: dict[str, Any], out_dir: str | None = None) -> str:
    return _save_json(data, "statistics", out_dir)


async def save_game_stats_teams(data: dict[str, Any], out_dir: str | None = None) -> str:
    return _save_json(data, "game_stats_teams", out_dir)


async def save_game_stats_players(data: dict[str, Any], out_dir: str | None = None) -> str:
    return _save_json(data, "game_stats_players", out_dir)


async def save_h2h(data: dict[str, Any], out_dir: str | None = None) -> str:
    return _save_json(data, "h2h", out_dir)


async def save_players(data: dict[str, Any], out_dir: str | None = None) -> str:
    return _save_json(data, "players", out_dir)


async def save_standings(data: dict[str, Any], out_dir: str | None = None) -> str:
    return _save_json(data, "standings", out_dir)


async def save_bookmakers(data: dict[str, Any], out_dir: str | None = None) -> str:
    return _save_json(data, "bookmakers", out_dir)


async def save_bets(data: dict[str, Any], out_dir: str | None = None) -> str:
    return _save_json(data, "bets", out_dir)
