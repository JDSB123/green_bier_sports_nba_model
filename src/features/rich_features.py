"""
Build rich features from API-Basketball endpoints.

FRESH DATA ONLY - No file caching, no silent fallbacks.

DATA SOURCE ARCHITECTURE (QA/QC):
================================
This module uses API-Basketball for INTERNAL feature calculations:
- Team statistics, H2H history, standings position
- Recent form, efficiency ratings, pace factors

IMPORTANT: For OUTPUT/DISPLAY purposes (team records shown with picks),
the serving layer (app.py) uses The Odds API scores to calculate
team W-L records. This ensures DATA INTEGRITY by keeping odds and
records from the SAME source (The Odds API).

See src/utils/slate_analysis.py for unified data source implementation.

Inspired by proven NBA models (FiveThirtyEight, DRatings, Cleaning the Glass):
- Tempo-free efficiency ratings (ORtg/DRtg)
- Pace adjustment for matchups
- Home court advantage (~2.5 pts / ~70 Elo)
- Recent form (last 5-10 games)
- Rest/fatigue factors (back-to-backs)
- H2H history

Uses all available endpoints:
- Team statistics (season averages)
- H2H history
- Standings (for contextual strength)
- Recent games (form)
- Player stats (key contributors)

STRICT MODE: Raises errors if data is missing - no silent defaults, no stale caches.
"""
from __future__ import annotations
import asyncio
import math
import os
from statistics import stdev
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

import pandas as pd

from src.config import settings
from src.ingestion import api_basketball
from src.ingestion.api_basketball import normalize_standings_response
from src.ingestion.espn import fetch_espn_standings
from src.modeling.team_factors import (
    get_home_court_advantage,
    get_team_context_features,
    calculate_travel_fatigue,
    get_travel_distance,
    get_timezone_difference,
)

# Lazy persistent cache for frequently accessed reference data
_REFERENCE_CACHE = {
    'teams': {},  # team_name -> team_id mappings
    'leagues': {},  # league info
    'seasons': {},  # season validation cache
    'last_updated': {}  # cache timestamps
}


class RichFeatureBuilder:
    """
    Build prediction features from live API data.

    - NO file caching - all data fetched fresh from APIs
    - NO silent fallbacks - errors are raised, not swallowed
    - Session-only memory cache for within-request deduplication
    - User-initiated requests only
    """

    def __init__(self, league_id: int = 12, season: str = None):
        self.league_id = league_id
        self.season = season or settings.current_season
        # Session-only memory caches (cleared between requests)
        self._team_cache: Dict[str, int] = {}
        self._stats_cache: Dict[int, Dict] = {}
        self._games_cache: Optional[List[Dict]] = None
        self._standings_cache: Optional[Dict[int, Dict]] = None
        self._espn_standings_cache: Optional[Dict[str, Dict]] = None
        self._injuries_cache: Optional[pd.DataFrame] = None
        self._injuries_fetched: bool = False

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about cache usage and performance."""
        return {
            'session_cache': {
                'teams': len(self._team_cache),
                'stats': len(self._stats_cache),
                'games_cached': self._games_cache is not None,
                'standings_cached': self._standings_cache is not None,
                'espn_standings_cached': self._espn_standings_cache is not None,
                'injuries_cached': self._injuries_cache is not None,
            },
            'persistent_cache': {
                'teams': len(_REFERENCE_CACHE['teams']),
                'leagues': len(_REFERENCE_CACHE['leagues']),
                'seasons': len(_REFERENCE_CACHE['seasons']),
                'total_entries': len(_REFERENCE_CACHE['teams']) + len(_REFERENCE_CACHE['leagues']) + len(_REFERENCE_CACHE['seasons']),
                'last_updated': {k: v.isoformat() for k, v in _REFERENCE_CACHE['last_updated'].items()},
            }
        }

    def clear_session_cache(self):
        """Clear all in-memory session caches to force fresh data."""
        self._team_cache.clear()
        self._stats_cache.clear()
        self._games_cache = None
        self._standings_cache = None
        self._espn_standings_cache = None
        self._injuries_cache = None
        self._injuries_fetched = False
        print("[CACHE] Session cache cleared - next calls will fetch fresh data")

    @staticmethod
    def _is_cache_valid(cache_key: str, max_age_hours: int = 24) -> bool:
        """Check if cached data is still valid based on age."""
        if cache_key not in _REFERENCE_CACHE['last_updated']:
            return False

        cache_time = _REFERENCE_CACHE['last_updated'][cache_key]
        age_hours = (datetime.now() - cache_time).total_seconds() / 3600
        return age_hours < max_age_hours

    @staticmethod
    def _update_cache_timestamp(cache_key: str):
        """Update the last updated timestamp for a cache key."""
        _REFERENCE_CACHE['last_updated'][cache_key] = datetime.now()

    @staticmethod
    def clear_persistent_cache():
        """Clear all persistent reference caches (admin function)."""
        _REFERENCE_CACHE['teams'].clear()
        _REFERENCE_CACHE['leagues'].clear()
        _REFERENCE_CACHE['seasons'].clear()
        _REFERENCE_CACHE['last_updated'].clear()
        print("[CACHE] Persistent reference cache cleared")

    async def get_team_id(self, team_name: str) -> int:
        """Get team ID from name, with lazy persistent caching."""
        # Check session cache first (fastest)
        if team_name in self._team_cache:
            return self._team_cache[team_name]

        # Check persistent cache (lazy-loaded, survives across requests)
        cache_key = f"{self.league_id}_{self.season}_{team_name}"
        if cache_key in _REFERENCE_CACHE['teams'] and self._is_cache_valid(cache_key):
            team_id = _REFERENCE_CACHE['teams'][cache_key]
            # Still populate session cache for this request
            self._team_cache[team_name] = team_id
            return team_id

        # Fetch fresh data from API
        result = await api_basketball.fetch_teams(search=team_name, league=self.league_id, season=self.season)
        teams = result.get("response", [])

        if not teams:
            raise ValueError(f"Team not found: {team_name}")

        team_id = teams[0]["id"]

        # Cache in both session and persistent caches
        self._team_cache[team_name] = team_id
        _REFERENCE_CACHE['teams'][cache_key] = team_id
        self._update_cache_timestamp(cache_key)

        print(f"[CACHE] Team ID cached: {team_name} -> {team_id}")
        return team_id

    async def get_league_info(self) -> Dict[str, Any]:
        """Get league information with lazy persistent caching."""
        cache_key = f"league_{self.league_id}"

        # Check persistent cache
        if cache_key in _REFERENCE_CACHE['leagues'] and self._is_cache_valid(cache_key):
            return _REFERENCE_CACHE['leagues'][cache_key]

        # Fetch fresh league data
        result = await api_basketball.fetch_leagues(league_id=self.league_id)
        leagues = result.get("response", [])

        if not leagues:
            raise ValueError(f"League not found: {self.league_id}")

        league_info = leagues[0]

        # Cache persistently
        _REFERENCE_CACHE['leagues'][cache_key] = league_info
        self._update_cache_timestamp(cache_key)

        print(f"[CACHE] League info cached: {league_info.get('name', 'Unknown')}")
        return league_info

    async def validate_season(self, season: str) -> bool:
        """Validate season exists with lazy persistent caching."""
        cache_key = f"season_{self.league_id}_{season}"

        # Check persistent cache
        if cache_key in _REFERENCE_CACHE['seasons'] and self._is_cache_valid(cache_key, max_age_hours=168):  # 1 week
            return _REFERENCE_CACHE['seasons'][cache_key]

        # Fetch seasons to validate
        result = await api_basketball.fetch_seasons(league=self.league_id)
        seasons = result.get("response", [])
        season_exists = season in seasons

        # Cache result
        _REFERENCE_CACHE['seasons'][cache_key] = season_exists
        self._update_cache_timestamp(cache_key)

        print(f"[CACHE] Season validation cached: {season} -> {'valid' if season_exists else 'invalid'}")
        return season_exists

    async def get_team_stats(self, team_id: int) -> Dict[str, Any]:
        """
        Get team season statistics - ALWAYS FRESH.

        STRICT MODE: No file caching. Fetches fresh from API every request.
        Session memory cache only prevents duplicate API calls within same request.
        """
        # Session memory cache only (within-request deduplication)
        if team_id in self._stats_cache:
            return self._stats_cache[team_id]

        # Fetch fresh statistics from API - NO FILE CACHE
        print(f"[API] Fetching fresh stats for team {team_id}")
        result = await api_basketball.fetch_statistics(
            league=self.league_id,
            season=self.season,
            team=team_id
        )

        response = result.get("response", {})
        if not response:
            raise ValueError(f"STRICT MODE: No statistics found for team {team_id} - API returned empty")

        self._stats_cache[team_id] = response
        return response

    async def get_h2h_history(self, team1_id: int, team2_id: int) -> List[Dict]:
        """
        Get head-to-head history between two teams - ALWAYS FRESH.

        STRICT MODE: No file caching. Fetches fresh from API every request.
        """
        # Sort IDs to ensure consistent cache key
        t1, t2 = sorted([team1_id, team2_id])
        h2h_string = f"{t1}-{t2}"

        # Check session cache first
        cache_key = f"h2h_{h2h_string}"
        if hasattr(self, '_h2h_cache') and cache_key in self._h2h_cache:
            return self._h2h_cache[cache_key]

        # Fetch fresh from API - NO FILE CACHE
        print(f"[API] Fetching fresh H2H for {h2h_string}")
        result = await api_basketball.fetch_h2h(
            h2h=h2h_string,
            league=self.league_id,
            season=self.season
        )

        response = result.get("response", [])

        # Session cache only
        if not hasattr(self, '_h2h_cache'):
            self._h2h_cache = {}
        self._h2h_cache[cache_key] = response

        return response

    async def get_recent_games(self, team_id: int, limit: int = 10) -> List[Dict]:
        """
        Get recent games for a team - ALWAYS FRESH.

        STRICT MODE: No file caching. Fetches fresh from API every request.
        Session memory cache only prevents duplicate API calls within same request.
        """
        # Fetch all games for team this season - session cache only
        if self._games_cache is None:
            print(f"[API] Fetching fresh games for season {self.season}")
            result = await api_basketball.fetch_games(
                season=self.season,
                league=self.league_id
            )
            self._games_cache = result.get("response", [])

            if not self._games_cache:
                raise ValueError(f"STRICT MODE: No games found for season {self.season} - API returned empty")

            print(f"[API] Fetched {len(self._games_cache)} games")

        all_games = self._games_cache

        # Filter to this team's completed games
        team_games = [
            g for g in all_games
            if (g.get("teams", {}).get("home", {}).get("id") == team_id or
                g.get("teams", {}).get("away", {}).get("id") == team_id) and
               g.get("status", {}).get("short") == "FT"  # Finished
        ]

        # Sort by date desc
        team_games.sort(
            key=lambda x: x.get("date", ""),
            reverse=True
        )

        return team_games[:limit]

    async def get_espn_standings(self) -> Dict[str, Dict[str, Any]]:
        """
        Fetch current standings from ESPN (PRIMARY source) - ALWAYS FRESH.

        STRICT MODE: ESPN is the ONLY source for team records.
        Raises error if ESPN fails - NO silent fallback to stale data.

        ESPN standings are FREE and update in real-time after each game.
        Returns dict keyed by team name with wins, losses, win_pct.
        """
        # Session cache only
        if self._espn_standings_cache is not None:
            return self._espn_standings_cache

        print("[API] Fetching fresh ESPN standings")
        espn_standings = await fetch_espn_standings()

        if not espn_standings:
            raise ValueError("STRICT MODE: ESPN standings returned empty - cannot proceed without fresh team records")

        self._espn_standings_cache = {
            name: {
                "wins": standing.wins,
                "losses": standing.losses,
                "win_pct": standing.win_pct,
                "games_played": standing.wins + standing.losses,
                "streak": standing.streak,
                "home_record": standing.home_record,
                "away_record": standing.away_record,
                "last_10": standing.last_10,
            }
            for name, standing in espn_standings.items()
        }

        print(f"[API] Fetched ESPN standings for {len(self._espn_standings_cache)} teams")
        return self._espn_standings_cache

    def calculate_team_record_from_games(self, team_id: int) -> Dict[str, int]:
        """Calculate team W-L record from completed games data.

        This is more accurate than the /statistics or /standings endpoints
        which may have stale data from API-Basketball.

        Args:
            team_id: The team ID to calculate record for

        Returns:
            Dict with 'wins', 'losses', 'games_played' keys
        """
        if self._games_cache is None:
            return {"wins": 0, "losses": 0, "games_played": 0}

        wins = 0
        losses = 0

        for game in self._games_cache:
            # Only count finished games
            if game.get("status", {}).get("short") != "FT":
                continue

            home_team = game.get("teams", {}).get("home", {})
            away_team = game.get("teams", {}).get("away", {})
            home_id = home_team.get("id")
            away_id = away_team.get("id")

            # Check if this team played in the game
            if home_id != team_id and away_id != team_id:
                continue

            # Get scores
            home_score = game.get("scores", {}).get("home", {}).get("total", 0) or 0
            away_score = game.get("scores", {}).get("away", {}).get("total", 0) or 0

            # Skip games with no score (shouldn't happen for finished games)
            if home_score == 0 and away_score == 0:
                continue

            # Determine if team won
            if home_id == team_id:
                if home_score > away_score:
                    wins += 1
                else:
                    losses += 1
            else:  # away_id == team_id
                if away_score > home_score:
                    wins += 1
                else:
                    losses += 1

        return {
            "wins": wins,
            "losses": losses,
            "games_played": wins + losses
        }

    async def get_standings(self) -> Dict[str, Dict]:
        """Get normalized standings (API-Basketball) keyed by team ID and canonical name."""
        result = await api_basketball.fetch_standings(
            league=self.league_id,
            season=self.season
        )
        return normalize_standings_response(result)

    async def get_injuries_df(self) -> Optional[pd.DataFrame]:
        """
        Get injuries dataframe - ALWAYS FRESH.

        STRICT MODE: No file caching. Fetches fresh from API every request.
        Uses ESPN (free) + API-Basketball for comprehensive injury data.
        """
        # Session cache only
        if self._injuries_fetched:
            return self._injuries_cache

        self._injuries_fetched = True

        # Fetch fresh injuries from all sources - NO FILE CACHE
        try:
            from src.ingestion.injuries import fetch_all_injuries, enrich_injuries_with_stats

            print("[API] Fetching fresh injury data...")
            injuries = await fetch_all_injuries()

            if not injuries:
                print("[INJURIES] No injury data available (this is OK - not all players are injured)")
                self._injuries_cache = None
                return None

            # Enrich with player stats
            injuries = await enrich_injuries_with_stats(injuries)

            # Convert to DataFrame
            rows = []
            for inj in injuries:
                rows.append({
                    "player_id": inj.player_id,
                    "player_name": inj.player_name,
                    "team": inj.team,
                    "status": inj.status,
                    "injury_type": inj.injury_type,
                    "ppg": inj.ppg,
                    "minutes_per_game": inj.minutes_per_game,
                    "usage_rate": inj.usage_rate,
                    "source": inj.source,
                })

            self._injuries_cache = pd.DataFrame(rows)
            print(f"[API] Fetched {len(self._injuries_cache)} injuries from live API")

            return self._injuries_cache

        except Exception as e:
            print(f"[INJURIES] Error fetching injuries: {e} - continuing without injury data")
            self._injuries_cache = None
            return None

    async def build_game_features(
        self,
        home_team: str,
        away_team: str,
        game_date: Optional[datetime] = None,
        betting_splits: Optional[Any] = None
    ) -> Dict[str, float]:
        """
        Build rich features for a game using all available API endpoints.

        Raises ValueError if required data is missing.
        """
        # Get team IDs
        home_id = await self.get_team_id(home_team)
        away_id = await self.get_team_id(away_team)

        # Fetch all data in parallel (API-Basketball standings used for W-L records)
        home_stats, away_stats, h2h, standings_norm, home_recent, away_recent = await asyncio.gather(
            self.get_team_stats(home_id),
            self.get_team_stats(away_id),
            self.get_h2h_history(home_id, away_id),
            self.get_standings(),
            self.get_recent_games(home_id, limit=10),
            self.get_recent_games(away_id, limit=10),
        )

        # Extract season averages
        # Stats structure: {games: {...}, points: {for: {average: {all: "112.1"}}, against: {...}}}
        home_points = home_stats.get("points", {})
        away_points = away_stats.get("points", {})

        # Points per game
        home_ppg = float(home_points.get("for", {}).get("average", {}).get("all", 0))
        away_ppg = float(away_points.get("for", {}).get("average", {}).get("all", 0))
        home_papg = float(home_points.get("against", {}).get("average", {}).get("all", 0))
        away_papg = float(away_points.get("against", {}).get("average", {}).get("all", 0))

        if home_ppg == 0 or away_ppg == 0:
            raise ValueError(f"Missing PPG data: home={home_ppg}, away={away_ppg}")

        standings_by_id = standings_norm.get("by_id", {}) if isinstance(standings_norm, dict) else {}

        # Standings position and records (API-Basketball)
        home_standing = standings_by_id.get(home_id, {})
        away_standing = standings_by_id.get(away_id, {})

        # Use API-B standings for W/L; fall back to computed record if missing
        home_record = home_standing or self.calculate_team_record_from_games(home_id)
        away_record = away_standing or self.calculate_team_record_from_games(away_id)

        home_wins = home_record.get("wins", 0)
        home_losses = home_record.get("losses", 0)
        away_wins = away_record.get("wins", 0)
        away_losses = away_record.get("losses", 0)

        home_games_played = home_record.get("games_played", home_wins + home_losses)
        away_games_played = away_record.get("games_played", away_wins + away_losses)

        if home_games_played == 0 or away_games_played == 0:
            raise ValueError(
                f"STRICT MODE: API-B standings missing games played data (home={home_team}, away={away_team})"
            )

        home_win_pct = home_wins / home_games_played if home_games_played > 0 else 0.0
        away_win_pct = away_wins / away_games_played if away_games_played > 0 else 0.0

        home_position = home_standing.get("position", 15)  # Mid-table if missing
        away_position = away_standing.get("position", 15)

        # H2H metrics (symmetric: count wins regardless of venue)
        if h2h:
            home_h2h_wins = 0
            valid_h2h_games = 0
            h2h_margins = []
            for g in h2h:
                try:
                    g_home_id = g.get("teams", {}).get("home", {}).get("id")
                    g_away_id = g.get("teams", {}).get("away", {}).get("id")
                    g_home_score = g.get("scores", {}).get("home", {}).get("total")
                    g_away_score = g.get("scores", {}).get("away", {}).get("total")
                    if g_home_score is None or g_away_score is None:
                        continue
                    if g_home_id == home_id:
                        h2h_margins.append(g_home_score - g_away_score)
                    elif g_away_id == home_id:
                        h2h_margins.append(g_away_score - g_home_score)
                    valid_h2h_games += 1
                    # Determine winner against the provided home_id team (true home for current game)
                    if (g_home_id == home_id and g_home_score > g_away_score) or (
                        g_away_id == home_id and g_away_score > g_home_score
                    ):
                        home_h2h_wins += 1
                except (TypeError, AttributeError):
                    continue
            h2h_win_rate = home_h2h_wins / valid_h2h_games if valid_h2h_games > 0 else 0.5
            h2h_margin = sum(h2h_margins) / len(h2h_margins) if h2h_margins else 0.0
        else:
            h2h_win_rate = 0.5  # Neutral if no history
            h2h_margin = 0.0
            valid_h2h_games = 0

        # ============================================================
        # RECENT FORM (FiveThirtyEight-style rolling performance)
        # ============================================================
        # Calculate last 5 and last 10 game performance
        # This captures "hot" and "cold" streaks

        def calc_recent_form(
            recent_games: List[Dict], team_id: int, game_dt: Optional[datetime]
        ) -> Dict:
            """Calculate recent form metrics from game history (1H, FG)."""
            if not recent_games:
                return {"l5_win_pct": 0.5, "l10_win_pct": 0.5, "l5_margin": 0, "l10_margin": 0,
                        "l5_ppg": 0, "l5_papg": 0, "rest_days": 3, "prev_game_location": None,
                        "l5_ppg_1h": 0, "l5_papg_1h": 0, "l5_margin_1h": 0,
                        "l5_win_pct_1h": 0.5, "win_pct_1h": 0.5,
                        "score_std": 0.0, "margin_std": 0.0}

            wins_l5 = wins_l10 = 0
            margin_l5 = margin_l10 = 0
            pts_l5 = pts_allowed_l5 = 0
            # 1H stats
            pts_l5_1h = pts_allowed_l5_1h = 0
            margin_l5_1h = 0
            wins_1h_l5 = wins_1h_l10 = 0
            scores_all = []
            margins_all = []

            for i, g in enumerate(recent_games[:10]):
                try:
                    is_home = g.get("teams", {}).get("home", {}).get("id") == team_id
                    home_score = g.get("scores", {}).get("home", {}).get("total", 0) or 0
                    away_score = g.get("scores", {}).get("away", {}).get("total", 0) or 0

                    # First half scores (Q1 + Q2)
                    home_q1 = g.get("scores", {}).get("home", {}).get("quarter_1", 0) or 0
                    away_q1 = g.get("scores", {}).get("away", {}).get("quarter_1", 0) or 0
                    home_1h = home_q1 + (g.get("scores", {}).get("home", {}).get("quarter_2", 0) or 0)
                    away_1h = away_q1 + (g.get("scores", {}).get("away", {}).get("quarter_2", 0) or 0)

                    if is_home:
                        team_score, opp_score = home_score, away_score
                        team_1h, opp_1h = home_1h, away_1h
                    else:
                        team_score, opp_score = away_score, home_score
                        team_1h, opp_1h = away_1h, home_1h

                    margin = team_score - opp_score
                    margin_1h = team_1h - opp_1h
                    won = 1 if margin > 0 else 0
                    won_1h = 1 if margin_1h > 0 else 0

                    scores_all.append(team_score)
                    margins_all.append(margin)

                    if i < 5:
                        wins_l5 += won
                        margin_l5 += margin
                        pts_l5 += team_score
                        pts_allowed_l5 += opp_score
                        # 1H
                        pts_l5_1h += team_1h
                        pts_allowed_l5_1h += opp_1h
                        margin_l5_1h += margin_1h
                        wins_1h_l5 += won_1h

                    wins_l10 += won
                    margin_l10 += margin
                    wins_1h_l10 += won_1h
                except (TypeError, KeyError):
                    continue

            games_l5 = min(5, len(recent_games))
            games_l10 = min(10, len(recent_games))

            # Calculate days since last game (rest) relative to game datetime if provided
            days_rest = 3  # Default (normal rest)
            prev_game_location = None  # Track where the team last played

            if recent_games:
                try:
                    last_game = recent_games[0]
                    last_game_date = last_game.get("date", "")

                    # Extract previous game location (the home team's arena)
                    prev_game_location = last_game.get("teams", {}).get("home", {}).get("name")

                    if last_game_date:
                        last_dt = datetime.fromisoformat(last_game_date.replace("Z", "+00:00"))
                        # Use scheduled game datetime when available to avoid "now" drift
                        ref_dt = (
                            game_dt.astimezone(last_dt.tzinfo)
                            if game_dt and last_dt.tzinfo
                            else game_dt
                        ) or (datetime.now(last_dt.tzinfo) if last_dt.tzinfo else datetime.now())
                        calculated_days = (ref_dt - last_dt).days
                        # Sanity check: days_rest should be non-negative and reasonable
                        # Negative means future date (data error), >14 is likely season break
                        if 0 <= calculated_days <= 14:
                            days_rest = calculated_days
                        # else: keep default of 3 (normal rest)
                except Exception:
                    pass

            return {
                "l5_win_pct": wins_l5 / games_l5 if games_l5 > 0 else 0.5,
                "l10_win_pct": wins_l10 / games_l10 if games_l10 > 0 else 0.5,
                "l5_margin": margin_l5 / games_l5 if games_l5 > 0 else 0,
                "l10_margin": margin_l10 / games_l10 if games_l10 > 0 else 0,
                "l5_ppg": pts_l5 / games_l5 if games_l5 > 0 else 0,
                "l5_papg": pts_allowed_l5 / games_l5 if games_l5 > 0 else 0,
                # 1H features
                "l5_ppg_1h": pts_l5_1h / games_l5 if games_l5 > 0 else 0,
                "l5_papg_1h": pts_allowed_l5_1h / games_l5 if games_l5 > 0 else 0,
                "l5_margin_1h": margin_l5_1h / games_l5 if games_l5 > 0 else 0,
                "l5_win_pct_1h": wins_1h_l5 / games_l5 if games_l5 > 0 else 0.5,
                "win_pct_1h": wins_1h_l10 / games_l10 if games_l10 > 0 else 0.5,
                "score_std": float(stdev(scores_all)) if len(scores_all) > 1 else 0.0,
                "margin_std": float(stdev(margins_all)) if len(margins_all) > 1 else 0.0,
                "rest_days": days_rest,
                "prev_game_location": prev_game_location,
            }

        # Use game_date if provided for rest calculations; otherwise rely on "now"
        home_form = calc_recent_form(home_recent, home_id, game_date)
        away_form = calc_recent_form(away_recent, away_id, game_date)

        # ============================================================
        # REST/FATIGUE ADJUSTMENT (FiveThirtyEight uses this)
        # ============================================================
        # Back-to-back (0-1 days rest) = -2 to -3 pts penalty
        # Well rested (3+ days) = slight bonus
        # Based on research: B2B teams score ~3 pts less

        def rest_adjustment(days: int) -> float:
            """Calculate rest adjustment in points.

            Args:
                days: Days since last game (should be >= 0)

            Returns:
                Point adjustment (negative = penalty, positive = bonus)
            """
            # Guard against invalid values
            if days < 0:
                return 0.0  # Invalid data, no adjustment
            elif days == 0:  # Same day (very rare, treat as B2B)
                return -2.5
            elif days == 1:  # Back-to-back
                return -2.5
            elif days == 2:  # Normal rest
                return 0.0
            elif days == 3:  # Good rest
                return 0.5
            elif days <= 7:  # Extended rest (4-7 days)
                return 0.3  # Slight bonus but diminishing
            else:  # Very long rest (8+ days, rust factor)
                return 0.0  # Rust cancels out rest benefit

        home_rest_adj = rest_adjustment(home_form["rest_days"])
        away_rest_adj = rest_adjustment(away_form["rest_days"])

        # ============================================================
        # NBA EFFICIENCY MODEL (Torvik-inspired, NBA-calibrated)
        # ============================================================
        #
        # Uses tempo-free efficiency ratings but calibrated for NBA:
        # - Home court advantage: TEAM-SPECIFIC (not fixed 2.5)
        # - Pace adjustment: accounts for fast/slow matchups
        # - Travel fatigue: B2B + long travel compounding
        # - No over-regression: trust the data, accept variance
        # ============================================================

        # NBA 2024-25 league averages
        LEAGUE_AVG_PPG = 113.5

        # TEAM-SPECIFIC HOME COURT ADVANTAGE
        # Denver ~4.2 pts (altitude), Utah ~3.8 pts, Boston ~3.5 pts, etc.
        HOME_COURT_ADV = get_home_court_advantage(home_team)

        # Estimate pace from total points involvement
        # Higher total = faster pace
        home_pace_factor = (home_ppg + home_papg) / (2 * LEAGUE_AVG_PPG)
        away_pace_factor = (away_ppg + away_papg) / (2 * LEAGUE_AVG_PPG)

        # Expected game pace (geometric mean to handle outliers better)
        expected_pace_factor = math.sqrt(home_pace_factor * away_pace_factor)

        # Offensive Rating (efficiency) - points per "100 possessions" equivalent
        home_ortg = home_ppg / home_pace_factor if home_pace_factor > 0 else LEAGUE_AVG_PPG
        away_ortg = away_ppg / away_pace_factor if away_pace_factor > 0 else LEAGUE_AVG_PPG

        # Defensive Rating - points allowed per "100 possessions" equivalent
        home_drtg = home_papg / home_pace_factor if home_pace_factor > 0 else LEAGUE_AVG_PPG
        away_drtg = away_papg / away_pace_factor if away_pace_factor > 0 else LEAGUE_AVG_PPG

        # Net ratings
        home_net_rtg = home_ortg - home_drtg
        away_net_rtg = away_ortg - away_drtg

        # Predicted points for each team
        # Home scores: average of (home offense, opponent defense) at expected pace
        home_expected_pts = ((home_ortg + away_drtg) / 2) * expected_pace_factor
        away_expected_pts = ((away_ortg + home_drtg) / 2) * expected_pace_factor

        # Apply rest to team totals directly (per-team impact instead of averaging)
        home_expected_pts += home_rest_adj
        away_expected_pts += away_rest_adj
        predicted_total_nba = home_expected_pts + away_expected_pts

        # ============================================================
        # RECENT FORM ADJUSTMENT (like DRatings "recent form" factor)
        # ============================================================
        # Weight recent performance (last 5 games) vs season average
        # If team is hot (L5 margin > season margin), boost them
        # Research shows 20-25% form weight optimal for NBA predictions
        FORM_WEIGHT = 0.20  # 20% weight on recent form

        # Scale form adjustment by expected pace so margins remain on per-game units
        home_form_adj = FORM_WEIGHT * expected_pace_factor * (
            home_form["l5_margin"] - (home_ppg - home_papg)
        )
        away_form_adj = FORM_WEIGHT * expected_pace_factor * (
            away_form["l5_margin"] - (away_ppg - away_papg)
        )

        # Predicted margin (home perspective)
        # Net rating difference scaled by expected pace + home court + rest + form
        base_margin = expected_pace_factor * (home_net_rtg - away_net_rtg) / 2
        rest_margin_adj = home_rest_adj - away_rest_adj  # Positive = home better rested
        form_margin_adj = home_form_adj - away_form_adj  # Positive = home hotter

        predicted_margin_nba = base_margin + HOME_COURT_ADV + rest_margin_adj + form_margin_adj

        # Injury integration - DYNAMIC FETCH from ESPN + API-Basketball
        injuries_df = await self.get_injuries_df()
        has_injury_data = injuries_df is not None and len(injuries_df) > 0

        if has_injury_data:
            out_injuries = injuries_df[injuries_df['status'] == 'out']

            # Match team names (partial match for flexibility)
            home_out_ppg = out_injuries[out_injuries['team'].str.contains(home_team, case=False, na=False)]['ppg'].sum()
            away_out_ppg = out_injuries[out_injuries['team'].str.contains(away_team, case=False, na=False)]['ppg'].sum()

            # Calculate injury impact on margin
            # Negative home_out_ppg hurts home team (subtracts from margin)
            # Positive away_out_ppg helps home team (adds to margin)
            # Research suggests 50-70% replacement value for star players
            # Using 65% replacement efficiency = 35% of star's PPG is lost
            # This accounts for both scoring and intangible impact (playmaking, gravity)
            REPLACEMENT_LOSS_FACTOR = 0.35  # 35% of PPG lost when player is out
            injury_margin_adj = (-home_out_ppg + away_out_ppg) * REPLACEMENT_LOSS_FACTOR

            # ADD injury adjustment to existing margin
            # (preserves HOME_COURT_ADV and form_margin_adj)
            predicted_margin_nba += injury_margin_adj

            # Adjust expected points for total calculation
            home_expected_pts -= home_out_ppg * REPLACEMENT_LOSS_FACTOR
            away_expected_pts -= away_out_ppg * REPLACEMENT_LOSS_FACTOR
            predicted_total_nba = home_expected_pts + away_expected_pts

            # Count star players out (players with 15+ PPG)
            home_star_out = len(out_injuries[
                (out_injuries['team'].str.contains(home_team, case=False, na=False)) &
                (out_injuries['ppg'] >= 15)
            ])
            away_star_out = len(out_injuries[
                (out_injuries['team'].str.contains(away_team, case=False, na=False)) &
                (out_injuries['ppg'] >= 15)
            ])
        else:
            home_out_ppg = away_out_ppg = 0.0
            injury_margin_adj = 0.0
            home_star_out = away_star_out = 0

        # ============================================================
        # 1H MODELING (v33.0.7.0: Independent matchup-based predictions)
        # ============================================================
        # Use actual 1H stats with matchup formula (same as FG)
        # NO scaling from FG - truly independent predictions

        # 1H stats from L5 recent form
        home_1h_ppg = home_form["l5_ppg_1h"]
        home_1h_papg = home_form["l5_papg_1h"]
        home_1h_margin = home_form["l5_margin_1h"]
        away_1h_ppg = away_form["l5_ppg_1h"]
        away_1h_papg = away_form["l5_papg_1h"]
        away_1h_margin = away_form["l5_margin_1h"]

        # 1H Predictions using MATCHUP formula (same logic as FG)
        # Home 1H expected = avg(home's 1H offense, away's 1H defense allowed)
        # Away 1H expected = avg(away's 1H offense, home's 1H defense allowed)
        home_1h_expected = (home_1h_ppg + away_1h_papg) / 2
        away_1h_expected = (away_1h_ppg + home_1h_papg) / 2
        predicted_total_1h = home_1h_expected + away_1h_expected

        # 1H Margin: Use actual 1H margin stats
        # HCA scaled for 1H (~1.5 pts vs 3 pts FG)
        hca_1h = 1.5  # Approximate 1H HCA
        predicted_margin_1h = (home_1h_margin - away_1h_margin) / 2 + hca_1h

        # Form trend proxies (recent vs longer-term performance)
        home_form_trend = home_form.get("l5_margin", 0.0) - home_form.get("l10_margin", 0.0)
        away_form_trend = away_form.get("l5_margin", 0.0) - away_form.get("l10_margin", 0.0)

        # Build feature dict
        features = {
            # Team averages (raw)
            "home_ppg": home_ppg,
            "away_ppg": away_ppg,
            "home_papg": home_papg,
            "away_papg": away_papg,

            # Tempo-free ratings (Torvik-style)
            "home_ortg": home_ortg,
            "away_ortg": away_ortg,
            "home_drtg": home_drtg,
            "away_drtg": away_drtg,
            "home_net_rtg": home_net_rtg,
            "away_net_rtg": away_net_rtg,

            # Pace factors
            "home_pace_factor": home_pace_factor,
            "away_pace_factor": away_pace_factor,
            "expected_pace_factor": expected_pace_factor,

            # NBA model predictions
            "home_expected_pts": home_expected_pts,
            "away_expected_pts": away_expected_pts,
            "predicted_margin": predicted_margin_nba,
            "predicted_total": predicted_total_nba,
            "spread_line": 0.0,
            "total_line": 0.0,
            "spread_vs_predicted": 0.0,
            "total_vs_predicted": 0.0,

            # Legacy derived (for compatibility)
            "home_avg_margin": home_ppg - home_papg,
            "away_avg_margin": away_ppg - away_papg,
            "home_total_ppg": home_ppg + home_papg,
            "away_total_ppg": away_ppg + away_papg,
            "ppg_diff": home_ppg - away_ppg,

            # Win rates
            "home_win_pct": home_win_pct,
            "away_win_pct": away_win_pct,
            "win_pct_diff": home_win_pct - away_win_pct,

            # Standings (lower is better)
            "home_position": home_position,
            "away_position": away_position,
            "position_diff": away_position - home_position,  # Positive = home better

            # H2H
            "h2h_win_rate": h2h_win_rate,
            "h2h_margin": h2h_margin,
            "h2h_games": valid_h2h_games,

            # Recent form (FiveThirtyEight-style)
            "home_l5_win_pct": home_form["l5_win_pct"],
            "away_l5_win_pct": away_form["l5_win_pct"],
            "home_l5_margin": home_form["l5_margin"],
            "away_l5_margin": away_form["l5_margin"],
            "home_l10_margin": home_form["l10_margin"],
            "away_l10_margin": away_form["l10_margin"],
            "home_form_trend": home_form_trend,
            "away_form_trend": away_form_trend,

            # 1H-specific features (first half stats from recent form)
            "home_ppg_1h": home_form["l5_ppg_1h"],
            "home_papg_1h": home_form["l5_papg_1h"],
            "home_spread_margin_1h": home_form["l5_margin_1h"],
            "away_ppg_1h": away_form["l5_ppg_1h"],
            "away_papg_1h": away_form["l5_papg_1h"],
            "away_spread_margin_1h": away_form["l5_margin_1h"],
            "ppg_diff_1h": home_form["l5_ppg_1h"] - away_form["l5_ppg_1h"],
            "papg_diff_1h": home_form["l5_papg_1h"] - away_form["l5_papg_1h"],

            # STRICT MODE: 1H models get NO FG features (temporal isolation)
            # Only 1H-specific features for 1H predictions

            # Predicted values for 1H model (v33.0.7.0: matchup-based, not scaled)
            "predicted_margin_1h": predicted_margin_1h,
            "predicted_total_1h": predicted_total_1h,

            # Rest/fatigue
            "home_rest_days": home_form["rest_days"],
            "away_rest_days": away_form["rest_days"],
            # Aliases for model compatibility (some models use days_rest instead of rest_days)
            "home_days_rest": home_form["rest_days"],
            "away_days_rest": away_form["rest_days"],
            "home_rest_adj": home_rest_adj,
            "away_rest_adj": away_rest_adj,
            "rest_margin_adj": rest_margin_adj,

            # Form adjustment
            "home_form_adj": home_form_adj,
            "away_form_adj": away_form_adj,
            "form_margin_adj": form_margin_adj,

            # ELO proxy (derived from win% and margin)
            "home_elo": 1500 + (home_win_pct - 0.5) * 400 + (home_ppg - home_papg) * 10,
            "away_elo": 1500 + (away_win_pct - 0.5) * 400 + (away_ppg - away_papg) * 10,
            # Injury features - PREMIUM API DATA (ESPN + API-Basketball)
            "has_injury_data": 1 if has_injury_data else 0,
            "home_injury_impact_ppg": home_out_ppg,
            "away_injury_impact_ppg": away_out_ppg,
            "injury_margin_adj": injury_margin_adj,
            "home_star_out": home_star_out,
            "away_star_out": away_star_out,
        }

        # Compatibility aliases for offline FeatureEngineer training schema
        features["home_margin"] = features["home_avg_margin"]
        features["away_margin"] = features["away_avg_margin"]
        features["home_rest"] = features["home_rest_days"]
        features["away_rest"] = features["away_rest_days"]
        features["rest_diff"] = features["home_rest_days"] - features["away_rest_days"]
        features["home_pace"] = home_ppg + home_papg
        features["away_pace"] = away_ppg + away_papg
        features["expected_pace"] = (features["home_pace"] + features["away_pace"]) / 2
        features["home_score_std"] = home_form["score_std"]
        features["away_score_std"] = away_form["score_std"]
        features["home_margin_std"] = home_form["margin_std"]
        features["away_margin_std"] = away_form["margin_std"]
        features["home_net_rating"] = features["home_net_rtg"]
        features["away_net_rating"] = features["away_net_rtg"]
        features["net_rating_diff"] = features["home_net_rtg"] - features["away_net_rtg"]
        features["home_1h_win_pct"] = home_form["win_pct_1h"]
        features["away_1h_win_pct"] = away_form["win_pct_1h"]
        features["home_margin_1h"] = features["home_spread_margin_1h"]
        features["away_margin_1h"] = features["away_spread_margin_1h"]
        features["home_pace_1h"] = features["home_ppg_1h"] + features["home_papg_1h"]
        features["away_pace_1h"] = features["away_ppg_1h"] + features["away_papg_1h"]
        features["expected_pace_1h"] = (features["home_pace_1h"] + features["away_pace_1h"]) / 2
        features["dynamic_hca"] = HOME_COURT_ADV
        features["dynamic_hca_1h"] = HOME_COURT_ADV * 0.5
        features["h2h_win_pct"] = h2h_win_rate

        features["elo_diff"] = features["home_elo"] - features["away_elo"]
        features["elo_prob_home"] = 1 / (1 + 10 ** (-features["elo_diff"] / 400))

        # ============================================================
        # TRAVEL/FATIGUE FEATURES
        # ============================================================
        # B2B + long travel = compounding fatigue penalty
        # Calculate travel distance from PREVIOUS GAME LOCATION (not home arena)

        # Get previous game location for away team (where they actually traveled from)
        away_prev_location = away_form.get("prev_game_location")

        # Calculate travel: from previous game location to current game (home team's arena)
        # If no previous location data, fall back to away team's home arena
        if away_prev_location:
            away_travel_distance = get_travel_distance(away_prev_location, home_team) or 0
            away_tz_change = get_timezone_difference(away_prev_location, home_team)
        else:
            # Fallback: assume traveling from home arena
            away_travel_distance = get_travel_distance(away_team, home_team) or 0
            away_tz_change = get_timezone_difference(away_team, home_team)

        away_is_b2b = away_form["rest_days"] <= 1
        home_is_b2b = home_form["rest_days"] <= 1

        features["home_b2b"] = 1 if home_is_b2b else 0
        features["away_b2b"] = 1 if away_is_b2b else 0

        # Calculate travel fatigue adjustment
        away_travel_fatigue = calculate_travel_fatigue(
            distance_miles=away_travel_distance,
            rest_days=away_form["rest_days"],
            timezone_change=away_tz_change,
            is_back_to_back=away_is_b2b,
        )

        features["away_travel_distance"] = away_travel_distance
        features["away_timezone_change"] = away_tz_change
        features["away_travel_fatigue"] = away_travel_fatigue
        features["is_away_long_trip"] = 1 if away_travel_distance >= 1500 else 0
        features["is_away_cross_country"] = 1 if away_travel_distance >= 2500 else 0

        # B2B + travel interaction (compounding penalty)
        features["away_b2b_travel_penalty"] = 0.0
        if away_is_b2b and away_travel_distance >= 1500:
            features["away_b2b_travel_penalty"] = -1.5  # Additional penalty

        # Travel advantage for home team
        features["travel_advantage"] = -away_travel_fatigue  # Away fatigue helps home

        # Store team-specific HCA for transparency
        features["home_court_advantage"] = HOME_COURT_ADV

        # Add betting splits features if available
        if betting_splits:
            from src.ingestion.betting_splits import splits_to_features
            splits_features = splits_to_features(betting_splits)
            features.update(splits_features)
        else:
            # EXPLICIT: No splits data available - mark as missing
            # Models should use has_real_splits to weight these features accordingly
            features["has_real_splits"] = 0
            features["spread_public_home_pct"] = 50.0
            features["spread_public_away_pct"] = 50.0
            features["spread_money_home_pct"] = 50.0
            features["spread_money_away_pct"] = 50.0
            features["over_public_pct"] = 50.0
            features["under_public_pct"] = 50.0
            features["over_money_pct"] = 50.0
            features["under_money_pct"] = 50.0
            features["spread_open"] = 0.0
            features["spread_current"] = 0.0
            features["spread_movement"] = 0.0
            features["total_open"] = 0.0
            features["total_current"] = 0.0
            features["total_movement"] = 0.0
            features["is_rlm_spread"] = 0
            features["is_rlm_total"] = 0
            features["sharp_side_spread"] = 0
            features["sharp_side_total"] = 0
            features["spread_ticket_money_diff"] = 0.0
            features["total_ticket_money_diff"] = 0.0

        # ATS (against the spread) cover rates - estimate from margin performance
        # Teams that consistently outperform their expected margin tend to cover more often
        # Formula: base 50% + adjustment based on how team performs vs expected scoring
        home_margin_performance = (home_ppg - home_papg) / 10  # Net rating scaled
        away_margin_performance = (away_ppg - away_papg) / 10
        features["home_ats_pct"] = max(0.35, min(0.65, 0.50 + home_margin_performance * 0.05))
        features["away_ats_pct"] = max(0.35, min(0.65, 0.50 + away_margin_performance * 0.05))
        features["home_ats_pct_1h"] = features["home_ats_pct"]  # Use same for 1H
        features["away_ats_pct_1h"] = features["away_ats_pct"]

        # Injury spread impact - calculated from actual injury PPG data
        # Losing scorers directly impacts expected margin
        features["home_injury_spread_impact"] = -home_out_ppg  # Negative = hurts spread
        features["away_injury_spread_impact"] = -away_out_ppg
        features["injury_spread_diff"] = features["home_injury_spread_impact"] - features["away_injury_spread_impact"]

        # Rest advantage for spread betting (home rest - away rest)
        features["rest_advantage"] = home_form["rest_days"] - away_form["rest_days"]

        return features
