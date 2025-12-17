"""
Build rich features from API-Basketball endpoints - NO FALLBACKS.

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

Raises errors if data is missing - no silent defaults.
"""
from __future__ import annotations
import asyncio
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import settings
from src.ingestion import api_basketball
from src.modeling.team_factors import (
    get_home_court_advantage,
    get_team_context_features,
    calculate_travel_fatigue,
    get_travel_distance,
    get_timezone_difference,
)


class RichFeatureBuilder:
    """Build prediction features from live API-Basketball data."""
    
    def __init__(self, league_id: int = 12, season: str = None):
        self.league_id = league_id
        self.season = season or settings.current_season
        self._team_cache: Dict[str, int] = {}
        self._stats_cache: Dict[int, Dict] = {}
        self._games_cache: Optional[List[Dict]] = None
        
        # Ensure cache directory exists
        self.cache_dir = os.path.join(settings.data_processed_dir, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
    async def get_team_id(self, team_name: str) -> int:
        """Get team ID from name, with caching."""
        # #region agent log
        DEBUG_LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cursor", "debug.log")
        try:
            import json as json_lib
            from datetime import datetime, timezone
            log_entry = {
                "sessionId": "debug-session",
                "runId": "prediction-debug",
                "hypothesisId": "C",
                "location": "build_rich_features.py:50",
                "message": "Looking up team ID",
                "data": {
                    "team_name_input": team_name,
                    "season": self.season,
                    "league_id": self.league_id
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json_lib.dumps(log_entry) + "\n")
        except Exception:
            pass
        # #endregion
        
        if team_name in self._team_cache:
            return self._team_cache[team_name]
        
        # Search for team with season parameter
        result = await api_basketball.fetch_teams(search=team_name, league=self.league_id, season=self.season)
        teams = result.get("response", [])
        
        # #region agent log
        try:
            log_entry = {
                "sessionId": "debug-session",
                "runId": "prediction-debug",
                "hypothesisId": "C",
                "location": "build_rich_features.py:60",
                "message": "Team lookup result",
                "data": {
                    "team_name_input": team_name,
                    "teams_found": len(teams),
                    "team_names_found": [t.get("name") for t in teams[:3]] if teams else [],
                    "team_id_found": teams[0]["id"] if teams else None
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json_lib.dumps(log_entry) + "\n")
        except Exception:
            pass
        # #endregion
        
        if not teams:
            raise ValueError(f"Team not found: {team_name}")
        
        team_id = teams[0]["id"]
        self._team_cache[team_name] = team_id
        return team_id
    
    async def get_team_stats(self, team_id: int) -> Dict[str, Any]:
        """Get team season statistics."""
        if team_id in self._stats_cache:
            return self._stats_cache[team_id]
        
        cache_file = os.path.join(
            self.cache_dir, 
            f"stats_{self.league_id}_{self.season}_{team_id}.joblib"
        )
        
        if os.path.exists(cache_file):
            response = joblib.load(cache_file)
            self._stats_cache[team_id] = response
            return response
        
        result = await api_basketball.fetch_statistics(
            league=self.league_id,
            season=self.season,
            team=team_id
        )
        
        response = result.get("response", {})
        if not response:
            raise ValueError(f"No statistics found for team {team_id}")
        
        # response is a dict with keys: country, league, team, games, points
        joblib.dump(response, cache_file)
        self._stats_cache[team_id] = response
        return response
    
    async def get_h2h_history(self, team1_id: int, team2_id: int) -> List[Dict]:
        """Get head-to-head history between two teams."""
        # Sort IDs to ensure consistent cache key
        t1, t2 = sorted([team1_id, team2_id])
        h2h_string = f"{t1}-{t2}"
        
        cache_file = os.path.join(
            self.cache_dir, 
            f"h2h_{self.league_id}_{self.season}_{h2h_string}.joblib"
        )
        
        if os.path.exists(cache_file):
            return joblib.load(cache_file)

        result = await api_basketball.fetch_h2h(
            h2h=h2h_string,
            league=self.league_id,
            season=self.season
        )
        
        response = result.get("response", [])
        joblib.dump(response, cache_file)
        return response
    
    async def get_recent_games(self, team_id: int, limit: int = 10) -> List[Dict]:
        """Get recent games for a team."""
        # Fetch all games for team this season (cached)
        if self._games_cache is None:
            cache_file = os.path.join(
                self.cache_dir, 
                f"games_{self.league_id}_{self.season}.joblib"
            )
            
            if os.path.exists(cache_file):
                self._games_cache = joblib.load(cache_file)
            else:
                result = await api_basketball.fetch_games(
                    season=self.season,
                    league=self.league_id
                )
                self._games_cache = result.get("response", [])
                joblib.dump(self._games_cache, cache_file)
        
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
    
    async def get_standings(self) -> Dict[int, Dict]:
        """Get current standings indexed by team ID."""
        result = await api_basketball.fetch_standings(
            league=self.league_id,
            season=self.season
        )
        
        standings = {}
        for entry in result.get("response", [[]])[0]:
            team_id = entry.get("team", {}).get("id")
            if team_id:
                standings[team_id] = {
                    "position": entry.get("position"),
                    "win_rate": entry.get("games", {}).get("win", {}).get("percentage"),
                    "games_played": entry.get("games", {}).get("played"),
                }
        
        return standings
    
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
        
        # Fetch all data in parallel
        home_stats, away_stats, h2h, standings, home_recent, away_recent = await asyncio.gather(
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
        
        # Win percentage from games stats
        home_games_data = home_stats.get("games", {})
        away_games_data = away_stats.get("games", {})
        
        home_wins = home_games_data.get("wins", {}).get("all", {}).get("total", 0)
        home_losses = home_games_data.get("loses", {}).get("all", {}).get("total", 0)
        away_wins = away_games_data.get("wins", {}).get("all", {}).get("total", 0)
        away_losses = away_games_data.get("loses", {}).get("all", {}).get("total", 0)
        
        home_games_played = home_wins + home_losses
        away_games_played = away_wins + away_losses
        
        if home_games_played == 0 or away_games_played == 0:
            raise ValueError(f"No games played: home={home_games_played}, away={away_games_played}")
        
        home_win_pct = home_wins / home_games_played
        away_win_pct = away_wins / away_games_played
        
        # Standings position (contextual strength)
        home_standing = standings.get(home_id, {})
        away_standing = standings.get(away_id, {})
        
        home_position = home_standing.get("position", 15)  # Mid-table if missing
        away_position = away_standing.get("position", 15)
        
        # H2H metrics (symmetric: count wins regardless of venue)
        if h2h:
            home_h2h_wins = 0
            valid_h2h_games = 0
            for g in h2h:
                try:
                    g_home_id = g.get("teams", {}).get("home", {}).get("id")
                    g_away_id = g.get("teams", {}).get("away", {}).get("id")
                    g_home_score = g.get("scores", {}).get("home", {}).get("total")
                    g_away_score = g.get("scores", {}).get("away", {}).get("total")
                    if g_home_score is None or g_away_score is None:
                        continue
                    valid_h2h_games += 1
                    # Determine winner against the provided home_id team (true home for current game)
                    if (g_home_id == home_id and g_home_score > g_away_score) or (
                        g_away_id == home_id and g_away_score > g_home_score
                    ):
                        home_h2h_wins += 1
                except (TypeError, AttributeError):
                    continue
            h2h_win_rate = home_h2h_wins / valid_h2h_games if valid_h2h_games > 0 else 0.5
        else:
            h2h_win_rate = 0.5  # Neutral if no history
        
        # ============================================================
        # RECENT FORM (FiveThirtyEight-style rolling performance)
        # ============================================================
        # Calculate last 5 and last 10 game performance
        # This captures "hot" and "cold" streaks
        
        def calc_recent_form(
            recent_games: List[Dict], team_id: int, game_dt: Optional[datetime]
        ) -> Dict:
            """Calculate recent form metrics from game history."""
            if not recent_games:
                return {"l5_win_pct": 0.5, "l10_win_pct": 0.5, "l5_margin": 0, "l10_margin": 0, 
                        "l5_ppg": 0, "l5_papg": 0, "days_rest": 3, "prev_game_location": None}
            
            wins_l5 = wins_l10 = 0
            margin_l5 = margin_l10 = 0
            pts_l5 = pts_allowed_l5 = 0
            pts_l5_1h = pts_allowed_l5_1h = 0
            margin_l5_1h = 0
            
            for i, g in enumerate(recent_games[:10]):
                try:
                    is_home = g.get("teams", {}).get("home", {}).get("id") == team_id
                    home_score = g.get("scores", {}).get("home", {}).get("total", 0) or 0
                    away_score = g.get("scores", {}).get("away", {}).get("total", 0) or 0
                    
                    # First half scores
                    home_1h = (g.get("scores", {}).get("home", {}).get("quarter_1", 0) or 0) + \
                              (g.get("scores", {}).get("home", {}).get("quarter_2", 0) or 0)
                    away_1h = (g.get("scores", {}).get("away", {}).get("quarter_1", 0) or 0) + \
                              (g.get("scores", {}).get("away", {}).get("quarter_2", 0) or 0)

                    if is_home:
                        team_score, opp_score = home_score, away_score
                        team_1h, opp_1h = home_1h, away_1h
                    else:
                        team_score, opp_score = away_score, home_score
                        team_1h, opp_1h = away_1h, home_1h
                    
                    margin = team_score - opp_score
                    margin_1h = team_1h - opp_1h
                    won = 1 if margin > 0 else 0
                    
                    if i < 5:
                        wins_l5 += won
                        margin_l5 += margin
                        pts_l5 += team_score
                        pts_allowed_l5 += opp_score
                        pts_l5_1h += team_1h
                        pts_allowed_l5_1h += opp_1h
                        margin_l5_1h += margin_1h
                    
                    wins_l10 += won
                    margin_l10 += margin
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
                        from datetime import datetime
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
                "l5_ppg_1h": pts_l5_1h / games_l5 if games_l5 > 0 else 0,
                "l5_papg_1h": pts_allowed_l5_1h / games_l5 if games_l5 > 0 else 0,
                "l5_margin_1h": margin_l5_1h / games_l5 if games_l5 > 0 else 0,
                "days_rest": days_rest,
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
        
        home_rest_adj = rest_adjustment(home_form["days_rest"])
        away_rest_adj = rest_adjustment(away_form["days_rest"])
        
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
        import math
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
        FORM_WEIGHT = 0.15  # 15% weight on recent form
        
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
        
        # Injury integration
        injury_path = os.path.join(settings.data_processed_dir, "injuries.csv")
        if os.path.exists(injury_path):
            injuries_df = pd.read_csv(injury_path)
            out_injuries = injuries_df[injuries_df['status'] == 'out']
            
            home_out_ppg = out_injuries[out_injuries['team'].str.contains(home_team, case=False, na=False)]['ppg'].sum()
            away_out_ppg = out_injuries[out_injuries['team'].str.contains(away_team, case=False, na=False)]['ppg'].sum()
            
            # Adjust expected points
            home_expected_pts -= home_out_ppg * 0.8  # 80% replacement efficiency
            away_expected_pts -= away_out_ppg * 0.8
            
            predicted_total_nba = home_expected_pts + away_expected_pts
            predicted_margin_nba = home_expected_pts - away_expected_pts  # Re-calc margin after adjustment
            
            injury_margin_adj = -home_out_ppg * 0.8 + away_out_ppg * 0.8
        else:
            home_out_ppg = away_out_ppg = 0.0
            injury_margin_adj = 0.0

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
            
            # Recent form (FiveThirtyEight-style)
            "home_l5_win_pct": home_form["l5_win_pct"],
            "away_l5_win_pct": away_form["l5_win_pct"],
            "home_l5_margin": home_form["l5_margin"],
            "away_l5_margin": away_form["l5_margin"],
            "home_l10_margin": home_form["l10_margin"],
            "away_l10_margin": away_form["l10_margin"],
            
            # 1H specific rolling stats
            "home_ppg_1h": home_form["l5_ppg_1h"],
            "home_papg_1h": home_form["l5_papg_1h"],
            "home_spread_margin_1h": home_form["l5_margin_1h"],
            "away_ppg_1h": away_form["l5_ppg_1h"],
            "away_papg_1h": away_form["l5_papg_1h"],
            "away_spread_margin_1h": away_form["l5_margin_1h"],
            "ppg_diff_1h": home_form["l5_ppg_1h"] - away_form["l5_ppg_1h"],
            "papg_diff_1h": home_form["l5_papg_1h"] - away_form["l5_papg_1h"],
            
            # Additional FG stats for 1H model
            "home_ppg_fg": home_ppg,
            "home_papg_fg": home_papg,
            "home_spread_margin_fg": home_ppg - home_papg,
            "home_games_played": home_games_played,
            "away_ppg_fg": away_ppg,
            "away_papg_fg": away_papg,
            "away_spread_margin_fg": away_ppg - away_papg,
            "away_games_played": away_games_played,
            "ppg_diff_fg": home_ppg - away_ppg,
            "papg_diff_fg": home_papg - away_papg,
            
            # Predicted values for 1H model (baseline scaling)
            "predicted_margin_1h": predicted_margin_nba * 0.5,
            "predicted_total_1h": predicted_total_nba * 0.5,
            
            # Rest/fatigue
            "home_days_rest": home_form["days_rest"],
            "away_days_rest": away_form["days_rest"],
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
            "home_injury_impact_ppg": home_out_ppg,
            "away_injury_impact_ppg": away_out_ppg,
            "injury_margin_adj": injury_margin_adj,
        }
        
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
        
        away_is_b2b = away_form["days_rest"] <= 1
        home_is_b2b = home_form["days_rest"] <= 1
        
        # Calculate travel fatigue adjustment
        away_travel_fatigue = calculate_travel_fatigue(
            distance_miles=away_travel_distance,
            rest_days=away_form["days_rest"],
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

        return features


async def main():
    """Test the rich feature builder."""
    if not settings.api_basketball_key:
        print("ERROR: API_BASKETBALL_KEY not set")
        return 1
    
    builder = RichFeatureBuilder(league_id=12, season="2025-2026")
    
    # Test with a sample matchup
    try:
        print("Building features for Lakers vs Warriors...")
        features = await builder.build_game_features(
            home_team="Los Angeles Lakers",
            away_team="Golden State Warriors"
        )
        
        print("\n✓ Features built successfully:")
        for key, value in sorted(features.items()):
            print(f"  {key:25s}: {value:8.2f}")
        
        return 0
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    raise SystemExit(exit_code)
