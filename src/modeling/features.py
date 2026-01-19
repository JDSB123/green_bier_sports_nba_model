from __future__ import annotations

from typing import Dict, Optional, Tuple, List
import logging

import numpy as np
import pandas as pd

from src.modeling.team_factors import (
    get_home_court_advantage,
    get_travel_distance,
    get_timezone_difference,
    calculate_travel_fatigue,
)
from src.modeling.period_features import PERIOD_SCALING
from src.modeling.season_utils import (
    is_crossing_offseason,
)

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering with additional predictive signals."""

    def __init__(self, lookback: int = 10):
        self.lookback = lookback
        self.team_stats_cache: Dict[str, pd.DataFrame] = {}

    def compute_team_rolling_stats(
        self,
        games_df: pd.DataFrame,
        team: str,
        as_of_date: pd.Timestamp,
    ) -> Dict[str, float]:
        """
        Compute comprehensive rolling stats for a team.

        Args:
            games_df: DataFrame with historical games (must be sorted by date)
            team: Team name to compute stats for
            as_of_date: Compute stats as of this date (exclusive - only prior games)

        Returns:
            Dictionary of team statistics, or empty dict if insufficient data

        Raises:
            ValueError: If games_df is not properly sorted or missing required columns
        """
        # Validate input
        required_cols = ["date", "home_team", "away_team", "home_score", "away_score"]
        missing_cols = [col for col in required_cols if col not in games_df.columns]
        if missing_cols:
            raise ValueError(f"games_df missing required columns: {missing_cols}")

        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(games_df["date"]):
            raise ValueError(f"games_df['date'] must be datetime, got {games_df['date'].dtype}")

        # Validate date ordering (this is critical for temporal integrity)
        if len(games_df) > 1 and not games_df["date"].is_monotonic_increasing:
            logger.warning(
                f"games_df is not sorted by date for team {team}. "
                f"This could cause temporal leakage. Sorting now."
            )
            games_df = games_df.sort_values("date")

        # Get all games for this team before the date
        home_games = games_df[
            (games_df["home_team"] == team) & (games_df["date"] < as_of_date)
        ].copy()
        away_games = games_df[
            (games_df["away_team"] == team) & (games_df["date"] < as_of_date)
        ].copy()

        # Normalize to team perspective
        home_games["team_score"] = home_games["home_score"]
        home_games["opp_score"] = home_games["away_score"]
        home_games["is_home"] = 1

        away_games["team_score"] = away_games["away_score"]
        away_games["opp_score"] = away_games["home_score"]
        away_games["is_home"] = 0

        all_games = pd.concat(
            [
                home_games[["date", "team_score", "opp_score", "is_home"]],
                away_games[["date", "team_score", "opp_score", "is_home"]],
            ]
        ).sort_values("date", ascending=False)

        # Check for minimum data - return empty instead of defaulting
        if len(all_games) < 3:
            logger.warning(
                f"Insufficient game history for {team} as of {as_of_date}: "
                f"{len(all_games)} games (need 3+). Returning empty stats."
            )
            return {}

        # Last N games
        recent = all_games.head(self.lookback)

        # Basic stats
        stats = {
            "ppg": recent["team_score"].mean(),
            "papg": recent["opp_score"].mean(),
            "margin": (recent["team_score"] - recent["opp_score"]).mean(),
            "win_pct": (recent["team_score"] > recent["opp_score"]).mean(),
            "games_played": len(all_games),
        }

        # Pace / total points
        stats["pace"] = recent["team_score"].mean() + recent["opp_score"].mean()

        # Variance / consistency
        stats["score_std"] = recent["team_score"].std()
        stats["margin_std"] = (recent["team_score"] - recent["opp_score"]).std()

        # Recent form (last 3 vs last 10)
        if len(recent) >= 5:
            last3 = recent.head(3)
            stats["form_3g"] = (last3["team_score"] - last3["opp_score"]).mean()
            stats["form_trend"] = stats["form_3g"] - stats["margin"]

        # Home/away splits
        home_recent = all_games[all_games["is_home"] == 1].head(5)
        away_recent = all_games[all_games["is_home"] == 0].head(5)

        if len(home_recent) >= 2:
            stats["home_ppg"] = home_recent["team_score"].mean()
            stats["home_margin"] = (
                home_recent["team_score"] - home_recent["opp_score"]
            ).mean()
        if len(away_recent) >= 2:
            stats["away_ppg"] = away_recent["team_score"].mean()
            stats["away_margin"] = (
                away_recent["team_score"] - away_recent["opp_score"]
            ).mean()

        # Rest days average
        if len(all_games) >= 2:
            stats["avg_rest"] = all_games["date"].diff(-1).dt.days.dropna().mean()

        # === NEW: CLUTCH PERFORMANCE FEATURES ===
        # Filter close games (margin <= 5 points)
        all_games_full = pd.concat([home_games, away_games])
        all_games_full["margin"] = all_games_full["team_score"] - all_games_full["opp_score"]
        close_games = all_games_full[abs(all_games_full["margin"]) <= 5]

        if len(close_games) >= 3:
            stats["clutch_win_pct"] = (close_games["margin"] > 0).mean()
            stats["clutch_margin"] = close_games["margin"].mean()
            stats["clutch_games_played"] = len(close_games)
        else:
            stats["clutch_win_pct"] = stats["win_pct"]  # Default to overall
            stats["clutch_margin"] = 0
            stats["clutch_games_played"] = 0

        # === NEW: OPPONENT-ADJUSTED METRICS ===
        # Calculate strength of opponent faced
        if len(all_games_full) >= 5:
            # Approximate opponent strength by points allowed
            opp_strength = all_games_full["opp_score"].mean()
            league_avg = 110  # Approximate NBA average

            # Adjust offensive rating
            stats["adj_offensive_rating"] = stats["ppg"] * (league_avg / opp_strength) if opp_strength > 0 else stats["ppg"]

            # Calculate defensive rating (lower is better)
            stats["defensive_rating"] = stats["papg"]
            stats["adj_defensive_rating"] = stats["papg"] * (opp_strength / league_avg) if opp_strength > 0 else stats["papg"]

            # Net rating
            stats["net_rating"] = stats["adj_offensive_rating"] - stats["adj_defensive_rating"]
        else:
            stats["adj_offensive_rating"] = stats["ppg"]
            stats["adj_defensive_rating"] = stats["papg"]
            stats["net_rating"] = stats["margin"]

        # === NEW: CONSISTENCY METRICS ===
        if len(recent) >= 5:
            # Win streak indicator
            recent_sorted = all_games.head(self.lookback).sort_values("date")
            recent_sorted["won"] = recent_sorted["team_score"] > recent_sorted["opp_score"]

            current_streak = 0
            for won in recent_sorted["won"].values[::-1]:  # Most recent first
                if won:
                    current_streak += 1
                else:
                    break

            stats["win_streak"] = current_streak
            stats["consistency"] = 1.0 - (stats["margin_std"] / (abs(stats["margin"]) + 1))  # Higher = more consistent
        else:
            stats["win_streak"] = 0
            stats["consistency"] = 0.5

        return stats

    def compute_period_rolling_stats(
        self,
        games_df: pd.DataFrame,
        team: str,
        as_of_date: pd.Timestamp,
        period: str = "fg",
    ) -> Dict[str, float]:
        """
        Compute period-specific rolling stats (1H or FG).

        This method computes INDEPENDENT statistics for each period using
        actual historical data for that period, not scaled from FG stats.

        Args:
            games_df: DataFrame with historical games (must have quarter columns)
            team: Team name to compute stats for
            as_of_date: Compute stats as of this date (exclusive)
            period: "1h" or "fg"

        Returns:
            Dictionary of period-specific statistics
        """
        # Check for required quarter columns for sub-game periods
        quarter_cols = ["home_q1", "home_q2", "home_q3", "home_q4",
                        "away_q1", "away_q2", "away_q3", "away_q4"]

        has_quarter_data = all(col in games_df.columns for col in quarter_cols)

        if period == "1h" and not has_quarter_data:
            logger.warning(
                "Quarter columns not available for 1H stats. "
                f"Missing columns will result in empty stats."
            )
            return {}

        # Get all games for this team before the date
        home_games = games_df[
            (games_df["home_team"] == team) & (games_df["date"] < as_of_date)
        ].copy()
        away_games = games_df[
            (games_df["away_team"] == team) & (games_df["date"] < as_of_date)
        ].copy()

        if len(home_games) + len(away_games) < 3:
            return {}

        # Compute period-specific scores - NO ZERO-FILLING!
        # Games with missing quarter data are EXCLUDED, not corrupted with zeros.
        if period == "1h":
            if has_quarter_data:
                # Only include games where Q1 AND Q2 data exists
                home_games = home_games.dropna(subset=["home_q1", "home_q2", "away_q1", "away_q2"])
                away_games = away_games.dropna(subset=["away_q1", "away_q2", "home_q1", "home_q2"])
                if len(home_games) + len(away_games) < 3:
                    logger.debug(f"Insufficient 1H data for {team}: {len(home_games)} home, {len(away_games)} away games with 1H scores")
                    return {}
                home_games["team_score"] = home_games["home_q1"] + home_games["home_q2"]
                home_games["opp_score"] = home_games["away_q1"] + home_games["away_q2"]
                away_games["team_score"] = away_games["away_q1"] + away_games["away_q2"]
                away_games["opp_score"] = away_games["home_q1"] + away_games["home_q2"]
            else:
                return {}
        elif period == "fg":
            home_games["team_score"] = home_games["home_score"]
            home_games["opp_score"] = home_games["away_score"]
            away_games["team_score"] = away_games["away_score"]
            away_games["opp_score"] = away_games["home_score"]
        else:
            raise ValueError(f"Unsupported period: {period}")

        home_games["is_home"] = 1
        away_games["is_home"] = 0

        all_games = pd.concat(
            [
                home_games[["date", "team_score", "opp_score", "is_home"]],
                away_games[["date", "team_score", "opp_score", "is_home"]],
            ]
        ).sort_values("date", ascending=False)

        # Drop rows with NaN scores
        all_games = all_games.dropna(subset=["team_score", "opp_score"])

        if len(all_games) < 3:
            return {}

        # Period suffix for feature names
        suffix = f"_{period}" if period != "fg" else ""

        # Last N games
        recent = all_games.head(self.lookback)

        stats = {}

        # Basic stats
        stats[f"ppg{suffix}"] = recent["team_score"].mean()
        stats[f"papg{suffix}"] = recent["opp_score"].mean()
        stats[f"margin{suffix}"] = (recent["team_score"] - recent["opp_score"]).mean()
        stats[f"win_pct{suffix}"] = (recent["team_score"] > recent["opp_score"]).mean()
        stats[f"pace{suffix}"] = recent["team_score"].mean() + recent["opp_score"].mean()

        # Volatility
        stats[f"margin_std{suffix}"] = (recent["team_score"] - recent["opp_score"]).std()

        # Last 5 performance
        last5 = recent.head(5)
        if len(last5) >= 3:
            stats[f"l5_margin{suffix}"] = (last5["team_score"] - last5["opp_score"]).mean()
            stats[f"l5_ppg{suffix}"] = last5["team_score"].mean()
            stats[f"l5_papg{suffix}"] = last5["opp_score"].mean()

        # Last 10 performance
        last10 = recent.head(10)
        if len(last10) >= 5:
            stats[f"l10_margin{suffix}"] = (last10["team_score"] - last10["opp_score"]).mean()

        # Period lead/win rate
        stats[f"lead_pct{suffix}"] = stats.get(f"win_pct{suffix}", 0.5)

        # Efficiency ratings (simple version for periods)
        ppg = stats.get(f"ppg{suffix}", 25)
        papg = stats.get(f"papg{suffix}", 25)
        if ppg > 0 and papg > 0:
            stats[f"ortg{suffix}"] = ppg / (ppg + papg) * 100
            stats[f"drtg{suffix}"] = papg / (ppg + papg) * 100
            stats[f"net_rtg{suffix}"] = stats[f"ortg{suffix}"] - stats[f"drtg{suffix}"]

        # Over rate (total > average)
        avg_total = stats[f"pace{suffix}"] if f"pace{suffix}" in stats else 50
        over_count = ((recent["team_score"] + recent["opp_score"]) > avg_total).sum()
        stats[f"over_pct{suffix}"] = over_count / len(recent) if len(recent) > 0 else 0.5

        return stats

    def build_all_period_features(
        self,
        game: pd.Series,
        historical_df: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Build features for ALL periods (Q1, 1H, FG) independently.

        This is the main entry point for building period-independent features.
        Each period's features are computed from that period's historical data.

        Args:
            game: Game row with team names and date
            historical_df: Historical games DataFrame (must have quarter data)

        Returns:
            Combined dictionary with features for all periods
        """
        home_team = game["home_team"]
        away_team = game["away_team"]
        game_date = pd.to_datetime(game["date"])

        features: Dict[str, float] = {}

        # Build features for each period independently
        for period in ["1h", "fg"]:
            home_period_stats = self.compute_period_rolling_stats(
                historical_df, home_team, game_date, period
            )
            away_period_stats = self.compute_period_rolling_stats(
                historical_df, away_team, game_date, period
            )

            if not home_period_stats or not away_period_stats:
                # Skip this period if insufficient data
                logger.debug(f"Insufficient {period} data for {home_team} vs {away_team}")
                continue

            suffix = f"_{period}" if period != "fg" else ""

            # Add home team stats
            for key, val in home_period_stats.items():
                features[f"home_{key}"] = val

            # Add away team stats
            for key, val in away_period_stats.items():
                features[f"away_{key}"] = val

            # Compute differentials
            ppg_key = f"ppg{suffix}"
            margin_key = f"margin{suffix}"
            pace_key = f"pace{suffix}"

            if ppg_key in home_period_stats and ppg_key in away_period_stats:
                features[f"ppg_diff{suffix}"] = (
                    home_period_stats[ppg_key] - away_period_stats[ppg_key]
                )
            if margin_key in home_period_stats and margin_key in away_period_stats:
                features[f"margin_diff{suffix}"] = (
                    home_period_stats[margin_key] - away_period_stats[margin_key]
                )
            if pace_key in home_period_stats and pace_key in away_period_stats:
                features[f"expected_pace{suffix}"] = (
                    home_period_stats[pace_key] + away_period_stats[pace_key]
                ) / 2

            # Compute period-specific predicted margin and total
            scaling = PERIOD_SCALING.get(period, PERIOD_SCALING["fg"])

            # Get dynamic HCA and scale for period
            base_hca = get_home_court_advantage(home_team)
            period_hca = base_hca * scaling["hca_factor"]
            features[f"dynamic_hca{suffix}"] = period_hca

            # Predicted margin for this period (using actual period margin stats)
            home_margin = home_period_stats[margin_key]
            away_margin = away_period_stats[margin_key]
            features[f"predicted_margin{suffix}"] = (
                (home_margin - away_margin) / 2 + period_hca
            )

            # Predicted total for this period using SOPHISTICATED EFFICIENCY MODEL
            # Match the methodology from rich_features.py for consistency
            papg_key = f"papg{suffix}"
            ppg_key = f"ppg{suffix}"

            # Get efficiency ratings (ORTG/DRTG equivalent for periods)
            home_ortg = home_period_stats[ppg_key] / max(home_period_stats.get(f"pace{suffix}", 50), 1) * 100 if home_period_stats.get(f"pace{suffix}", 0) > 0 else home_period_stats[ppg_key]
            away_ortg = away_period_stats[ppg_key] / max(away_period_stats.get(f"pace{suffix}", 50), 1) * 100 if away_period_stats.get(f"pace{suffix}", 0) > 0 else away_period_stats[ppg_key]
            home_drtg = home_period_stats[papg_key] / max(home_period_stats.get(f"pace{suffix}", 50), 1) * 100 if home_period_stats.get(f"pace{suffix}", 0) > 0 else home_period_stats[papg_key]
            away_drtg = away_period_stats[papg_key] / max(away_period_stats.get(f"pace{suffix}", 50), 1) * 100 if away_period_stats.get(f"pace{suffix}", 0) > 0 else away_period_stats[papg_key]

            # Expected pace factor (geometric mean for better outlier handling)
            home_pace = home_period_stats.get(f"pace{suffix}", 50)
            away_pace = away_period_stats.get(f"pace{suffix}", 50)
            expected_pace_factor = (home_pace * away_pace) ** 0.5 / 50 if home_pace > 0 and away_pace > 0 else 1.0

            # Home expected points = avg(home offense + away defense) × pace factor
            home_expected_pts = ((home_ortg + away_drtg) / 2) * expected_pace_factor
            away_expected_pts = ((away_ortg + home_drtg) / 2) * expected_pace_factor

            features[f"predicted_total{suffix}"] = home_expected_pts + away_expected_pts

        return features

    def compute_rest_days(
        self,
        games_df: pd.DataFrame,
        team: str,
        game_date: pd.Timestamp,
        default_rest: Optional[int] = None,
    ) -> int:
        """
        Days since last game (0 = back-to-back).

        Args:
            games_df: Historical games DataFrame
            team: Team name
            game_date: Date of current game
            default_rest: Rest days to return if no previous games found.
                          If None, raises ValueError instead.

        Returns:
            Number of rest days (0 = back-to-back)

        Notes:
            Large gaps across the NBA offseason are expected. We suppress the
            "unusually long rest" warning when the gap crosses an offseason,
            but we still return the true number of days since last game (no cap).
        """
        mask = ((games_df["home_team"] == team) | (games_df["away_team"] == team)) & (
            games_df["date"] < game_date
        )
        team_games = games_df[mask].sort_values("date", ascending=False)

        if len(team_games) == 0:
            if default_rest is None:
                raise ValueError(
                    f"No previous games found for {team} before {game_date}. "
                    f"Cannot compute rest days. Set default_rest parameter or ensure historical data exists."
                )
            logger.warning(
                f"No previous games for {team} before {game_date}. "
                f"Using default rest days: {default_rest}"
            )
            return default_rest

        last_game = team_games.iloc[0]["date"]

        rest_days = max(0, (game_date - last_game).days - 1)

        # Sanity check: extremely long rest is suspicious EXCEPT across offseason
        if rest_days > 30:
            if is_crossing_offseason(last_game, game_date):
                logger.debug(
                    f"Long rest for {team} crosses offseason: {rest_days} days "
                    f"(last game: {last_game}, current: {game_date})."
                )
            else:
                logger.warning(
                    f"Unusually long rest for {team}: {rest_days} days "
                    f"(last game: {last_game}, current: {game_date}). "
                    f"Possible data gap or incorrect date?"
                )

        return rest_days

    def get_previous_game_location(
        self,
        games_df: pd.DataFrame,
        team: str,
        game_date: pd.Timestamp,
    ) -> Tuple[Optional[str], int]:
        """
        Get the location (arena city) of team's previous game.
        
        Returns:
            Tuple of (previous_location_team, rest_days)
            - previous_location_team: The team whose arena the previous game was at
            - rest_days: Days since that game
        """
        mask = ((games_df["home_team"] == team) | (games_df["away_team"] == team)) & (
            games_df["date"] < game_date
        )
        team_games = games_df[mask].sort_values("date", ascending=False)

        if len(team_games) == 0:
            return None, 3

        last_game = team_games.iloc[0]
        last_game_date = last_game["date"]
        rest_days = max(0, (game_date - last_game_date).days - 1)
        
        # The location is the home team's arena
        previous_location = last_game["home_team"]
        
        return previous_location, rest_days

    def compute_travel_features(
        self,
        games_df: pd.DataFrame,
        team: str,
        opponent: str,
        game_date: pd.Timestamp,
        is_home: bool,
    ) -> Dict[str, float]:
        """
        Compute travel-related features for a team.
        
        Args:
            games_df: Historical games DataFrame
            team: The team we're computing features for
            opponent: The opponent team
            game_date: Date of the current game
            is_home: Whether this team is playing at home
            
        Returns:
            Dict of travel features
        """
        features = {
            "travel_distance": 0.0,
            "timezone_change": 0,
            "travel_fatigue": 0.0,
            "is_long_trip": 0,
            "is_cross_country": 0,
        }
        
        # Get previous game location
        prev_location, rest_days = self.get_previous_game_location(
            games_df, team, game_date
        )
        
        if prev_location is None:
            return features
        
        # Current game location (opponent's arena if away, own arena if home)
        current_location = team if is_home else opponent
        
        # Calculate travel from previous location to current location
        distance = get_travel_distance(prev_location, current_location)
        
        if distance is not None:
            features["travel_distance"] = distance
            features["is_long_trip"] = 1 if distance >= 1500 else 0
            features["is_cross_country"] = 1 if distance >= 2500 else 0
            
            tz_change = get_timezone_difference(prev_location, current_location)
            features["timezone_change"] = tz_change
            
            is_b2b = rest_days <= 1
            features["travel_fatigue"] = calculate_travel_fatigue(
                distance_miles=distance,
                rest_days=rest_days,
                timezone_change=tz_change,
                is_back_to_back=is_b2b,
            )
        
        return features

    def compute_dynamic_hca(
        self,
        games_df: pd.DataFrame,
        home_team: str,
        away_team: str,
        game_date: pd.Timestamp,
        home_rest: int,
    ) -> float:
        """
        Calculate dynamic home court advantage.

        Adjusts base HCA for situational factors.
        """
        # Get base team-specific HCA
        base_hca = get_home_court_advantage(home_team)

        # Adjust for back-to-back
        if home_rest == 0:
            base_hca -= 1.5  # Tired home team

        # Adjust for long rest (3+ days)
        if home_rest >= 3:
            base_hca += 0.5  # Well-rested home team

        # Season phase adjustment (HCA is weaker early in season)
        # Count games played by home team
        home_games_played = len(games_df[
            ((games_df["home_team"] == home_team) | (games_df["away_team"] == home_team)) &
            (games_df["date"] < game_date)
        ])

        if home_games_played < 10:
            base_hca *= 0.7  # Reduce HCA for first 10 games
        elif home_games_played < 20:
            base_hca *= 0.85  # Moderate reduction for games 10-20

        return base_hca

    def compute_h2h_stats(
        self,
        games_df: pd.DataFrame,
        home_team: str,
        away_team: str,
        as_of_date: pd.Timestamp,
        n_games: int = 5,
    ) -> Dict[str, float]:
        """Head-to-head history."""
        mask = (
            (
                (games_df["home_team"] == home_team)
                & (games_df["away_team"] == away_team)
            )
            | (
                (games_df["home_team"] == away_team)
                & (games_df["away_team"] == home_team)
            )
        ) & (games_df["date"] < as_of_date)
        h2h = games_df[mask].sort_values("date", ascending=False).head(n_games)

        if len(h2h) == 0:
            return {"h2h_games": 0, "h2h_margin": 0}

        # From home team's perspective
        margins = []
        for _, g in h2h.iterrows():
            if g["home_team"] == home_team:
                margins.append(g["home_score"] - g["away_score"])
            else:
                margins.append(g["away_score"] - g["home_score"])

        return {
            "h2h_games": len(h2h),
            "h2h_margin": np.mean(margins),
        }

    def compute_enhanced_h2h(
        self,
        games_df: pd.DataFrame,
        home_team: str,
        away_team: str,
        as_of_date: pd.Timestamp,
        n_games: int = 10,
    ) -> Dict[str, float]:
        """
        Enhanced head-to-head features with more history.
        
        Returns additional H2H metrics:
        - h2h_home_win_pct: Home team's win rate vs away team when playing at home
        - h2h_recent_margin: Last 3 H2H games margin
        - h2h_total_avg: Average total points in H2H games
        - h2h_cover_pct: Approximate ATS record in H2H
        """
        mask = (
            (
                (games_df["home_team"] == home_team)
                & (games_df["away_team"] == away_team)
            )
            | (
                (games_df["home_team"] == away_team)
                & (games_df["away_team"] == home_team)
            )
        ) & (games_df["date"] < as_of_date)
        h2h = games_df[mask].sort_values("date", ascending=False).head(n_games)

        if len(h2h) == 0:
            return {
                "h2h_games": 0,
                "h2h_margin": 0,
                "h2h_home_win_pct": 0.5,
                "h2h_recent_margin": 0,
                "h2h_total_avg": 220,
                "h2h_cover_pct": 0.5,
            }

        # Compute features from home team's perspective
        margins = []
        totals = []
        home_wins = 0
        home_at_home_wins = 0
        home_at_home_games = 0
        
        for _, g in h2h.iterrows():
            total = g["home_score"] + g["away_score"]
            totals.append(total)
            
            if g["home_team"] == home_team:
                margin = g["home_score"] - g["away_score"]
                margins.append(margin)
                home_at_home_games += 1
                if margin > 0:
                    home_wins += 1
                    home_at_home_wins += 1
            else:
                margin = g["away_score"] - g["home_score"]
                margins.append(margin)
                if margin > 0:
                    home_wins += 1

        # Recent margin (last 3 games)
        recent_margin = np.mean(margins[:3]) if len(margins) >= 3 else np.mean(margins)
        
        # Home team's H2H win rate when playing at home
        h2h_home_win_pct = home_at_home_wins / home_at_home_games if home_at_home_games > 0 else 0.5

        return {
            "h2h_games": len(h2h),
            "h2h_margin": np.mean(margins),
            "h2h_home_win_pct": h2h_home_win_pct,
            "h2h_recent_margin": recent_margin,
            "h2h_total_avg": np.mean(totals),
            "h2h_cover_pct": home_wins / len(h2h) if len(h2h) > 0 else 0.5,
        }

    def compute_sos_features(
        self,
        team: str,
        games_df: pd.DataFrame,
        as_of_date: pd.Timestamp,
        n_recent: int = 10,
    ) -> Dict[str, float]:
        """
        Compute strength of schedule features.
        
        Args:
            team: Team name
            games_df: Historical games DataFrame
            as_of_date: Compute as of this date
            n_recent: Number of recent games for recent SOS
        
        Returns:
            Dictionary with SOS features:
            - sos_rating: Average opponent win percentage
            - recent_sos: SOS of last N games
            - opp_avg_margin: Average opponent margin
            - opp_avg_ppg: Average opponent PPG
        """
        # Get all completed games for this team before the date
        mask = (
            ((games_df["home_team"] == team) | (games_df["away_team"] == team))
            & (games_df["date"] < as_of_date)
        )
        team_games = games_df[mask].sort_values("date", ascending=False)
        
        if len(team_games) < 5:
            return {
                "sos_rating": 0.5,
                "recent_sos": 0.5,
                "opp_avg_margin": 0,
                "opp_avg_ppg": 110,
            }
        
        # Collect opponent stats
        opp_win_pcts = []
        opp_margins = []
        opp_ppgs = []
        
        for _, game in team_games.iterrows():
            # Determine opponent
            if game["home_team"] == team:
                opponent = game["away_team"]
            else:
                opponent = game["home_team"]
            
            # Get opponent's record before this game
            opp_games_before = games_df[
                ((games_df["home_team"] == opponent) | (games_df["away_team"] == opponent))
                & (games_df["date"] < game["date"])
            ]
            
            if len(opp_games_before) >= 3:
                # Calculate opponent's win percentage
                opp_wins = 0
                opp_margin_sum = 0
                opp_ppg_sum = 0
                
                for _, og in opp_games_before.iterrows():
                    if og["home_team"] == opponent:
                        opp_score = og["home_score"]
                        other_score = og["away_score"]
                    else:
                        opp_score = og["away_score"]
                        other_score = og["home_score"]
                    
                    if opp_score > other_score:
                        opp_wins += 1
                    opp_margin_sum += opp_score - other_score
                    opp_ppg_sum += opp_score
                
                opp_win_pcts.append(opp_wins / len(opp_games_before))
                opp_margins.append(opp_margin_sum / len(opp_games_before))
                opp_ppgs.append(opp_ppg_sum / len(opp_games_before))
        
        if not opp_win_pcts:
            return {
                "sos_rating": 0.5,
                "recent_sos": 0.5,
                "opp_avg_margin": 0,
                "opp_avg_ppg": 110,
            }
        
        # Calculate SOS metrics
        sos_rating = np.mean(opp_win_pcts)
        recent_sos = np.mean(opp_win_pcts[:n_recent]) if len(opp_win_pcts) >= n_recent else sos_rating
        
        return {
            "sos_rating": sos_rating,
            "recent_sos": recent_sos,
            "opp_avg_margin": np.mean(opp_margins),
            "opp_avg_ppg": np.mean(opp_ppgs),
        }

    def build_game_features(
        self,
        game: pd.Series,
        historical_df: pd.DataFrame,
    ) -> Dict[str, float]:
        """Build all features for a game prediction."""
        home_team = game["home_team"]
        away_team = game["away_team"]
        game_date = pd.to_datetime(game["date"])

        features: Dict[str, float] = {}

        # Team rolling stats
        home_stats = self.compute_team_rolling_stats(
            historical_df, home_team, game_date
        )
        away_stats = self.compute_team_rolling_stats(
            historical_df, away_team, game_date
        )

        if not home_stats or not away_stats:
            return {}

        # Home team features
        for key, val in home_stats.items():
            features[f"home_{key}"] = val

        # Away team features
        for key, val in away_stats.items():
            features[f"away_{key}"] = val

        # Differential features
        features["ppg_diff"] = home_stats["ppg"] - away_stats["ppg"]
        features["papg_diff"] = home_stats["papg"] - away_stats["papg"]
        features["margin_diff"] = home_stats["margin"] - away_stats["margin"]
        features["win_pct_diff"] = home_stats["win_pct"] - away_stats["win_pct"]
        features["pace_diff"] = home_stats["pace"] - away_stats["pace"]

        # === NEW: Advanced differential features ===
        features["clutch_diff"] = home_stats.get("clutch_win_pct", 0.5) - away_stats.get("clutch_win_pct", 0.5)
        features["net_rating_diff"] = home_stats.get("net_rating", 0) - away_stats.get("net_rating", 0)
        features["consistency_diff"] = home_stats.get("consistency", 0.5) - away_stats.get("consistency", 0.5)
        features["win_streak_diff"] = home_stats.get("win_streak", 0) - away_stats.get("win_streak", 0)

        # Adjusted ratings
        features["adj_offensive_diff"] = home_stats.get("adj_offensive_rating", 110) - away_stats.get("adj_offensive_rating", 110)
        features["adj_defensive_diff"] = away_stats.get("adj_defensive_rating", 110) - home_stats.get("adj_defensive_rating", 110)  # Lower is better for defense

        # Rest days
        # Use default_rest=3 for production safety (early season games may lack history)
        home_rest = self.compute_rest_days(historical_df, home_team, game_date, default_rest=3)
        away_rest = self.compute_rest_days(historical_df, away_team, game_date, default_rest=3)
        features["home_rest"] = home_rest
        features["away_rest"] = away_rest
        features["rest_diff"] = home_rest - away_rest
        features["home_b2b"] = 1 if home_rest == 0 else 0
        features["away_b2b"] = 1 if away_rest == 0 else 0

        # Dynamic Home Court Advantage
        # Base HCA is ~3.0 points, but adjust for context
        base_hca = 3.0

        # Adjust for rest advantage/disadvantage
        if home_rest == 0:  # Home team on back-to-back
            base_hca -= 1.5  # Tired home team, reduced HCA
        elif home_rest >= 3 and away_rest <= 1:  # Home rested, away tired
            base_hca += 0.5  # Extra advantage

        # Adjust for rest differential (home team advantage)
        if features["rest_diff"] >= 2:  # Home has 2+ days more rest
            base_hca += 0.5
        elif features["rest_diff"] <= -2:  # Away has 2+ days more rest
            base_hca -= 0.5

        features["dynamic_hca"] = base_hca

        # ============================================================
        # TRAVEL/FATIGUE FEATURES
        # ============================================================
        # Away team travel (from previous game location to this game)
        away_travel = self.compute_travel_features(
            historical_df, away_team, home_team, game_date, is_home=False
        )
        features["away_travel_distance"] = away_travel["travel_distance"]
        features["away_timezone_change"] = away_travel["timezone_change"]
        features["away_travel_fatigue"] = away_travel["travel_fatigue"]
        features["is_away_long_trip"] = away_travel["is_long_trip"]
        features["is_away_cross_country"] = away_travel["is_cross_country"]
        
        # B2B + travel interaction (compounding penalty)
        features["away_b2b_travel_penalty"] = 0.0
        if features["away_b2b"] == 1 and away_travel["travel_distance"] >= 1500:
            features["away_b2b_travel_penalty"] = -1.5
        
        # Travel advantage (away fatigue helps home team)
        features["travel_advantage"] = -away_travel["travel_fatigue"]

        # === UPDATED: Dynamic home court advantage ===
        hca = self.compute_dynamic_hca(
            historical_df, home_team, away_team, game_date, home_rest
        )
        features["home_court_advantage"] = hca

        # Head-to-head (basic)
        h2h = self.compute_h2h_stats(historical_df, home_team, away_team, game_date)
        features.update(h2h)

        # Enhanced Head-to-head
        enhanced_h2h = self.compute_enhanced_h2h(historical_df, home_team, away_team, game_date)
        features.update(enhanced_h2h)

        # Strength of Schedule
        home_sos = self.compute_sos_features(home_team, historical_df, game_date)
        away_sos = self.compute_sos_features(away_team, historical_df, game_date)
        features["home_sos_rating"] = home_sos["sos_rating"]
        features["away_sos_rating"] = away_sos["sos_rating"]
        features["sos_diff"] = home_sos["sos_rating"] - away_sos["sos_rating"]
        features["home_recent_sos"] = home_sos["recent_sos"]
        features["away_recent_sos"] = away_sos["recent_sos"]
        features["home_opp_avg_margin"] = home_sos["opp_avg_margin"]
        features["away_opp_avg_margin"] = away_sos["opp_avg_margin"]


        # === Enhanced predicted margin ===
        # Removed explicit rest term - rest is already accounted for in:
        # 1. compute_dynamic_hca() adjusts HCA for home B2B and rest differential
        # 2. calculate_travel_fatigue() mitigates fatigue when rest >= 2 days
        features["predicted_margin"] = (
            (home_stats["margin"] - away_stats["margin"]) / 2
            + hca  # Dynamic home court advantage (already includes rest adjustments)
            - away_travel["travel_fatigue"]  # Away team fatigue (already has rest mitigation)
            + features.get("clutch_diff", 0) * 0.3  # Clutch factor
            + features.get("net_rating_diff", 0) * 0.2  # Net rating
        )

        # Predicted total using matchup-based formula
        # Home expected points = average of (home's offense, away's defense allowed)
        # Away expected points = average of (away's offense, home's defense allowed)
        # This properly accounts for the matchup instead of just averaging pace
        home_expected_pts = (home_stats["ppg"] + away_stats["papg"]) / 2
        away_expected_pts = (away_stats["ppg"] + home_stats["papg"]) / 2
        features["predicted_total"] = home_expected_pts + away_expected_pts

        # === FIRST HALF PREDICTIONS (v33.0.7.0: Independent 1H Stats) ===
        # Compute 1H-specific stats using actual historical 1H data
        home_1h_stats = self.compute_period_rolling_stats(
            historical_df, home_team, game_date, period="1h"
        )
        away_1h_stats = self.compute_period_rolling_stats(
            historical_df, away_team, game_date, period="1h"
        )

        # Scale HCA for 1H (~1.5 pts vs 3 pts for FG)
        hca_1h = hca * 0.5

        # v33.1.0: Graceful degradation - skip 1H features if data unavailable
        # This allows FG markets to work even if 1H data is missing
        if not home_1h_stats:
            logger.warning(
                f"Insufficient 1H historical data for {home_team}. "
                f"Skipping 1H predictions (FG predictions will still work). "
                f"Team needs at least 3 games with Q1/Q2 data for 1H markets."
            )
            # Return features WITHOUT 1H predictions (FG predictions still work)
            return features
        if not away_1h_stats:
            logger.warning(
                f"Insufficient 1H historical data for {away_team}. "
                f"Skipping 1H predictions (FG predictions will still work). "
                f"Team needs at least 3 games with Q1/Q2 data for 1H markets."
            )
            # Return features WITHOUT 1H predictions (FG predictions still work)
            return features

        # 1H Margin: Use actual 1H margin stats (not scaled FG)
        home_1h_margin = home_1h_stats["margin_1h"]
        away_1h_margin = away_1h_stats["margin_1h"]
        features["predicted_margin_1h"] = (
            (home_1h_margin - away_1h_margin) / 2
            + hca_1h
            - away_travel["travel_fatigue"] * 0.5
        )

        # 1H Total: Matchup-based formula (same logic as FG)
        # Home 1H expected = avg of (home's 1H offense, away's 1H defense allowed)
        # Away 1H expected = avg of (away's 1H offense, home's 1H defense allowed)
        home_1h_ppg = home_1h_stats["ppg_1h"]
        home_1h_papg = home_1h_stats["papg_1h"]
        away_1h_ppg = away_1h_stats["ppg_1h"]
        away_1h_papg = away_1h_stats["papg_1h"]

        home_1h_expected = (home_1h_ppg + away_1h_papg) / 2
        away_1h_expected = (away_1h_ppg + home_1h_papg) / 2
        features["predicted_total_1h"] = home_1h_expected + away_1h_expected

        # Store 1H stats for model features
        features["home_ppg_1h"] = home_1h_ppg
        features["home_papg_1h"] = home_1h_papg
        features["away_ppg_1h"] = away_1h_ppg
        features["away_papg_1h"] = away_1h_papg
        features["home_margin_1h"] = home_1h_margin
        features["away_margin_1h"] = away_1h_margin

        # Form features
        if "form_trend" in home_stats and "form_trend" in away_stats:
            features["home_form_trend"] = home_stats["form_trend"]
            features["away_form_trend"] = away_stats["form_trend"]
            features["form_diff"] = (
                home_stats.get("form_3g", 0) - away_stats.get("form_3g", 0)
            )

        # *** LINE AS FEATURE - ALL PERIODS ***
        # Full Game lines
        if "spread_line" in game and pd.notna(game["spread_line"]):
            spread_line = game["spread_line"]
            features["spread_line"] = spread_line
            features["fg_spread_line"] = spread_line  # Alias for consistency
            features["spread_vs_predicted"] = (
                features["predicted_margin"] - (-spread_line)
            )

        if "total_line" in game and pd.notna(game["total_line"]):
            total_line = game["total_line"]
            features["total_line"] = total_line
            features["fg_total_line"] = total_line  # Alias for consistency
            features["total_vs_predicted"] = (
                features["predicted_total"] - total_line
            )

        # First Half lines - PREMIUM API DATA
        if "fh_spread_line" in game and pd.notna(game["fh_spread_line"]):
            fh_spread = game["fh_spread_line"]
            features["fh_spread_line"] = fh_spread
            features["1h_spread_line"] = fh_spread  # Alias
            if "predicted_margin_1h" in features:
                features["fh_spread_vs_predicted"] = features["predicted_margin_1h"] - (-fh_spread)

        if "fh_total_line" in game and pd.notna(game["fh_total_line"]):
            fh_total = game["fh_total_line"]
            features["fh_total_line"] = fh_total
            features["1h_total_line"] = fh_total  # Alias
            if "predicted_total_1h" in features:
                features["fh_total_vs_predicted"] = features["predicted_total_1h"] - fh_total

        # *** INJURY IMPACT FEATURES ***
        # Injury features - has_injury_data indicates whether we have real API data
        features["has_injury_data"] = game.get("has_injury_data", 0)
        features["home_injury_spread_impact"] = game.get(
            "home_injury_spread_impact", 0
        )
        features["away_injury_spread_impact"] = game.get(
            "away_injury_spread_impact", 0
        )
        features["injury_spread_diff"] = (
            features["home_injury_spread_impact"]
            - features["away_injury_spread_impact"]
        )
        features["home_star_out"] = game.get("home_star_out", 0)
        features["away_star_out"] = game.get("away_star_out", 0)

        # *** RLM / SHARP MONEY FEATURES ***
        # IMPORTANT: has_real_splits indicates whether we have REAL betting splits data
        # If 0, the splits features are defaults/missing and should be weighted accordingly
        features["has_real_splits"] = game.get("has_real_splits", 0)
        features["is_rlm_spread"] = game.get("is_rlm_spread", 0)
        features["sharp_side_spread"] = game.get("sharp_side_spread", 0)
        # NOTE: Default 50 means "no data" - models should check has_real_splits
        features["spread_public_home_pct"] = game.get("spread_public_home_pct", 50)
        features["spread_ticket_money_diff"] = game.get(
            "spread_ticket_money_diff", 0
        )
        features["spread_movement"] = game.get("spread_movement", 0)

        # Totals RLM
        features["is_rlm_total"] = game.get("is_rlm_total", 0)
        features["sharp_side_total"] = game.get("sharp_side_total", 0)

        return features

    def build_features_dataframe(
        self,
        games_df: pd.DataFrame,
        historical_df: pd.DataFrame,
        odds_df: pd.DataFrame = None,  # Keep signature consistent with call
    ) -> pd.DataFrame:
        """Build features for a dataframe of games."""
        games_df = games_df.copy()
        if "game_date" in games_df.columns and "date" not in games_df.columns:
            games_df.rename(columns={"game_date": "date"}, inplace=True)

        all_features = []
        for _, game in games_df.iterrows():
            features = self.build_game_features(game, historical_df)
            if features:
                # Add identifiers for joining
                features["home_team"] = game["home_team"]
                features["away_team"] = game["away_team"]
                features["game_date"] = game["date"]
                all_features.append(features)

        if not all_features:
            return pd.DataFrame()

        return pd.DataFrame(all_features)
