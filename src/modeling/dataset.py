"""
Dataset builder for NBA prediction models.

Links odds data with game outcomes to create training datasets.
"""
from __future__ import annotations
import os
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

from src.config import settings
from src.modeling.features import FeatureEngineer
from src.utils.team_names import normalize_team_name


class DatasetBuilder:
    """
    Build training datasets by linking odds to outcomes.

    Creates labeled datasets for:
    - Spreads: Did home team cover the spread?
    - Totals: Did the game go over the total?
    """

    # Team name normalization is now provided by single source:
    # src.utils.team_names.normalize_team_name (line 63)
    #
    # This ensures consistent handling of team names across all modules.

    def __init__(self, feature_engineer: Optional[FeatureEngineer] = None):
        self.feature_engineer = feature_engineer or FeatureEngineer()

    def load_odds_data(self, path: Optional[str] = None) -> pd.DataFrame:
        """Load and normalize odds data."""
        path = path or os.path.join(settings.data_processed_dir, "odds_the_odds.csv")
        if not os.path.exists(path):
            return pd.DataFrame()

        df = pd.read_csv(path)
        df["home_team"] = df["home_team"].apply(normalize_team_name)
        df["away_team"] = df["away_team"].apply(normalize_team_name)
        df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
        return df

    def load_outcomes_data(self, path: Optional[str] = None) -> pd.DataFrame:
        """Load game outcomes data."""
        # Try multiple sources
        paths_to_try = [
            path,
            os.path.join(settings.data_processed_dir, "game_outcomes.csv"),
            os.path.join(settings.data_processed_dir, "historical_games.csv"),
        ]

        for p in paths_to_try:
            if p and os.path.exists(p):
                df = pd.read_csv(p)
                df["home_team"] = df["home_team"].apply(normalize_team_name)
                df["away_team"] = df["away_team"].apply(normalize_team_name)
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                return df

        return pd.DataFrame()

    def _get_consensus_line(
        self,
        odds_df: pd.DataFrame,
        event_id: str,
        market: str,
        participant: str,
    ) -> Optional[float]:
        """Get consensus line for a market/participant."""
        filtered = odds_df[
            (odds_df["event_id"] == event_id) &
            (odds_df["market"] == market) &
            (odds_df["participant"] == participant)
        ]
        lines = filtered["line"].dropna()
        if len(lines) == 0:
            return None
        return lines.mean()

    def link_odds_to_outcomes(
        self,
        odds_df: pd.DataFrame,
        outcomes_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Link odds data to game outcomes.

        Matches on team names and date (within 1 day tolerance).
        """
        linked_rows = []

        # Get unique events from odds
        events = odds_df.groupby("event_id").first().reset_index()

        for _, event in events.iterrows():
            event_id = event["event_id"]
            home_team = event["home_team"]
            away_team = event["away_team"]
            start_time = pd.to_datetime(event["start_time"])

            if pd.isna(start_time):
                continue

            # Find matching outcome (within 1 day)
            matching_outcomes = outcomes_df[
                (outcomes_df["home_team"] == home_team) &
                (outcomes_df["away_team"] == away_team) &
                (abs((outcomes_df["date"] - start_time).dt.total_seconds()) < 86400 * 2)
            ]

            if len(matching_outcomes) == 0:
                continue

            outcome = matching_outcomes.iloc[0]

            # Get consensus lines
            spread_line = self._get_consensus_line(odds_df, event_id, "spreads", home_team)
            total_line = self._get_consensus_line(odds_df, event_id, "totals", "Over")

            # Calculate targets
            home_margin = outcome["home_margin"]
            total_score = outcome["total_score"]

            linked_rows.append({
                "event_id": event_id,
                "game_date": start_time,
                "home_team": home_team,
                "away_team": away_team,
                "home_score": outcome["home_score"],
                "away_score": outcome["away_score"],
                "home_margin": home_margin,
                "total_score": total_score,
                # Spread data
                "spread_line": spread_line,
                "spread_covered": (
                    1 if spread_line is not None and home_margin > -spread_line else 0
                ),
                "spread_push": (
                    1 if spread_line is not None and home_margin == -spread_line else 0
                ),
                # Totals data
                "total_line": total_line,
                "went_over": (
                    1 if total_line is not None and total_score > total_line else 0
                ),
                "total_push": (
                    1 if total_line is not None and total_score == total_line else 0
                ),
            })

        return pd.DataFrame(linked_rows)

    def build_training_dataset(
        self,
        odds_path: Optional[str] = None,
        outcomes_path: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Build complete training dataset with features and labels.

        Returns DataFrame ready for model training.
        """
        # Load data
        odds_df = self.load_odds_data(odds_path)
        outcomes_df = self.load_outcomes_data(outcomes_path)

        if odds_df.empty or outcomes_df.empty:
            print("Warning: Missing odds or outcomes data")
            return pd.DataFrame()

        # Link odds to outcomes
        linked_df = self.link_odds_to_outcomes(odds_df, outcomes_df)

        if linked_df.empty:
            print("Warning: No games could be linked between odds and outcomes")
            return pd.DataFrame()

        # Build features for each game (team form, rest, head-to-head, line features)
        features_df = self.feature_engineer.build_features_dataframe(
            linked_df,
            outcomes_df,
            odds_df,
        )

        # Merge features with targets
        training_df = pd.concat(
            [
                linked_df.reset_index(drop=True),
                features_df.drop(
                    columns=["home_team", "away_team", "game_date"], errors="ignore"
                ).reset_index(drop=True),
            ],
            axis=1,
        )

        # Ensure we always have a canonical date column for downstream consumers
        if "date" not in training_df.columns and "game_date" in training_df.columns:
            training_df["date"] = pd.to_datetime(training_df["game_date"], errors="coerce")

        # Optionally enrich with betting splits / RLM features if available
        splits_path = os.path.join(settings.data_processed_dir, "betting_splits.csv")
        if os.path.exists(splits_path):
            try:
                splits_df = pd.read_csv(splits_path)
                if "event_id" in splits_df.columns:
                    # Map string sharp side to numeric indicator to align with feature_config
                    mapping_spread = {"home": 1, "away": -1}
                    mapping_total = {"over": 1, "under": -1}

                    splits_features = splits_df[
                        [
                            "event_id",
                            "is_rlm_spread",
                            "is_rlm_total",
                            "sharp_spread_side",
                            "sharp_total_side",
                        ]
                    ].copy()

                    splits_features["sharp_side_spread"] = (
                        splits_features["sharp_spread_side"]
                        .map(mapping_spread)
                        .fillna(0)
                        .astype(int)
                    )
                    splits_features["sharp_side_total"] = (
                        splits_features["sharp_total_side"]
                        .map(mapping_total)
                        .fillna(0)
                        .astype(int)
                    )

                    splits_features = splits_features.drop(
                        columns=["sharp_spread_side", "sharp_total_side"], errors="ignore"
                    )

                    training_df = training_df.merge(
                        splits_features, on="event_id", how="left"
                    )

                    for col in ["is_rlm_spread", "is_rlm_total", "sharp_side_spread", "sharp_side_total"]:
                        if col in training_df.columns:
                            training_df[col] = training_df[col].fillna(0)
            except Exception:
                # If anything goes wrong, continue without betting splits enrichment
                pass

        # Save if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            training_df.to_csv(output_path, index=False)
            print(f"Training dataset saved to {output_path}")

        return training_df

    def split_train_test(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        date_column: str = "game_date",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into train/test by date (temporal split).

        Uses most recent games for testing to simulate real prediction.
        """
        df = df.sort_values(date_column)
        split_idx = int(len(df) * (1 - test_size))
        return df.iloc[:split_idx], df.iloc[split_idx:]

    def get_feature_columns(self, target: str = "spreads") -> List[str]:
        """Get list of feature columns for a prediction target."""
        base_features = [
            "home_ppg", "home_papg", "home_total_ppg", "home_win_pct", "home_avg_margin",
            "away_ppg", "away_papg", "away_total_ppg", "away_win_pct", "away_avg_margin",
            "home_rest_days", "away_rest_days", "rest_advantage",
            "h2h_win_pct", "h2h_avg_margin",
            "win_pct_diff", "ppg_diff",
        ]

        if target == "spreads":
            base_features.extend([
                "predicted_margin",
                "spreads_consensus_line",
                "spreads_line_std",
            ])
        elif target == "totals":
            base_features.extend([
                "predicted_total",
                "totals_consensus_line",
                "totals_line_std",
            ])

        return base_features
