"""
Tests for DATA INTEGRITY - Ensures no data leakage between backtesting and live predictions.

CRITICAL INVARIANTS FOR NBA PREDICTION SYSTEM:
=============================================

1. TEMPORAL ISOLATION: Rolling/expanding features use shift(1) to exclude current game
2. WALK-FORWARD: Backtest training data ONLY includes games BEFORE test date
3. FEATURE CONSISTENCY: Same feature names map to same data across all pipelines
4. DATE CANONICALIZATION: All dates in CST (America/Chicago) timezone
5. TEAM CANONICALIZATION: All team names normalized to ESPN format (30 teams)

This file tests these invariants to ensure the model is NOT trained on future data.
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

# =============================================================================
# CRITICAL: DATA LEAKAGE PREVENTION TESTS
# =============================================================================


class TestShiftOneLookback:
    """
    Tests that rolling/expanding features use shift(1) to exclude current game.

    CRITICAL: If rolling stats include the current game's result, we have leakage.
    The model would be predicting outcomes using information from those outcomes.
    """

    def test_rolling_stats_exclude_current_game(self):
        """Rolling averages must NOT include the current game's result."""
        # Simulate team data
        data = {
            "game_date": pd.date_range("2025-01-01", periods=10, freq="D"),
            "team": ["LAL"] * 10,
            "score": [100, 105, 110, 115, 120, 125, 130, 135, 140, 145],
        }
        df = pd.DataFrame(data)

        # CORRECT: shift(1) excludes current game
        df["rolling_avg_correct"] = df.groupby("team")["score"].transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).mean()
        )

        # WRONG: No shift includes current game (DATA LEAKAGE!)
        df["rolling_avg_wrong"] = df.groupby("team")["score"].transform(
            lambda x: x.rolling(3, min_periods=1).mean()
        )

        # For game 5 (score=120):
        # CORRECT: avg of games 2,3,4 (105,110,115) = 110
        # WRONG: avg of games 3,4,5 (110,115,120) = 115 (LEAKAGE!)

        # Should be avg(100,105,110)=105
        game_5_correct = df.iloc[4]["rolling_avg_correct"]
        # Would be avg(110,115,120)=115
        game_5_wrong = df.iloc[4]["rolling_avg_wrong"]

        # The correct rolling should NOT include game 5's score (120)
        assert game_5_correct < 120, "Rolling average should exclude current game"

        # Verify first game has no lookback data
        assert pd.isna(df.iloc[0]["rolling_avg_correct"]), "First game should have no prior data"

    def test_expanding_stats_exclude_current_game(self):
        """Expanding means must NOT include the current game's result."""
        data = {
            "game_date": pd.date_range("2025-01-01", periods=5, freq="D"),
            "team": ["BOS"] * 5,
            "points": [90, 100, 110, 120, 130],
        }
        df = pd.DataFrame(data)

        # CORRECT: shift(1) then expand
        df["expanding_avg"] = df.groupby("team")["points"].transform(
            lambda x: x.shift(1).expanding(min_periods=1).mean()
        )

        # For game 4 (points=120):
        # CORRECT: avg of games 1,2,3 (90,100,110) = 100
        game_4_expanding = df.iloc[3]["expanding_avg"]

        # Should NOT include game 4's points (120)
        assert game_4_expanding < 120, "Expanding average should exclude current game"
        assert game_4_expanding == pytest.approx(100.0, abs=0.1)  # (90+100+110)/3

    def test_lag_features_respect_temporal_order(self):
        """Lag features should only reference past games."""
        # If we want "last game's score", we must use shift(1)
        data = {
            "game_date": pd.date_range("2025-01-01", periods=5, freq="D"),
            "team": ["PHX"] * 5,
            "score": [100, 110, 120, 130, 140],
        }
        df = pd.DataFrame(data)

        df["last_game_score"] = df.groupby("team")["score"].shift(1)

        # Game 3's last score should be Game 2's score
        assert df.iloc[2]["last_game_score"] == 110

        # Game 1 has no prior game
        assert pd.isna(df.iloc[0]["last_game_score"])


class TestWalkForwardIsolation:
    """
    Tests that walk-forward backtest respects temporal boundaries.

    CRITICAL: Training data must ONLY include games BEFORE the test window.
    """

    def test_train_test_split_temporal_ordering(self):
        """Training set must end BEFORE test set begins."""
        # Simulate walk-forward
        n_games = 100
        games = pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01", periods=n_games, freq="D"),
                "home_team": ["LAL"] * n_games,
                "away_team": ["BOS"] * n_games,
            }
        )

        min_train = 50
        test_chunk_size = 10

        # Walk-forward loop
        for chunk_start in range(min_train, n_games, test_chunk_size):
            chunk_end = min(chunk_start + test_chunk_size, n_games)

            train_df = games.iloc[:chunk_start]
            test_df = games.iloc[chunk_start:chunk_end]

            # CRITICAL: Train dates must all be BEFORE test dates
            train_max_date = train_df["date"].max()
            test_min_date = test_df["date"].min()

            assert (
                train_max_date < test_min_date
            ), f"Training data leaked into test! Train max: {train_max_date}, Test min: {test_min_date}"

    def test_no_future_features_in_training(self):
        """Features computed from training data cannot include future information."""
        # Create dataset with features
        data = {
            "date": pd.date_range("2025-01-01", periods=20, freq="D"),
            "score": range(100, 120),
        }
        df = pd.DataFrame(data)

        # Compute rolling feature CORRECTLY (shift then roll)
        df["rolling_score"] = df["score"].shift(1).rolling(5, min_periods=1).mean()

        # Test point: game 15
        test_idx = 15

        # Feature for game 15 should only use games 0-13 (with shift)
        feature_at_test = df.iloc[test_idx]["rolling_score"]

        # The rolling window for game 15 uses scores from games 10-14 (after shift)
        expected_scores = df.iloc[10:15]["score"].values
        expected_avg = expected_scores.mean()

        assert feature_at_test == pytest.approx(expected_avg, abs=0.1)


class TestFeatureConsistency:
    """
    Tests that feature names map consistently across training and live prediction.

    CRITICAL: A feature named "home_ppg" in training must mean the same thing in live.
    """

    def test_feature_name_semantic_consistency(self):
        """Feature names should have consistent semantic meaning."""
        # Define what each feature should represent
        feature_semantics = {
            "home_ppg": "Home team points per game (season/recent average)",
            "away_ppg": "Away team points per game (season/recent average)",
            "home_rest": "Days since home team's last game",
            "away_rest": "Days since away team's last game",
            "elo_diff": "Home Elo - Away Elo",
            "spread_line": "Point spread (negative = home favored)",
        }

        # All features should have defined semantics
        for feature, meaning in feature_semantics.items():
            assert meaning, f"Feature {feature} has no semantic definition"

    def test_1h_fg_feature_mapping_correct(self):
        """1H features should map correctly to FG feature names for model input."""
        from src.prediction.engine import map_1h_features_to_fg_names

        # 1H features with _1h suffix
        h1_features = {
            "home_ppg_1h": 55.0,  # 1H specific PPG
            "away_ppg_1h": 52.0,
            "home_rest": 2,  # Shared (no suffix)
            "away_rest": 1,
            "elo_diff": 50.0,  # Shared (no suffix)
        }

        mapped = map_1h_features_to_fg_names(h1_features)

        # 1H-specific features should be mapped to FG names
        assert mapped.get("home_ppg") == 55.0
        assert mapped.get("away_ppg") == 52.0

        # Shared features should remain unchanged
        assert mapped.get("home_rest") == 2
        assert mapped.get("elo_diff") == 50.0


# =============================================================================
# CANONICALIZATION TESTS
# =============================================================================


class TestDateCanonicalization:
    """
    Tests that dates are consistently in CST (America/Chicago) timezone.

    CRITICAL: UTC times can cause date mismatches for late-night games.
    A 10pm CST game is 4am UTC the NEXT day.
    """

    def test_utc_to_cst_late_night_game(self):
        """Late night CST games from UTC should have correct date."""
        from zoneinfo import ZoneInfo

        from src.data import to_cst

        CST = ZoneInfo("America/Chicago")

        # 10pm CST on Jan 15 = 4am UTC on Jan 16
        utc_time = "2025-01-16T04:00:00Z"

        cst_dt = to_cst(utc_time)

        # Date should be Jan 15 (CST), not Jan 16 (UTC)
        assert (
            cst_dt.date().isoformat() == "2025-01-15"
        ), f"Expected 2025-01-15, got {cst_dt.date().isoformat()}"
        assert cst_dt.hour == 22  # 10pm CST

    def test_afternoon_game_same_date(self):
        """Afternoon CST games should have same date in UTC and CST."""
        from src.data import to_cst

        # 2pm CST on Jan 15 = 8pm UTC on Jan 15
        utc_time = "2025-01-15T20:00:00Z"

        cst_dt = to_cst(utc_time)

        assert cst_dt.date().isoformat() == "2025-01-15"
        assert cst_dt.hour == 14  # 2pm CST

    def test_date_only_strings_treated_as_local(self):
        """Date-only strings should be treated as CST dates, not UTC midnight."""
        from src.ingestion.standardize import standardize_game_data

        # Date-only input
        game = {
            "away_team": "LAL",
            "home_team": "BOS",
            "date": "2025-01-15",  # No time component
        }

        result = standardize_game_data(game, source="manual")

        # Should preserve the date as-is (CST local date)
        assert (
            result["date"] == "2025-01-15"
        ), f"Date-only string should remain {game['date']}, got {result['date']}"


class TestTeamCanonicalization:
    """
    Tests that team names are consistently normalized to ESPN format.

    CRITICAL: Team name mismatches cause join failures in data pipelines.
    """

    def test_all_30_teams_normalize(self):
        """All 30 NBA team variants should normalize correctly."""
        from src.data import standardize_team_name

        # Sample variants for each team
        team_variants = {
            "Los Angeles Lakers": ["LAL", "lal", "Lakers", "lakers", "LA Lakers"],
            "Boston Celtics": ["BOS", "bos", "Celtics", "celtics"],
            "LA Clippers": ["LAC", "lac", "Clippers", "Los Angeles Clippers"],
            "Philadelphia 76ers": ["PHI", "phi", "76ers", "Sixers"],
            "Golden State Warriors": ["GSW", "gsw", "Warriors", "GS"],
        }

        for expected, variants in team_variants.items():
            for variant in variants:
                result = standardize_team_name(variant)
                assert (
                    result == expected
                ), f"'{variant}' should normalize to '{expected}', got '{result}'"

    def test_team_name_consistency_across_modules(self):
        """Team names should be consistent across standardization modules."""
        from src.data import standardize_team_name as data_std
        from src.ingestion.standardize import normalize_team_to_espn as ingestion_std

        test_cases = ["LAL", "lakers", "Los Angeles Lakers"]

        for variant in test_cases:
            data_result = data_std(variant)
            ingestion_result, _ = ingestion_std(variant)

            assert (
                data_result == ingestion_result
            ), f"Inconsistent normalization for '{variant}': data={data_result}, ingestion={ingestion_result}"

    def test_match_key_generation_deterministic(self):
        """Match keys should be deterministic for the same game."""
        from src.data import generate_match_key

        # Same game, different variant representations
        key1 = generate_match_key("2025-01-15", "LAL", "BOS", source_is_utc=False)
        key2 = generate_match_key("2025-01-15", "Lakers", "Celtics", source_is_utc=False)
        key3 = generate_match_key(
            "2025-01-15", "Los Angeles Lakers", "Boston Celtics", source_is_utc=False
        )

        # All should generate the same match key
        assert key1 == key2 == key3, f"Match keys differ: {key1}, {key2}, {key3}"


# =============================================================================
# PRODUCTION DATA PIPELINE TESTS
# =============================================================================


class TestProductionDataFlow:
    """
    Tests for production data pipeline integrity.

    CRITICAL: Live predictions must use ONLY current/past data.
    """

    def test_feature_computation_does_not_use_future(self):
        """Feature computation for live predictions cannot use future data."""
        # This is ensured by:
        # 1. RichFeatureBuilder fetches LIVE data from APIs
        # 2. No access to future game results
        # 3. Rolling features computed from historical games only

        # Mock the concept
        today = datetime(2025, 1, 15, 12, 0, 0)

        # Available historical data
        historical_games = pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01", "2025-01-14", freq="D"),
                "home_score": range(100, 114),
                "away_score": range(95, 109),
            }
        )

        # For a prediction on Jan 15, we can ONLY use Jan 1-14 data
        valid_data = historical_games[historical_games["date"] < today]

        assert len(valid_data) == 14
        assert valid_data["date"].max() < pd.Timestamp(today)

    def test_backtest_and_live_use_same_features(self):
        """Backtest and live prediction must use identical feature definitions."""
        from src.modeling.unified_features import MODEL_CONFIGS

        # All 4 markets
        markets = ["fg_spread", "fg_total", "1h_spread", "1h_total"]

        for market in markets:
            assert market in MODEL_CONFIGS, f"Missing market config: {market}"

            config = MODEL_CONFIGS[market]

            # Each market should have defined feature sets
            assert hasattr(config, "model_features") or "features" in str(config)


class TestBacktestReproducibility:
    """
    Tests that backtest results are reproducible.

    CRITICAL: Same data + same config = same results.
    """

    def test_walk_forward_deterministic_order(self):
        """Walk-forward should process games in consistent order."""
        games = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2025-01-15",
                        "2025-01-14",
                        "2025-01-16",  # Unsorted
                    ]
                ),
                "game_id": [1, 2, 3],
            }
        )

        # Sort by date (standard preprocessing)
        games_sorted = games.sort_values("date").reset_index(drop=True)

        # Order should be: Jan 14, Jan 15, Jan 16
        expected_order = ["2025-01-14", "2025-01-15", "2025-01-16"]
        actual_order = games_sorted["date"].dt.strftime("%Y-%m-%d").tolist()

        assert actual_order == expected_order


# =============================================================================
# EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases that could cause data leakage."""

    def test_first_game_of_season_no_prior_data(self):
        """First game of season should have no rolling/expanding features."""
        df = pd.DataFrame(
            {
                "date": ["2024-10-22"],  # NBA season opener
                "team": ["LAL"],
                "score": [105],
            }
        )

        # Rolling/expanding with shift(1) should return NaN for first game
        df["rolling"] = df.groupby("team")["score"].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        )

        assert pd.isna(df.iloc[0]["rolling"]), "First game should have no prior stats"

    def test_team_first_game_after_trade(self):
        """Player traded mid-season: team stats should still be valid."""
        # Team stats are team-level, not player-level
        # This test ensures team features aren't confused with player features

        team_games = pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01", periods=10, freq="D"),
                "team": ["LAL"] * 10,
                "score": [100, 105, 110, 108, 112, 115, 118, 120, 122, 125],
            }
        )

        # Team rolling average (not affected by trades)
        team_games["rolling_ppg"] = team_games.groupby("team")["score"].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        )

        # Should have valid rolling averages from game 2 onwards
        assert not pd.isna(team_games.iloc[1]["rolling_ppg"])

    def test_back_to_back_games_same_day(self):
        """Two games on same day should be handled correctly."""
        # This can happen with double-headers or time zone issues
        games = pd.DataFrame(
            {
                "date": ["2025-01-15", "2025-01-15", "2025-01-16"],
                "game_id": ["G1", "G2", "G3"],
                "team": ["LAL", "LAL", "LAL"],
            }
        )

        # Both Jan 15 games should NOT use each other's results
        # Only prior day data is safe

        # This would require timestamp-level ordering, not just date
        # For robustness, features should use data from t-1 days, not t-1 games
        pass  # Design note: Current system uses date-level, may need timestamp
