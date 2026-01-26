"""
Test prediction invariants to prevent bet_side/confidence bugs.

These tests verify that bet_side and confidence always refer to the same outcome.
This prevents the critical bug where API returns {bet_side: "home", confidence: 0.72}
but confidence actually refers to the AWAY probability.

FIXED in v33.1.0: Edge-based confidence with signal conflict detection.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.config import PROJECT_ROOT
from src.prediction import UnifiedPredictionEngine


@pytest.fixture(scope="module")
def engine():
    """Load production prediction engine."""
    models_dir = PROJECT_ROOT / "models" / "production"
    if not models_dir.exists():
        pytest.skip(f"Models directory not found: {models_dir}")

    return UnifiedPredictionEngine(models_dir=models_dir)


@pytest.fixture
def sample_features():
    """Sample features for testing - complete feature set."""
    from src.modeling.unified_features import get_feature_defaults

    # Start with all defaults
    features = get_feature_defaults()

    # Override with test values for key features
    features.update(
        {
            # Core stats
            "home_ppg": 115.0,
            "home_papg": 110.0,
            "home_margin": 5.0,
            "home_win_pct": 0.65,
            "away_ppg": 108.0,
            "away_papg": 112.0,
            "away_margin": -4.0,
            "away_win_pct": 0.45,
            # Differentials
            "ppg_diff": 7.0,
            "win_pct_diff": 0.20,
            # Rest
            "home_rest": 2.0,
            "away_rest": 1.0,
            "rest_diff": 1.0,
            "home_b2b": 0.0,
            "away_b2b": 1.0,
            "home_rest_adj": 0.0,
            "away_rest_adj": -0.5,
            "rest_margin_adj": 0.5,
            # HCA
            "dynamic_hca": 3.0,
            "home_court_advantage": 3.0,
            # Travel
            "away_travel_distance": 1500.0,
            "away_timezone_change": 2.0,
            "away_travel_fatigue": 1.5,
            "is_away_long_trip": 1.0,
            "is_away_cross_country": 1.0,
            "away_b2b_travel_penalty": 1.0,
            "travel_advantage": 2.0,
            # Efficiency
            "home_ortg": 115.0,
            "home_drtg": 110.0,
            "home_net_rtg": 5.0,
            "away_ortg": 108.0,
            "away_drtg": 112.0,
            "away_net_rtg": -4.0,
            "net_rating_diff": 9.0,
            # Form
            "home_l5_margin": 6.0,
            "away_l5_margin": -3.0,
            "home_l10_margin": 5.0,
            "away_l10_margin": -4.0,
            "home_margin_std": 10.0,
            "away_margin_std": 12.0,
            "home_score_std": 8.0,
            "away_score_std": 9.0,
            # Pace
            "home_pace": 102.0,
            "away_pace": 100.0,
            "expected_pace": 101.0,
            # Predicted values (critical for edge calculation)
            "predicted_margin": 8.0,  # Model predicts home wins by 8
            "predicted_total": 225.0,
            # 1H predicted values (required for 1H edge calculation)
            "predicted_margin_1h": 4.0,
            "predicted_total_1h": 112.0,
            # Market line features (cross-market lines required by all models)
            # These are normally injected by predict_all_markets(), but when calling
            # predict_full_game() or predict_first_half() directly, tests must provide them
            "fg_spread_line": -5.0,
            "fg_total_line": 225.0,
            "1h_spread_line": -2.5,
            "1h_total_line": 112.0,
            "spread_line": -5.0,  # Alias
            "total_line": 225.0,  # Alias
        }
    )

    return features


class TestSpreadInvariants:
    """Test spread prediction invariants."""

    def test_fg_spread_bet_side_matches_confidence(self, engine, sample_features):
        """CRITICAL: bet_side and confidence must refer to the same outcome."""
        spread_line = -5.0  # Home favored by 5

        pred = engine.predict_full_game(
            sample_features,
            spread_line=spread_line,
            total_line=225.0,
        )

        spread_pred = pred.get("spread")
        if not spread_pred or not spread_pred.get("passes_filter"):
            pytest.skip("Prediction filtered out (not enough edge/confidence)")

        bet_side = spread_pred["bet_side"]
        confidence = spread_pred["confidence"]
        home_prob = spread_pred["home_cover_prob"]
        away_prob = spread_pred["away_cover_prob"]

        # INVARIANT: confidence must equal the probability of the bet_side
        if bet_side == "home":
            assert (
                abs(confidence - home_prob) < 0.001
            ), f"bet_side='home' but confidence={confidence:.3f} != home_cover_prob={home_prob:.3f}"
        else:
            assert (
                abs(confidence - away_prob) < 0.001
            ), f"bet_side='away' but confidence={confidence:.3f} != away_cover_prob={away_prob:.3f}"

    def test_1h_spread_bet_side_matches_confidence(self, engine, sample_features):
        """CRITICAL: 1H spread invariant (same as FG)."""
        spread_line = -2.5  # Home favored by 2.5 in 1H

        pred = engine.predict_first_half(
            sample_features,
            spread_line=spread_line,
            total_line=112.0,
        )

        spread_pred = pred.get("spread")
        if not spread_pred or not spread_pred.get("passes_filter"):
            pytest.skip("Prediction filtered out")

        bet_side = spread_pred["bet_side"]
        confidence = spread_pred["confidence"]
        home_prob = spread_pred["home_cover_prob"]
        away_prob = spread_pred["away_cover_prob"]

        # INVARIANT: confidence must equal the probability of the bet_side
        if bet_side == "home":
            assert (
                abs(confidence - home_prob) < 0.001
            ), f"1H: bet_side='home' but confidence={confidence:.3f} != home_cover_prob={home_prob:.3f}"
        else:
            assert (
                abs(confidence - away_prob) < 0.001
            ), f"1H: bet_side='away' but confidence={confidence:.3f} != away_cover_prob={away_prob:.3f}"

    def test_spread_signals_agree_field_present(self, engine, sample_features):
        """Signal agreement field must be present and accurate.

        Note: As of v33.1.5 we use EDGE-ONLY filtering. Signal conflicts
        are tracked for diagnostics but do NOT filter predictions.
        The bet_side always follows the edge-based prediction_side.
        """
        spread_line = -5.0

        pred = engine.predict_full_game(
            sample_features,
            spread_line=spread_line,
            total_line=225.0,
        )

        spread_pred = pred.get("spread")
        if not spread_pred:
            pytest.skip("No spread prediction")

        # These fields must exist for diagnostics
        assert "signals_agree" in spread_pred
        assert "classifier_side" in spread_pred
        assert "prediction_side" in spread_pred

        # signals_agree must accurately reflect the comparison
        signals_agree = spread_pred["signals_agree"]
        expected_agree = spread_pred["classifier_side"] == spread_pred["prediction_side"]
        assert signals_agree == expected_agree, (
            f"signals_agree={signals_agree} but classifier_side={spread_pred['classifier_side']} "
            f"vs prediction_side={spread_pred['prediction_side']}"
        )


class TestTotalInvariants:
    """Test total prediction invariants."""

    def test_fg_total_bet_side_matches_confidence(self, engine, sample_features):
        """CRITICAL: bet_side and confidence must refer to the same outcome."""
        total_line = 220.0

        pred = engine.predict_full_game(
            sample_features,
            spread_line=-5.0,
            total_line=total_line,
        )

        total_pred = pred.get("total")
        if not total_pred or not total_pred.get("passes_filter"):
            pytest.skip("Prediction filtered out")

        bet_side = total_pred["bet_side"]
        confidence = total_pred["confidence"]
        over_prob = total_pred["over_prob"]
        under_prob = total_pred["under_prob"]

        # INVARIANT: confidence must equal the probability of the bet_side
        if bet_side == "over":
            assert (
                abs(confidence - over_prob) < 0.001
            ), f"bet_side='over' but confidence={confidence:.3f} != over_prob={over_prob:.3f}"
        else:
            assert (
                abs(confidence - under_prob) < 0.001
            ), f"bet_side='under' but confidence={confidence:.3f} != under_prob={under_prob:.3f}"

    def test_1h_total_bet_side_matches_confidence(self, engine, sample_features):
        """CRITICAL: 1H total invariant (same as FG)."""
        total_line = 110.0

        pred = engine.predict_first_half(
            sample_features,
            spread_line=-2.5,
            total_line=total_line,
        )

        total_pred = pred.get("total")
        if not total_pred or not total_pred.get("passes_filter"):
            pytest.skip("Prediction filtered out")

        bet_side = total_pred["bet_side"]
        confidence = total_pred["confidence"]
        over_prob = total_pred["over_prob"]
        under_prob = total_pred["under_prob"]

        # INVARIANT: confidence must equal the probability of the bet_side
        if bet_side == "over":
            assert (
                abs(confidence - over_prob) < 0.001
            ), f"1H: bet_side='over' but confidence={confidence:.3f} != over_prob={over_prob:.3f}"
        else:
            assert (
                abs(confidence - under_prob) < 0.001
            ), f"1H: bet_side='under' but confidence={confidence:.3f} != under_prob={under_prob:.3f}"

    def test_total_signals_agree_field_present(self, engine, sample_features):
        """Signal agreement field must be present and accurate.

        Note: As of v33.1.5 we use EDGE-ONLY filtering. Signal conflicts
        are tracked for diagnostics but do NOT filter predictions.
        The bet_side always follows the edge-based prediction_side.
        """
        total_line = 220.0

        pred = engine.predict_full_game(
            sample_features,
            spread_line=-5.0,
            total_line=total_line,
        )

        total_pred = pred.get("total")
        if not total_pred:
            pytest.skip("No total prediction")

        # These fields must exist for diagnostics
        assert "signals_agree" in total_pred
        assert "classifier_side" in total_pred
        assert "prediction_side" in total_pred

        # signals_agree must accurately reflect the comparison
        signals_agree = total_pred["signals_agree"]
        expected_agree = total_pred["classifier_side"] == total_pred["prediction_side"]
        assert signals_agree == expected_agree, (
            f"signals_agree={signals_agree} but classifier_side={total_pred['classifier_side']} "
            f"vs prediction_side={total_pred['prediction_side']}"
        )


class TestEdgeCalculations:
    """Test edge calculation consistency."""

    def test_spread_edge_matches_prediction(self, engine, sample_features):
        """Edge should match predicted_margin + spread_line."""
        spread_line = -5.0
        predicted_margin = sample_features["predicted_margin"]  # 8.0

        pred = engine.predict_full_game(
            sample_features,
            spread_line=spread_line,
            total_line=225.0,
        )

        spread_pred = pred.get("spread")
        if not spread_pred:
            pytest.skip("No spread prediction")

        raw_edge = spread_pred["raw_edge"]
        expected_edge = predicted_margin + spread_line  # 8.0 + (-5.0) = 3.0

        assert abs(raw_edge - expected_edge) < 0.001, (
            f"Edge calculation wrong: raw_edge={raw_edge:.1f}, "
            f"expected={expected_edge:.1f} (predicted_margin={predicted_margin} + spread_line={spread_line})"
        )

    def test_total_edge_matches_prediction(self, engine, sample_features):
        """Edge should match predicted_total - total_line."""
        total_line = 220.0
        predicted_total = sample_features["predicted_total"]  # 225.0

        pred = engine.predict_full_game(
            sample_features,
            spread_line=-5.0,
            total_line=total_line,
        )

        total_pred = pred.get("total")
        if not total_pred:
            pytest.skip("No total prediction")

        raw_edge = total_pred["raw_edge"]
        expected_edge = predicted_total - total_line  # 225.0 - 220.0 = 5.0

        assert abs(raw_edge - expected_edge) < 0.001, (
            f"Edge calculation wrong: raw_edge={raw_edge:.1f}, "
            f"expected={expected_edge:.1f} (predicted_total={predicted_total} - total_line={total_line})"
        )
