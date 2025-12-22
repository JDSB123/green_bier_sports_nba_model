"""Tests for the UnifiedPredictionEngine and PeriodPredictor - v6.0.

Tests all 9 INDEPENDENT markets:
- First Quarter: Spread, Total, Moneyline
- First Half: Spread, Total, Moneyline
- Full Game: Spread, Total, Moneyline
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path


class TestPeriodPredictor:
    """Tests for the PeriodPredictor class."""

    @pytest.fixture
    def mock_spread_model(self):
        """Create mock spread model."""
        model = MagicMock()
        # Return probabilities [away_cover, home_cover]
        model.predict_proba.return_value = np.array([[0.40, 0.60]])
        return model

    @pytest.fixture
    def mock_total_model(self):
        """Create mock total model."""
        model = MagicMock()
        # Return probabilities [under, over]
        model.predict_proba.return_value = np.array([[0.45, 0.55]])
        return model

    @pytest.fixture
    def mock_moneyline_model(self):
        """Create mock moneyline model."""
        model = MagicMock()
        # Return probabilities [away_win, home_win]
        model.predict_proba.return_value = np.array([[0.35, 0.65]])
        return model

    @pytest.fixture
    def sample_features(self):
        """Sample features for testing."""
        return {
            "home_ppg": 115.0,
            "away_ppg": 110.0,
            "home_papg": 108.0,
            "away_papg": 112.0,
            "predicted_margin": 5.0,
            "predicted_total": 225.0,
            "predicted_margin_1h": 2.5,
            "predicted_total_1h": 112.5,
            "predicted_margin_q1": 1.2,
            "predicted_total_q1": 56.0,
            "home_win_pct": 0.65,
            "away_win_pct": 0.45,
            "home_rest_days": 2,
            "away_rest_days": 1,
        }

    @pytest.fixture
    def period_predictor(self, mock_spread_model, mock_total_model, mock_moneyline_model):
        """Create a PeriodPredictor instance for testing."""
        from src.prediction.engine import PeriodPredictor

        return PeriodPredictor(
            period="fg",
            spread_model=mock_spread_model,
            spread_features=["home_ppg", "away_ppg", "predicted_margin"],
            total_model=mock_total_model,
            total_features=["home_ppg", "away_ppg", "predicted_total"],
            moneyline_model=mock_moneyline_model,
            moneyline_features=["home_win_pct", "away_win_pct"],
        )

    def test_predict_spread_returns_correct_structure(self, period_predictor, sample_features):
        """Test spread prediction returns expected keys."""
        result = period_predictor.predict_spread(sample_features, spread_line=-3.5)

        assert "home_cover_prob" in result
        assert "away_cover_prob" in result
        assert "predicted_margin" in result
        assert "confidence" in result
        assert "bet_side" in result
        assert "edge" in result
        assert "passes_filter" in result

    def test_predict_spread_home_favorite(self, period_predictor, sample_features):
        """Test spread prediction when home team is favorite."""
        result = period_predictor.predict_spread(sample_features, spread_line=-3.5)

        assert result["home_cover_prob"] == 0.60
        assert result["away_cover_prob"] == 0.40
        assert result["bet_side"] == "home"
        # Edge = predicted_margin - spread_line = 5.0 - (-3.5) = 8.5
        assert result["edge"] == pytest.approx(8.5, rel=0.01)

    def test_predict_spread_missing_features_raises(self, mock_spread_model, mock_total_model, mock_moneyline_model):
        """Test that missing features raises ValueError in STRICT MODE."""
        from src.prediction.engine import PeriodPredictor

        predictor = PeriodPredictor(
            period="fg",
            spread_model=mock_spread_model,
            spread_features=["home_ppg", "away_ppg", "nonexistent_feature"],
            total_model=mock_total_model,
            total_features=["home_ppg"],
            moneyline_model=mock_moneyline_model,
            moneyline_features=["home_win_pct"],
        )

        features = {"home_ppg": 110.0, "away_ppg": 105.0}

        with pytest.raises(ValueError, match="MISSING.*REQUIRED FEATURES"):
            predictor.predict_spread(features, spread_line=-3.5)

    def test_predict_total_returns_correct_structure(self, period_predictor, sample_features):
        """Test total prediction returns expected keys."""
        result = period_predictor.predict_total(sample_features, total_line=220.0)

        assert "over_prob" in result
        assert "under_prob" in result
        assert "predicted_total" in result
        assert "confidence" in result
        assert "bet_side" in result
        assert "edge" in result
        assert "passes_filter" in result

    def test_predict_total_over(self, period_predictor, sample_features):
        """Test total prediction when model predicts over."""
        result = period_predictor.predict_total(sample_features, total_line=220.0)

        assert result["over_prob"] == 0.55
        assert result["under_prob"] == 0.45
        assert result["bet_side"] == "over"
        # Edge = predicted_total - total_line = 225.0 - 220.0 = 5.0
        assert result["edge"] == pytest.approx(5.0, rel=0.01)

    def test_predict_moneyline_returns_correct_structure(self, period_predictor, sample_features):
        """Test moneyline prediction returns expected keys."""
        result = period_predictor.predict_moneyline(
            sample_features,
            home_ml_odds=-150,
            away_ml_odds=130,
        )

        assert "home_win_prob" in result
        assert "away_win_prob" in result
        assert "confidence" in result
        assert "recommended_bet" in result
        assert "passes_filter" in result

    def test_predict_moneyline_home_favorite(self, period_predictor, sample_features):
        """Test moneyline prediction when home team is favorite."""
        result = period_predictor.predict_moneyline(
            sample_features,
            home_ml_odds=-150,
            away_ml_odds=130,
        )

        assert result["home_win_prob"] == 0.65
        assert result["away_win_prob"] == 0.35
        assert result["recommended_bet"] == "home"


class TestUnifiedPredictionEngine:
    """Tests for the UnifiedPredictionEngine class."""

    def test_engine_init_requires_all_models_strict_mode(self, tmp_path):
        """Test engine fails when models are missing in strict mode."""
        from src.prediction.engine import UnifiedPredictionEngine, ModelNotFoundError

        empty_dir = tmp_path / "empty_models"
        empty_dir.mkdir()

        with pytest.raises(ModelNotFoundError):
            UnifiedPredictionEngine(models_dir=empty_dir, require_all=True)

    def test_get_model_info_returns_correct_structure(self, tmp_path):
        """Test get_model_info returns expected structure."""
        from src.prediction.engine import UnifiedPredictionEngine

        # Create engine directly without calling __init__
        engine = UnifiedPredictionEngine.__new__(UnifiedPredictionEngine)
        engine.models_dir = tmp_path
        engine.q1_predictor = MagicMock()
        engine.h1_predictor = MagicMock()
        engine.fg_predictor = MagicMock()
        engine.loaded_models = {
            "q1_spread": True, "q1_total": True, "q1_moneyline": True,
            "1h_spread": True, "1h_total": True, "1h_moneyline": True,
            "fg_spread": True, "fg_total": True, "fg_moneyline": True,
        }

        info = engine.get_model_info()

        assert "version" in info
        assert info["version"] == "6.5"
        assert "markets" in info
        assert info["markets"] == 9
        assert "markets_list" in info

    def test_predict_all_markets_returns_all_periods(self):
        """Test predict_all_markets returns all 3 periods with 3 markets each."""
        from src.prediction.engine import UnifiedPredictionEngine

        # Create mock engine with mock predictors
        engine = UnifiedPredictionEngine.__new__(UnifiedPredictionEngine)
        engine.models_dir = Path("/mock")
        engine.loaded_models = {}

        # Create mock period prediction return values
        mock_period_result = {
            "spread": {"passes_filter": True, "confidence": 0.6},
            "total": {"passes_filter": True, "confidence": 0.55},
            "moneyline": {"passes_filter": True, "confidence": 0.65}
        }

        mock_predictor = MagicMock()
        engine.q1_predictor = mock_predictor
        engine.h1_predictor = mock_predictor
        engine.fg_predictor = mock_predictor

        # Mock the predict methods to return full period results
        with patch.object(engine, 'predict_quarter', return_value=mock_period_result), \
             patch.object(engine, 'predict_first_half', return_value=mock_period_result), \
             patch.object(engine, 'predict_full_game', return_value=mock_period_result):

            features = {"home_ppg": 110, "away_ppg": 105, "predicted_margin": 5}

            result = engine.predict_all_markets(
                features=features,
                fg_spread_line=-3.5,
                fg_total_line=220.0,
                fh_spread_line=-1.5,
                fh_total_line=110.0,
                q1_spread_line=-0.5,
                q1_total_line=55.0,
                home_ml_odds=-150,
                away_ml_odds=130,
            )

            assert "first_quarter" in result
            assert "first_half" in result
            assert "full_game" in result

            for period in ["first_quarter", "first_half", "full_game"]:
                assert "spread" in result[period]
                assert "total" in result[period]
                assert "moneyline" in result[period]


class TestModelNotFoundError:
    """Tests for the ModelNotFoundError exception."""

    def test_model_not_found_error_is_exception(self):
        """Test ModelNotFoundError is an Exception."""
        from src.prediction.engine import ModelNotFoundError

        error = ModelNotFoundError("Model not found")
        assert isinstance(error, Exception)
        assert str(error) == "Model not found"


class TestConfidenceCalculation:
    """Tests for confidence calculation utilities.

    The confidence function uses entropy-based calculation, not simple probability.
    It maps probability strength * certainty to a range [0.5, 0.95].
    """

    def test_confidence_from_probabilities_balanced(self):
        """Test confidence calculation with balanced probabilities."""
        from src.prediction.confidence import calculate_confidence_from_probabilities

        # 50/50 = maximum entropy = minimum confidence
        confidence = calculate_confidence_from_probabilities(0.5, 0.5)
        assert confidence == pytest.approx(0.5, rel=0.01)

    def test_confidence_from_probabilities_strong_home(self):
        """Test confidence calculation with strong home probability."""
        from src.prediction.confidence import calculate_confidence_from_probabilities

        # 70% home should give higher confidence than 50/50
        confidence = calculate_confidence_from_probabilities(0.7, 0.3)
        assert confidence > 0.5  # More confident than random
        assert confidence < 0.95  # Below max cap

    def test_confidence_from_probabilities_strong_away(self):
        """Test confidence calculation with strong away probability."""
        from src.prediction.confidence import calculate_confidence_from_probabilities

        # 30% home (70% away) should give same confidence as 70/30
        confidence_away = calculate_confidence_from_probabilities(0.3, 0.7)
        confidence_home = calculate_confidence_from_probabilities(0.7, 0.3)
        assert confidence_away == pytest.approx(confidence_home, rel=0.01)

    def test_confidence_extreme_probability(self):
        """Test confidence with extreme probability approaches max."""
        from src.prediction.confidence import calculate_confidence_from_probabilities

        # Near-certain prediction (99%)
        confidence = calculate_confidence_from_probabilities(0.99, 0.01)
        assert confidence > 0.8  # High confidence
        assert confidence <= 0.95  # Capped at max

    def test_confidence_increases_with_probability_difference(self):
        """Test that confidence increases as probability difference increases."""
        from src.prediction.confidence import calculate_confidence_from_probabilities

        conf_55 = calculate_confidence_from_probabilities(0.55, 0.45)
        conf_65 = calculate_confidence_from_probabilities(0.65, 0.35)
        conf_80 = calculate_confidence_from_probabilities(0.80, 0.20)

        assert conf_55 < conf_65 < conf_80


class TestFilterThresholds:
    """Tests for filter threshold application."""

    def test_spread_filter_low_confidence(self):
        """Test spread prediction fails filter on low confidence."""
        from src.prediction.engine import PeriodPredictor

        # Create predictor with mock model that returns low probability
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.48, 0.52]])  # Low confidence

        predictor = PeriodPredictor(
            period="fg",
            spread_model=mock_model,
            spread_features=["home_ppg"],
            total_model=MagicMock(),
            total_features=["home_ppg"],
            moneyline_model=MagicMock(),
            moneyline_features=["home_ppg"],
        )

        features = {"home_ppg": 110, "predicted_margin": 5}
        result = predictor.predict_spread(features, spread_line=-3.5)

        # 52% confidence is below 55% threshold
        assert result["passes_filter"] is False
        assert "confidence" in result["filter_reason"].lower()

    def test_spread_filter_passes_with_high_confidence_and_edge(self):
        """Test spread prediction passes filter with high confidence and edge."""
        from src.prediction.engine import PeriodPredictor

        # Create predictor with mock model that returns very high probability
        # Entropy-based confidence at 0.90 prob gives ~0.72 confidence
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.10, 0.90]])

        predictor = PeriodPredictor(
            period="fg",
            spread_model=mock_model,
            spread_features=["home_ppg"],
            total_model=MagicMock(),
            total_features=["home_ppg"],
            moneyline_model=MagicMock(),
            moneyline_features=["home_ppg"],
        )

        # Big edge: predicted_margin=8, spread_line=-3.5, edge=11.5
        features = {"home_ppg": 110, "predicted_margin": 8}
        result = predictor.predict_spread(features, spread_line=-3.5)

        # High confidence (90% model prob -> ~0.72 entropy-based confidence)
        # Large edge (11.5 > 1.0 threshold) - should pass both filters
        assert result["passes_filter"] is True
        assert result["filter_reason"] is None
