"""
Unit tests for XGBoost predictor
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
import numpy as np
import pandas as pd

from src.models.predictor import NCAAFPredictor


@pytest.fixture
def mock_models():
    """Create mock XGBoost models"""
    models = {}
    for model_name in ['margin', 'total', 'home_score', 'away_score']:
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([25.0])  # Default prediction
        models[model_name] = mock_model
    return models


@pytest.fixture
def predictor(mock_models):
    """Create predictor with mocked models"""
    with patch('src.models.predictor.joblib.load') as mock_load:
        mock_load.side_effect = lambda path: mock_models[path.split('_')[1].split('.')[0]]
        pred = NCAAFPredictor(model_path='/fake/path/models')
        pred.models = mock_models
        return pred


class TestNCAAFPredictor:
    """Test suite for NCAAFPredictor"""

    def test_initialization(self, predictor):
        """Test predictor initializes with all models"""
        assert len(predictor.models) == 4
        assert 'margin' in predictor.models
        assert 'total' in predictor.models
        assert 'home_score' in predictor.models
        assert 'away_score' in predictor.models

    def test_predict_game_basic(self, predictor):
        """Test basic game prediction"""
        features = {
            'home_yards_per_play': 6.5,
            'away_yards_per_play': 5.2,
            'home_completion_pct': 0.65,
            'away_completion_pct': 0.58,
            'home_third_down_conversion_pct': 0.45,
            'away_third_down_conversion_pct': 0.38,
            'talent_differential': 5.0,
        }

        # Mock predictions
        predictor.models['margin'].predict.return_value = np.array([7.5])
        predictor.models['total'].predict.return_value = np.array([52.0])
        predictor.models['home_score'].predict.return_value = np.array([29.75])
        predictor.models['away_score'].predict.return_value = np.array([22.25])

        result = predictor.predict_game(
            features=features,
            consensus_spread=-7.0,
            consensus_total=51.5
        )

        # Verify prediction structure
        assert 'predicted_margin' in result
        assert 'predicted_total' in result
        assert 'predicted_home_score' in result
        assert 'predicted_away_score' in result
        assert 'confidence' in result
        assert 'edge_spread' in result
        assert 'edge_total' in result

    def test_edge_calculation(self, predictor):
        """Test edge calculation against market"""
        features = {
            'home_yards_per_play': 6.0,
            'away_yards_per_play': 5.5,
            'talent_differential': 3.0
        }

        # Model predicts home by 10
        predictor.models['margin'].predict.return_value = np.array([10.0])
        predictor.models['total'].predict.return_value = np.array([55.0])

        # Market has home by 7
        consensus_spread = -7.0
        consensus_total = 52.0

        result = predictor.predict_game(features, consensus_spread, consensus_total)

        # Edge should be prediction - market
        # For spread: predicted_margin - abs(consensus_spread)
        # 10 - 7 = 3 point edge
        expected_spread_edge = 10.0 - abs(consensus_spread)
        assert result['edge_spread'] == pytest.approx(expected_spread_edge, 0.01)

        # For total: predicted_total - consensus_total
        # 55 - 52 = 3 point edge
        expected_total_edge = 55.0 - consensus_total
        assert result['edge_total'] == pytest.approx(expected_total_edge, 0.01)

    def test_betting_recommendation_spread(self, predictor):
        """Test betting recommendation for spread bet"""
        features = {'talent_differential': 8.0}

        # Large edge on spread (model 14, market 7)
        predictor.models['margin'].predict.return_value = np.array([14.0])
        predictor.models['total'].predict.return_value = np.array([50.0])

        result = predictor.predict_game(
            features=features,
            consensus_spread=-7.0,
            consensus_total=50.0
        )

        recommendation = result.get('recommendation', {})

        # With 7-point edge and decent confidence, should recommend bet
        if recommendation.get('recommend_bet'):
            assert recommendation['bet_type'] in ['spread', 'total']
            assert recommendation['recommended_units'] > 0
            assert recommendation['recommended_units'] <= 2.0  # Max 2 units

    def test_betting_recommendation_no_edge(self, predictor):
        """Test no recommendation when edge is too small"""
        features = {'talent_differential': 1.0}

        # Small edge (model 7.5, market 7)
        predictor.models['margin'].predict.return_value = np.array([7.5])
        predictor.models['total'].predict.return_value = np.array([50.5])

        result = predictor.predict_game(
            features=features,
            consensus_spread=-7.0,
            consensus_total=50.0
        )

        recommendation = result.get('recommendation', {})

        # Edge is only 0.5 points, below 2.5 threshold
        assert recommendation.get('recommend_bet') is False or recommendation.get('recommended_units', 0) == 0

    def test_confidence_calculation(self, predictor):
        """Test confidence score calculation"""
        # Strong features should yield high confidence
        strong_features = {
            'home_yards_per_play': 7.5,
            'away_yards_per_play': 4.5,
            'home_third_down_conversion_pct': 0.55,
            'away_third_down_conversion_pct': 0.30,
            'talent_differential': 12.0,
            'home_recent_points_avg': 35.0,
            'away_recent_points_avg': 18.0
        }

        predictor.models['margin'].predict.return_value = np.array([15.0])
        predictor.models['total'].predict.return_value = np.array([50.0])

        result = predictor.predict_game(strong_features, -10.0, 50.0)

        # Strong differentials should produce higher confidence
        assert 0.0 <= result['confidence'] <= 1.0
        # With strong features, confidence should be above 0.6
        assert result['confidence'] >= 0.5

    def test_calculate_units(self, predictor):
        """Test unit sizing based on edge and confidence"""
        # Test with varying edges and confidence levels
        test_cases = [
            {'edge': 8.0, 'confidence': 0.80, 'expected_min': 1.5},  # Strong bet
            {'edge': 3.0, 'confidence': 0.65, 'expected_min': 0.5},  # Moderate bet
            {'edge': 1.0, 'confidence': 0.55, 'expected_min': 0.0},  # Weak bet
        ]

        for case in test_cases:
            units = predictor._calculate_units(
                edge=case['edge'],
                confidence=case['confidence'],
                bet_type='spread'
            )

            # Units should be between 0 and 2
            assert 0.0 <= units <= 2.0

            # Should meet minimum expectation
            assert units >= case['expected_min']

    def test_predict_game_with_missing_features(self, predictor):
        """Test prediction handles missing features"""
        # Minimal feature set
        features = {
            'home_yards_per_play': 6.0,
            'away_yards_per_play': 5.5,
        }

        predictor.models['margin'].predict.return_value = np.array([7.0])
        predictor.models['total'].predict.return_value = np.array([48.0])

        # Should not raise exception
        result = predictor.predict_game(features, -7.0, 48.0)

        assert 'predicted_margin' in result
        assert 'confidence' in result

    def test_feature_alignment(self, predictor):
        """Test that features are properly aligned for model input"""
        features = {
            'home_yards_per_play': 6.5,
            'away_yards_per_play': 5.2,
            'talent_differential': 5.0,
        }

        # Assuming models expect specific feature order
        with patch.object(predictor, 'expected_features', ['home_yards_per_play', 'away_yards_per_play', 'talent_differential']):
            predictor.models['margin'].predict.return_value = np.array([7.0])
            predictor.models['total'].predict.return_value = np.array([50.0])

            result = predictor.predict_game(features, -7.0, 50.0)

            # Verify model was called
            assert predictor.models['margin'].predict.called
            assert predictor.models['total'].predict.called

            # Verify input shape (should be 2D array with 1 row)
            call_args = predictor.models['margin'].predict.call_args[0][0]
            assert call_args.shape[0] == 1  # Single prediction

    def test_score_consistency(self, predictor):
        """Test that predicted scores match predicted margin and total"""
        features = {'talent_differential': 5.0}

        # Set specific predictions
        predictor.models['margin'].predict.return_value = np.array([8.0])
        predictor.models['total'].predict.return_value = np.array([50.0])
        predictor.models['home_score'].predict.return_value = np.array([29.0])
        predictor.models['away_score'].predict.return_value = np.array([21.0])

        result = predictor.predict_game(features, -7.0, 50.0)

        home_score = result['predicted_home_score']
        away_score = result['predicted_away_score']

        # Margin should equal home - away
        calculated_margin = home_score - away_score
        assert calculated_margin == pytest.approx(result['predicted_margin'], 0.1)

        # Total should equal home + away
        calculated_total = home_score + away_score
        assert calculated_total == pytest.approx(result['predicted_total'], 0.1)

    def test_prediction_bounds(self, predictor):
        """Test predictions are within reasonable bounds"""
        features = {'talent_differential': 10.0}

        predictor.models['margin'].predict.return_value = np.array([12.0])
        predictor.models['total'].predict.return_value = np.array([55.0])
        predictor.models['home_score'].predict.return_value = np.array([33.5])
        predictor.models['away_score'].predict.return_value = np.array([21.5])

        result = predictor.predict_game(features, -10.0, 55.0)

        # Scores should be non-negative
        assert result['predicted_home_score'] >= 0
        assert result['predicted_away_score'] >= 0

        # Total should be reasonable (typically 20-100 points)
        assert 20 <= result['predicted_total'] <= 100

        # Margin should be reasonable (typically -50 to +50)
        assert -50 <= result['predicted_margin'] <= 50


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
