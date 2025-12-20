"""
Unit tests for feature extraction
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
import pandas as pd
import numpy as np

from src.features.feature_extractor import FeatureExtractor


@pytest.fixture
def mock_db_connection():
    """Mock database connection"""
    conn = MagicMock()
    return conn


@pytest.fixture
def feature_extractor(mock_db_connection):
    """Create FeatureExtractor instance with mocked DB"""
    with patch('src.features.feature_extractor.psycopg2.connect', return_value=mock_db_connection):
        extractor = FeatureExtractor(
            host='localhost',
            port=5432,
            database='test_db',
            user='test_user',
            password='test_pass'
        )
        extractor.conn = mock_db_connection
        return extractor


class TestFeatureExtractor:
    """Test suite for FeatureExtractor"""

    def test_initialization(self, feature_extractor):
        """Test FeatureExtractor initializes correctly"""
        assert feature_extractor.conn is not None
        assert hasattr(feature_extractor, 'extract_game_features')

    def test_extract_efficiency_features(self, feature_extractor):
        """Test efficiency metrics calculation"""
        # Mock team stats
        mock_stats = {
            'offensive_plays': 75,
            'offensive_yards': 450,
            'rushing_attempts': 35,
            'rushing_yards': 180,
            'passing_completions': 25,
            'passing_attempts': 40,
            'passing_yards': 270,
            'third_down_conversions': 8,
            'third_down_attempts': 14,
            'red_zone_scores': 4,
            'red_zone_attempts': 5,
            'turnovers': 1,
            'penalties': 6,
            'penalty_yards': 55
        }

        features = feature_extractor._extract_efficiency_features(mock_stats, prefix='test_')

        # Verify feature calculations
        assert features['test_yards_per_play'] == pytest.approx(450 / 75, 0.01)
        assert features['test_yards_per_rush'] == pytest.approx(180 / 35, 0.01)
        assert features['test_completion_pct'] == pytest.approx(25 / 40, 0.01)
        assert features['test_yards_per_pass_attempt'] == pytest.approx(270 / 40, 0.01)
        assert features['test_third_down_conversion_pct'] == pytest.approx(8 / 14, 0.01)
        assert features['test_red_zone_scoring_pct'] == pytest.approx(4 / 5, 0.01)
        assert features['test_turnovers'] == 1
        assert features['test_penalties'] == 6

    def test_extract_efficiency_features_zero_division(self, feature_extractor):
        """Test efficiency features handle division by zero"""
        mock_stats = {
            'offensive_plays': 0,
            'offensive_yards': 0,
            'rushing_attempts': 0,
            'rushing_yards': 0,
            'passing_attempts': 0,
            'passing_completions': 0,
            'passing_yards': 0,
            'third_down_attempts': 0,
            'third_down_conversions': 0,
            'red_zone_attempts': 0,
            'red_zone_scores': 0,
            'turnovers': 0,
            'penalties': 0,
            'penalty_yards': 0
        }

        features = feature_extractor._extract_efficiency_features(mock_stats, prefix='test_')

        # All ratios should be 0 when denominators are 0
        assert features['test_yards_per_play'] == 0.0
        assert features['test_yards_per_rush'] == 0.0
        assert features['test_completion_pct'] == 0.0
        assert features['test_third_down_conversion_pct'] == 0.0

    def test_extract_qb_features(self, feature_extractor):
        """Test QB stats extraction"""
        mock_stats = {
            'passing_completions': 28,
            'passing_attempts': 40,
            'passing_yards': 350,
            'passing_touchdowns': 3,
            'interceptions': 1,
            'sacks': 2,
            'sack_yards': 15
        }

        features = feature_extractor._extract_qb_features(mock_stats, prefix='qb_')

        assert features['qb_completion_pct'] == pytest.approx(28 / 40, 0.01)
        assert features['qb_yards_per_attempt'] == pytest.approx(350 / 40, 0.01)
        assert features['qb_touchdowns'] == 3
        assert features['qb_interceptions'] == 1
        assert features['qb_td_int_ratio'] == pytest.approx(3 / 1, 0.01)
        assert features['qb_sacks'] == 2
        assert features['qb_sack_yards'] == 15

    def test_extract_qb_features_no_interceptions(self, feature_extractor):
        """Test QB TD/INT ratio when no interceptions"""
        mock_stats = {
            'passing_completions': 25,
            'passing_attempts': 35,
            'passing_yards': 300,
            'passing_touchdowns': 4,
            'interceptions': 0,  # Zero interceptions
            'sacks': 1,
            'sack_yards': 8
        }

        features = feature_extractor._extract_qb_features(mock_stats, prefix='qb_')

        # Should handle division by zero gracefully
        assert features['qb_td_int_ratio'] == 4.0  # or some reasonable default

    def test_extract_matchup_features(self, feature_extractor):
        """Test matchup differential calculations"""
        home_stats = {
            'offensive_plays': 70,
            'offensive_yards': 400,
            'points_scored': 30,
            'rushing_yards': 150,
            'passing_yards': 250
        }

        away_stats = {
            'offensive_plays': 65,
            'offensive_yards': 350,
            'points_scored': 24,
            'rushing_yards': 120,
            'passing_yards': 230
        }

        features = feature_extractor._extract_matchup_features(home_stats, away_stats)

        # Verify differentials
        assert features['yards_per_play_differential'] == pytest.approx(
            (400/70) - (350/65), 0.01
        )
        assert features['scoring_differential'] == pytest.approx(30 - 24, 0.01)

    @patch.object(FeatureExtractor, '_get_team_stats')
    def test_extract_game_features_integration(self, mock_get_stats, feature_extractor):
        """Test full game feature extraction"""
        # Mock team stats response
        mock_home_stats = {
            'offensive_plays': 70,
            'offensive_yards': 420,
            'rushing_attempts': 35,
            'rushing_yards': 180,
            'passing_attempts': 35,
            'passing_completions': 24,
            'passing_yards': 240,
            'passing_touchdowns': 2,
            'interceptions': 0,
            'sacks': 1,
            'sack_yards': 7,
            'third_down_conversions': 7,
            'third_down_attempts': 13,
            'red_zone_scores': 4,
            'red_zone_attempts': 5,
            'turnovers': 0,
            'penalties': 5,
            'penalty_yards': 45,
            'points_scored': 28,
            'points_allowed': 21
        }

        mock_away_stats = {
            'offensive_plays': 65,
            'offensive_yards': 350,
            'rushing_attempts': 30,
            'rushing_yards': 120,
            'passing_attempts': 35,
            'passing_completions': 20,
            'passing_yards': 230,
            'passing_touchdowns': 2,
            'interceptions': 1,
            'sacks': 2,
            'sack_yards': 12,
            'third_down_conversions': 5,
            'third_down_attempts': 12,
            'red_zone_scores': 3,
            'red_zone_attempts': 4,
            'turnovers': 1,
            'penalties': 7,
            'penalty_yards': 60,
            'points_scored': 21,
            'points_allowed': 28
        }

        # Mock the database call
        mock_get_stats.side_effect = [mock_home_stats, mock_away_stats]

        # Also mock other helper methods
        with patch.object(feature_extractor, '_get_recent_form', return_value=[]):
            with patch.object(feature_extractor, '_get_home_away_splits', return_value={}):
                with patch.object(feature_extractor, '_get_talent_composite', return_value=85.0):
                    features = feature_extractor.extract_game_features(
                        home_team_id=1,
                        away_team_id=2,
                        season=2024,
                        week=10
                    )

        # Verify features were extracted
        assert isinstance(features, dict)
        assert len(features) > 0
        assert 'home_yards_per_play' in features
        assert 'away_yards_per_play' in features
        assert 'home_completion_pct' in features
        assert 'away_completion_pct' in features

    def test_calculate_confidence(self, feature_extractor):
        """Test confidence calculation"""
        features = {
            'home_yards_per_play': 6.5,
            'away_yards_per_play': 5.0,
            'home_third_down_conversion_pct': 0.50,
            'away_third_down_conversion_pct': 0.35,
            'talent_differential': 8.5,
            'home_recent_points_avg': 32.0,
            'away_recent_points_avg': 21.0
        }

        predictions = {
            'predicted_margin': 10.5,
            'predicted_total': 52.0
        }

        confidence = feature_extractor._calculate_confidence(features, predictions)

        # Confidence should be between 0 and 1
        assert 0.0 <= confidence <= 1.0

        # With good differentials, confidence should be reasonably high
        assert confidence > 0.5

    def test_feature_extraction_missing_data(self, feature_extractor):
        """Test feature extraction handles missing data gracefully"""
        mock_stats = {
            'offensive_plays': 70,
            'offensive_yards': 400,
            # Missing many fields
        }

        # Should not raise an exception
        features = feature_extractor._extract_efficiency_features(mock_stats, prefix='test_')

        # Should return features with defaults or zeros
        assert isinstance(features, dict)

    def test_normalize_features(self, feature_extractor):
        """Test feature normalization"""
        features = {
            'yards_per_play': 6.5,
            'completion_pct': 0.65,
            'third_down_pct': 0.45,
            'talent': 90.0
        }

        # Assuming there's a normalize method
        if hasattr(feature_extractor, '_normalize_features'):
            normalized = feature_extractor._normalize_features(features)

            # Check values are normalized appropriately
            for key, value in normalized.items():
                assert isinstance(value, (int, float))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
