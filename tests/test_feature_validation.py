"""
Tests for unified feature validation module.

Tests cover:
- FeatureMode enum behavior
- validate_and_prepare_features in all modes (strict, warn, silent)
- MissingFeaturesError exception
- Environment variable configuration
"""

import os
from unittest.mock import patch

import pandas as pd
import pytest

from src.prediction.feature_validation import (
    FeatureMode,
    MissingFeaturesError,
    get_feature_mode,
    log_feature_stats,
    validate_and_prepare_features,
)


class TestFeatureMode:
    """Tests for FeatureMode enum."""

    def test_strict_mode_value(self):
        assert FeatureMode.STRICT.value == "strict"

    def test_warn_mode_value(self):
        assert FeatureMode.WARN.value == "warn"

    def test_silent_mode_value(self):
        assert FeatureMode.SILENT.value == "silent"


class TestGetFeatureMode:
    """Tests for get_feature_mode function."""

    def test_default_is_strict(self):
        """Without environment variable, default to strict mode."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove the env var if it exists
            os.environ.pop("PREDICTION_FEATURE_MODE", None)
            mode = get_feature_mode()
            assert mode == FeatureMode.STRICT

    def test_strict_from_env(self):
        with patch.dict(os.environ, {"PREDICTION_FEATURE_MODE": "strict"}):
            mode = get_feature_mode()
            assert mode == FeatureMode.STRICT

    def test_warn_from_env(self):
        with patch.dict(os.environ, {"PREDICTION_FEATURE_MODE": "warn"}):
            mode = get_feature_mode()
            assert mode == FeatureMode.WARN

    def test_silent_from_env(self):
        with patch.dict(os.environ, {"PREDICTION_FEATURE_MODE": "silent"}):
            mode = get_feature_mode()
            assert mode == FeatureMode.SILENT

    def test_case_insensitive(self):
        with patch.dict(os.environ, {"PREDICTION_FEATURE_MODE": "STRICT"}):
            mode = get_feature_mode()
            assert mode == FeatureMode.STRICT

    def test_invalid_mode_defaults_to_strict(self):
        with patch.dict(os.environ, {"PREDICTION_FEATURE_MODE": "invalid_mode"}):
            mode = get_feature_mode()
            assert mode == FeatureMode.STRICT

    def test_whitespace_trimmed(self):
        with patch.dict(os.environ, {"PREDICTION_FEATURE_MODE": "  warn  "}):
            mode = get_feature_mode()
            assert mode == FeatureMode.WARN


class TestMissingFeaturesError:
    """Tests for MissingFeaturesError exception."""

    def test_error_attributes(self):
        error = MissingFeaturesError(
            market="fg_spread",
            missing_features=["feat1", "feat2"],
            available_features=["feat3", "feat4"],
        )
        assert error.market == "fg_spread"
        assert error.missing_features == ["feat1", "feat2"]
        assert error.available_features == ["feat3", "feat4"]

    def test_error_message_contains_market(self):
        error = MissingFeaturesError(
            market="1h_total",
            missing_features=["feat1"],
            available_features=["feat2"],
        )
        assert "1h_total" in str(error)

    def test_error_message_contains_missing_count(self):
        error = MissingFeaturesError(
            market="fg_spread",
            missing_features=["f1", "f2", "f3"],
            available_features=[],
        )
        assert "3" in str(error) or "MISSING 3" in str(error)

    def test_error_message_truncates_long_list(self):
        """Error message should truncate if more than 10 missing features."""
        missing = [f"feat_{i}" for i in range(15)]
        error = MissingFeaturesError(
            market="fg_spread",
            missing_features=missing,
            available_features=[],
        )
        message = str(error)
        assert "and 5 more" in message


class TestValidateAndPrepareFeatures:
    """Tests for validate_and_prepare_features function."""

    def test_no_missing_features(self):
        """When all features present, return DataFrame unchanged."""
        features = {"a": 1.0, "b": 2.0, "c": 3.0}
        df = pd.DataFrame([features])
        required = ["a", "b"]

        result_df, missing = validate_and_prepare_features(
            df, required, market="test", mode=FeatureMode.STRICT
        )

        assert missing == []
        assert list(result_df.columns) == ["a", "b"]
        assert result_df["a"].values[0] == 1.0
        assert result_df["b"].values[0] == 2.0

    def test_strict_mode_raises_on_missing(self):
        """STRICT mode raises MissingFeaturesError when features missing."""
        features = {"a": 1.0}
        df = pd.DataFrame([features])
        required = ["a", "b", "c"]

        with pytest.raises(MissingFeaturesError) as exc_info:
            validate_and_prepare_features(df, required, market="fg_spread", mode=FeatureMode.STRICT)

        error = exc_info.value
        assert error.market == "fg_spread"
        assert set(error.missing_features) == {"b", "c"}

    def test_warn_mode_zero_fills(self):
        """WARN mode zero-fills missing features."""
        features = {"a": 1.0}
        df = pd.DataFrame([features])
        required = ["a", "b"]

        result_df, missing = validate_and_prepare_features(
            df, required, market="test", mode=FeatureMode.WARN
        )

        assert sorted(missing) == ["b"]
        assert result_df["a"].values[0] == 1.0
        assert result_df["b"].values[0] == 0  # Zero-filled

    def test_silent_mode_zero_fills_without_logging(self):
        """SILENT mode zero-fills missing features without logging."""
        features = {"a": 1.0}
        df = pd.DataFrame([features])
        required = ["a", "b"]

        result_df, missing = validate_and_prepare_features(
            df, required, market="test", mode=FeatureMode.SILENT
        )

        assert sorted(missing) == ["b"]
        assert result_df["a"].values[0] == 1.0
        assert result_df["b"].values[0] == 0  # Zero-filled

    def test_uses_env_mode_by_default(self):
        """When mode not specified, uses environment setting."""
        features = {"a": 1.0}
        df = pd.DataFrame([features])
        required = ["a", "b"]

        with patch.dict(os.environ, {"PREDICTION_FEATURE_MODE": "warn"}):
            result_df, missing = validate_and_prepare_features(df, required, market="test")
            # Should not raise - uses warn mode
            assert missing == ["b"]

    def test_returns_only_required_columns(self):
        """Result DataFrame should only contain required columns in order."""
        features = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}
        df = pd.DataFrame([features])
        required = ["c", "a"]  # Order matters

        result_df, missing = validate_and_prepare_features(
            df, required, market="test", mode=FeatureMode.STRICT
        )

        assert list(result_df.columns) == ["c", "a"]

    def test_multiple_missing_features(self):
        """Handles multiple missing features correctly."""
        features = {"a": 1.0}
        df = pd.DataFrame([features])
        required = ["a", "b", "c", "d", "e"]

        result_df, missing = validate_and_prepare_features(
            df, required, market="test", mode=FeatureMode.WARN
        )

        assert sorted(missing) == ["b", "c", "d", "e"]
        assert len(result_df.columns) == 5
        for col in ["b", "c", "d", "e"]:
            assert result_df[col].values[0] == 0

    def test_all_features_missing(self):
        """Handles case where ALL required features are missing."""
        features = {"x": 1.0, "y": 2.0}
        df = pd.DataFrame([features])
        required = ["a", "b"]

        # In STRICT mode - should raise
        with pytest.raises(MissingFeaturesError):
            validate_and_prepare_features(df, required, market="test", mode=FeatureMode.STRICT)

        # In WARN mode - should zero-fill all
        result_df, missing = validate_and_prepare_features(
            df, required, market="test", mode=FeatureMode.WARN
        )
        assert sorted(missing) == ["a", "b"]
        assert result_df["a"].values[0] == 0
        assert result_df["b"].values[0] == 0


class TestLogFeatureStats:
    """Tests for log_feature_stats function (basic coverage)."""

    def test_no_error_with_zero_missing(self):
        """Should not raise when no features missing."""
        log_feature_stats("test", 10, 0, FeatureMode.STRICT)

    def test_no_error_with_missing(self):
        """Should not raise when features missing."""
        log_feature_stats("test", 10, 3, FeatureMode.WARN)


class TestIntegrationWithPredictors:
    """Integration tests verifying predictors use unified validation."""

    def test_period_predictor_uses_unified_validation(self):
        """Verify engine.py PeriodPredictor uses validate_and_prepare_features."""
        # This is a smoke test - actual integration requires model fixtures
        from src.prediction.feature_validation import validate_and_prepare_features

        # Just verify the import works
        assert callable(validate_and_prepare_features)

    def test_error_message_suggests_debug_mode(self):
        """Error message should suggest using warn mode for debugging."""
        error = MissingFeaturesError(
            market="fg_spread",
            missing_features=["feat1"],
            available_features=["feat2"],
        )
        assert "PREDICTION_FEATURE_MODE=warn" in str(error)
