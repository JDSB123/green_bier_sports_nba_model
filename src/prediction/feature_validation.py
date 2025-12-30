"""
Unified Feature Validation for NBA Prediction Engine.

This module provides consistent feature validation behavior across ALL predictors:
- PeriodPredictor (engine.py)
- SpreadPredictor (spreads/predictor.py)
- TotalPredictor (totals/predictor.py)


CONFIGURATION:
    Environment variable: PREDICTION_FEATURE_MODE
    Values:
        - "strict" (default): Raise ValueError on missing features - RECOMMENDED FOR PRODUCTION
        - "warn": Log warning and zero-fill missing features - for debugging
        - "silent": Zero-fill without logging - NOT RECOMMENDED

RATIONALE:
    Previous implementation had inconsistent behavior:
    - engine.py used STRICT mode (fail on missing)
    - legacy predictors used PERMISSIVE mode (zero-fill silently)

    This caused the same prediction to succeed or fail depending on code path.
    Unified behavior ensures predictable, debuggable predictions.
"""
import os
import logging
from typing import List, Set, Tuple, Literal
from enum import Enum

import pandas as pd

logger = logging.getLogger(__name__)


class FeatureMode(Enum):
    """Feature validation mode."""
    STRICT = "strict"   # Raise ValueError on missing features
    WARN = "warn"       # Log warning, zero-fill missing features
    SILENT = "silent"   # Zero-fill without logging (NOT RECOMMENDED)


class MissingFeaturesError(ValueError):
    """
    Raised when required features are missing in STRICT mode.

    Attributes:
        market: The market type (e.g., "fg_spread", "1h_total")
        missing_features: List of missing feature names
        available_features: List of available feature names
    """
    def __init__(
        self,
        market: str,
        missing_features: List[str],
        available_features: List[str],
    ):
        self.market = market
        self.missing_features = missing_features
        self.available_features = available_features

        # Truncate for readability
        missing_display = sorted(missing_features)[:10]
        missing_suffix = f"... and {len(missing_features) - 10} more" if len(missing_features) > 10 else ""

        message = (
            f"[{market}] MISSING {len(missing_features)} REQUIRED FEATURES: {missing_display}{missing_suffix}. "
            f"Feature pipeline is broken - fix data ingestion, do not zero-fill. "
            f"Set PREDICTION_FEATURE_MODE=warn to debug with zero-filled features."
        )
        super().__init__(message)


def get_feature_mode() -> FeatureMode:
    """
    Get the current feature validation mode from environment.

    Returns:
        FeatureMode enum value (defaults to STRICT)
    """
    mode_str = os.getenv("PREDICTION_FEATURE_MODE", "strict").lower().strip()

    try:
        return FeatureMode(mode_str)
    except ValueError:
        logger.warning(
            f"Invalid PREDICTION_FEATURE_MODE='{mode_str}'. "
            f"Valid options: strict, warn, silent. Defaulting to 'strict'."
        )
        return FeatureMode.STRICT


def validate_and_prepare_features(
    feature_df: pd.DataFrame,
    required_features: List[str],
    market: str,
    mode: FeatureMode | None = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Validate features and prepare DataFrame for prediction.

    This is the SINGLE source of truth for feature validation across all predictors.

    Args:
        feature_df: DataFrame containing feature values (single row)
        required_features: List of required feature column names
        market: Market identifier for error messages (e.g., "fg_spread", "1h_total")
        mode: Override feature mode (defaults to environment setting)

    Returns:
        Tuple of (prepared_df, missing_features_list)
        - prepared_df: DataFrame with required features (zero-filled if needed)
        - missing_features_list: List of features that were missing (empty if none)

    Raises:
        MissingFeaturesError: In STRICT mode when features are missing
    """
    if mode is None:
        mode = get_feature_mode()

    # Find missing features
    available = set(feature_df.columns)
    required = set(required_features)
    missing = required - available
    missing_list = sorted(missing)

    # No missing features - return as-is
    if not missing:
        return feature_df[required_features], []

    # Handle based on mode
    if mode == FeatureMode.STRICT:
        raise MissingFeaturesError(
            market=market,
            missing_features=missing_list,
            available_features=sorted(available),
        )

    elif mode == FeatureMode.WARN:
        logger.warning(
            f"[{market}] Zero-filling {len(missing)} missing features: {missing_list[:5]}..."
            f"{' (and more)' if len(missing) > 5 else ''} "
            f"Set PREDICTION_FEATURE_MODE=strict to fail on missing features."
        )

    # mode == SILENT or WARN: zero-fill missing features
    for col in missing:
        feature_df[col] = 0

    return feature_df[required_features], missing_list


def log_feature_stats(
    market: str,
    total_features: int,
    missing_count: int,
    mode: FeatureMode,
) -> None:
    """
    Log feature statistics for debugging.

    Args:
        market: Market identifier
        total_features: Total required features
        missing_count: Number of missing features
        mode: Current feature mode
    """
    if missing_count == 0:
        logger.debug(f"[{market}] All {total_features} features present")
    else:
        pct_missing = (missing_count / total_features) * 100
        logger.info(
            f"[{market}] {missing_count}/{total_features} features missing ({pct_missing:.1f}%) "
            f"[mode={mode.value}]"
        )
