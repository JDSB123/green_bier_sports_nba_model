"""
Feature configuration helpers for NBA prediction model training.

All feature definitions live in unified_features.py (single source of truth).
This module provides training-specific helper functions.
"""
from __future__ import annotations
import logging
from typing import List

from src.modeling.unified_features import (
    UNIFIED_FEATURE_NAMES,
    FEATURE_DEFAULTS,
    REQUIRED_FEATURES,
    LEAKY_FEATURES_BLACKLIST,
)

logger = logging.getLogger(__name__)


def get_spreads_features() -> List[str]:
    """Get all features for spreads prediction model."""
    return UNIFIED_FEATURE_NAMES.copy()


def get_totals_features() -> List[str]:
    """Get all features for totals prediction model."""
    return UNIFIED_FEATURE_NAMES.copy()


def get_all_features() -> List[str]:
    """Get complete list of all available features."""
    return UNIFIED_FEATURE_NAMES.copy()


def remove_leaky_features(features: List[str]) -> List[str]:
    """
    Remove any features that are known to cause data leakage.

    CRITICAL: Call this before training any model to ensure no leaky features
    are included.
    """
    blacklist_set = set(LEAKY_FEATURES_BLACKLIST)
    clean_features = [f for f in features if f not in blacklist_set]

    removed = set(features) - set(clean_features)
    if removed:
        logger.warning(
            f"LEAKAGE PREVENTION: Removed {len(removed)} leaky features: {sorted(removed)}"
        )

    return clean_features


def filter_available_features(
    requested: List[str],
    available_columns: List[str],
    min_required_pct: float = 0.3,
    critical_features: List[str] = None,
    exclude_leaky: bool = True,
) -> List[str]:
    """
    Filter feature list to only those present in the data.

    Args:
        requested: List of requested feature names
        available_columns: List of column names actually present in data
        min_required_pct: Minimum % of requested features that must be available
        critical_features: List of feature names that MUST be present
        exclude_leaky: If True, automatically remove leaky features

    Returns:
        List of features that are both requested and available

    Raises:
        ValueError: If critical features are missing or insufficient features available
    """
    available_set = set(available_columns)
    requested_set = set(requested)

    missing = requested_set - available_set
    available = requested_set & available_set

    if missing:
        missing_pct = len(missing) / len(requested) * 100
        logger.info(
            f"Feature filtering: {len(available)}/{len(requested)} features available "
            f"({len(missing)} missing)"
        )
        if missing_pct > 50:
            logger.warning(f"Many features missing ({missing_pct:.0f}%): {sorted(list(missing)[:10])}...")

    if critical_features is None:
        critical_features = REQUIRED_FEATURES

    if critical_features:
        critical_set = set(critical_features)
        missing_critical = critical_set - available_set
        if missing_critical:
            raise ValueError(
                f"CRITICAL FEATURES MISSING: {sorted(missing_critical)}. "
                f"Cannot proceed without these features."
            )

    available_pct = len(available) / len(requested) if requested else 0
    if available_pct < min_required_pct:
        raise ValueError(
            f"Insufficient features available: {len(available)}/{len(requested)} "
            f"({available_pct:.1%} < {min_required_pct:.1%} required). "
            f"Missing: {sorted(list(missing)[:20])}..."
        )

    logger.info(f"Using {len(available)}/{len(requested)} requested features ({available_pct:.1%})")

    result = [f for f in requested if f in available_set]

    if exclude_leaky:
        result = remove_leaky_features(result)

    return result


__all__ = [
    "get_spreads_features",
    "get_totals_features",
    "get_all_features",
    "filter_available_features",
    "remove_leaky_features",
    "UNIFIED_FEATURE_NAMES",
    "FEATURE_DEFAULTS",
    "REQUIRED_FEATURES",
    "LEAKY_FEATURES_BLACKLIST",
]
