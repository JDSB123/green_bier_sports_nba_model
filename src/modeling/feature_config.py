"""
Feature configuration for NBA prediction models.

SINGLE SOURCE OF TRUTH: All features are now defined in unified_features.py

This module provides backward-compatible imports and helper functions.
For new code, use:
    from src.modeling.unified_features import (
        UNIFIED_FEATURE_NAMES,
        MODEL_REGISTRY,
        get_model_config,
        validate_features,
    )
"""
from __future__ import annotations
import logging
from typing import List

# Import from unified source of truth
from src.modeling.unified_features import (
    UNIFIED_FEATURE_NAMES,
    FEATURE_DEFAULTS,
    REQUIRED_FEATURES,
    FeatureCategory,
    get_features_by_category,
    validate_features,
    LEAKY_FEATURES_BLACKLIST,
)

logger = logging.getLogger(__name__)


# =============================================================================
# BACKWARD COMPATIBLE FEATURE GROUPS
# =============================================================================
# These are explicitly defined to match original semantics exactly

CORE_TEAM_FEATURES = get_features_by_category(FeatureCategory.CORE)
REST_FEATURES = get_features_by_category(FeatureCategory.REST)
TRAVEL_FEATURES = get_features_by_category(FeatureCategory.TRAVEL)
INJURY_FEATURES = get_features_by_category(FeatureCategory.INJURY)
H2H_FEATURES = get_features_by_category(FeatureCategory.H2H)

# RLM_FEATURES - Live predictions fetch real splits from Action Network
# Training data has placeholders, but models include these for live use
RLM_FEATURES = [
    "is_rlm_spread", "sharp_side_spread",
    "spread_public_home_pct", "spread_ticket_money_diff",
    "spread_movement",
    "is_rlm_total", "sharp_side_total",
    "over_public_pct", "total_ticket_money_diff",
]

# ATS_FEATURES - Against The Spread performance (separate from RLM)
ATS_FEATURES = [
    "home_ats_pct", "away_ats_pct",
]

# TOTALS_FEATURES - Explicitly include predicted_total (original had 3 features)
TOTALS_FEATURES = [
    "home_total_ppg", "away_total_ppg",
    "predicted_total",
]


# =============================================================================
# UNIFIED FEATURE FUNCTIONS
# =============================================================================

def get_spreads_features() -> List[str]:
    """Get all features for spreads prediction model.

    Now returns the UNIFIED feature set (same as totals).
    """
    return UNIFIED_FEATURE_NAMES.copy()


def get_totals_features() -> List[str]:
    """Get all features for totals prediction model.

    Now returns the UNIFIED feature set (same as spreads).
    """
    return UNIFIED_FEATURE_NAMES.copy()


def get_all_features() -> List[str]:
    """Get complete list of all available features."""
    return UNIFIED_FEATURE_NAMES.copy()


def remove_leaky_features(features: List[str]) -> List[str]:
    """
    Remove any features that are known to cause data leakage.

    CRITICAL: Call this before training any model to ensure no leaky features
    are included. Leaky features are those computed from the game's actual
    outcome (box scores, final scores) rather than pre-game data.

    Args:
        features: List of feature names to filter

    Returns:
        List with leaky features removed
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
    min_required_pct: float = 0.3,  # Reduced from 0.5 since we have many features
    critical_features: List[str] = None,
    exclude_leaky: bool = True,  # NEW: Auto-exclude leaky features
) -> List[str]:
    """
    Filter feature list to only those present in the data.

    Args:
        requested: List of requested feature names
        available_columns: List of column names actually present in data
        min_required_pct: Minimum % of requested features that must be available
        critical_features: List of feature names that MUST be present
        exclude_leaky: If True, automatically remove features from LEAKY_FEATURES_BLACKLIST

    Returns:
        List of features that are both requested and available

    Raises:
        ValueError: If critical features are missing or insufficient features available
    """
    available_set = set(available_columns)
    requested_set = set(requested)

    # Find missing features
    missing = requested_set - available_set
    available = requested_set & available_set

    # Log the filtering results
    if missing:
        missing_pct = len(missing) / len(requested) * 100
        logger.info(
            f"Feature filtering: {len(available)}/{len(requested)} features available "
            f"({len(missing)} missing)"
        )
        if missing_pct > 50:
            logger.warning(f"Many features missing ({missing_pct:.0f}%): {sorted(list(missing)[:10])}...")

    # Use default critical features if not specified
    if critical_features is None:
        critical_features = REQUIRED_FEATURES

    # Check for critical features
    if critical_features:
        critical_set = set(critical_features)
        missing_critical = critical_set - available_set
        if missing_critical:
            raise ValueError(
                f"CRITICAL FEATURES MISSING: {sorted(missing_critical)}. "
                f"Cannot proceed without these features."
            )

    # Check if we have enough features
    available_pct = len(available) / len(requested) if requested else 0
    if available_pct < min_required_pct:
        raise ValueError(
            f"Insufficient features available: {len(available)}/{len(requested)} "
            f"({available_pct:.1%} < {min_required_pct:.1%} required). "
            f"Missing: {sorted(list(missing)[:20])}..."
        )

    logger.info(f"Using {len(available)}/{len(requested)} requested features ({available_pct:.1%})")

    # Build final feature list in original request order
    result = [f for f in requested if f in available_set]

    # CRITICAL: Remove leaky features if enabled
    if exclude_leaky:
        result = remove_leaky_features(result)

    return result


# =============================================================================
# RE-EXPORTS FOR BACKWARD COMPATIBILITY
# =============================================================================

__all__ = [
    "CORE_TEAM_FEATURES",
    "REST_FEATURES",
    "TRAVEL_FEATURES",
    "INJURY_FEATURES",
    "RLM_FEATURES",
    "H2H_FEATURES",
    "ATS_FEATURES",
    "TOTALS_FEATURES",
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
