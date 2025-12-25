"""
Feature configuration for NBA prediction models.

Centralizes feature definitions used across training, backtesting, and prediction.
This ensures consistency and makes it easy to add/remove features in one place.
"""
from __future__ import annotations
import logging
from typing import List

logger = logging.getLogger(__name__)


# =============================================================================
# FEATURE GROUPS
# =============================================================================

# Core team performance stats
CORE_TEAM_FEATURES = [
    "home_ppg", "home_papg", "home_avg_margin",
    "away_ppg", "away_papg", "away_avg_margin",
    "home_elo", "away_elo", "elo_diff",
    "predicted_margin", "win_pct_diff", "ppg_diff",
]

# Rest and scheduling features
REST_FEATURES = [
    "home_rest_days", "away_rest_days", "rest_advantage",
    "home_b2b", "away_b2b",
]

# Travel and fatigue features (NEW in v1.3)
TRAVEL_FEATURES = [
    "away_travel_distance", "away_timezone_change", "away_travel_fatigue",
    "is_away_long_trip", "is_away_cross_country",
    "away_b2b_travel_penalty", "travel_advantage",
    "home_court_advantage",  # Team-specific HCA (Denver ~4.2, etc.)
]

# Injury impact features
INJURY_FEATURES = [
    "home_injury_spread_impact", "away_injury_spread_impact",
    "injury_spread_diff", "home_star_out", "away_star_out",
    "home_injury_total_impact", "away_injury_total_impact",
    "injury_total_diff",
]

# RLM (Reverse Line Movement) and sharp money features
RLM_FEATURES = [
    "is_rlm_spread", "sharp_side_spread",
    "spread_public_home_pct", "spread_ticket_money_diff",
    "spread_movement",
    "is_rlm_total", "sharp_side_total",
    "over_public_pct", "total_ticket_money_diff",
]

# Head-to-head features
H2H_FEATURES = [
    "h2h_games", "h2h_margin", "h2h_win_rate",
]

# ATS (Against The Spread) features
ATS_FEATURES = [
    "home_ats_pct", "away_ats_pct",
]

# Totals-specific features
TOTALS_FEATURES = [
    "home_total_ppg", "away_total_ppg",
    "predicted_total",
]


# =============================================================================
# MODEL-SPECIFIC FEATURE SETS
# =============================================================================

def get_spreads_features() -> List[str]:
    """Get all features for spreads prediction model."""
    return (
        CORE_TEAM_FEATURES +
        REST_FEATURES +
        TRAVEL_FEATURES +  # NEW: Travel/fatigue features
        ATS_FEATURES +
        [f for f in INJURY_FEATURES if "spread" in f or "star" in f] +
        [f for f in RLM_FEATURES if "spread" in f]
    )


def get_totals_features() -> List[str]:
    """Get all features for totals prediction model."""
    return (
        ["home_ppg", "home_papg", "away_ppg", "away_papg", "home_elo", "away_elo"] +
        TOTALS_FEATURES +
        ["home_rest_days", "away_rest_days", "home_b2b", "away_b2b"] +
        ["away_travel_fatigue", "travel_advantage"] +  # NEW: Travel impacts pace
        [f for f in INJURY_FEATURES if "total" in f] +
        [f for f in RLM_FEATURES if "total" in f or "over" in f]
    )


def get_moneyline_features() -> List[str]:
    """Get all features for moneyline prediction model."""
    return [
        "home_ppg", "away_ppg", "home_avg_margin", "away_avg_margin",
        "home_elo", "away_elo", "elo_diff", "predicted_margin", "predicted_total",
    ]


def get_all_features() -> List[str]:
    """Get complete list of all available features."""
    all_features = set(
        CORE_TEAM_FEATURES +
        REST_FEATURES +
        TRAVEL_FEATURES +
        INJURY_FEATURES +
        RLM_FEATURES +
        H2H_FEATURES +
        ATS_FEATURES +
        TOTALS_FEATURES
    )
    return sorted(list(all_features))


def filter_available_features(
    requested: List[str],
    available_columns: List[str],
    min_required_pct: float = 0.5,
    critical_features: List[str] = None,
) -> List[str]:
    """
    Filter feature list to only those present in the data.

    Args:
        requested: List of requested feature names
        available_columns: List of column names actually present in data
        min_required_pct: Minimum % of requested features that must be available (default 0.5)
        critical_features: List of feature names that MUST be present (raises error if missing)

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
        logger.warning(
            f"Feature filtering: {len(missing)}/{len(requested)} ({missing_pct:.1f}%) features unavailable"
        )
        logger.warning(f"Missing features: {sorted(missing)}")

    # Check for critical features
    if critical_features:
        critical_set = set(critical_features)
        missing_critical = critical_set - available_set
        if missing_critical:
            raise ValueError(
                f"CRITICAL FEATURES MISSING: {sorted(missing_critical)}. "
                f"Cannot proceed without these features. Available: {sorted(available_set)}"
            )

    # Check if we have enough features
    available_pct = len(available) / len(requested)
    if available_pct < min_required_pct:
        raise ValueError(
            f"Insufficient features available: {len(available)}/{len(requested)} "
            f"({available_pct:.1%} < {min_required_pct:.1%} required). "
            f"Missing: {sorted(missing)}"
        )

    logger.info(f"Using {len(available)}/{len(requested)} requested features ({available_pct:.1%})")

    # Return in original request order
    return [f for f in requested if f in available_set]
