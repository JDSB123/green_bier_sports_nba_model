"""
UNIFIED FEATURE SPECIFICATION - SINGLE SOURCE OF TRUTH

This module defines ALL features used by the NBA prediction models.
Both 1H and FG markets use IDENTICAL feature sets for consistency.

Architecture:
- All 4 models (1h_spread, 1h_total, fg_spread, fg_total) use the SAME features
- Feature values are computed fresh at prediction time (no caching)
- Betting splits are fetched live from Action Network
- No stale data - everything is computed on demand

Usage:
    from src.modeling.unified_features import UNIFIED_FEATURES, MODEL_REGISTRY
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


# =============================================================================
# LEAKY FEATURES BLACKLIST - NEVER USE THESE IN TRAINING
# =============================================================================
# These features are computed FROM the game's actual outcome (box scores, final scores)
# and therefore contain future information that leaks the answer to the model.
#
# IMPORTANT DISTINCTION (2026-01-20 FIX):
# - LEAKY (blacklist): Features derived from THIS GAME's actual result (scores, box scores)
# - NOT LEAKY (allow): Model's own predictions computed BEFORE the game from historical data
#   e.g., predicted_margin, predicted_total are computed at inference time from live API stats
#
# The 2026-01-18 commit incorrectly blacklisted predicted_margin/predicted_total which are
# MODEL OUTPUTS needed for edge calculation, not leaky inputs. This broke FG predictions.

LEAKY_FEATURES_BLACKLIST = [
    # ==========================================================================
    # ACTUAL GAME RESULTS (definitely leaky - these are the answers)
    # ==========================================================================
    "home_score", "away_score",
    "home_1h", "away_1h",
    "home_q1", "away_q1",
    "fg_margin", "actual_margin",

    # ==========================================================================
    # BOX SCORE STATS FROM THIS GAME (leaky - computed from final stats)
    # ==========================================================================
    "home_off_rtg", "away_off_rtg",  # Game-specific offensive rating
    "home_def_rtg", "away_def_rtg",  # Game-specific defensive rating
    "home_fgm", "away_fgm",
    "home_dreb", "away_dreb",
    "home_efg_pct", "away_efg_pct",

    # ==========================================================================
    # DO NOT BLACKLIST THESE - They are model outputs, not leaky inputs:
    # - predicted_margin: Model's prediction of home team margin (computed pre-game)
    # - predicted_total: Model's prediction of total points (computed pre-game)
    # - spread_vs_predicted: Edge calculation (predicted_margin vs spread_line)
    # - total_vs_predicted: Edge calculation (predicted_total vs total_line)
    # - home_ppg, away_ppg: Rolling averages from PRIOR games (properly lagged)
    # - home_elo, away_elo: Pre-game Elo ratings (updated AFTER prior games)
    # ==========================================================================
]


# =============================================================================
# FEATURE CATEGORIES
# =============================================================================

class FeatureCategory(Enum):
    """Feature categories for organization and documentation."""
    CORE = "core"           # Basic team stats (PPG, margin, win%)
    EFFICIENCY = "efficiency"  # Advanced stats (ORtg, DRtg, net rating)
    FORM = "form"           # Recent performance (L5, L10 trends)
    REST = "rest"           # Rest days, B2B
    TRAVEL = "travel"       # Distance, timezone, fatigue
    INJURY = "injury"       # Injury impact on scoring
    BETTING = "betting"     # Public betting splits, RLM
    H2H = "h2h"             # Head-to-head history
    ELO = "elo"             # ELO ratings
    MARKET = "market"       # Spread/total lines


# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================

@dataclass(frozen=True)
class Feature:
    """Definition of a single feature."""
    name: str
    category: FeatureCategory
    description: str
    default: float = 0.0
    required: bool = False  # If True, prediction fails without this feature


# -----------------------------------------------------------------------------
# CORE TEAM STATS - Basic scoring and win rates
# -----------------------------------------------------------------------------
CORE_FEATURES = [
    Feature("home_ppg", FeatureCategory.CORE,
            "Home team points per game", default=110.0, required=True),
    Feature("home_papg", FeatureCategory.CORE,
            "Home team points allowed per game", default=110.0, required=True),
    Feature("home_margin", FeatureCategory.CORE,
            "Home team average margin", default=0.0, required=True),
    Feature("away_ppg", FeatureCategory.CORE,
            "Away team points per game", default=110.0, required=True),
    Feature("away_papg", FeatureCategory.CORE,
            "Away team points allowed per game", default=110.0, required=True),
    Feature("away_margin", FeatureCategory.CORE,
            "Away team average margin", default=0.0, required=True),
    Feature("home_win_pct", FeatureCategory.CORE,
            "Home team win percentage", default=0.5, required=True),
    Feature("away_win_pct", FeatureCategory.CORE,
            "Away team win percentage", default=0.5, required=True),
    Feature("win_pct_diff", FeatureCategory.CORE,
            "Win percentage differential (home - away)", default=0.0),
    Feature("ppg_diff", FeatureCategory.CORE,
            "PPG differential (home - away)", default=0.0),
    Feature("home_pace", FeatureCategory.CORE,
            "Home team pace (possessions per game)", default=100.0),
    Feature("away_pace", FeatureCategory.CORE,
            "Away team pace (possessions per game)", default=100.0),
    Feature("expected_pace", FeatureCategory.CORE,
            "Expected game pace", default=100.0),
    Feature("predicted_margin", FeatureCategory.CORE,
            "Model predicted margin", default=0.0),
    Feature("predicted_total", FeatureCategory.CORE,
            "Model predicted total", default=220.0),
]

# -----------------------------------------------------------------------------
# EFFICIENCY RATINGS - LEAK-SAFE PRE-GAME METRICS
# -----------------------------------------------------------------------------
# These are derived from pre-game rolling averages or season-to-date PPG/PAPG
# (see scripts/data_unified_feature_complete.py) and are safe for training.
EFFICIENCY_FEATURES = [
    Feature("home_ortg", FeatureCategory.EFFICIENCY,
            "Home offensive rating (pregame)", default=110.0),
    Feature("home_drtg", FeatureCategory.EFFICIENCY,
            "Home defensive rating (pregame)", default=110.0),
    Feature("home_net_rtg", FeatureCategory.EFFICIENCY,
            "Home net rating (pregame)", default=0.0),
    Feature("away_ortg", FeatureCategory.EFFICIENCY,
            "Away offensive rating (pregame)", default=110.0),
    Feature("away_drtg", FeatureCategory.EFFICIENCY,
            "Away defensive rating (pregame)", default=110.0),
    Feature("away_net_rtg", FeatureCategory.EFFICIENCY,
            "Away net rating (pregame)", default=0.0),
    Feature("net_rating_diff", FeatureCategory.EFFICIENCY,
            "Net rating differential", default=0.0),
]

# -----------------------------------------------------------------------------
# RECENT FORM - Last 5 and 10 game performance
# -----------------------------------------------------------------------------
FORM_FEATURES = [
    Feature("home_l5_margin", FeatureCategory.FORM,
            "Home margin last 5 games", default=0.0),
    Feature("away_l5_margin", FeatureCategory.FORM,
            "Away margin last 5 games", default=0.0),
    Feature("home_l10_margin", FeatureCategory.FORM,
            "Home margin last 10 games", default=0.0),
    Feature("away_l10_margin", FeatureCategory.FORM,
            "Away margin last 10 games", default=0.0),
    Feature("home_form_trend", FeatureCategory.FORM,
            "Home form trend (L5 vs L10)", default=0.0),
    Feature("away_form_trend", FeatureCategory.FORM,
            "Away form trend (L5 vs L10)", default=0.0),
    Feature("home_margin_std", FeatureCategory.FORM,
            "Home margin standard deviation", default=10.0),
    Feature("away_margin_std", FeatureCategory.FORM,
            "Away margin standard deviation", default=10.0),
    Feature("home_score_std", FeatureCategory.FORM,
            "Home scoring standard deviation", default=10.0),
    Feature("away_score_std", FeatureCategory.FORM,
            "Away scoring standard deviation", default=10.0),
]

# -----------------------------------------------------------------------------
# REST AND SCHEDULING
# -----------------------------------------------------------------------------
# NOTE: Feature names must match what features.py generates (home_rest, not home_rest_days)
REST_FEATURES = [
    Feature("home_rest", FeatureCategory.REST,
            "Home team days of rest", default=2.0),
    Feature("away_rest", FeatureCategory.REST,
            "Away team days of rest", default=2.0),
    Feature("rest_diff", FeatureCategory.REST,
            "Rest differential (home - away)", default=0.0),
    Feature("home_b2b", FeatureCategory.REST,
            "Home playing back-to-back (0/1)", default=0.0),
    Feature("away_b2b", FeatureCategory.REST,
            "Away playing back-to-back (0/1)", default=0.0),
    Feature("home_rest_adj", FeatureCategory.REST,
            "Home rest adjustment factor", default=0.0),
    Feature("away_rest_adj", FeatureCategory.REST,
            "Away rest adjustment factor", default=0.0),
    Feature("rest_margin_adj", FeatureCategory.REST,
            "Rest-based margin adjustment", default=0.0),
]

# -----------------------------------------------------------------------------
# TRAVEL AND FATIGUE
# -----------------------------------------------------------------------------
TRAVEL_FEATURES = [
    Feature("away_travel_distance", FeatureCategory.TRAVEL,
            "Away team travel distance (miles)", default=0.0),
    Feature("away_timezone_change", FeatureCategory.TRAVEL,
            "Away team timezone change (hours)", default=0.0),
    Feature("away_travel_fatigue", FeatureCategory.TRAVEL,
            "Away team travel fatigue score", default=0.0),
    Feature("is_away_long_trip", FeatureCategory.TRAVEL,
            "Is away team on long trip (0/1)", default=0.0),
    Feature("is_away_cross_country", FeatureCategory.TRAVEL,
            "Is cross-country travel (0/1)", default=0.0),
    Feature("away_b2b_travel_penalty", FeatureCategory.TRAVEL,
            "B2B + travel compound penalty", default=0.0),
    Feature("travel_advantage", FeatureCategory.TRAVEL,
            "Home travel advantage score", default=0.0),
    Feature("home_court_advantage", FeatureCategory.TRAVEL,
            "Team-specific HCA (e.g., Denver ~4.2)", default=3.0),
    Feature("dynamic_hca", FeatureCategory.TRAVEL,
            "Dynamic home court advantage", default=3.0),
]

# -----------------------------------------------------------------------------
# INJURY IMPACT
# -----------------------------------------------------------------------------
INJURY_FEATURES = [
    Feature("has_injury_data", FeatureCategory.INJURY,
            "Injury data available (0/1)", default=0.0),
    Feature("home_injury_impact_ppg", FeatureCategory.INJURY,
            "Home PPG lost to injuries", default=0.0),
    Feature("away_injury_impact_ppg", FeatureCategory.INJURY,
            "Away PPG lost to injuries", default=0.0),
    Feature("injury_margin_adj", FeatureCategory.INJURY,
            "Injury-based margin adjustment", default=0.0),
    Feature("home_star_out", FeatureCategory.INJURY,
            "Home star player out (0/1)", default=0.0),
    Feature("away_star_out", FeatureCategory.INJURY,
            "Away star player out (0/1)", default=0.0),
    Feature("home_injury_spread_impact", FeatureCategory.INJURY,
            "Home spread impact from injuries", default=0.0),
    Feature("away_injury_spread_impact", FeatureCategory.INJURY,
            "Away spread impact from injuries", default=0.0),
    Feature("injury_spread_diff", FeatureCategory.INJURY,
            "Net injury spread impact", default=0.0),
    Feature("home_injury_total_impact", FeatureCategory.INJURY,
            "Home total impact from injuries", default=0.0),
    Feature("away_injury_total_impact", FeatureCategory.INJURY,
            "Away total impact from injuries", default=0.0),
    Feature("injury_total_diff", FeatureCategory.INJURY,
            "Net injury total impact", default=0.0),
]

# -----------------------------------------------------------------------------
# BETTING SPLITS AND RLM - Sharp money signals
# -----------------------------------------------------------------------------
# NOTE: Training data has placeholder values (50/50), but LIVE predictions
# fetch real splits from Action Network. Models include these features so
# live predictions can leverage real RLM/sharp money signals.
# -----------------------------------------------------------------------------
BETTING_FEATURES = [
    Feature("has_real_splits", FeatureCategory.BETTING,
            "Real splits data available (0/1)", default=0.0),
    # Spread public betting
    Feature("spread_public_home_pct", FeatureCategory.BETTING,
            "% of public on home spread", default=50.0),
    Feature("spread_public_away_pct", FeatureCategory.BETTING,
            "% of public on away spread", default=50.0),
    Feature("spread_money_home_pct", FeatureCategory.BETTING,
            "% of money on home spread", default=50.0),
    Feature("spread_money_away_pct", FeatureCategory.BETTING,
            "% of money on away spread", default=50.0),
    Feature("spread_ticket_money_diff", FeatureCategory.BETTING,
            "Ticket vs money divergence (spread)", default=0.0),
    # Spread line movement
    Feature("spread_open", FeatureCategory.BETTING,
            "Opening spread line", default=0.0),
    Feature("spread_current", FeatureCategory.BETTING,
            "Current spread line", default=0.0),
    Feature("spread_movement", FeatureCategory.BETTING,
            "Spread line movement", default=0.0),
    # Spread RLM signals
    Feature("is_rlm_spread", FeatureCategory.BETTING,
            "Reverse line movement on spread (0/1)", default=0.0),
    Feature("sharp_side_spread", FeatureCategory.BETTING,
            "Sharp money side on spread (-1/0/1)", default=0.0),
    # Total public betting
    Feature("over_public_pct", FeatureCategory.BETTING,
            "% of public on over", default=50.0),
    Feature("under_public_pct", FeatureCategory.BETTING,
            "% of public on under", default=50.0),
    Feature("over_money_pct", FeatureCategory.BETTING,
            "% of money on over", default=50.0),
    Feature("under_money_pct", FeatureCategory.BETTING,
            "% of money on under", default=50.0),
    Feature("total_ticket_money_diff", FeatureCategory.BETTING,
            "Ticket vs money divergence (total)", default=0.0),
    # Total line movement
    Feature("total_open", FeatureCategory.BETTING,
            "Opening total line", default=220.0),
    Feature("total_current", FeatureCategory.BETTING,
            "Current total line", default=220.0),
    Feature("total_movement", FeatureCategory.BETTING,
            "Total line movement", default=0.0),
    # Total RLM signals
    Feature("is_rlm_total", FeatureCategory.BETTING,
            "Reverse line movement on total (0/1)", default=0.0),
    Feature("sharp_side_total", FeatureCategory.BETTING,
            "Sharp money side on total (-1/0/1)", default=0.0),
]

# -----------------------------------------------------------------------------
# HEAD-TO-HEAD HISTORY - LEAK-SAFE (PRE-GAME ONLY)
# -----------------------------------------------------------------------------
# Computed from prior matchups only (see scripts/data_unified_feature_complete.py).
H2H_FEATURES = [
    Feature("h2h_games", FeatureCategory.H2H,
            "Number of H2H games this season", default=0.0),
    Feature("h2h_margin", FeatureCategory.H2H,
            "Average H2H margin (last 5)", default=0.0),
    Feature("h2h_win_rate", FeatureCategory.H2H,
            "H2H win rate (last 5)", default=0.5),
]

# -----------------------------------------------------------------------------
# ELO RATINGS
# -----------------------------------------------------------------------------
ELO_FEATURES = [
    Feature("home_elo", FeatureCategory.ELO,
            "Home team ELO rating", default=1500.0),
    Feature("away_elo", FeatureCategory.ELO,
            "Away team ELO rating", default=1500.0),
    Feature("elo_diff", FeatureCategory.ELO,
            "ELO differential (home - away)", default=0.0),
    Feature("elo_prob_home", FeatureCategory.ELO,
            "ELO-based home win probability", default=0.5),
]

# -----------------------------------------------------------------------------
# STANDINGS
# -----------------------------------------------------------------------------
STANDINGS_FEATURES = [
    Feature("home_position", FeatureCategory.CORE,
            "Home team standings position (1-15)", default=8.0),
    Feature("away_position", FeatureCategory.CORE,
            "Away team standings position (1-15)", default=8.0),
    Feature("position_diff", FeatureCategory.CORE,
            "Position differential", default=0.0),
]

# -----------------------------------------------------------------------------
# ATS PERFORMANCE
# -----------------------------------------------------------------------------
ATS_FEATURES = [
    Feature("home_ats_pct", FeatureCategory.BETTING,
            "Home ATS win percentage", default=0.5),
    Feature("away_ats_pct", FeatureCategory.BETTING,
            "Away ATS win percentage", default=0.5),
    Feature("home_over_pct", FeatureCategory.BETTING,
            "Home over percentage", default=0.5),
    Feature("away_over_pct", FeatureCategory.BETTING,
            "Away over percentage", default=0.5),
]

# -----------------------------------------------------------------------------
# MARKET LINES (Added at prediction time) - FG ONLY
# NOTE: 1H-specific lines (1h_spread_line, 1h_total_line) are NOT included
# in unified training to avoid 23% NaN imputation in historical FG data.
# They are injected at PREDICTION time for 1H models via map_1h_features_to_fg_names()
# -----------------------------------------------------------------------------
MARKET_FEATURES = [
    Feature("spread_line", FeatureCategory.MARKET,
            "Current spread line", default=0.0),
    Feature("total_line", FeatureCategory.MARKET,
            "Current total line", default=220.0),
    Feature("spread_vs_predicted", FeatureCategory.MARKET,
            "Spread vs model prediction diff", default=0.0),
    Feature("total_vs_predicted", FeatureCategory.MARKET,
            "Total vs model prediction diff", default=0.0),
]

# 1H-Specific Market Features (injected at prediction time, NOT in training)
H1_MARKET_FEATURES = [
    Feature("1h_spread_line", FeatureCategory.MARKET,
            "Current 1H spread line", default=0.0),
    Feature("1h_total_line", FeatureCategory.MARKET,
            "Current 1H total line", default=110.0),
]

# -----------------------------------------------------------------------------
# TOTALS-SPECIFIC
# -----------------------------------------------------------------------------
TOTALS_SPECIFIC_FEATURES = [
    Feature("home_total_ppg", FeatureCategory.CORE,
            "Home contribution to total", default=110.0),
    Feature("away_total_ppg", FeatureCategory.CORE,
            "Away contribution to total", default=110.0),
]


# =============================================================================
# UNIFIED FEATURE LIST - ALL MODELS USE THIS
# =============================================================================

ALL_FEATURES: List[Feature] = (
    CORE_FEATURES +
    EFFICIENCY_FEATURES +
    FORM_FEATURES +
    REST_FEATURES +
    TRAVEL_FEATURES +
    INJURY_FEATURES +
    BETTING_FEATURES +
    H2H_FEATURES +
    ELO_FEATURES +
    STANDINGS_FEATURES +
    ATS_FEATURES +
    MARKET_FEATURES +
    TOTALS_SPECIFIC_FEATURES
)

# Feature name list for quick access
UNIFIED_FEATURE_NAMES: List[str] = [f.name for f in ALL_FEATURES]

# Feature defaults dict
FEATURE_DEFAULTS: Dict[str, float] = {f.name: f.default for f in ALL_FEATURES}

# Required features
REQUIRED_FEATURES: List[str] = [f.name for f in ALL_FEATURES if f.required]


# =============================================================================
# MODEL REGISTRY - All 4 models with identical configuration
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for a single model."""
    market_key: str          # e.g., "1h_spread", "fg_total"
    period: str              # "1h" or "fg"
    market_type: str         # "spread" or "total"
    label_column: str        # Column name for training labels
    line_column: str         # Column name for the betting line
    model_file: str          # Filename for saved model
    features: List[str] = field(
        default_factory=lambda: UNIFIED_FEATURE_NAMES.copy())


MODEL_REGISTRY: Dict[str, ModelConfig] = {
    "1h_spread": ModelConfig(
        market_key="1h_spread",
        period="1h",
        market_type="spread",
        label_column="1h_spread_covered",
        line_column="1h_spread_line",
        model_file="1h_spread_model.joblib",
    ),
    "1h_total": ModelConfig(
        market_key="1h_total",
        period="1h",
        market_type="total",
        label_column="1h_total_over",
        line_column="1h_total_line",
        model_file="1h_total_model.joblib",
    ),
    "fg_spread": ModelConfig(
        market_key="fg_spread",
        period="fg",
        market_type="spread",
        label_column="spread_covered",
        line_column="spread_line",
        model_file="fg_spread_model.joblib",
    ),
    "fg_total": ModelConfig(
        market_key="fg_total",
        period="fg",
        market_type="total",
        label_column="total_over",
        line_column="total_line",
        model_file="fg_total_model.joblib",
    ),
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_feature_names() -> List[str]:
    """Get list of all feature names."""
    return UNIFIED_FEATURE_NAMES.copy()


def get_feature_defaults() -> Dict[str, float]:
    """Get dict of feature name -> default value."""
    return FEATURE_DEFAULTS.copy()


def get_features_by_category(category: FeatureCategory) -> List[str]:
    """Get feature names for a specific category."""
    return [f.name for f in ALL_FEATURES if f.category == category]


def get_model_config(market_key: str) -> ModelConfig:
    """Get configuration for a specific model."""
    if market_key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown market: {market_key}. Valid: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[market_key]


def get_all_market_keys() -> List[str]:
    """Get all market keys."""
    return list(MODEL_REGISTRY.keys())


def validate_features(features: Dict[str, float]) -> Dict[str, float]:
    """
    Validate and fill missing features with defaults.

    Args:
        features: Dict of feature name -> value

    Returns:
        Complete feature dict with all required features filled
    """
    result = get_feature_defaults()
    result.update(features)

    # Check required features have non-default values
    missing_required = []
    for feat_name in REQUIRED_FEATURES:
        if feat_name not in features:
            missing_required.append(feat_name)

    if missing_required:
        import logging
        logging.warning(
            f"Required features missing (using defaults): {missing_required}")

    return result


# =============================================================================
# FEATURE COUNT SUMMARY
# =============================================================================

def print_feature_summary():
    """Print summary of all features."""
    print("=" * 60)
    print("UNIFIED FEATURE SPECIFICATION")
    print("=" * 60)
    print(f"Total Features: {len(ALL_FEATURES)}")
    print(f"Required Features: {len(REQUIRED_FEATURES)}")
    print()

    # Count by category
    for category in FeatureCategory:
        count = len(get_features_by_category(category))
        print(f"  {category.value.upper():12s}: {count:3d} features")

    print()
    print("All 4 models use IDENTICAL feature sets.")
    print("=" * 60)


if __name__ == "__main__":
    print_feature_summary()
