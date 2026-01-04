"""
Period-specific feature definitions for 1H and FG markets.

Q1 markets are removed; only 1H + FG are supported.

Each period (1H, FG) has INDEPENDENT features computed from
historical data for that specific period. No cross-period dependencies.

Key Principle: Features for 1H predictions come from 1H historical stats,
and FG from FG stats. This prevents leakage and allows each model to learn
period-specific patterns.
"""
from __future__ import annotations
from typing import Dict, List
from dataclasses import dataclass, field


# =============================================================================
# SCALING FACTORS - Empirically derived from NBA data
# =============================================================================
# These are used for rest/travel adjustments, NOT for deriving predictions
PERIOD_SCALING = {
    "1h": {
        "hca_factor": 0.5,         # 1H HCA ~1.5 pts (vs 3 pts FG)
        "rest_factor": 0.5,        # Rest impact moderate for 1H
        "travel_factor": 0.5,      # Travel fatigue builds
        "scoring_pct": 0.485,      # 1H is ~48.5% of game scoring
    },
    "fg": {
        "hca_factor": 1.0,         # Full HCA
        "rest_factor": 1.0,        # Full rest impact
        "travel_factor": 1.0,      # Full travel impact
        "scoring_pct": 1.0,        # Full game
    },
}


# =============================================================================
# PERIOD-SPECIFIC FEATURE DEFINITIONS
# =============================================================================

@dataclass
class PeriodFeatureConfig:
    """Configuration for a specific period's features."""
    period: str  # "1h" or "fg"

    # Suffixes for period-specific columns in training data
    score_suffix: str = ""  # e.g., "_1h", "" for FG

    # Core statistical features (computed from period-specific historical data)
    core_features: List[str] = field(default_factory=list)

    # Spread-specific features
    spread_features: List[str] = field(default_factory=list)

    # Total-specific features
    total_features: List[str] = field(default_factory=list)

    # Context features (rest, travel - same for all periods but scaled)
    context_features: List[str] = field(default_factory=list)


# First Half Features
H1_FEATURES = PeriodFeatureConfig(
    period="1h",
    score_suffix="_1h",
    core_features=[
        # 1H-specific rolling stats
        "home_ppg_1h",            # Home team 1H PPG
        "home_papg_1h",           # Home team 1H points allowed
        "home_margin_1h",         # Home team 1H avg margin
        "away_ppg_1h",            # Away team 1H PPG
        "away_papg_1h",           # Away team 1H points allowed
        "away_margin_1h",         # Away team 1H avg margin
        # Differentials
        "ppg_diff_1h",            # 1H PPG differential
        "margin_diff_1h",         # 1H margin differential
        # Win rates in 1H
        "home_1h_win_pct",        # Home leading at half rate
        "away_1h_win_pct",        # Away leading at half rate
        # Pace in 1H
        "home_pace_1h",           # Home total 1H points
        "away_pace_1h",           # Away total 1H points
        # Recent form
        "home_l5_margin_1h",      # 1H margin last 5
        "away_l5_margin_1h",
        "home_l10_margin_1h",     # 1H margin last 10
        "away_l10_margin_1h",
        # Consistency
        "home_margin_std_1h",     # 1H margin volatility
        "away_margin_std_1h",
        # Efficiency (1H specific)
        "home_ortg_1h",           # 1H offensive rating
        "away_ortg_1h",
        "home_drtg_1h",           # 1H defensive rating
        "away_drtg_1h",
    ],
    spread_features=[
        # 1H spread-specific
        "predicted_margin_1h",    # Model's 1H margin prediction
        "1h_spread_line",         # Market 1H spread line
        "spread_vs_predicted_1h", # Model vs market disagreement
        # 1H ATS performance
        "home_ats_pct_1h",        # 1H cover rate
        "away_ats_pct_1h",
        # Spread movement
        "spread_movement_1h",
    ],
    total_features=[
        # 1H total-specific
        "predicted_total_1h",     # Model's 1H total prediction
        "1h_total_line",          # Market 1H total line
        "total_vs_predicted_1h",  # Model vs market disagreement
        # 1H over/under tendencies
        "home_over_pct_1h",       # 1H over rate
        "away_over_pct_1h",
        # Combined pace
        "expected_pace_1h",       # Expected 1H combined scoring
    ],
    context_features=[
        # Scaled for 1H
        "dynamic_hca_1h",         # 1H-scaled HCA
        "home_rest_adj_1h",       # Rest adjustment (scaled)
        "away_rest_adj_1h",
        "travel_fatigue_1h",      # Travel impact (scaled)
        # Raw
        "home_rest_days",
        "away_rest_days",
        "home_b2b",
        "away_b2b",
        # H2H
        "h2h_margin_1h",          # H2H 1H margin if available
    ],
)


# Full Game Features
FG_FEATURES = PeriodFeatureConfig(
    period="fg",
    score_suffix="",  # No suffix for full game
    core_features=[
        # FG rolling stats
        "home_ppg",               # Home team PPG
        "home_papg",              # Home team points allowed
        "home_margin",            # Home team avg margin
        "away_ppg",               # Away team PPG
        "away_papg",              # Away team points allowed
        "away_margin",            # Away team avg margin
        # Differentials
        "ppg_diff",               # PPG differential
        "margin_diff",            # Margin differential
        # Win rates
        "home_win_pct",           # Home team win rate
        "away_win_pct",           # Away team win rate
        "win_pct_diff",           # Win % differential
        # Pace
        "home_pace",              # Home total points
        "away_pace",              # Away total points
        # Recent form
        "home_l5_margin",         # Margin last 5
        "away_l5_margin",
        "home_l10_margin",        # Margin last 10
        "away_l10_margin",
        "home_form_trend",        # Form trend (L3 vs L10)
        "away_form_trend",
        # Consistency
        "home_margin_std",        # Margin volatility
        "away_margin_std",
        # Advanced ratings
        "home_ortg",              # Offensive rating
        "away_ortg",
        "home_drtg",              # Defensive rating
        "away_drtg",
        "home_net_rtg",           # Net rating
        "away_net_rtg",
        "net_rating_diff",
        # Clutch
        "home_clutch_win_pct",
        "away_clutch_win_pct",
        "clutch_diff",
    ],
    spread_features=[
        # FG spread-specific
        "predicted_margin",       # Model's margin prediction
        "spread_line",            # Market spread line
        "spread_vs_predicted",    # Model vs market disagreement
        # ATS performance
        "home_ats_pct",           # Cover rate
        "away_ats_pct",
        # Market signals
        "spread_movement",
        "spread_opening_line",
        "is_rlm_spread",          # Reverse line movement
        "sharp_side_spread",
        "spread_public_home_pct",
        # Injury
        "home_injury_spread_impact",
        "away_injury_spread_impact",
        "injury_spread_diff",
    ],
    total_features=[
        # FG total-specific
        "predicted_total",        # Model's total prediction
        "total_line",             # Market total line
        "total_vs_predicted",     # Model vs market disagreement
        # Over/under tendencies
        "home_over_pct",          # Over rate
        "away_over_pct",
        # Combined pace
        "expected_pace",          # Expected combined scoring
        # Market signals
        "total_movement",
        "total_opening_line",
        "is_rlm_total",
        "sharp_side_total",
    ],
    context_features=[
        # Full impact
        "dynamic_hca",            # Full HCA
        "home_court_advantage",   # Team-specific base HCA
        "home_rest_days",
        "away_rest_days",
        "rest_diff",
        "home_b2b",
        "away_b2b",
        # Travel
        "away_travel_distance",
        "away_travel_fatigue",
        "travel_advantage",
        "away_b2b_travel_penalty",
    ],
)


# =============================================================================
# COMBINED FEATURE LISTS FOR EACH MODEL
# =============================================================================

def get_model_features(period: str, market: str) -> List[str]:
    """
    Get the full feature list for a specific model.

    Args:
        period: "1h" or "fg"
        market: "spread" or "total"

    Returns:
        List of feature column names
    """
    config = {
        "1h": H1_FEATURES,
        "fg": FG_FEATURES,
    }.get(period)

    if config is None:
        raise ValueError(f"Unknown period: {period}")

    # Start with core features
    features = list(config.core_features)

    # Add market-specific features
    if market == "spread":
        features.extend(config.spread_features)
    elif market == "total":
        features.extend(config.total_features)
    else:
        raise ValueError(f"Unknown market: {market}")

    # Add context features
    features.extend(config.context_features)

    return features


# =============================================================================
# ALL 4 MODEL CONFIGURATIONS
# =============================================================================

MODEL_CONFIGS: Dict[str, Dict] = {
    # First Half Models (v33.1.0: standardized to .joblib format, bundled features)
    "1h_spread": {
        "period": "1h",
        "market": "spread",
        "label_col": "1h_spread_covered",
        "line_col": "1h_spread_line",
        "model_file": "1h_spread_model.joblib",
        "features": get_model_features("1h", "spread"),
    },
    "1h_total": {
        "period": "1h",
        "market": "total",
        "label_col": "1h_total_over",
        "line_col": "1h_total_line",
        "model_file": "1h_total_model.joblib",
        "features": get_model_features("1h", "total"),
    },
    # Full Game Models
    "fg_spread": {
        "period": "fg",
        "market": "spread",
        "label_col": "spread_covered",
        "line_col": "spread_line",
        "model_file": "fg_spread_model.joblib",
        "features": get_model_features("fg", "spread"),
    },
    "fg_total": {
        "period": "fg",
        "market": "total",
        "label_col": "total_over",
        "line_col": "total_line",
        "model_file": "fg_total_model.joblib",
        "features": get_model_features("fg", "total"),
    },
}


def get_all_market_keys() -> List[str]:
    """Return all active market keys (1H + FG)."""
    return list(MODEL_CONFIGS.keys())


def get_period_markets(period: str) -> List[str]:
    """Get all market keys for a specific period."""
    return [k for k, v in MODEL_CONFIGS.items() if v["period"] == period]
