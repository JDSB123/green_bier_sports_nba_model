"""
Period-specific feature definitions for Q1, 1H, and FG markets.

NBA v6.0: Each period (Q1, 1H, FG) has INDEPENDENT features computed from
historical data for that specific period. No cross-period dependencies.

Key Principle: Features for Q1 predictions come from Q1 historical stats,
features for 1H come from 1H historical stats, and FG from FG stats.
This prevents leakage and allows each model to learn period-specific patterns.
"""
from __future__ import annotations
from typing import Dict, List
from dataclasses import dataclass, field


# =============================================================================
# SCALING FACTORS - Empirically derived from NBA data
# =============================================================================
# These are used for rest/travel adjustments, NOT for deriving predictions
PERIOD_SCALING = {
    "q1": {
        "hca_factor": 0.25,        # Q1 HCA ~0.75 pts (vs 3 pts FG)
        "rest_factor": 0.25,       # Rest impact reduced for Q1
        "travel_factor": 0.3,      # Travel fatigue lower early in game
        "scoring_pct": 0.25,       # Q1 is ~25% of game scoring
    },
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
    period: str  # "q1", "1h", or "fg"

    # Suffixes for period-specific columns in training data
    score_suffix: str = ""  # e.g., "_q1", "_1h", "" for FG

    # Core statistical features (computed from period-specific historical data)
    core_features: List[str] = field(default_factory=list)

    # Spread-specific features
    spread_features: List[str] = field(default_factory=list)

    # Total-specific features
    total_features: List[str] = field(default_factory=list)

    # Moneyline-specific features
    moneyline_features: List[str] = field(default_factory=list)

    # Context features (rest, travel - same for all periods but scaled)
    context_features: List[str] = field(default_factory=list)


# First Quarter Features
Q1_FEATURES = PeriodFeatureConfig(
    period="q1",
    score_suffix="_q1",
    core_features=[
        # Q1-specific rolling stats (computed from historical Q1 data)
        "home_ppg_q1",           # Home team Q1 PPG (last N games)
        "home_papg_q1",          # Home team Q1 points allowed
        "home_margin_q1",        # Home team Q1 avg margin
        "away_ppg_q1",           # Away team Q1 PPG
        "away_papg_q1",          # Away team Q1 points allowed
        "away_margin_q1",        # Away team Q1 avg margin
        # Differentials
        "ppg_diff_q1",           # Q1 PPG differential
        "margin_diff_q1",        # Q1 margin differential
        # Win rates in Q1
        "home_q1_win_pct",       # Home team Q1 win rate
        "away_q1_win_pct",       # Away team Q1 win rate
        # Pace in Q1
        "home_pace_q1",          # Home total Q1 points (off + def)
        "away_pace_q1",          # Away total Q1 points
        # Recent form (last 5 games Q1 performance)
        "home_l5_margin_q1",     # Home Q1 margin last 5
        "away_l5_margin_q1",     # Away Q1 margin last 5
        # Consistency
        "home_margin_std_q1",    # Q1 margin volatility
        "away_margin_std_q1",
    ],
    spread_features=[
        # Q1 spread-specific
        "predicted_margin_q1",    # Model's Q1 margin prediction
        "q1_spread_line",         # Market Q1 spread line
        "spread_vs_predicted_q1", # Model vs market disagreement
        # Q1 ATS performance
        "home_ats_pct_q1",        # Q1 cover rate
        "away_ats_pct_q1",
    ],
    total_features=[
        # Q1 total-specific
        "predicted_total_q1",     # Model's Q1 total prediction
        "q1_total_line",          # Market Q1 total line
        "total_vs_predicted_q1",  # Model vs market disagreement
        # Q1 over/under tendencies
        "home_over_pct_q1",       # Q1 over rate
        "away_over_pct_q1",
        # Combined pace
        "expected_pace_q1",       # Expected Q1 combined scoring
    ],
    moneyline_features=[
        # Q1 moneyline-specific
        "ml_prob_home_q1",        # Q1 home win probability estimate
        "ml_elo_diff_q1",         # Elo difference (can be shared)
        "ml_momentum_q1",         # Recent Q1 momentum
        "home_q1_lead_pct",       # How often home leads after Q1
        "away_q1_lead_pct",
    ],
    context_features=[
        # Scaled for Q1
        "dynamic_hca_q1",         # Q1-scaled HCA
        "home_rest_adj_q1",       # Rest adjustment (scaled)
        "away_rest_adj_q1",
        "travel_fatigue_q1",      # Travel impact (scaled)
        # Raw (unscaled) for reference
        "home_rest_days",
        "away_rest_days",
        "home_b2b",
        "away_b2b",
    ],
)


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
    moneyline_features=[
        # 1H moneyline-specific
        "ml_prob_home_1h",        # 1H home lead probability
        "ml_elo_diff",            # Elo difference (shared)
        "ml_momentum_1h",         # Recent 1H momentum
        "home_1h_lead_pct",       # How often home leads at half
        "away_1h_lead_pct",
        "ml_pythagorean_diff_1h", # Pythagorean expectation for 1H
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
    moneyline_features=[
        # FG moneyline-specific
        "ml_estimated_home_prob", # Home win probability
        "ml_elo_diff",            # Elo difference
        "ml_pythagorean_diff",    # Pythagorean expectation
        "ml_momentum_diff",       # Momentum difference
        "ml_win_prob_diff",       # Win probability difference
        "ml_h2h_factor",          # H2H factor
        # H2H
        "h2h_margin",
        "h2h_home_win_pct",
        "h2h_games",
        # SOS
        "home_sos_rating",
        "away_sos_rating",
        "sos_diff",
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
        period: "q1", "1h", or "fg"
        market: "spread", "total", or "moneyline"

    Returns:
        List of feature column names
    """
    config = {
        "q1": Q1_FEATURES,
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
    elif market == "moneyline":
        features.extend(config.moneyline_features)
    else:
        raise ValueError(f"Unknown market: {market}")

    # Add context features
    features.extend(config.context_features)

    return features


# =============================================================================
# ALL 9 MODEL CONFIGURATIONS
# =============================================================================

MODEL_CONFIGS: Dict[str, Dict] = {
    # First Quarter Models
    "q1_spread": {
        "period": "q1",
        "market": "spread",
        "label_col": "q1_spread_covered",
        "line_col": "q1_spread_line",
        "model_file": "q1_spread_model.joblib",
        "features": get_model_features("q1", "spread"),
    },
    "q1_total": {
        "period": "q1",
        "market": "total",
        "label_col": "q1_total_over",
        "line_col": "q1_total_line",
        "model_file": "q1_total_model.joblib",
        "features": get_model_features("q1", "total"),
    },
    "q1_moneyline": {
        "period": "q1",
        "market": "moneyline",
        "label_col": "home_q1_win",
        "line_col": None,
        "model_file": "q1_moneyline_model.joblib",
        "features": get_model_features("q1", "moneyline"),
    },
    # First Half Models
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
    "1h_moneyline": {
        "period": "1h",
        "market": "moneyline",
        "label_col": "home_1h_win",
        "line_col": None,
        "model_file": "1h_moneyline_model.joblib",
        "features": get_model_features("1h", "moneyline"),
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
    "fg_moneyline": {
        "period": "fg",
        "market": "moneyline",
        "label_col": "home_win",
        "line_col": None,
        "model_file": "fg_moneyline_model.joblib",
        "features": get_model_features("fg", "moneyline"),
    },
}


def get_all_market_keys() -> List[str]:
    """Return all 9 market keys."""
    return list(MODEL_CONFIGS.keys())


def get_period_markets(period: str) -> List[str]:
    """Get all market keys for a specific period."""
    return [k for k, v in MODEL_CONFIGS.items() if v["period"] == period]
