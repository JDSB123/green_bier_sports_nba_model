"""
Period-specific feature definitions for 1H and FG markets.

DEPRECATED: This module now imports from unified_features.py
All feature definitions are centralized there.

For new code, use:
    from src.modeling.unified_features import (
        UNIFIED_FEATURE_NAMES,
        MODEL_REGISTRY,
        get_model_config,
    )
"""
from __future__ import annotations
from typing import Dict, List
from dataclasses import dataclass, field

# Import from unified source of truth
from src.modeling.unified_features import (
    UNIFIED_FEATURE_NAMES,
    MODEL_REGISTRY,
    get_model_config,
    get_all_market_keys,
    get_feature_names,
    FEATURE_DEFAULTS,
)


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
# LEGACY COMPATIBILITY - Forward to unified features
# =============================================================================

@dataclass
class PeriodFeatureConfig:
    """Configuration for a specific period's features.
    
    DEPRECATED: Use MODEL_REGISTRY from unified_features.py instead.
    """
    period: str  # "1h" or "fg"
    score_suffix: str = ""
    core_features: List[str] = field(default_factory=list)
    spread_features: List[str] = field(default_factory=list)
    total_features: List[str] = field(default_factory=list)
    context_features: List[str] = field(default_factory=list)


# Legacy configs - now just wrappers around unified features
H1_FEATURES = PeriodFeatureConfig(
    period="1h",
    score_suffix="_1h",
    core_features=UNIFIED_FEATURE_NAMES,
    spread_features=[],
    total_features=[],
    context_features=[],
)

FG_FEATURES = PeriodFeatureConfig(
    period="fg",
    score_suffix="",
    core_features=UNIFIED_FEATURE_NAMES,
    spread_features=[],
    total_features=[],
    context_features=[],
)


def get_model_features(period: str, market: str) -> List[str]:
    """
    Get the full feature list for a specific model.
    
    All models now use the SAME unified feature set.

    Args:
        period: "1h" or "fg"
        market: "spread" or "total"

    Returns:
        List of feature column names (same for all markets)
    """
    # All models use identical features now
    return get_feature_names()


# =============================================================================
# MODEL CONFIGURATIONS - Now from unified source
# =============================================================================

MODEL_CONFIGS: Dict[str, Dict] = {
    market_key: {
        "period": config.period,
        "market": config.market_type,
        "label_col": config.label_column,
        "line_col": config.line_column,
        "model_file": config.model_file,
        "features": config.features,
    }
    for market_key, config in MODEL_REGISTRY.items()
}


def get_period_markets(period: str) -> List[str]:
    """Get all market keys for a specific period."""
    return [k for k, v in MODEL_CONFIGS.items() if v["period"] == period]


# Re-export for backward compatibility
__all__ = [
    "PERIOD_SCALING",
    "PeriodFeatureConfig",
    "H1_FEATURES",
    "FG_FEATURES",
    "get_model_features",
    "MODEL_CONFIGS",
    "get_all_market_keys",
    "get_period_markets",
]
