"""
NBA Prediction Module - v6.0

Production prediction engine for 4 INDEPENDENT markets:
- First Half: Spread, Total
- Full Game: Spread (60.6% acc), Total (59.2% acc)

STRICT MODE: All 4 models must exist. No fallbacks. No silent failures.
"""
# Unified engine (THE ONLY ENTRY POINT)
from src.prediction.engine import UnifiedPredictionEngine, ModelNotFoundError

# Market-specific modules (for direct access if needed)
from src.prediction.spreads import SpreadPredictor
from src.prediction.totals import TotalPredictor

# Model loading (used by engine)
from src.prediction.models import (
    load_spread_model,
    load_total_model,
    load_first_half_spread_model,
    load_first_half_total_model,
)

__all__ = [
    # Engine (PRIMARY ENTRY POINT)
    "UnifiedPredictionEngine",
    "ModelNotFoundError",
    # Predictors
    "SpreadPredictor",
    "TotalPredictor",
    # Model loaders
    "load_spread_model",
    "load_total_model",
    "load_first_half_spread_model",
    "load_first_half_total_model",
]
