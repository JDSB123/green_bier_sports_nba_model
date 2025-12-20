"""
NBA Prediction Module - v6.0

Production prediction engine for 9 INDEPENDENT markets:
- First Quarter: Spread, Total, Moneyline
- First Half: Spread, Total, Moneyline
- Full Game: Spread (60.6% acc), Total (59.2% acc), Moneyline (65.5% acc)

STRICT MODE: All 9 models must exist. No fallbacks. No silent failures.
"""
# Unified engine (THE ONLY ENTRY POINT)
from src.prediction.engine import UnifiedPredictionEngine, ModelNotFoundError

# Market-specific modules (for direct access if needed)
from src.prediction.spreads import SpreadPredictor
from src.prediction.totals import TotalPredictor
from src.prediction.moneyline import MoneylinePredictor

# Model loading (used by engine)
from src.prediction.models import (
    load_spread_model,
    load_total_model,
    load_first_half_spread_model,
    load_first_half_total_model,
    load_moneyline_model,
)

__all__ = [
    # Engine (PRIMARY ENTRY POINT)
    "UnifiedPredictionEngine",
    "ModelNotFoundError",
    # Predictors
    "SpreadPredictor",
    "TotalPredictor",
    "MoneylinePredictor",
    # Model loaders
    "load_spread_model",
    "load_total_model",
    "load_first_half_spread_model",
    "load_first_half_total_model",
    "load_moneyline_model",
]
