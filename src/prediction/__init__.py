"""
NBA Prediction Module - v5.1 FINAL

Production prediction engine for 3 PROVEN ROE markets:
- Full Game Spread (60.6% acc, +15.7% ROI)
- Full Game Total (59.2% acc, +13.1% ROI)
- Full Game Moneyline (65.5% acc, +25.1% ROI)

STRICT MODE: All models must exist. No fallbacks. No silent failures.
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
