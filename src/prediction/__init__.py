"""
NBA Prediction Module

Market-based modular prediction system for all markets (spreads, totals, moneyline)
across full game and first half with smart filtering.
"""
# Unified engine (recommended entry point)
from src.prediction.engine import UnifiedPredictionEngine

# Market-specific modules
from src.prediction import spreads, totals, moneyline

# Legacy support (old modular architecture)
from src.prediction.filters import (
    SpreadFilter,
    TotalFilter,
    FirstHalfSpreadFilter,
    FirstHalfTotalFilter,
    filter_predictions,
)
from src.prediction.models import load_spread_model, load_total_model
from src.prediction.predictor import PredictionEngine

__all__ = [
    # Unified engine (NEW - recommended)
    "UnifiedPredictionEngine",
    # Market modules
    "spreads",
    "totals",
    "moneyline",
    # Legacy (OLD - for compatibility)
    "SpreadFilter",
    "TotalFilter",
    "FirstHalfSpreadFilter",
    "FirstHalfTotalFilter",
    "filter_predictions",
    "load_spread_model",
    "load_total_model",
    "PredictionEngine",
]
