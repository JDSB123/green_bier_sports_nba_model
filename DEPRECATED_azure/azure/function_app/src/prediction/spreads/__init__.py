"""
Spreads prediction module (Full Game + First Half).
"""
from src.prediction.spreads.filters import FGSpreadFilter, FirstHalfSpreadFilter
from src.prediction.spreads.predictor import SpreadPredictor

__all__ = [
    "FGSpreadFilter",
    "FirstHalfSpreadFilter",
    "SpreadPredictor",
]
