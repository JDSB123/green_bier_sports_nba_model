"""
Totals prediction module (Full Game + First Half).
"""
from src.prediction.totals.filters import FGTotalFilter, FirstHalfTotalFilter
from src.prediction.totals.predictor import TotalPredictor

__all__ = [
    "FGTotalFilter",
    "FirstHalfTotalFilter",
    "TotalPredictor",
]
