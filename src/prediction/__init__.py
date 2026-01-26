"""
NBA Prediction Module - PRODUCTION STACK

Single entry point: UnifiedPredictionEngine
4 Markets: 1H Spread, 1H Total, FG Spread, FG Total
60 features per model (RichFeatureBuilder)

STRICT MODE: All 4 models must exist. No fallbacks. No silent failures.
"""

from src.prediction.engine import ModelNotFoundError, UnifiedPredictionEngine
from src.prediction.models import (
    load_first_half_spread_model,
    load_first_half_total_model,
    load_spread_model,
    load_total_model,
)

__all__ = [
    "UnifiedPredictionEngine",
    "ModelNotFoundError",
    "load_spread_model",
    "load_total_model",
    "load_first_half_spread_model",
    "load_first_half_total_model",
]
