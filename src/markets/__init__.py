"""
NBA Market Definitions.

Defines the 6 independent markets:
- FG Spread, FG Total, FG Moneyline
- 1H Spread, 1H Total, 1H Moneyline

DEPRECATION NOTICE (v33.1.5):
============================
The SpreadMarket, TotalMarket, and MoneylineMarket classes are NOT used
in production. Actual prediction filtering uses EDGE-ONLY logic in:
  - src/prediction/resolution.py::apply_thresholds()

The classes here are retained for potential future refactoring but the
apply_filters() method on BaseMarket has STALE confidence-based logic.
Do NOT use these classes for production filtering.
"""

from src.markets.base import BaseMarket, MarketPrediction
from src.markets.spread import SpreadMarket
from src.markets.total import TotalMarket
from src.markets.moneyline import MoneylineMarket

__all__ = [
    "BaseMarket",
    "MarketPrediction",
    "SpreadMarket",
    "TotalMarket",
    "MoneylineMarket",
]
