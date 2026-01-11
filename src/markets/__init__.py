"""
NBA Market Definitions.

Defines the 6 independent markets:
- FG Spread, FG Total, FG Moneyline
- 1H Spread, 1H Total, 1H Moneyline
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
