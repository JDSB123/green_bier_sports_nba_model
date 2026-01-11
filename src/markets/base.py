"""
Base market class for NBA betting markets.

Provides abstract interface for all market types.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class MarketPrediction:
    """Prediction for a single market."""
    
    # Market info
    market_key: str
    period: str  # "fg" or "1h"
    market_type: str  # "spread", "total", "moneyline"
    
    # Prediction
    side: str  # "home"/"away" for spread/ML, "over"/"under" for total
    probability: float  # P(side wins)
    confidence: float  # 0-1 confidence score
    
    # Edge calculation
    line: Optional[float] = None
    implied_prob: Optional[float] = None
    edge: Optional[float] = None  # probability - implied_prob
    
    # Kelly sizing
    kelly_fraction: Optional[float] = None
    
    # Filters
    passes_filter: bool = True
    filter_reason: Optional[str] = None


class BaseMarket(ABC):
    """
    Abstract base class for betting markets.
    
    Each market type (spread, total, moneyline) inherits from this
    and implements market-specific prediction logic.
    """
    
    # Class-level configuration
    MARKET_TYPE: str = "base"
    PERIODS: Tuple[str, ...] = ("fg", "1h")
    
    # Default -110 odds
    STANDARD_VIG = 0.0455  # ~4.55% vig
    WIN_PAYOUT = 100.0 / 110.0  # ~0.909
    BREAK_EVEN_PROB = 110.0 / 210.0  # ~52.38%
    
    def __init__(
        self,
        period: str = "fg",
        min_confidence: float = 0.0,
        min_edge: float = 0.0,
    ):
        """
        Initialize market.
        
        Args:
            period: "fg" (full game) or "1h" (first half)
            min_confidence: Minimum confidence to pass filter
            min_edge: Minimum edge vs implied probability to pass filter
        """
        if period not in self.PERIODS:
            raise ValueError(f"Invalid period: {period}. Must be one of {self.PERIODS}")
        
        self.period = period
        self.min_confidence = min_confidence
        self.min_edge = min_edge
        
        self.market_key = f"{period}_{self.MARKET_TYPE}"
    
    @abstractmethod
    def predict(
        self,
        features: Dict[str, float],
        line: Optional[float] = None,
        odds: Optional[int] = None,
    ) -> MarketPrediction:
        """
        Generate prediction for this market.
        
        Args:
            features: Dictionary of feature values
            line: Market line (spread/total) or None for moneyline
            odds: American odds for edge calculation
            
        Returns:
            MarketPrediction with all fields populated
        """
        pass
    
    @abstractmethod
    def compute_label(
        self,
        home_score: float,
        away_score: float,
        line: Optional[float] = None,
    ) -> Optional[int]:
        """
        Compute outcome label from scores.
        
        Args:
            home_score: Home team score (full game or 1H)
            away_score: Away team score
            line: Market line if applicable
            
        Returns:
            1 for home/over, 0 for away/under, None if push/invalid
        """
        pass
    
    def calculate_edge(
        self,
        model_prob: float,
        odds: int = -110,
    ) -> float:
        """
        Calculate edge vs market odds.
        
        Edge = Model probability - Implied probability
        
        Args:
            model_prob: Model's probability of the outcome
            odds: American odds (e.g., -110, +150)
            
        Returns:
            Edge (positive = value bet)
        """
        implied = self.american_to_implied(odds)
        return model_prob - implied
    
    def calculate_kelly(
        self,
        model_prob: float,
        odds: int = -110,
    ) -> float:
        """
        Calculate Kelly criterion bet fraction.
        
        Kelly = (bp - q) / b
        where b = decimal odds - 1, p = win prob, q = 1 - p
        
        Args:
            model_prob: Model's probability of winning
            odds: American odds
            
        Returns:
            Kelly fraction (0-1, or 0 if no edge)
        """
        decimal_odds = self.american_to_decimal(odds)
        b = decimal_odds - 1  # Profit per unit wagered
        p = model_prob
        q = 1 - p
        
        kelly = (b * p - q) / b if b > 0 else 0
        return max(0.0, kelly)  # Don't bet if negative edge
    
    def american_to_implied(self, odds: int) -> float:
        """
        Convert American odds to implied probability.
        
        Args:
            odds: American odds (e.g., -110, +150)
            
        Returns:
            Implied probability (0-1)
        """
        if odds < 0:
            return abs(odds) / (abs(odds) + 100)
        else:
            return 100 / (odds + 100)
    
    def american_to_decimal(self, odds: int) -> float:
        """
        Convert American odds to decimal odds.
        
        Args:
            odds: American odds
            
        Returns:
            Decimal odds (e.g., 1.91 for -110)
        """
        if odds < 0:
            return 1 + (100 / abs(odds))
        else:
            return 1 + (odds / 100)
    
    def apply_filters(
        self,
        prediction: MarketPrediction,
    ) -> MarketPrediction:
        """
        Apply filtering rules to prediction.
        
        Updates passes_filter and filter_reason fields.
        """
        # Check confidence
        if prediction.confidence < self.min_confidence:
            prediction.passes_filter = False
            prediction.filter_reason = f"Low confidence: {prediction.confidence:.1%} < {self.min_confidence:.1%}"
            return prediction
        
        # Check edge
        if prediction.edge is not None and prediction.edge < self.min_edge:
            prediction.passes_filter = False
            prediction.filter_reason = f"Low edge: {prediction.edge:.1%} < {self.min_edge:.1%}"
            return prediction
        
        prediction.passes_filter = True
        prediction.filter_reason = None
        return prediction
    
    def get_feature_columns(self) -> List[str]:
        """Get list of feature columns for this market."""
        # Base features common to all markets
        base = [
            "home_ppg", "away_ppg",
            "home_papg", "away_papg",
            "home_margin", "away_margin",
            "home_win_pct", "away_win_pct",
            "ppg_diff", "margin_diff", "win_pct_diff",
            "home_rest", "away_rest", "rest_diff",
            "home_b2b", "away_b2b",
            "dynamic_hca",
            "h2h_margin", "h2h_games",
            "away_travel_fatigue",
        ]
        
        # Period-specific prefixes
        if self.period == "1h":
            base = [f"{f}_1h" if not f.startswith(("home_", "away_", "h2h_")) else f 
                    for f in base]
        
        return base
