"""
Spread market implementation.

Predicts whether home team covers the spread.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from src.markets.base import BaseMarket, MarketPrediction
from src.prediction.confidence import calculate_confidence_from_binary_probability


class SpreadMarket(BaseMarket):
    """
    Spread betting market.
    
    Predicts whether the home team will cover the point spread.
    - Spread of -3.5 means home must win by 4+ to cover
    - Spread of +3.5 means home can lose by up to 3 and still cover
    """
    
    MARKET_TYPE = "spread"
    
    # NBA-specific spread parameters
    HOME_COVER_BASE_RATE = 0.50  # Historical home cover rate ~50%
    MARGIN_STD = 12.0  # Standard deviation of NBA margins
    
    def __init__(
        self,
        period: str = "fg",
        min_confidence: float = 0.55,
        min_edge: float = 0.02,
    ):
        """
        Initialize spread market.
        
        Args:
            period: "fg" or "1h"
            min_confidence: Minimum confidence threshold
            min_edge: Minimum edge vs implied probability
        """
        super().__init__(period, min_confidence, min_edge)
        
        # Adjust std for 1H (typically about half the variance)
        if period == "1h":
            self.margin_std = self.MARGIN_STD * 0.6
        else:
            self.margin_std = self.MARGIN_STD
    
    def predict(
        self,
        features: Dict[str, float],
        line: Optional[float] = None,
        odds: Optional[int] = None,
    ) -> MarketPrediction:
        """
        Generate spread prediction.
        
        Uses predicted margin vs line to calculate cover probability.
        
        Args:
            features: Must contain 'predicted_margin' or equivalent
            line: Spread line (e.g., -3.5)
            odds: American odds for edge calculation
        """
        if line is None:
            raise ValueError("Spread prediction requires line")
        
        # Get predicted margin
        margin_key = "predicted_margin_1h" if self.period == "1h" else "predicted_margin"
        predicted_margin = features.get(margin_key)
        
        if predicted_margin is None:
            raise ValueError(f"Missing required feature: {margin_key}")
        
        # Calculate probability of covering
        # Home covers if actual_margin > -line
        # P(cover) = P(margin > -line) using normal distribution
        from scipy.stats import norm
        
        # Edge = predicted_margin - (-line) = predicted_margin + line
        margin_edge = predicted_margin + line
        
        # Convert to probability using normal CDF
        home_cover_prob = float(norm.cdf(margin_edge / self.margin_std))
        
        # Determine side and confidence
        if home_cover_prob >= 0.5:
            side = "home"
            probability = home_cover_prob
        else:
            side = "away"
            probability = 1 - home_cover_prob
        
        confidence = calculate_confidence_from_binary_probability(probability)
        
        # Edge calculation
        edge = None
        kelly = None
        if odds is not None:
            edge = self.calculate_edge(probability, odds)
            kelly = self.calculate_kelly(probability, odds)
        
        prediction = MarketPrediction(
            market_key=self.market_key,
            period=self.period,
            market_type=self.MARKET_TYPE,
            side=side,
            probability=probability,
            confidence=confidence,
            line=line,
            implied_prob=self.american_to_implied(odds) if odds else None,
            edge=edge,
            kelly_fraction=kelly,
        )
        
        return self.apply_filters(prediction)
    
    def compute_label(
        self,
        home_score: float,
        away_score: float,
        line: Optional[float] = None,
    ) -> Optional[int]:
        """
        Compute spread outcome.
        
        Returns:
            1 if home covers, 0 if away covers, None if push
        """
        if line is None:
            return None
        
        actual_margin = home_score - away_score
        cover_margin = actual_margin + line  # Home covers if > 0
        
        if cover_margin > 0:
            return 1  # Home covers
        elif cover_margin < 0:
            return 0  # Away covers
        else:
            return None  # Push
    
    def get_feature_columns(self) -> List[str]:
        """Get spread-specific features."""
        base = super().get_feature_columns()
        
        if self.period == "1h":
            spread_features = [
                "predicted_margin_1h",
                "fh_spread_line",
                "home_ppg_1h", "away_ppg_1h",
                "home_margin_1h", "away_margin_1h",
            ]
        else:
            spread_features = [
                "predicted_margin",
                "spread_line",
                "spread_vs_predicted",
                "home_ats_pct", "away_ats_pct",
            ]
        
        return base + spread_features
    
    def derive_moneyline_prob(
        self,
        features: Dict[str, float],
    ) -> float:
        """
        Derive moneyline probability from spread prediction.
        
        Uses the predicted margin to calculate P(home wins).
        
        Returns:
            Probability of home team winning outright
        """
        margin_key = "predicted_margin_1h" if self.period == "1h" else "predicted_margin"
        predicted_margin = features.get(margin_key, 0)
        
        from scipy.stats import norm
        
        # P(home wins) = P(margin > 0)
        return float(norm.cdf(predicted_margin / self.margin_std))
