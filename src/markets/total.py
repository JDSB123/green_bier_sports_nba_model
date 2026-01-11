"""
Total (over/under) market implementation.

Predicts whether combined score goes over or under the line.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from src.markets.base import BaseMarket, MarketPrediction


class TotalMarket(BaseMarket):
    """
    Total (over/under) betting market.
    
    Predicts whether the combined score will go over or under the line.
    """
    
    MARKET_TYPE = "total"
    
    # NBA-specific total parameters
    TOTAL_STD = 15.0  # Standard deviation of NBA totals
    LEAGUE_AVG_TOTAL = 225.0  # Approximate league average
    
    def __init__(
        self,
        period: str = "fg",
        min_confidence: float = 0.55,
        min_edge: float = 0.02,
    ):
        """
        Initialize total market.
        
        Args:
            period: "fg" or "1h"
            min_confidence: Minimum confidence threshold
            min_edge: Minimum edge vs implied probability
        """
        super().__init__(period, min_confidence, min_edge)
        
        # Adjust for 1H (typically about half the variance)
        if period == "1h":
            self.total_std = self.TOTAL_STD * 0.55
            self.league_avg = self.LEAGUE_AVG_TOTAL * 0.48  # ~108 pts in 1H
        else:
            self.total_std = self.TOTAL_STD
            self.league_avg = self.LEAGUE_AVG_TOTAL
    
    def predict(
        self,
        features: Dict[str, float],
        line: Optional[float] = None,
        odds: Optional[int] = None,
    ) -> MarketPrediction:
        """
        Generate total prediction.
        
        Uses predicted total vs line to calculate over probability.
        
        Args:
            features: Must contain 'predicted_total' or equivalent
            line: Total line (e.g., 225.5)
            odds: American odds for edge calculation
        """
        if line is None:
            raise ValueError("Total prediction requires line")
        
        # Get predicted total
        total_key = "predicted_total_1h" if self.period == "1h" else "predicted_total"
        predicted_total = features.get(total_key)
        
        if predicted_total is None:
            raise ValueError(f"Missing required feature: {total_key}")
        
        # Calculate probability of over
        # Over hits if actual_total > line
        from scipy.stats import norm
        
        # Edge = predicted_total - line
        total_edge = predicted_total - line
        
        # Convert to probability using normal CDF
        # P(over) = P(total > line) = 1 - P(total < line)
        over_prob = float(1 - norm.cdf(-total_edge / self.total_std))
        
        # Determine side and confidence
        if over_prob >= 0.5:
            side = "over"
            probability = over_prob
        else:
            side = "under"
            probability = 1 - over_prob
        
        confidence = abs(over_prob - 0.5) * 2  # 0 to 1 scale
        
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
        Compute total outcome.
        
        Returns:
            1 if over, 0 if under, None if push
        """
        if line is None:
            return None
        
        actual_total = home_score + away_score
        
        if actual_total > line:
            return 1  # Over
        elif actual_total < line:
            return 0  # Under
        else:
            return None  # Push
    
    def get_feature_columns(self) -> List[str]:
        """Get total-specific features."""
        base = super().get_feature_columns()
        
        if self.period == "1h":
            total_features = [
                "predicted_total_1h",
                "fh_total_line",
                "home_ppg_1h", "away_ppg_1h",
                "home_papg_1h", "away_papg_1h",
            ]
        else:
            total_features = [
                "predicted_total",
                "total_line",
                "total_vs_predicted",
                "home_over_pct", "away_over_pct",
                "home_pace", "away_pace",
            ]
        
        return base + total_features
