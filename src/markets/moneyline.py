"""
Moneyline market implementation.

Predicts outright winner with both derived and independent model approaches.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from src.markets.base import BaseMarket, MarketPrediction
from src.prediction.confidence import calculate_confidence_from_binary_probability


class MoneylineMarket(BaseMarket):
    """
    Moneyline betting market.
    
    Predicts which team wins outright (no spread).
    
    Supports two approaches:
    1. Derived: Convert spread prediction to moneyline probability
    2. Independent: Train separate classifier on moneyline-specific features
    """
    
    MARKET_TYPE = "moneyline"
    
    # NBA-specific parameters
    MARGIN_STD = 12.0  # Standard deviation of NBA margins
    HOME_WIN_RATE = 0.57  # Historical NBA home win rate
    
    def __init__(
        self,
        period: str = "fg",
        min_confidence: float = 0.60,
        min_edge: float = 0.03,
        use_derived: bool = True,
    ):
        """
        Initialize moneyline market.
        
        Args:
            period: "fg" or "1h"
            min_confidence: Minimum confidence threshold
            min_edge: Minimum edge vs implied probability
            use_derived: If True, derive from margin; if False, use independent model
        """
        super().__init__(period, min_confidence, min_edge)
        
        self.use_derived = use_derived
        
        # Adjust std for 1H
        if period == "1h":
            self.margin_std = self.MARGIN_STD * 0.6
        else:
            self.margin_std = self.MARGIN_STD
    
    def predict(
        self,
        features: Dict[str, float],
        line: Optional[float] = None,  # Not used for moneyline
        odds: Optional[int] = None,
    ) -> MarketPrediction:
        """
        Generate moneyline prediction.
        
        Uses either derived approach (from margin) or independent classifier.
        
        Args:
            features: Must contain 'predicted_margin' or classifier probabilities
            line: Not used for moneyline
            odds: American odds for edge calculation
        """
        if self.use_derived:
            home_win_prob = self._derive_from_margin(features)
        else:
            home_win_prob = self._predict_independent(features)
        
        # Determine side and confidence
        if home_win_prob >= 0.5:
            side = "home"
            probability = home_win_prob
        else:
            side = "away"
            probability = 1 - home_win_prob
        
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
            line=None,  # Moneyline has no line
            implied_prob=self.american_to_implied(odds) if odds else None,
            edge=edge,
            kelly_fraction=kelly,
        )
        
        return self.apply_filters(prediction)
    
    def _derive_from_margin(self, features: Dict[str, float]) -> float:
        """
        Derive moneyline probability from predicted margin.
        
        Uses normal distribution: P(home wins) = P(margin > 0)
        
        Args:
            features: Must contain predicted_margin
            
        Returns:
            Probability of home team winning
        """
        from scipy.stats import norm
        
        margin_key = "predicted_margin_1h" if self.period == "1h" else "predicted_margin"
        predicted_margin = features.get(margin_key)
        
        if predicted_margin is None:
            # Fall back to computing from team stats
            home_margin = features.get("home_margin", 0)
            away_margin = features.get("away_margin", 0)
            hca = features.get("dynamic_hca", 3.0)
            
            predicted_margin = (home_margin - away_margin) / 2 + hca
        
        # P(home wins) = P(margin > 0)
        return float(norm.cdf(predicted_margin / self.margin_std))
    
    def _predict_independent(self, features: Dict[str, float]) -> float:
        """
        Use independent model features for prediction.
        
        This is a placeholder for when a trained classifier is used.
        Falls back to derived approach if model not available.
        
        Args:
            features: Game features
            
        Returns:
            Probability of home team winning
        """
        # Check for precomputed classifier probability
        if "ml_home_prob" in features:
            return features["ml_home_prob"]
        
        # Fall back to team strength comparison
        home_win_pct = features.get("home_win_pct", 0.5)
        away_win_pct = features.get("away_win_pct", 0.5)
        
        # Adjust for home court
        hca_bonus = 0.03  # ~3% home advantage
        
        # Simple log5 formula
        home_strength = home_win_pct + hca_bonus
        away_strength = away_win_pct
        
        if home_strength + away_strength == 0:
            return 0.5
        
        # Log5 method
        home_prob = (home_strength * (1 - away_strength)) / (
            home_strength * (1 - away_strength) + (1 - home_strength) * away_strength
        )
        
        return float(np.clip(home_prob, 0.01, 0.99))
    
    def compute_label(
        self,
        home_score: float,
        away_score: float,
        line: Optional[float] = None,
    ) -> Optional[int]:
        """
        Compute moneyline outcome.
        
        Returns:
            1 if home wins, 0 if away wins, None if tie
        """
        if home_score > away_score:
            return 1  # Home wins
        elif away_score > home_score:
            return 0  # Away wins
        else:
            return None  # Tie (rare in NBA, only in regulation)
    
    def get_feature_columns(self) -> List[str]:
        """Get moneyline-specific features."""
        base = super().get_feature_columns()
        
        if self.period == "1h":
            ml_features = [
                "predicted_margin_1h",
                "home_1h_win_pct", "away_1h_win_pct",
            ]
        else:
            ml_features = [
                "predicted_margin",
                "home_elo", "away_elo", "elo_diff",
                "h2h_win_rate",
            ]
        
        return base + ml_features
    
    def implied_spread_from_moneyline(
        self,
        home_odds: int,
        away_odds: int,
    ) -> float:
        """
        Calculate implied spread from moneyline odds.
        
        Uses the relationship between win probability and point spread
        to estimate the implied spread.
        
        Args:
            home_odds: American odds for home team
            away_odds: American odds for away team
            
        Returns:
            Implied spread for home team
        """
        from scipy.stats import norm
        
        # Convert to probabilities (remove vig by normalizing)
        home_prob = self.american_to_implied(home_odds)
        away_prob = self.american_to_implied(away_odds)
        
        # Normalize to remove vig
        total = home_prob + away_prob
        home_prob_true = home_prob / total
        
        # Convert probability to margin using inverse normal
        # P(home wins) = Φ(margin / σ)
        # margin = σ * Φ^(-1)(P(home wins))
        z = norm.ppf(home_prob_true)
        implied_margin = z * self.margin_std
        
        # Spread is negative of margin (spread of -3 means home favored by 3)
        return -implied_margin
