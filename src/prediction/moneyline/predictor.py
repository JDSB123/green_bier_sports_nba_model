"""
Moneyline prediction logic (Full Game + First Half).

NBA v6.0: All 9 markets with independent models.
- FG Moneyline: 65.5% accuracy, +25.1% ROI
- 1H Moneyline: 63.0% accuracy, +19.8% ROI
"""
from typing import Dict, Any, List
import pandas as pd
import math

from src.prediction.moneyline.filters import (
    FGMoneylineFilter,
    FirstHalfMoneylineFilter,
)
from src.prediction.confidence import calculate_confidence_from_probabilities


def american_odds_to_implied_prob(odds: int) -> float:
    """
    Convert American odds to implied probability.

    Args:
        odds: American odds (e.g., -110, +150)

    Returns:
        Implied probability (0-1)
    """
    if odds < 0:
        # Favorite (e.g., -200 = 200/(200+100) = 66.7%)
        return abs(odds) / (abs(odds) + 100)
    else:
        # Underdog (e.g., +150 = 100/(150+100) = 40%)
        return 100 / (odds + 100)


class MoneylinePredictor:
    """
    Moneyline predictor for Full Game and First Half.

    NBA v6.0: Both FG and 1H models required.
    - FG Moneyline: 65.5% accuracy, +25.1% ROI
    - 1H Moneyline: 63.0% accuracy, +19.8% ROI
    """

    def __init__(
        self,
        model,
        feature_columns: List[str],
        fh_model,
        fh_feature_columns: List[str],
    ):
        """
        Initialize moneyline predictor with ALL required models.

        Args:
            model: Trained FG moneyline model (REQUIRED)
            feature_columns: FG feature column names (REQUIRED)
            fh_model: Trained 1H model (REQUIRED - uses spread model for ML conversion)
            fh_feature_columns: 1H feature column names (REQUIRED)

        Raises:
            ValueError: If any model or features are None
        """
        # Validate ALL inputs - REQUIRED
        if model is None:
            raise ValueError("model (FG) is REQUIRED - cannot be None")
        if feature_columns is None:
            raise ValueError("feature_columns (FG) is REQUIRED - cannot be None")
        if fh_model is None:
            raise ValueError("fh_model is REQUIRED - cannot be None")
        if fh_feature_columns is None:
            raise ValueError("fh_feature_columns is REQUIRED - cannot be None")

        self.model = model
        self.feature_columns = feature_columns
        self.fh_model = fh_model
        self.fh_feature_columns = fh_feature_columns

        # Filters use defaults - these are config, not models
        self.fg_filter = FGMoneylineFilter()
        self.first_half_filter = FirstHalfMoneylineFilter()

    def predict_full_game(
        self,
        features: Dict[str, float],
        home_odds: int,
        away_odds: int,
    ) -> Dict[str, Any]:
        """
        Generate full game moneyline prediction.

        Args:
            features: Feature dictionary (REQUIRED)
            home_odds: Home team American odds (REQUIRED, e.g., -150)
            away_odds: Away team American odds (REQUIRED, e.g., +130)

        Returns:
            Prediction dictionary
        """
        # Prepare features
        feature_df = pd.DataFrame([features])
        missing = set(self.feature_columns) - set(feature_df.columns)
        for col in missing:
            feature_df[col] = 0
        X = feature_df[self.feature_columns]

        # Get predictions from dedicated moneyline model
        # Model is trained on home_win (1 = home win, 0 = away win)
        ml_proba = self.model.predict_proba(X)[0]
        home_win_prob = float(ml_proba[1])  # Class 1 = home win
        away_win_prob = float(ml_proba[0])  # Class 0 = away win
        
        confidence = calculate_confidence_from_probabilities(home_win_prob, away_win_prob)
        predicted_winner = "home" if home_win_prob > 0.5 else "away"

        # Calculate implied probabilities and edges
        home_implied_prob = american_odds_to_implied_prob(home_odds)
        away_implied_prob = american_odds_to_implied_prob(away_odds)
        home_edge = home_win_prob - home_implied_prob
        away_edge = away_win_prob - away_implied_prob

        # Determine recommended bet (highest positive edge)
        if home_edge > away_edge and home_edge > 0:
            recommended_bet = "home"
            bet_prob = home_win_prob
            bet_implied = home_implied_prob
        elif away_edge > 0:
            recommended_bet = "away"
            bet_prob = away_win_prob
            bet_implied = away_implied_prob
        else:
            recommended_bet = None
            bet_prob = None
            bet_implied = None

        # Apply filter
        passes_filter = True
        filter_reason = None
        if recommended_bet and bet_prob and bet_implied:
            passes_filter, filter_reason = self.fg_filter.should_bet(
                predicted_prob=bet_prob,
                implied_prob=bet_implied,
            )

        return {
            "home_win_prob": home_win_prob,
            "away_win_prob": away_win_prob,
            "predicted_winner": predicted_winner,
            "confidence": confidence,
            "home_implied_prob": home_implied_prob,
            "away_implied_prob": away_implied_prob,
            "home_edge": home_edge,
            "away_edge": away_edge,
            "recommended_bet": recommended_bet if passes_filter else None,
            "passes_filter": passes_filter,
            "filter_reason": filter_reason,
        }

    def predict_first_half(
        self,
        features: Dict[str, float],
        home_odds: int,
        away_odds: int,
    ) -> Dict[str, Any]:
        """
        Generate first half moneyline prediction.

        Args:
            features: Feature dictionary (REQUIRED)
            home_odds: Home team 1H American odds (REQUIRED)
            away_odds: Away team 1H American odds (REQUIRED)

        Returns:
            Prediction dictionary
        """
        # Prepare features - use 1H model ONLY
        feature_df = pd.DataFrame([features])
        missing = set(self.fh_feature_columns) - set(feature_df.columns)
        for col in missing:
            feature_df[col] = 0
        X = feature_df[self.fh_feature_columns]

        # Get spread cover probabilities (NOT win probabilities)
        spread_proba = self.fh_model.predict_proba(X)[0]
        home_cover_prob = float(spread_proba[1])
        away_cover_prob = float(spread_proba[0])
        
        # Convert spread cover probabilities to actual win probabilities
        # Use predicted margin for first half
        predicted_margin_1h = features.get("predicted_margin_1h", 0.0)
        k = 0.16  # NBA-specific constant
        
        if predicted_margin_1h != 0:
            # Use logistic function to convert margin to win probability
            home_win_prob_raw = 1.0 / (1.0 + math.exp(-k * predicted_margin_1h))
            home_win_prob = max(0.05, min(0.95, home_win_prob_raw))
            away_win_prob = 1.0 - home_win_prob
        else:
            # Fallback: adjust cover prob to win prob
            if home_cover_prob > 0.5:
                home_win_prob = 0.5 + (home_cover_prob - 0.5) * 1.3
                home_win_prob = min(0.95, home_win_prob)
            else:
                home_win_prob = 0.5 - (0.5 - home_cover_prob) * 1.3
                home_win_prob = max(0.05, home_win_prob)
            away_win_prob = 1.0 - home_win_prob
        
        confidence = calculate_confidence_from_probabilities(home_win_prob, away_win_prob)
        predicted_winner = "home" if home_win_prob > 0.5 else "away"

        # Calculate implied probabilities and edges
        home_implied_prob = american_odds_to_implied_prob(home_odds)
        away_implied_prob = american_odds_to_implied_prob(away_odds)
        home_edge = home_win_prob - home_implied_prob
        away_edge = away_win_prob - away_implied_prob

        # Determine recommended bet (highest positive edge)
        if home_edge > away_edge and home_edge > 0:
            recommended_bet = "home"
            bet_prob = home_win_prob
            bet_implied = home_implied_prob
        elif away_edge > 0:
            recommended_bet = "away"
            bet_prob = away_win_prob
            bet_implied = away_implied_prob
        else:
            recommended_bet = None
            bet_prob = None
            bet_implied = None

        # Apply filter
        passes_filter = True
        filter_reason = None
        if recommended_bet and bet_prob and bet_implied:
            passes_filter, filter_reason = self.first_half_filter.should_bet(
                predicted_prob=bet_prob,
                implied_prob=bet_implied,
            )

        return {
            "home_win_prob": home_win_prob,
            "away_win_prob": away_win_prob,
            "predicted_winner": predicted_winner,
            "confidence": confidence,
            "home_implied_prob": home_implied_prob,
            "away_implied_prob": away_implied_prob,
            "home_edge": home_edge,
            "away_edge": away_edge,
            "recommended_bet": recommended_bet if passes_filter else None,
            "passes_filter": passes_filter,
            "filter_reason": filter_reason,
        }
