"""
Moneyline prediction logic (Full Game + First Half).

STRICT MODE: No fallbacks. Each market requires its own trained model.
Only BACKTESTED markets supported: FG and 1H.
"""
from typing import Dict, Any
import pandas as pd

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
    Moneyline predictor for Full Game and First Half markets.

    STRICT MODE:
    - Each market uses its OWN dedicated model
    - NO fallbacks to other models
    - Missing model = immediate failure
    """

    def __init__(
        self,
        model,
        feature_columns: list,
        fh_model,
        fh_feature_columns: list,
    ):
        """
        Initialize moneyline predictor with ALL required models.

        Args:
            model: Trained FG moneyline/spread model (REQUIRED)
            feature_columns: FG feature column names (REQUIRED)
            fh_model: Trained 1H model (REQUIRED)
            fh_feature_columns: 1H feature column names (REQUIRED)

        Raises:
            ValueError: If any required model or features are None
        """
        # Validate ALL inputs - NO NONE ALLOWED
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

        # Get win probabilities
        spread_proba = self.model.predict_proba(X)[0]
        home_win_prob = float(spread_proba[1])
        away_win_prob = float(spread_proba[0])
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

        spread_proba = self.fh_model.predict_proba(X)[0]
        home_win_prob = float(spread_proba[1])
        away_win_prob = float(spread_proba[0])
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
