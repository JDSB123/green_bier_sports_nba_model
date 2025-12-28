"""
Moneyline prediction logic (Full Game + First Half).

NBA v6.0: All 9 markets with independent models.
- FG Moneyline: 65.5% accuracy, +25.1% ROI
- 1H Moneyline: 63.0% accuracy, +19.8% ROI

FEATURE VALIDATION:
    Uses unified validation from src/prediction/feature_validation.py
    Controlled by PREDICTION_FEATURE_MODE environment variable:
    - "strict" (default): Raise error on missing features
    - "warn": Log warning, zero-fill missing features
    - "silent": Zero-fill without logging
"""
from typing import Dict, Any, List
import logging
import pandas as pd
import math

from src.prediction.moneyline.filters import (
    FGMoneylineFilter,
    FirstHalfMoneylineFilter,
)
from src.prediction.confidence import calculate_confidence_from_probabilities
from src.prediction.feature_validation import validate_and_prepare_features

logger = logging.getLogger(__name__)


def validate_american_odds(odds: int, param_name: str = "odds") -> None:
    """
    Validate American odds to prevent division by zero and unrealistic values.

    Args:
        odds: American odds to validate
        param_name: Parameter name for error messages

    Raises:
        ValueError: If odds are invalid
    """
    if odds == 0:
        raise ValueError(f"{param_name} cannot be zero - invalid American odds")
    if odds > 0 and odds < 100:
        logger.warning(f"{param_name}={odds} is unusual - American odds are typically >= +100 or negative")


def american_odds_to_implied_prob(odds: int) -> float:
    """
    Convert American odds to implied probability.

    Args:
        odds: American odds (e.g., -110, +150). Must not be zero.

    Returns:
        Implied probability (0-1)

    Raises:
        ValueError: If odds is zero
    """
    if odds == 0:
        raise ValueError("Cannot convert odds=0 to implied probability - division by zero")

    if odds < 0:
        # Favorite (e.g., -200 = 200/(200+100) = 66.7%)
        return abs(odds) / (abs(odds) + 100)
    else:
        # Underdog (e.g., +150 = 100/(150+100) = 40%)
        return 100 / (odds + 100)


class MoneylinePredictor:
    """
    Moneyline predictor for Full Game and First Half.

    v34.0: Dedicated independent models for both FG and 1H moneyline.
    - FG Moneyline: Uses dedicated FG moneyline model with market signals
    - 1H Moneyline: Uses dedicated 1H moneyline model (not derived from spread)
    """

    def __init__(
        self,
        model,
        feature_columns: List[str],
        fh_model,
        fh_feature_columns: List[str],
        # v34.0: Dedicated 1H moneyline model (optional for backwards compat)
        fh_ml_model=None,
        fh_ml_feature_columns: List[str] = None,
    ):
        """
        Initialize moneyline predictor with models.

        Args:
            model: Trained FG moneyline model (REQUIRED)
            feature_columns: FG feature column names (REQUIRED)
            fh_model: Trained 1H spread model (used as fallback if fh_ml_model not provided)
            fh_feature_columns: 1H spread feature column names
            fh_ml_model: Trained dedicated 1H moneyline model (v34.0, preferred)
            fh_ml_feature_columns: 1H moneyline feature column names

        Raises:
            ValueError: If FG model or features are None
        """
        # FG model is always required
        if model is None:
            raise ValueError("model (FG) is REQUIRED - cannot be None")
        if feature_columns is None:
            raise ValueError("feature_columns (FG) is REQUIRED - cannot be None")

        self.model = model
        self.feature_columns = feature_columns

        # v34.0: Dedicated 1H moneyline model (REQUIRED - no fallback)
        self.fh_ml_model = fh_ml_model
        self.fh_ml_feature_columns = fh_ml_feature_columns or []
        self.has_dedicated_1h_model = fh_ml_model is not None

        # Legacy params kept for backwards compatibility but not used
        self.fh_model = fh_model
        self.fh_feature_columns = fh_feature_columns or []

        if self.has_dedicated_1h_model:
            logger.info("[1H_ML] Using dedicated 1H moneyline model (v34.0)")

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

        Raises:
            ValueError: If odds are invalid (zero)
        """
        # Validate odds
        validate_american_odds(home_odds, "home_odds")
        validate_american_odds(away_odds, "away_odds")

        # Prepare features using unified validation
        feature_df = pd.DataFrame([features])
        X, missing = validate_and_prepare_features(
            feature_df,
            self.feature_columns,
            market="fg_moneyline",
        )

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

        v34.0: Uses dedicated 1H moneyline model if available, otherwise
        falls back to spread-derived approach.

        Args:
            features: Feature dictionary (REQUIRED)
            home_odds: Home team 1H American odds (REQUIRED)
            away_odds: Away team 1H American odds (REQUIRED)

        Returns:
            Prediction dictionary

        Raises:
            ValueError: If odds are invalid (zero)
        """
        # Validate odds
        validate_american_odds(home_odds, "home_odds")
        validate_american_odds(away_odds, "away_odds")

        # v34.0: Require dedicated 1H moneyline model - NO fallback to derived approach
        if not self.has_dedicated_1h_model:
            raise ValueError(
                "1H moneyline model not loaded. Dedicated 1H ML model is REQUIRED in v34.0. "
                "Train with: python scripts/train_models.py"
            )

        # Prepare features for dedicated 1H ML model
        feature_df = pd.DataFrame([features])
        X, missing = validate_and_prepare_features(
            feature_df,
            self.fh_ml_feature_columns,
            market="1h_moneyline",
        )

        # Get predictions from dedicated 1H moneyline model
        # Model is trained on home_1h_win (1 = home leads at half, 0 = away leads)
        ml_proba = self.fh_ml_model.predict_proba(X)[0]
        home_win_prob = float(ml_proba[1])  # Class 1 = home leads
        away_win_prob = float(ml_proba[0])  # Class 0 = away leads

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
