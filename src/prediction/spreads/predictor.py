"""
Spread prediction logic (Full Game + First Half).

NBA v6.0: All 9 markets with independent models.

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

from src.prediction.spreads.filters import (
    FGSpreadFilter,
    FirstHalfSpreadFilter,
)
from src.prediction.confidence import calculate_confidence_from_probabilities
from src.prediction.feature_validation import validate_and_prepare_features

logger = logging.getLogger(__name__)


class SpreadPredictor:
    """
    Handles spread predictions for Full Game and First Half.

    NBA v6.0: Both FG and 1H models required.
    STRICT MODE: Missing model = immediate failure.
    """

    def __init__(
        self,
        fg_model,
        fg_feature_columns: List[str],
        fh_model,
        fh_feature_columns: List[str],
    ):
        """
        Initialize spread predictor with ALL required models.

        Args:
            fg_model: Trained FG spread model (REQUIRED)
            fg_feature_columns: FG feature column names (REQUIRED)
            fh_model: Trained 1H spread model (REQUIRED)
            fh_feature_columns: 1H feature column names (REQUIRED)

        Raises:
            ValueError: If any model or features are None
        """
        # Validate ALL inputs - REQUIRED
        if fg_model is None:
            raise ValueError("fg_model is REQUIRED - cannot be None")
        if fg_feature_columns is None:
            raise ValueError("fg_feature_columns is REQUIRED - cannot be None")
        if fh_model is None:
            raise ValueError("fh_model is REQUIRED - cannot be None")
        if fh_feature_columns is None:
            raise ValueError("fh_feature_columns is REQUIRED - cannot be None")

        self.fg_model = fg_model
        self.fg_feature_columns = fg_feature_columns
        self.fh_model = fh_model
        self.fh_feature_columns = fh_feature_columns
        
        # Filters use defaults - these are config, not models
        self.fg_filter = FGSpreadFilter()
        self.first_half_filter = FirstHalfSpreadFilter()

    def predict_full_game(
        self,
        features: Dict[str, float],
        spread_line: float,
    ) -> Dict[str, Any]:
        """
        Generate full game spread prediction.

        Args:
            features: Feature dictionary (REQUIRED, must contain predicted_margin)
            spread_line: Vegas spread line (REQUIRED)

        Returns:
            Prediction dictionary with probabilities, edge, filter status
        """
        # Validate required inputs
        if "predicted_margin" not in features:
            raise ValueError("predicted_margin is REQUIRED in features for FG spread predictions")

        # Prepare features using unified validation
        feature_df = pd.DataFrame([features])
        X, missing = validate_and_prepare_features(
            feature_df,
            self.fg_feature_columns,
            market="fg_spread",
        )

        # Get prediction
        spread_proba = self.fg_model.predict_proba(X)[0]
        home_cover_prob = float(spread_proba[1])
        away_cover_prob = float(spread_proba[0])
        confidence = calculate_confidence_from_probabilities(home_cover_prob, away_cover_prob)
        bet_side = "home" if home_cover_prob > 0.5 else "away"
        predicted_margin = features["predicted_margin"]
        # EDGE CALCULATION (v6.5 fix):
        # spread_line is HOME spread (negative = home favored)
        # predicted_margin is positive when home wins
        # edge = predicted_margin + spread_line
        # Example: spread_line = -7.5, predicted_margin = 2.1
        #   edge = 2.1 + (-7.5) = -5.4 → bet AWAY (they cover)
        edge = predicted_margin + spread_line

        passes_filter, filter_reason = self.fg_filter.should_bet(
            spread_line=spread_line,
            confidence=confidence,
        )

        return {
            "home_cover_prob": home_cover_prob,
            "away_cover_prob": away_cover_prob,
            "predicted_margin": predicted_margin,
            "confidence": confidence,
            "bet_side": bet_side,
            "edge": edge,
            "model_edge_pct": abs(confidence - 0.5),
            "passes_filter": passes_filter,
            "filter_reason": filter_reason,
        }

    def predict_first_half(
        self,
        features: Dict[str, float],
        spread_line: float,
    ) -> Dict[str, Any]:
        """
        Generate first half spread prediction.

        Args:
            features: Feature dictionary (REQUIRED, must contain predicted_margin_1h)
            spread_line: Vegas 1H spread line (REQUIRED)

        Returns:
            Prediction dictionary
        """
        # Validate required inputs - NO FALLBACK CALCULATIONS
        if "predicted_margin_1h" not in features:
            raise ValueError(
                "predicted_margin_1h is REQUIRED in features for 1H spread predictions. "
                "Do not use full-game margin with arbitrary multipliers."
            )

        # Use 1H model ONLY - no fallbacks
        feature_df = pd.DataFrame([features])
        X, missing = validate_and_prepare_features(
            feature_df,
            self.fh_feature_columns,
            market="1h_spread",
        )
        spread_proba = self.fh_model.predict_proba(X)[0]

        home_cover_prob = float(spread_proba[1])
        away_cover_prob = float(spread_proba[0])
        confidence = calculate_confidence_from_probabilities(home_cover_prob, away_cover_prob)
        bet_side = "home" if home_cover_prob > 0.5 else "away"
        predicted_margin_1h = features["predicted_margin_1h"]  # No fallback - already validated
        # EDGE CALCULATION (v6.5 fix): edge = predicted_margin + spread_line
        edge = predicted_margin_1h + spread_line

        passes_filter, filter_reason = self.first_half_filter.should_bet(
            spread_line=spread_line,
            confidence=confidence,
        )

        return {
            "home_cover_prob": home_cover_prob,
            "away_cover_prob": away_cover_prob,
            "predicted_margin": predicted_margin_1h,
            "confidence": confidence,
            "bet_side": bet_side,
            "edge": edge,
            "model_edge_pct": abs(confidence - 0.5),
            "passes_filter": passes_filter,
            "filter_reason": filter_reason,
        }
