"""
Spread prediction logic (Full Game + First Half + First Quarter).

STRICT MODE: No fallbacks. Each market requires its own trained model.
"""
from typing import Dict, Any
import pandas as pd

from src.prediction.spreads.filters import (
    FGSpreadFilter,
    FirstHalfSpreadFilter,
    FirstQuarterSpreadFilter,
)
from src.prediction.confidence import calculate_confidence_from_probabilities


class MissingModelError(Exception):
    """Raised when attempting to use a model that wasn't provided."""
    pass


class SpreadPredictor:
    """
    Handles spread predictions for Full Game, First Half, and First Quarter.

    STRICT MODE:
    - Each market uses its OWN dedicated model
    - NO fallbacks to other models
    - Missing model = immediate failure
    """

    def __init__(
        self,
        fg_model,
        fg_feature_columns: list,
        fh_model,
        fh_feature_columns: list,
        fq_model,
        fq_feature_columns: list,
    ):
        """
        Initialize spread predictor with ALL required models.

        Args:
            fg_model: Trained FG spread model (REQUIRED)
            fg_feature_columns: FG feature column names (REQUIRED)
            fh_model: Trained 1H spread model (REQUIRED)
            fh_feature_columns: 1H feature column names (REQUIRED)
            fq_model: Trained Q1 spread model (REQUIRED)
            fq_feature_columns: Q1 feature column names (REQUIRED)

        Raises:
            ValueError: If any required model or features are None
        """
        # Validate ALL inputs - NO NONE ALLOWED
        if fg_model is None:
            raise ValueError("fg_model is REQUIRED - cannot be None")
        if fg_feature_columns is None:
            raise ValueError("fg_feature_columns is REQUIRED - cannot be None")
        if fh_model is None:
            raise ValueError("fh_model is REQUIRED - cannot be None")
        if fh_feature_columns is None:
            raise ValueError("fh_feature_columns is REQUIRED - cannot be None")
        if fq_model is None:
            raise ValueError("fq_model is REQUIRED - cannot be None")
        if fq_feature_columns is None:
            raise ValueError("fq_feature_columns is REQUIRED - cannot be None")

        self.fg_model = fg_model
        self.fg_feature_columns = fg_feature_columns
        self.fh_model = fh_model
        self.fh_feature_columns = fh_feature_columns
        self.fq_model = fq_model
        self.fq_feature_columns = fq_feature_columns
        
        # Filters use defaults - these are config, not models
        self.fg_filter = FGSpreadFilter()
        self.first_half_filter = FirstHalfSpreadFilter()
        self.first_quarter_filter = FirstQuarterSpreadFilter()

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

        # Prepare features
        feature_df = pd.DataFrame([features])
        missing = set(self.fg_feature_columns) - set(feature_df.columns)
        for col in missing:
            feature_df[col] = 0
        X = feature_df[self.fg_feature_columns]

        # Get prediction
        spread_proba = self.fg_model.predict_proba(X)[0]
        home_cover_prob = float(spread_proba[1])
        away_cover_prob = float(spread_proba[0])
        confidence = calculate_confidence_from_probabilities(home_cover_prob, away_cover_prob)
        bet_side = "home" if home_cover_prob > 0.5 else "away"
        predicted_margin = features["predicted_margin"]
        edge = predicted_margin - spread_line

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

    def predict_first_quarter(
        self,
        features: Dict[str, float],
        spread_line: float,
    ) -> Dict[str, Any]:
        """
        Generate first quarter spread prediction.

        Args:
            features: Feature dictionary (REQUIRED)
            spread_line: Vegas Q1 spread line (REQUIRED)

        Returns:
            Prediction dictionary
        """
        # Use Q1 model ONLY - no fallbacks
        feature_df = pd.DataFrame([features])
        missing = set(self.fq_feature_columns) - set(feature_df.columns)
        for col in missing:
            feature_df[col] = 0
        X = feature_df[self.fq_feature_columns]
        spread_proba = self.fq_model.predict_proba(X)[0]

        home_cover_prob = float(spread_proba[1])
        away_cover_prob = float(spread_proba[0])
        confidence = calculate_confidence_from_probabilities(home_cover_prob, away_cover_prob)
        bet_side = "home" if home_cover_prob > 0.5 else "away"
        
        # Q1 predicted margin is REQUIRED in features
        if "predicted_margin_q1" not in features:
            raise ValueError("predicted_margin_q1 is REQUIRED in features for Q1 predictions")
        predicted_margin = features["predicted_margin_q1"]

        edge = predicted_margin - spread_line
        passes_filter, filter_reason = self.first_quarter_filter.should_bet(
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
        # Validate required inputs
        if "predicted_margin_1h" not in features:
            raise ValueError("predicted_margin_1h is REQUIRED in features for 1H spread predictions")

        # Use 1H model ONLY - no fallbacks
        feature_df = pd.DataFrame([features])
        missing = set(self.fh_feature_columns) - set(feature_df.columns)
        for col in missing:
            feature_df[col] = 0
        X = feature_df[self.fh_feature_columns]
        spread_proba = self.fh_model.predict_proba(X)[0]

        home_cover_prob = float(spread_proba[1])
        away_cover_prob = float(spread_proba[0])
        confidence = calculate_confidence_from_probabilities(home_cover_prob, away_cover_prob)
        bet_side = "home" if home_cover_prob > 0.5 else "away"
        predicted_margin = features["predicted_margin_1h"]
        edge = predicted_margin - spread_line

        passes_filter, filter_reason = self.first_half_filter.should_bet(
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
