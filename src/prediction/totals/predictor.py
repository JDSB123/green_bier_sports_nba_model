"""
Totals prediction logic for Full Game, First Half, and First Quarter.

STRICT MODE: No fallbacks. Each market requires its own trained model.
"""
from typing import Dict, Any
import pandas as pd

from src.prediction.totals.filters import (
    FGTotalFilter,
    FirstHalfTotalFilter,
    FirstQuarterTotalFilter,
)
from src.prediction.confidence import calculate_confidence_from_probabilities


class TotalPredictor:
    """
    Totals predictor for Full Game, First Half, and First Quarter markets.

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
        Initialize totals predictor with ALL required models.

        Args:
            fg_model: Trained FG total model (REQUIRED)
            fg_feature_columns: FG feature column names (REQUIRED)
            fh_model: Trained 1H total model (REQUIRED)
            fh_feature_columns: 1H feature column names (REQUIRED)
            fq_model: Trained Q1 total model (REQUIRED)
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
        self.fg_filter = FGTotalFilter()
        self.first_half_filter = FirstHalfTotalFilter()
        self.first_quarter_filter = FirstQuarterTotalFilter()

    def predict_full_game(
        self,
        features: Dict[str, float],
        total_line: float,
    ) -> Dict[str, Any]:
        """
        Generate full game total prediction.

        Args:
            features: Feature dictionary (REQUIRED, must contain predicted_total)
            total_line: Vegas FG total line (REQUIRED)

        Returns:
            Prediction dictionary
        """
        # Validate required inputs
        if "predicted_total" not in features:
            raise ValueError("predicted_total is REQUIRED in features for FG total predictions")

        # Prepare features
        feature_df = pd.DataFrame([features])
        missing = set(self.fg_feature_columns) - set(feature_df.columns)
        for col in missing:
            feature_df[col] = 0
        X = feature_df[self.fg_feature_columns]

        # Get prediction
        total_proba = self.fg_model.predict_proba(X)[0]
        over_prob = float(total_proba[1])
        under_prob = float(total_proba[0])
        confidence = calculate_confidence_from_probabilities(over_prob, under_prob)
        bet_side = "over" if over_prob > 0.5 else "under"
        predicted_total = features["predicted_total"]

        # Calculate edge
        if bet_side == "over":
            edge = predicted_total - total_line
        else:
            edge = total_line - predicted_total

        passes_filter, filter_reason = self.fg_filter.should_bet(confidence=confidence)

        return {
            "over_prob": over_prob,
            "under_prob": under_prob,
            "predicted_total": predicted_total,
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
        total_line: float,
    ) -> Dict[str, Any]:
        """
        Generate first half total prediction.

        Args:
            features: Feature dictionary (REQUIRED, must contain predicted_total_1h)
            total_line: Vegas 1H total line (REQUIRED)

        Returns:
            Prediction dictionary
        """
        # Validate required inputs
        if "predicted_total_1h" not in features:
            raise ValueError("predicted_total_1h is REQUIRED in features for 1H total predictions")

        # Use 1H model ONLY - no fallbacks
        feature_df = pd.DataFrame([features])
        missing = set(self.fh_feature_columns) - set(feature_df.columns)
        for col in missing:
            feature_df[col] = 0
        X = feature_df[self.fh_feature_columns]
        total_proba = self.fh_model.predict_proba(X)[0]

        over_prob = float(total_proba[1])
        under_prob = float(total_proba[0])
        confidence = calculate_confidence_from_probabilities(over_prob, under_prob)
        bet_side = "over" if over_prob > 0.5 else "under"
        predicted_total = features["predicted_total_1h"]

        # Calculate edge
        if bet_side == "over":
            edge = predicted_total - total_line
        else:
            edge = total_line - predicted_total

        passes_filter, filter_reason = self.first_half_filter.should_bet(confidence=confidence)

        return {
            "over_prob": over_prob,
            "under_prob": under_prob,
            "predicted_total": predicted_total,
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
        total_line: float,
    ) -> Dict[str, Any]:
        """
        Generate first quarter total prediction.

        Args:
            features: Feature dictionary (REQUIRED, must contain predicted_total_q1)
            total_line: Vegas Q1 total line (REQUIRED)

        Returns:
            Prediction dictionary
        """
        # Validate required inputs
        if "predicted_total_q1" not in features:
            raise ValueError("predicted_total_q1 is REQUIRED in features for Q1 total predictions")

        # Use Q1 model ONLY - no fallbacks
        feature_df = pd.DataFrame([features])
        missing = set(self.fq_feature_columns) - set(feature_df.columns)
        for col in missing:
            feature_df[col] = 0
        X = feature_df[self.fq_feature_columns]
        total_proba = self.fq_model.predict_proba(X)[0]

        over_prob = float(total_proba[1])
        under_prob = float(total_proba[0])
        confidence = calculate_confidence_from_probabilities(over_prob, under_prob)
        bet_side = "over" if over_prob > 0.5 else "under"
        predicted_total = features["predicted_total_q1"]

        # Calculate edge
        if bet_side == "over":
            edge = predicted_total - total_line
        else:
            edge = total_line - predicted_total

        passes_filter, filter_reason = self.first_quarter_filter.should_bet(confidence=confidence)

        return {
            "over_prob": over_prob,
            "under_prob": under_prob,
            "predicted_total": predicted_total,
            "confidence": confidence,
            "bet_side": bet_side,
            "edge": edge,
            "model_edge_pct": abs(confidence - 0.5),
            "passes_filter": passes_filter,
            "filter_reason": filter_reason,
        }
