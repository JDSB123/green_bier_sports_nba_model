"""
Totals prediction logic for full game and first half.
"""
from typing import Dict, Any, Optional
import pandas as pd

from src.prediction.totals.filters import FGTotalFilter, FirstHalfTotalFilter
from src.prediction.confidence import calculate_confidence_from_probabilities


class TotalPredictor:
    """
    Totals predictor for full game and first half markets.

    Uses separate models for FG and 1H trained on respective data.
    """

    def __init__(
        self,
        fg_model,
        fg_feature_columns: list,
        fh_model=None,
        fh_feature_columns: Optional[list] = None,
        fg_filter: Optional[FGTotalFilter] = None,
        first_half_filter: Optional[FirstHalfTotalFilter] = None,
    ):
        """
        Initialize totals predictor.

        Args:
            fg_model: Trained FG total model
            fg_feature_columns: List of FG feature column names
            fh_model: Trained 1H total model (None = fallback to FG model)
            fh_feature_columns: List of 1H feature column names
            fg_filter: FGTotalFilter instance (None = use defaults)
            first_half_filter: FirstHalfTotalFilter instance (None = use defaults)
        """
        self.fg_model = fg_model
        self.fg_feature_columns = fg_feature_columns
        self.fh_model = fh_model
        self.fh_feature_columns = fh_feature_columns or fg_feature_columns

        # Initialize filters
        self.fg_filter = fg_filter or FGTotalFilter()
        self.first_half_filter = first_half_filter or FirstHalfTotalFilter()

    def predict_full_game(
        self,
        features: Dict[str, float],
        total_line: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Generate full game total prediction.

        Args:
            features: Feature dictionary for the game
            total_line: Vegas FG total line

        Returns:
            Dictionary with:
                - over_prob: Probability over
                - under_prob: Probability under
                - predicted_total: Predicted total points
                - confidence: Max(over_prob, under_prob)
                - bet_side: "over" or "under"
                - edge: Predicted total - total line (if line provided)
                - model_edge_pct: abs(confidence - 0.5)
                - passes_filter: bool
                - filter_reason: str or None
        """
        # Create feature dataframe
        feature_df = pd.DataFrame([features])

        # Add missing features with 0
        missing = set(self.fg_feature_columns) - set(feature_df.columns)
        for col in missing:
            feature_df[col] = 0

        # Align columns
        X = feature_df[self.fg_feature_columns]

        # Get prediction
        total_proba = self.fg_model.predict_proba(X)[0]
        over_prob = float(total_proba[1])
        under_prob = float(total_proba[0])

        # Calculate confidence from probabilities using entropy-based approach
        # This accounts for uncertainty rather than using raw probability
        confidence = calculate_confidence_from_probabilities(over_prob, under_prob)

        # Determine bet side
        if over_prob > 0.5:
            bet_side = "over"
        else:
            bet_side = "under"

        # Get predicted total from features
        predicted_total = features.get("predicted_total", 0.0)

        # Calculate edge if total line provided
        edge = None
        if total_line is not None:
            if bet_side == "over":
                edge = predicted_total - total_line
            else:
                edge = total_line - predicted_total

        # Apply FG total filter
        passes_filter, filter_reason = self.fg_filter.should_bet(
            confidence=confidence,
        )

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
        first_half_total_line: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Generate first half total prediction.

        Uses dedicated 1H model if available, else falls back to FG model.

        Args:
            features: Feature dictionary for the game
            first_half_total_line: Vegas 1H total line

        Returns:
            Dictionary with 1H total prediction
        """
        # Use 1H model if available
        if self.fh_model is not None:
            feature_df = pd.DataFrame([features])
            missing = set(self.fh_feature_columns) - set(feature_df.columns)
            for col in missing:
                feature_df[col] = 0
            X = feature_df[self.fh_feature_columns]
            total_proba = self.fh_model.predict_proba(X)[0]
        else:
            # Fallback to FG model
            feature_df = pd.DataFrame([features])
            missing = set(self.fg_feature_columns) - set(feature_df.columns)
            for col in missing:
                feature_df[col] = 0
            X = feature_df[self.fg_feature_columns]
            total_proba = self.fg_model.predict_proba(X)[0]

        over_prob = float(total_proba[1])
        under_prob = float(total_proba[0])

        # Calculate confidence from probabilities using entropy-based approach
        # This accounts for uncertainty rather than using raw probability
        confidence = calculate_confidence_from_probabilities(over_prob, under_prob)

        # Determine bet side
        if over_prob > 0.5:
            bet_side = "over"
        else:
            bet_side = "under"

        # Get predicted total from features
        predicted_total_1h = features.get("predicted_total_1h", 0.0)

        # Calculate edge if 1H total line provided
        edge = None
        if first_half_total_line is not None:
            if bet_side == "over":
                edge = predicted_total_1h - first_half_total_line
            else:
                edge = first_half_total_line - predicted_total_1h

        # Apply 1H total filter
        passes_filter, filter_reason = self.first_half_filter.should_bet(
            confidence=confidence,
        )

        return {
            "over_prob": over_prob,
            "under_prob": under_prob,
            "predicted_total": predicted_total_1h,
            "confidence": confidence,
            "bet_side": bet_side,
            "edge": edge,
            "model_edge_pct": abs(confidence - 0.5),
            "passes_filter": passes_filter,
            "filter_reason": filter_reason,
        }
