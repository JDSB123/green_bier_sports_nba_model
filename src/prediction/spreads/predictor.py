"""
Spread prediction logic (Full Game + First Half).
"""
from typing import Dict, Any, Optional
import pandas as pd

from src.prediction.spreads.filters import FGSpreadFilter, FirstHalfSpreadFilter
from src.prediction.confidence import calculate_confidence_from_probabilities


class SpreadPredictor:
    """
    Handles spread predictions for both full game and first half.

    Uses separate models for FG and 1H trained on respective data.
    """

    def __init__(
        self,
        fg_model,
        fg_feature_columns: list,
        fh_model=None,
        fh_feature_columns: Optional[list] = None,
        fg_filter: Optional[FGSpreadFilter] = None,
        first_half_filter: Optional[FirstHalfSpreadFilter] = None,
    ):
        """
        Initialize spread predictor.

        Args:
            fg_model: Trained FG spread model
            fg_feature_columns: List of FG feature column names
            fh_model: Trained 1H spread model (None = fallback to FG model)
            fh_feature_columns: List of 1H feature column names
            fg_filter: Full game spread filter (None = use defaults)
            first_half_filter: First half spread filter (None = use defaults)
        """
        self.fg_model = fg_model
        self.fg_feature_columns = fg_feature_columns
        self.fh_model = fh_model
        self.fh_feature_columns = fh_feature_columns or fg_feature_columns
        self.fg_filter = fg_filter or FGSpreadFilter()
        self.first_half_filter = first_half_filter or FirstHalfSpreadFilter()

    def predict_full_game(
        self,
        features: Dict[str, float],
        spread_line: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Generate full game spread prediction.

        Args:
            features: Feature dictionary
            spread_line: Vegas spread line (home perspective)

        Returns:
            Prediction dictionary with probabilities, edge, filter status
        """
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

        # Calculate confidence from probabilities using entropy-based approach
        # This accounts for uncertainty rather than using raw probability
        confidence = calculate_confidence_from_probabilities(home_cover_prob, away_cover_prob)

        # Determine bet side
        if home_cover_prob > 0.5:
            bet_side = "home"
        else:
            bet_side = "away"

        # Get predicted margin
        predicted_margin = features.get("predicted_margin", 0.0)

        # Calculate edge
        edge = None
        if spread_line is not None:
            edge = predicted_margin - spread_line

        # Apply filter
        passes_filter = True
        filter_reason = None
        if spread_line is not None:
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
        spread_line: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Generate first half spread prediction.

        Uses dedicated 1H model if available, else falls back to FG model.

        Args:
            features: Feature dictionary
            spread_line: Vegas 1H spread line (home perspective)

        Returns:
            Prediction dictionary with probabilities, edge, filter status
        """
        # Use 1H model if available
        if self.fh_model is not None:
            feature_df = pd.DataFrame([features])
            missing = set(self.fh_feature_columns) - set(feature_df.columns)
            for col in missing:
                feature_df[col] = 0
            X = feature_df[self.fh_feature_columns]
            spread_proba = self.fh_model.predict_proba(X)[0]
        else:
            # Fallback to FG model
            feature_df = pd.DataFrame([features])
            missing = set(self.fg_feature_columns) - set(feature_df.columns)
            for col in missing:
                feature_df[col] = 0
            X = feature_df[self.fg_feature_columns]
            spread_proba = self.fg_model.predict_proba(X)[0]

        home_cover_prob = float(spread_proba[1])
        away_cover_prob = float(spread_proba[0])

        # Calculate confidence from probabilities using entropy-based approach
        # This accounts for uncertainty rather than using raw probability
        confidence = calculate_confidence_from_probabilities(home_cover_prob, away_cover_prob)

        # Determine bet side
        if home_cover_prob > 0.5:
            bet_side = "home"
        else:
            bet_side = "away"

        # Get predicted margin from features
        predicted_margin = features.get("predicted_margin_1h", 0.0)

        # Calculate edge
        edge = None
        if spread_line is not None:
            edge = predicted_margin - spread_line

        # Apply 1H filter
        passes_filter = True
        filter_reason = None
        if spread_line is not None:
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
