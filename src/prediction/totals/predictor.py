"""
Totals prediction logic (Full Game + First Half).

NBA v33.0: 4 markets with independent models (1H + FG).

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

from src.prediction.totals.filters import (
    FGTotalFilter,
    FirstHalfTotalFilter,
)
from src.prediction.feature_validation import validate_and_prepare_features
from src.prediction.resolution import resolve_total_two_signal
from src.config import filter_thresholds

logger = logging.getLogger(__name__)


class TotalPredictor:
    """
    Totals predictor for Full Game and First Half.

    Both FG and 1H models required. Missing model = immediate failure.
    """

    def __init__(
        self,
        fg_model,
        fg_feature_columns: List[str],
        fh_model,
        fh_feature_columns: List[str],
    ):
        """
        Initialize totals predictor with ALL required models.

        Args:
            fg_model: Trained FG total model (REQUIRED)
            fg_feature_columns: FG feature column names (REQUIRED)
            fh_model: Trained 1H total model (REQUIRED)
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
        self.fg_filter = FGTotalFilter()
        self.first_half_filter = FirstHalfTotalFilter()

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

        # Prepare features using unified validation
        feature_payload = dict(features)
        feature_payload["total_line"] = total_line
        feature_payload["fg_total_line"] = total_line
        feature_payload["total_vs_predicted"] = feature_payload["predicted_total"] - total_line

        feature_df = pd.DataFrame([feature_payload])
        X, missing = validate_and_prepare_features(
            feature_df,
            self.fg_feature_columns,
            market="fg_total",
        )

        # Get prediction
        total_proba = self.fg_model.predict_proba(X)[0]
        over_prob = float(total_proba[1])
        under_prob = float(total_proba[0])
        classifier_side = "over" if over_prob > 0.5 else "under"
        predicted_total = features["predicted_total"]

        # Calculate edge - ALWAYS use consistent formula
        # Positive edge = model predicts OVER the line
        # Negative edge = model predicts UNDER the line
        raw_edge = predicted_total - total_line
        prediction_side = "over" if raw_edge > 0 else "under"
        edge = abs(raw_edge)  # Use absolute value for display/filtering

        # v33.0.11.0: Classifier sanity check - detect broken/drifted models
        classifier_extreme = (over_prob > 0.99 or over_prob < 0.01)

        resolved = resolve_total_two_signal(
            over_prob=over_prob,
            under_prob=under_prob,
            classifier_side=classifier_side,
            prediction_side=prediction_side,
            edge_abs=edge,
            min_confidence=filter_thresholds.total_min_confidence,
            min_edge=filter_thresholds.total_min_edge,
            classifier_extreme=classifier_extreme,
        )

        return {
            "over_prob": over_prob,
            "under_prob": under_prob,
            "predicted_total": predicted_total,
            "confidence": resolved.confidence,
            "classifier_confidence": resolved.classifier_confidence,
            "bet_side": resolved.bet_side,  # Use prediction_side (from edge) as authoritative
            "edge": edge,
            "raw_edge": raw_edge,  # Signed edge for diagnostics
            "classifier_side": classifier_side,  # What the ML classifier said
            "prediction_side": prediction_side,  # What the point prediction said
            "signals_agree": resolved.signals_agree,
            "classifier_extreme": classifier_extreme,  # v33.0.11.0: True if classifier unreliable
            "model_edge_pct": abs(resolved.confidence - 0.5),
            "passes_filter": resolved.passes_filter,
            "filter_reason": resolved.filter_reason,
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
        # Validate required inputs - NO FALLBACK CALCULATIONS
        if "predicted_total_1h" not in features:
            raise ValueError(
                "predicted_total_1h is REQUIRED in features for 1H total predictions. "
                "Do not use full-game total with arbitrary multipliers."
            )

        # FIX: For 1H models, map 1H features to FG feature names that the model expects
        from src.prediction.engine import map_1h_features_to_fg_names
        features = map_1h_features_to_fg_names(features)

        # Use 1H model ONLY - no fallbacks
        feature_payload = dict(features)
        feature_payload["1h_total_line"] = total_line
        feature_payload["fh_total_line"] = total_line
        feature_payload["total_vs_predicted_1h"] = feature_payload["predicted_total_1h"] - total_line
        feature_payload["fh_total_vs_predicted"] = feature_payload["predicted_total_1h"] - total_line

        feature_df = pd.DataFrame([feature_payload])
        X, missing = validate_and_prepare_features(
            feature_df,
            self.fh_feature_columns,
            market="1h_total",
        )
        total_proba = self.fh_model.predict_proba(X)[0]

        over_prob = float(total_proba[1])
        under_prob = float(total_proba[0])
        classifier_side = "over" if over_prob > 0.5 else "under"
        predicted_total_1h = features["predicted_total_1h"]  # No fallback - already validated

        # Calculate edge - ALWAYS use consistent formula
        # Don't flip sign based on bet_side - maintain signed edge
        raw_edge = predicted_total_1h - total_line
        prediction_side = "over" if raw_edge > 0 else "under"
        edge = abs(raw_edge)  # Use absolute value for display/filtering

        # v33.0.11.0: Classifier sanity check - detect broken/drifted models
        classifier_extreme = (over_prob > 0.99 or over_prob < 0.01)

        resolved = resolve_total_two_signal(
            over_prob=over_prob,
            under_prob=under_prob,
            classifier_side=classifier_side,
            prediction_side=prediction_side,
            edge_abs=edge,
            min_confidence=filter_thresholds.fh_total_min_confidence,
            min_edge=filter_thresholds.fh_total_min_edge,
            classifier_extreme=classifier_extreme,
        )

        return {
            "over_prob": over_prob,
            "under_prob": under_prob,
            "predicted_total": predicted_total_1h,
            "confidence": resolved.confidence,
            "classifier_confidence": resolved.classifier_confidence,
            "bet_side": resolved.bet_side,  # Use prediction_side (from edge) as authoritative
            "edge": edge,
            "raw_edge": raw_edge,  # Signed edge for diagnostics
            "classifier_side": classifier_side,  # What the ML classifier said
            "prediction_side": prediction_side,  # What the point prediction said
            "signals_agree": resolved.signals_agree,
            "classifier_extreme": classifier_extreme,
            "model_edge_pct": abs(resolved.confidence - 0.5),
            "passes_filter": resolved.passes_filter,
            "filter_reason": resolved.filter_reason,
        }
