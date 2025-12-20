"""
Spread prediction logic (Full Game + First Half).

NBA v6.0: All 9 markets with independent models.
STRICT MODE: No fallbacks. Each market requires its own trained model.
"""
from typing import Dict, Any, List
import logging
import pandas as pd

from src.prediction.spreads.filters import (
    FGSpreadFilter,
    FirstHalfSpreadFilter,
)
from src.prediction.confidence import calculate_confidence_from_probabilities

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

        # Prepare features
        feature_df = pd.DataFrame([features])
        missing = set(self.fg_feature_columns) - set(feature_df.columns)
        if missing:
            logger.warning(f"[fg_spread] Zero-filling {len(missing)} missing features: {sorted(missing)[:5]}...")
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
        if missing:
            logger.warning(f"[1h_spread] Zero-filling {len(missing)} missing features: {sorted(missing)[:5]}...")
        for col in missing:
            feature_df[col] = 0
        X = feature_df[self.fh_feature_columns]
        spread_proba = self.fh_model.predict_proba(X)[0]

        home_cover_prob = float(spread_proba[1])
        away_cover_prob = float(spread_proba[0])
        confidence = calculate_confidence_from_probabilities(home_cover_prob, away_cover_prob)
        bet_side = "home" if home_cover_prob > 0.5 else "away"
        predicted_margin_1h = features.get("predicted_margin_1h", features.get("predicted_margin", 0) * 0.48)
        edge = predicted_margin_1h - spread_line

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
