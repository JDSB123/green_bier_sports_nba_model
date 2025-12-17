"""
Core prediction engine for spreads and totals (full game + first half).
"""
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np

from src.prediction.models import load_spread_model, load_total_model
from src.prediction.filters import (
    SpreadFilter,
    TotalFilter,
    FirstHalfSpreadFilter,
    FirstHalfTotalFilter,
)


class PredictionEngine:
    """
    Core prediction engine supporting spreads and totals (full game + first half).

    Modular design with smart filtering based on backtest validation.
    """

    def __init__(
        self,
        models_dir: Path,
        spread_filter: Optional[SpreadFilter] = None,
        total_filter: Optional[TotalFilter] = None,
        first_half_spread_filter: Optional[FirstHalfSpreadFilter] = None,
        first_half_total_filter: Optional[FirstHalfTotalFilter] = None,
    ):
        """
        Initialize prediction engine.

        Args:
            models_dir: Path to models directory
            spread_filter: SpreadFilter instance (None = use defaults)
            total_filter: TotalFilter instance (None = use defaults)
            first_half_spread_filter: FirstHalfSpreadFilter instance (None = use defaults)
            first_half_total_filter: FirstHalfTotalFilter instance (None = use defaults)

        Note:
            Currently uses FG models for 1H predictions (scaled appropriately).
            Dedicated 1H models would improve accuracy.
        """
        self.models_dir = models_dir

        # Load models (using FG models for both FG and 1H predictions)
        self.spread_model, self.spread_features = load_spread_model(models_dir)
        self.total_model, self.total_features = load_total_model(models_dir)

        # Initialize filters
        self.spread_filter = spread_filter or SpreadFilter()
        self.total_filter = total_filter or TotalFilter()
        self.first_half_spread_filter = first_half_spread_filter or FirstHalfSpreadFilter()
        self.first_half_total_filter = first_half_total_filter or FirstHalfTotalFilter()

    def predict_spread(
        self,
        features: Dict[str, float],
        spread_line: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Generate spread prediction for a game.

        Args:
            features: Feature dictionary for the game
            spread_line: Vegas spread line (home perspective)

        Returns:
            Dictionary with:
                - home_cover_prob: Probability home covers
                - away_cover_prob: Probability away covers
                - predicted_margin: Predicted margin (home perspective)
                - confidence: Max(home_cover_prob, away_cover_prob)
                - bet_side: "home" or "away"
                - edge: Predicted margin - spread line (if line provided)
                - model_edge_pct: abs(confidence - 0.5)
                - passes_filter: bool
                - filter_reason: str or None
        """
        # Create feature dataframe
        feature_df = pd.DataFrame([features])

        # Add missing features with 0
        missing = set(self.spread_features) - set(feature_df.columns)
        for col in missing:
            feature_df[col] = 0

        # Align columns
        X = feature_df[self.spread_features]

        # Get prediction
        spread_proba = self.spread_model.predict_proba(X)[0]
        home_cover_prob = float(spread_proba[1])
        away_cover_prob = float(spread_proba[0])

        # Determine bet side
        if home_cover_prob > 0.5:
            bet_side = "home"
            confidence = home_cover_prob
        else:
            bet_side = "away"
            confidence = away_cover_prob

        # Get predicted margin from features
        predicted_margin = features.get("predicted_margin", 0.0)

        # Calculate edge if spread line provided
        edge = None
        if spread_line is not None:
            edge = predicted_margin - spread_line

        # Apply smart filter
        passes_filter = True
        filter_reason = None
        if spread_line is not None:
            passes_filter, filter_reason = self.spread_filter.should_bet(
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

    def predict_total(
        self,
        features: Dict[str, float],
        total_line: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Generate totals prediction for a game.

        Args:
            features: Feature dictionary for the game
            total_line: Vegas total line

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
        missing = set(self.total_features) - set(feature_df.columns)
        for col in missing:
            feature_df[col] = 0

        # Align columns
        X = feature_df[self.total_features]

        # Get prediction
        total_proba = self.total_model.predict_proba(X)[0]
        over_prob = float(total_proba[1])
        under_prob = float(total_proba[0])

        # Determine bet side
        if over_prob > 0.5:
            bet_side = "over"
            confidence = over_prob
        else:
            bet_side = "under"
            confidence = under_prob

        # Get predicted total from features
        predicted_total = features.get("predicted_total", 0.0)

        # Calculate edge if total line provided
        edge = None
        if total_line is not None:
            if bet_side == "over":
                edge = predicted_total - total_line
            else:
                edge = total_line - predicted_total

        # Apply smart filter
        passes_filter, filter_reason = self.total_filter.should_bet(
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

    def predict_game(
        self,
        features: Dict[str, float],
        spread_line: Optional[float] = None,
        total_line: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Generate both spread and total predictions for a game.

        Args:
            features: Feature dictionary for the game
            spread_line: Vegas spread line (home perspective)
            total_line: Vegas total line

        Returns:
            Dictionary with both spread and total predictions
        """
        spread_pred = self.predict_spread(features, spread_line)
        total_pred = self.predict_total(features, total_line)

        return {
            "spread": spread_pred,
            "total": total_pred,
        }

    def predict_first_half_spread(
        self,
        features: Dict[str, float],
        first_half_spread_line: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Generate first half spread prediction.

        Note: Uses FG spread model with scaled predictions (~50% of FG).

        Args:
            features: Feature dictionary for the game
            first_half_spread_line: Vegas 1H spread line (home perspective)

        Returns:
            Dictionary with 1H spread prediction
        """
        # Use FG model
        feature_df = pd.DataFrame([features])
        missing = set(self.spread_features) - set(feature_df.columns)
        for col in missing:
            feature_df[col] = 0
        X = feature_df[self.spread_features]

        # Get prediction
        spread_proba = self.spread_model.predict_proba(X)[0]
        home_cover_prob = float(spread_proba[1])
        away_cover_prob = float(spread_proba[0])

        # Determine bet side
        if home_cover_prob > 0.5:
            bet_side = "home"
            confidence = home_cover_prob
        else:
            bet_side = "away"
            confidence = away_cover_prob

        # Scale FG predicted margin to 1H (~50%)
        fg_predicted_margin = features.get("predicted_margin", 0.0)
        predicted_margin_1h = fg_predicted_margin * 0.5

        # Calculate edge if 1H spread line provided
        edge = None
        if first_half_spread_line is not None:
            edge = predicted_margin_1h - first_half_spread_line

        # Apply 1H spread filter
        passes_filter = True
        filter_reason = None
        if first_half_spread_line is not None:
            passes_filter, filter_reason = self.first_half_spread_filter.should_bet(
                spread_line=first_half_spread_line,
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

    def predict_first_half_total(
        self,
        features: Dict[str, float],
        first_half_total_line: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Generate first half total prediction.

        Note: Uses FG totals model with scaled predictions (~50% of FG).

        Args:
            features: Feature dictionary for the game
            first_half_total_line: Vegas 1H total line

        Returns:
            Dictionary with 1H total prediction
        """
        # Use FG model
        feature_df = pd.DataFrame([features])
        missing = set(self.total_features) - set(feature_df.columns)
        for col in missing:
            feature_df[col] = 0
        X = feature_df[self.total_features]

        # Get prediction
        total_proba = self.total_model.predict_proba(X)[0]
        over_prob = float(total_proba[1])
        under_prob = float(total_proba[0])

        # Determine bet side
        if over_prob > 0.5:
            bet_side = "over"
            confidence = over_prob
        else:
            bet_side = "under"
            confidence = under_prob

        # Scale FG predicted total to 1H (~50%)
        fg_predicted_total = features.get("predicted_total", 0.0)
        predicted_total_1h = fg_predicted_total * 0.5

        # Calculate edge if 1H total line provided
        edge = None
        if first_half_total_line is not None:
            if bet_side == "over":
                edge = predicted_total_1h - first_half_total_line
            else:
                edge = first_half_total_line - predicted_total_1h

        # Apply 1H total filter
        passes_filter, filter_reason = self.first_half_total_filter.should_bet(
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

    def predict_game_all_markets(
        self,
        features: Dict[str, float],
        spread_line: Optional[float] = None,
        total_line: Optional[float] = None,
        first_half_spread_line: Optional[float] = None,
        first_half_total_line: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Generate predictions for all markets (FG + 1H).

        Args:
            features: Feature dictionary for the game
            spread_line: Vegas FG spread line (home perspective)
            total_line: Vegas FG total line
            first_half_spread_line: Vegas 1H spread line (home perspective)
            first_half_total_line: Vegas 1H total line

        Returns:
            Dictionary with all market predictions:
                - full_game: {spread, total}
                - first_half: {spread, total}
        """
        # Full game predictions
        fg_predictions = self.predict_game(features, spread_line, total_line)

        # First half predictions
        fh_spread = self.predict_first_half_spread(features, first_half_spread_line)
        fh_total = self.predict_first_half_total(features, first_half_total_line)

        return {
            "full_game": fg_predictions,
            "first_half": {
                "spread": fh_spread,
                "total": fh_total,
            },
        }
