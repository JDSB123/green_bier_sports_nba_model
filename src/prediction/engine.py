"""
Unified prediction engine composing all market predictors.

STRICT MODE: No fallbacks, no silent failures. All models must exist.
If a model is missing, initialization FAILS LOUDLY.

Supports 4 BACKTESTED markets:
- Full Game: Spread, Total
- First Half: Spread, Total
"""
from pathlib import Path
from typing import Dict, Any

from src.prediction.spreads import SpreadPredictor
from src.prediction.totals import TotalPredictor
from src.prediction.models import (
    load_spread_model,
    load_total_model,
    load_first_half_spread_model,
    load_first_half_total_model,
)


class ModelNotFoundError(Exception):
    """Raised when a required model file is missing."""
    pass


class UnifiedPredictionEngine:
    """
    Unified prediction engine for NBA betting markets.

    STRICT MODE:
    - 4 REQUIRED models: FG spreads, FG totals, 1H spreads, 1H totals
    - NO fallbacks - missing required model = crash

    Supports 4 BACKTESTED markets:
    - Full Game: Spread (60.6% acc), Total (59.2% acc)
    - First Half: Spread (55.9% acc), Total (58.1% acc)
    """

    def __init__(self, models_dir: Path):
        """
        Initialize unified prediction engine.

        Args:
            models_dir: Path to models directory (must contain 4 required models:
                       spreads, totals, 1H spread, 1H total)

        Raises:
            ModelNotFoundError: If ANY required model is missing
        """
        self.models_dir = Path(models_dir)
        
        if not self.models_dir.exists():
            raise ModelNotFoundError(
                f"Models directory does not exist: {self.models_dir}\n"
                f"Run: python scripts/train_models.py"
            )

        # Load 4 REQUIRED models (spreads, totals, 1H spread, 1H total)
        # Full Game models (REQUIRED)
        fg_spread_model, fg_spread_features = self._load_required_model(
            load_spread_model, "Full Game Spread"
        )
        fg_total_model, fg_total_features = self._load_required_model(
            load_total_model, "Full Game Total"
        )

        # First Half models (REQUIRED)
        fh_spread_model, fh_spread_features = self._load_required_model(
            load_first_half_spread_model, "First Half Spread"
        )
        fh_total_model, fh_total_features = self._load_required_model(
            load_first_half_total_model, "First Half Total"
        )

        # Initialize predictors with EXPLICIT models
        self.spread_predictor = SpreadPredictor(
            fg_model=fg_spread_model,
            fg_feature_columns=fg_spread_features,
            fh_model=fh_spread_model,
            fh_feature_columns=fh_spread_features,
        )
        self.total_predictor = TotalPredictor(
            fg_model=fg_total_model,
            fg_feature_columns=fg_total_features,
            fh_model=fh_total_model,
            fh_feature_columns=fh_total_features,
        )

    def _load_required_model(self, loader_func, model_name: str):
        """Load a model - FAIL LOUDLY if missing."""
        try:
            return loader_func(self.models_dir)
        except FileNotFoundError as e:
            raise ModelNotFoundError(
                f"REQUIRED MODEL MISSING: {model_name}\n"
                f"Details: {e}\n"
                f"Models directory: {self.models_dir}\n"
                f"Run: python scripts/train_models.py"
            ) from e

    def predict_full_game(
        self,
        features: Dict[str, float],
        spread_line: float,
        total_line: float,
    ) -> Dict[str, Any]:
        """
        Generate predictions for all full game markets.

        ALL INPUTS REQUIRED - no defaults, no None.

        Args:
            features: Feature dictionary (REQUIRED)
            spread_line: Vegas FG spread line (REQUIRED)
            total_line: Vegas FG total line (REQUIRED)

        Returns:
            Predictions for FG Spread and Total markets
        """
        return {
            "spread": self.spread_predictor.predict_full_game(features, spread_line),
            "total": self.total_predictor.predict_full_game(features, total_line),
        }

    def predict_first_half(
        self,
        features: Dict[str, float],
        spread_line: float,
        total_line: float,
    ) -> Dict[str, Any]:
        """
        Generate predictions for all first half markets.

        ALL INPUTS REQUIRED - no defaults, no None.

        Args:
            features: Feature dictionary (REQUIRED)
            spread_line: Vegas 1H spread line (REQUIRED)
            total_line: Vegas 1H total line (REQUIRED)

        Returns:
            Predictions for 1H Spread and Total markets
        """
        return {
            "spread": self.spread_predictor.predict_first_half(features, spread_line),
            "total": self.total_predictor.predict_first_half(features, total_line),
        }

    def predict_all_markets(
        self,
        features: Dict[str, float],
        # Full game lines - ALL REQUIRED
        fg_spread_line: float,
        fg_total_line: float,
        # First half lines - ALL REQUIRED
        fh_spread_line: float,
        fh_total_line: float,
    ) -> Dict[str, Any]:
        """
        Generate predictions for ALL 4 BACKTESTED markets.

        ALL 4 LINE PARAMETERS REQUIRED - no defaults, no None.

        Args:
            features: Feature dictionary (REQUIRED)
            fg_spread_line: Vegas FG spread line (REQUIRED)
            fg_total_line: Vegas FG total line (REQUIRED)
            fh_spread_line: Vegas 1H spread line (REQUIRED)
            fh_total_line: Vegas 1H total line (REQUIRED)

        Returns:
            Predictions for all 4 markets (2 periods x 2 bet types)
        """
        return {
            "full_game": self.predict_full_game(
                features,
                spread_line=fg_spread_line,
                total_line=fg_total_line,
            ),
            "first_half": self.predict_first_half(
                features,
                spread_line=fh_spread_line,
                total_line=fh_total_line,
            ),
        }
