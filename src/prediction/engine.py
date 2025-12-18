"""
Unified prediction engine composing all market predictors.

STRICT MODE: No fallbacks, no silent failures. All models must exist.
If a model is missing, initialization FAILS LOUDLY.

Supports 6 BACKTESTED markets:
- Full Game: Spread, Total, Moneyline
- First Half: Spread, Total, Moneyline
"""
from pathlib import Path
from typing import Dict, Any

from src.prediction.spreads import SpreadPredictor
from src.prediction.totals import TotalPredictor
from src.prediction.moneyline import MoneylinePredictor
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
    - ALL 4 models must exist at initialization
    - NO fallbacks - 1H predictions require 1H models
    - NO silent failures - missing model = crash

    Supports 6 BACKTESTED markets:
    - Full Game: Spread (60.6% acc), Total (59.2% acc), Moneyline (65.5% acc)
    - First Half: Spread (55.9% acc), Total (58.1% acc), Moneyline (63.0% acc)
    """

    def __init__(self, models_dir: Path):
        """
        Initialize unified prediction engine.

        Args:
            models_dir: Path to models directory (must contain ALL 4 required models)

        Raises:
            ModelNotFoundError: If ANY required model is missing
        """
        self.models_dir = Path(models_dir)
        
        if not self.models_dir.exists():
            raise ModelNotFoundError(
                f"Models directory does not exist: {self.models_dir}\n"
                f"Run: python scripts/train_models.py"
            )

        # Load ALL 4 models - NO TRY/EXCEPT, FAIL IF MISSING
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
        self.moneyline_predictor = MoneylinePredictor(
            model=fg_spread_model,
            feature_columns=fg_spread_features,
            fh_model=fh_spread_model,
            fh_feature_columns=fh_spread_features,
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
        home_ml_odds: int,
        away_ml_odds: int,
    ) -> Dict[str, Any]:
        """
        Generate predictions for all full game markets.

        ALL INPUTS REQUIRED - no defaults, no None.

        Args:
            features: Feature dictionary (REQUIRED)
            spread_line: Vegas FG spread line (REQUIRED)
            total_line: Vegas FG total line (REQUIRED)
            home_ml_odds: Home team moneyline odds (REQUIRED)
            away_ml_odds: Away team moneyline odds (REQUIRED)

        Returns:
            Predictions for all 3 FG markets
        """
        return {
            "spread": self.spread_predictor.predict_full_game(features, spread_line),
            "total": self.total_predictor.predict_full_game(features, total_line),
            "moneyline": self.moneyline_predictor.predict_full_game(
                features, home_ml_odds, away_ml_odds
            ),
        }

    def predict_first_half(
        self,
        features: Dict[str, float],
        spread_line: float,
        total_line: float,
        home_ml_odds: int,
        away_ml_odds: int,
    ) -> Dict[str, Any]:
        """
        Generate predictions for all first half markets.

        ALL INPUTS REQUIRED - no defaults, no None.

        Args:
            features: Feature dictionary (REQUIRED)
            spread_line: Vegas 1H spread line (REQUIRED)
            total_line: Vegas 1H total line (REQUIRED)
            home_ml_odds: Home team 1H moneyline odds (REQUIRED)
            away_ml_odds: Away team 1H moneyline odds (REQUIRED)

        Returns:
            Predictions for all 3 1H markets
        """
        return {
            "spread": self.spread_predictor.predict_first_half(features, spread_line),
            "total": self.total_predictor.predict_first_half(features, total_line),
            "moneyline": self.moneyline_predictor.predict_first_half(
                features, home_ml_odds, away_ml_odds
            ),
        }

    def predict_all_markets(
        self,
        features: Dict[str, float],
        # Full game lines - ALL REQUIRED
        fg_spread_line: float,
        fg_total_line: float,
        fg_home_ml_odds: int,
        fg_away_ml_odds: int,
        # First half lines - ALL REQUIRED
        fh_spread_line: float,
        fh_total_line: float,
        fh_home_ml_odds: int,
        fh_away_ml_odds: int,
    ) -> Dict[str, Any]:
        """
        Generate predictions for ALL 6 BACKTESTED markets.

        ALL 8 LINE/ODDS PARAMETERS REQUIRED - no defaults, no None.

        Args:
            features: Feature dictionary (REQUIRED)
            fg_spread_line: Vegas FG spread line (REQUIRED)
            fg_total_line: Vegas FG total line (REQUIRED)
            fg_home_ml_odds: Home team FG moneyline odds (REQUIRED)
            fg_away_ml_odds: Away team FG moneyline odds (REQUIRED)
            fh_spread_line: Vegas 1H spread line (REQUIRED)
            fh_total_line: Vegas 1H total line (REQUIRED)
            fh_home_ml_odds: Home team 1H moneyline odds (REQUIRED)
            fh_away_ml_odds: Away team 1H moneyline odds (REQUIRED)

        Returns:
            Predictions for all 6 markets (2 periods x 3 bet types)
        """
        return {
            "full_game": self.predict_full_game(
                features,
                spread_line=fg_spread_line,
                total_line=fg_total_line,
                home_ml_odds=fg_home_ml_odds,
                away_ml_odds=fg_away_ml_odds,
            ),
            "first_half": self.predict_first_half(
                features,
                spread_line=fh_spread_line,
                total_line=fh_total_line,
                home_ml_odds=fh_home_ml_odds,
                away_ml_odds=fh_away_ml_odds,
            ),
        }
