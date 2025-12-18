"""
Unified prediction engine composing all market predictors.

STRICT MODE: No fallbacks, no silent failures. All models must exist.
If a model is missing, initialization FAILS LOUDLY.
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
    load_first_quarter_spread_model,
    load_first_quarter_total_model,
    load_first_quarter_moneyline_model,
)


class ModelNotFoundError(Exception):
    """Raised when a required model file is missing."""
    pass


class UnifiedPredictionEngine:
    """
    Unified prediction engine for all NBA betting markets.

    STRICT MODE:
    - ALL models must exist at initialization
    - NO fallbacks - 1H predictions require 1H models
    - NO silent failures - missing model = crash

    Supports 9 markets:
    - Full Game: Spread, Total, Moneyline
    - First Half: Spread, Total, Moneyline
    - First Quarter: Spread, Total, Moneyline
    """

    def __init__(self, models_dir: Path):
        """
        Initialize unified prediction engine.

        Args:
            models_dir: Path to models directory (must contain ALL required models)

        Raises:
            ModelNotFoundError: If ANY required model is missing
        """
        self.models_dir = Path(models_dir)
        
        if not self.models_dir.exists():
            raise ModelNotFoundError(
                f"Models directory does not exist: {self.models_dir}\n"
                f"Run: python scripts/train_models.py"
            )

        # Load ALL models - NO TRY/EXCEPT, FAIL IF MISSING
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

        # First Quarter models (REQUIRED)
        fq_spread_model, fq_spread_features = self._load_required_model(
            load_first_quarter_spread_model, "First Quarter Spread"
        )
        fq_total_model, fq_total_features = self._load_required_model(
            load_first_quarter_total_model, "First Quarter Total"
        )
        fq_moneyline_model, fq_moneyline_features = self._load_required_model(
            load_first_quarter_moneyline_model, "First Quarter Moneyline"
        )

        # Initialize predictors with EXPLICIT models - NO OR CHAINS
        self.spread_predictor = SpreadPredictor(
            fg_model=fg_spread_model,
            fg_feature_columns=fg_spread_features,
            fh_model=fh_spread_model,
            fh_feature_columns=fh_spread_features,
            fq_model=fq_spread_model,
            fq_feature_columns=fq_spread_features,
        )
        self.total_predictor = TotalPredictor(
            fg_model=fg_total_model,
            fg_feature_columns=fg_total_features,
            fh_model=fh_total_model,
            fh_feature_columns=fh_total_features,
            fq_model=fq_total_model,
            fq_feature_columns=fq_total_features,
        )
        self.moneyline_predictor = MoneylinePredictor(
            model=fg_spread_model,
            feature_columns=fg_spread_features,
            fh_model=fh_spread_model,
            fh_feature_columns=fh_spread_features,
            fq_model=fq_moneyline_model,
            fq_feature_columns=fq_moneyline_features,
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
            Predictions for all FG markets
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
            Predictions for all 1H markets
        """
        return {
            "spread": self.spread_predictor.predict_first_half(features, spread_line),
            "total": self.total_predictor.predict_first_half(features, total_line),
            "moneyline": self.moneyline_predictor.predict_first_half(
                features, home_ml_odds, away_ml_odds
            ),
        }

    def predict_first_quarter(
        self,
        features: Dict[str, float],
        spread_line: float,
        total_line: float,
        home_ml_odds: int,
        away_ml_odds: int,
    ) -> Dict[str, Any]:
        """
        Generate predictions for all first quarter markets.

        ALL INPUTS REQUIRED - no defaults, no None.

        Args:
            features: Feature dictionary (REQUIRED)
            spread_line: Vegas Q1 spread line (REQUIRED)
            total_line: Vegas Q1 total line (REQUIRED)
            home_ml_odds: Home team Q1 moneyline odds (REQUIRED)
            away_ml_odds: Away team Q1 moneyline odds (REQUIRED)

        Returns:
            Predictions for all Q1 markets
        """
        return {
            "spread": self.spread_predictor.predict_first_quarter(features, spread_line),
            "total": self.total_predictor.predict_first_quarter(features, total_line),
            "moneyline": self.moneyline_predictor.predict_first_quarter(
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
        # First quarter lines - ALL REQUIRED
        q1_spread_line: float,
        q1_total_line: float,
        q1_home_ml_odds: int,
        q1_away_ml_odds: int,
    ) -> Dict[str, Any]:
        """
        Generate predictions for ALL 9 markets.

        ALL 12 LINE/ODDS PARAMETERS REQUIRED - no defaults, no None.

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
            q1_spread_line: Vegas Q1 spread line (REQUIRED)
            q1_total_line: Vegas Q1 total line (REQUIRED)
            q1_home_ml_odds: Home team Q1 moneyline odds (REQUIRED)
            q1_away_ml_odds: Away team Q1 moneyline odds (REQUIRED)

        Returns:
            Predictions for all 9 markets (3 periods x 3 bet types)
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
            "first_quarter": self.predict_first_quarter(
                features,
                spread_line=q1_spread_line,
                total_line=q1_total_line,
                home_ml_odds=q1_home_ml_odds,
                away_ml_odds=q1_away_ml_odds,
            ),
        }
