"""
Unified prediction engine composing all market predictors.

Provides single interface for predicting all markets (spreads, totals, moneyline)
across both full game and first half.
"""
from pathlib import Path
from typing import Dict, Any, Optional

from src.prediction.spreads import SpreadPredictor
from src.prediction.totals import TotalPredictor
from src.prediction.moneyline import MoneylinePredictor
from src.prediction.models import (
    load_spread_model,
    load_total_model,
    load_first_half_spread_model,
    load_first_half_total_model,
)


class UnifiedPredictionEngine:
    """
    Unified prediction engine for all NBA betting markets.

    Composes three market-specific predictors:
    - SpreadPredictor (FG + 1H)
    - TotalPredictor (FG + 1H)
    - MoneylinePredictor (FG + 1H)

    Each predictor has its own smart filtering based on backtest validation.
    """

    def __init__(
        self,
        models_dir: Path,
        spread_predictor: Optional[SpreadPredictor] = None,
        total_predictor: Optional[TotalPredictor] = None,
        moneyline_predictor: Optional[MoneylinePredictor] = None,
    ):
        """
        Initialize unified prediction engine.

        Args:
            models_dir: Path to models directory
            spread_predictor: SpreadPredictor instance (None = use defaults)
            total_predictor: TotalPredictor instance (None = use defaults)
            moneyline_predictor: MoneylinePredictor instance (None = use defaults)
        """
        self.models_dir = models_dir

        # Load FG models
        fg_spread_model, fg_spread_features = load_spread_model(models_dir)
        fg_total_model, fg_total_features = load_total_model(models_dir)

        # Load 1H models (if available)
        try:
            fh_spread_model, fh_spread_features = load_first_half_spread_model(models_dir)
        except FileNotFoundError:
            fh_spread_model, fh_spread_features = None, None

        try:
            fh_total_model, fh_total_features = load_first_half_total_model(models_dir)
        except FileNotFoundError:
            fh_total_model, fh_total_features = None, None

        # Initialize market predictors
        self.spread_predictor = spread_predictor or SpreadPredictor(
            fg_model=fg_spread_model,
            fg_feature_columns=fg_spread_features,
            fh_model=fh_spread_model,
            fh_feature_columns=fh_spread_features,
        )
        self.total_predictor = total_predictor or TotalPredictor(
            fg_model=fg_total_model,
            fg_feature_columns=fg_total_features,
            fh_model=fh_total_model,
            fh_feature_columns=fh_total_features,
        )
        self.moneyline_predictor = moneyline_predictor or MoneylinePredictor(
            model=fg_spread_model,  # Moneyline uses FG spread model
            feature_columns=fg_spread_features,
        )

    def predict_full_game(
        self,
        features: Dict[str, float],
        spread_line: Optional[float] = None,
        total_line: Optional[float] = None,
        home_ml_odds: Optional[int] = None,
        away_ml_odds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate predictions for all full game markets.

        Args:
            features: Feature dictionary for the game
            spread_line: Vegas FG spread line (home perspective)
            total_line: Vegas FG total line
            home_ml_odds: Home team moneyline odds (American)
            away_ml_odds: Away team moneyline odds (American)

        Returns:
            Dictionary with predictions for all FG markets:
                {
                    "spread": {...},
                    "total": {...},
                    "moneyline": {...}
                }
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
        spread_line: Optional[float] = None,
        total_line: Optional[float] = None,
        home_ml_odds: Optional[int] = None,
        away_ml_odds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate predictions for all first half markets.

        Args:
            features: Feature dictionary for the game
            spread_line: Vegas 1H spread line (home perspective)
            total_line: Vegas 1H total line
            home_ml_odds: Home team 1H moneyline odds (American)
            away_ml_odds: Away team 1H moneyline odds (American)

        Returns:
            Dictionary with predictions for all 1H markets:
                {
                    "spread": {...},
                    "total": {...},
                    "moneyline": {...}
                }
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
        # Full game lines
        fg_spread_line: Optional[float] = None,
        fg_total_line: Optional[float] = None,
        fg_home_ml_odds: Optional[int] = None,
        fg_away_ml_odds: Optional[int] = None,
        # First half lines
        fh_spread_line: Optional[float] = None,
        fh_total_line: Optional[float] = None,
        fh_home_ml_odds: Optional[int] = None,
        fh_away_ml_odds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate predictions for ALL markets (FG + 1H, spreads + totals + moneyline).

        Args:
            features: Feature dictionary for the game
            fg_spread_line: Vegas FG spread line (home perspective)
            fg_total_line: Vegas FG total line
            fg_home_ml_odds: Home team FG moneyline odds
            fg_away_ml_odds: Away team FG moneyline odds
            fh_spread_line: Vegas 1H spread line (home perspective)
            fh_total_line: Vegas 1H total line
            fh_home_ml_odds: Home team 1H moneyline odds
            fh_away_ml_odds: Away team 1H moneyline odds

        Returns:
            Dictionary with predictions for all markets:
                {
                    "full_game": {
                        "spread": {...},
                        "total": {...},
                        "moneyline": {...}
                    },
                    "first_half": {
                        "spread": {...},
                        "total": {...},
                        "moneyline": {...}
                    }
                }
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
