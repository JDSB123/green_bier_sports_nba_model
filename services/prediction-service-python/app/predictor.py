"""
NBA Prediction Service - STRICT MODE

STRICT MODE: All inputs required. No fallbacks. No silent failures.

Supports 6 BACKTESTED markets:
- Full Game: Spread (60.6%), Total (59.2%), Moneyline (65.5%)
- First Half: Spread (55.9%), Total (58.1%), Moneyline (63.0%)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from app.adapter import load_prediction_engine
from app.config import settings

logger = logging.getLogger(__name__)


class PredictionService:
    """
    STRICT MODE prediction service.
    
    ALL inputs are REQUIRED. No Optional parameters.
    Missing input = immediate failure with clear error message.
    
    6 BACKTESTED markets only - no Q1 (not validated).
    """

    def __init__(self, models_dir: str | Path | None = None):
        self.models_dir = Path(models_dir or settings.models_dir).expanduser()
        self._engine = self._load_engine()

    def _load_engine(self) -> "UnifiedPredictionEngine":
        """
        Load the STRICT MODE unified prediction engine.
        
        Raises:
            ModelNotFoundError: If ANY required model is missing
            FileNotFoundError: If models directory does not exist
        """
        logger.info("Loading STRICT MODE prediction engine from %s", self.models_dir)
        engine = load_prediction_engine(str(self.models_dir))
        logger.info("Successfully loaded prediction engine with all 4 models")
        return engine

    def predict_full_game(
        self,
        features: Dict[str, float],
        spread_line: float,
        total_line: float,
        home_ml_odds: int,
        away_ml_odds: int,
    ) -> Dict[str, Any]:
        """
        Generate full-game predictions for all 3 markets.
        
        ALL INPUTS REQUIRED - no defaults, no None.
        """
        return self._engine.predict_full_game(
            features=features,
            spread_line=spread_line,
            total_line=total_line,
            home_ml_odds=home_ml_odds,
            away_ml_odds=away_ml_odds,
        )

    def predict_first_half(
        self,
        features: Dict[str, float],
        spread_line: float,
        total_line: float,
        home_ml_odds: int,
        away_ml_odds: int,
    ) -> Dict[str, Any]:
        """
        Generate first-half predictions for all 3 markets.
        
        ALL INPUTS REQUIRED - no defaults, no None.
        """
        return self._engine.predict_first_half(
            features=features,
            spread_line=spread_line,
            total_line=total_line,
            home_ml_odds=home_ml_odds,
            away_ml_odds=away_ml_odds,
        )

    def predict_all_markets(
        self,
        features: Dict[str, float],
        # Full game - ALL REQUIRED
        fg_spread_line: float,
        fg_total_line: float,
        fg_home_ml_odds: int,
        fg_away_ml_odds: int,
        # First half - ALL REQUIRED
        fh_spread_line: float,
        fh_total_line: float,
        fh_home_ml_odds: int,
        fh_away_ml_odds: int,
    ) -> Dict[str, Any]:
        """
        Generate predictions for ALL 6 BACKTESTED markets.
        
        ALL 8 LINE/ODDS PARAMETERS REQUIRED.
        """
        return self._engine.predict_all_markets(
            features=features,
            fg_spread_line=fg_spread_line,
            fg_total_line=fg_total_line,
            fg_home_ml_odds=fg_home_ml_odds,
            fg_away_ml_odds=fg_away_ml_odds,
            fh_spread_line=fh_spread_line,
            fh_total_line=fh_total_line,
            fh_home_ml_odds=fh_home_ml_odds,
            fh_away_ml_odds=fh_away_ml_odds,
        )

    def build_recommendations(
        self,
        predictions: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Generate a flattened list of actionable recommendations from predictions.
        
        Only includes bets that pass their respective filters.
        """
        recommendations: List[Dict[str, Any]] = []

        for scope in ["full_game", "first_half"]:
            scope_data = predictions.get(scope, {})
            if not scope_data:
                continue
                
            # Spread recommendation
            spread = scope_data.get("spread")
            if spread and spread.get("passes_filter"):
                recommendations.append({
                    "market": "spread",
                    "scope": scope,
                    "bet": spread["bet_side"],
                    "edge": spread["edge"],
                    "confidence": spread["confidence"],
                    "model_edge_pct": spread["model_edge_pct"],
                })

            # Total recommendation
            total = scope_data.get("total")
            if total and total.get("passes_filter"):
                recommendations.append({
                    "market": "total",
                    "scope": scope,
                    "bet": total["bet_side"],
                    "edge": total["edge"],
                    "confidence": total["confidence"],
                    "model_edge_pct": total["model_edge_pct"],
                })

            # Moneyline recommendation
            ml = scope_data.get("moneyline")
            if ml and ml.get("passes_filter") and ml.get("recommended_bet"):
                bet = ml["recommended_bet"]
                if bet == "home":
                    edge = ml["home_edge"]
                    predicted = ml["home_win_prob"]
                    implied = ml["home_implied_prob"]
                else:
                    edge = ml["away_edge"]
                    predicted = ml["away_win_prob"]
                    implied = ml["away_implied_prob"]
                    
                recommendations.append({
                    "market": "moneyline",
                    "scope": scope,
                    "bet": bet,
                    "edge": edge,
                    "confidence": ml["confidence"],
                    "predicted_prob": predicted,
                    "implied_prob": implied,
                })

        return recommendations


# Singleton instance used by FastAPI handlers
# NOTE: This will FAIL LOUDLY at import time if models are missing
prediction_service = PredictionService()
