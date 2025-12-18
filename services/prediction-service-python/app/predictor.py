"""
NBA Prediction Engine v5.0

Bridges the containerized FastAPI service to the proven v4 predictors so that
Docker deployments serve real betting edges instead of placeholders.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.adapter import load_v4_predictors
from app.config import settings

logger = logging.getLogger(__name__)


class PredictionEngine:
    """
    Thin orchestrator around the v4 prediction stack.
    """

    def __init__(self, models_dir: str | Path | None = None):
        self.models_dir = Path(models_dir or settings.models_dir).expanduser()
        self._load_v4_predictors()

    def _load_v4_predictors(self) -> None:
        """
        Load v4 predictors and keep references for reuse.
        """
        try:
            predictors = load_v4_predictors(str(self.models_dir))
        except FileNotFoundError as exc:
            logger.error("Prediction models directory is missing: %s", exc)
            raise

        self._spread_total_engine = predictors.spread_total_engine
        self._moneyline_predictor = predictors.moneyline_predictor
        logger.info("Loaded v4 prediction artifacts from %s", self.models_dir)

    def predict_full_game(
        self,
        *,
        features: Dict[str, float],
        spread_line: Optional[float] = None,
        total_line: Optional[float] = None,
        home_ml_odds: Optional[int] = None,
        away_ml_odds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate full-game predictions for spread, total, and moneyline markets.
        """
        fg_predictions = self._spread_total_engine.predict_game(
            features=features,
            spread_line=spread_line,
            total_line=total_line,
        )

        moneyline = None
        if home_ml_odds is not None or away_ml_odds is not None:
            moneyline = self._moneyline_predictor.predict_full_game(
                features=features,
                home_odds=home_ml_odds,
                away_odds=away_ml_odds,
            )

        return {
            "spread": fg_predictions.get("spread"),
            "total": fg_predictions.get("total"),
            "moneyline": moneyline,
        }

    def predict_first_half(
        self,
        *,
        features: Dict[str, float],
        spread_line_1h: Optional[float] = None,
        total_line_1h: Optional[float] = None,
        home_ml_odds_1h: Optional[int] = None,
        away_ml_odds_1h: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate first-half predictions for spread, total, and moneyline markets.
        """
        fh_spread = self._spread_total_engine.predict_first_half_spread(
            features=features,
            first_half_spread_line=spread_line_1h,
        )
        fh_total = self._spread_total_engine.predict_first_half_total(
            features=features,
            first_half_total_line=total_line_1h,
        )

        moneyline = None
        if home_ml_odds_1h is not None or away_ml_odds_1h is not None:
            moneyline = self._moneyline_predictor.predict_first_half(
                features=features,
                home_odds=home_ml_odds_1h,
                away_odds=away_ml_odds_1h,
            )

        return {
            "spread": fh_spread,
            "total": fh_total,
            "moneyline": moneyline,
        }

    def build_recommendations(
        self,
        *,
        full_game: Dict[str, Any],
        first_half: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate a flattened list of actionable recommendations from predictions.
        """
        recommendations: List[Dict[str, Any]] = []

        recommendations += list(
            filter(
                None,
                [
                    self._spread_total_recommendation("spread", "full_game", full_game.get("spread")),
                    self._spread_total_recommendation("total", "full_game", full_game.get("total")),
                    self._moneyline_recommendation("moneyline", "full_game", full_game.get("moneyline")),
                ],
            )
        )

        if first_half:
            recommendations += list(
                filter(
                    None,
                    [
                        self._spread_total_recommendation("spread", "first_half", first_half.get("spread")),
                        self._spread_total_recommendation("total", "first_half", first_half.get("total")),
                        self._moneyline_recommendation("moneyline", "first_half", first_half.get("moneyline")),
                    ],
                )
            )

        return recommendations

    @staticmethod
    def _spread_total_recommendation(
        market: str,
        scope: str,
        payload: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Convert spread/total prediction payloads into a recommendation dict.
        """
        if not payload:
            return None
        if payload.get("passes_filter") is False:
            return None

        bet_side = payload.get("bet_side")
        edge = payload.get("edge")
        if not bet_side or edge is None:
            return None

        return {
            "market": market,
            "scope": scope,
            "bet": bet_side,
            "edge": edge,
            "confidence": payload.get("confidence"),
            "model_edge_pct": payload.get("model_edge_pct"),
        }

    @staticmethod
    def _moneyline_recommendation(
        market: str,
        scope: str,
        payload: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Convert moneyline payload into a recommendation dict.
        """
        if not payload:
            return None

        bet = payload.get("recommended_bet")
        if not bet or payload.get("passes_filter") is False:
            return None

        if bet == "home":
            edge = payload.get("home_edge")
            implied = payload.get("home_implied_prob")
            predicted = payload.get("home_win_prob")
        else:
            edge = payload.get("away_edge")
            implied = payload.get("away_implied_prob")
            predicted = payload.get("away_win_prob")

        if edge is None:
            return None

        return {
            "market": market,
            "scope": scope,
            "bet": bet,
            "edge": edge,
            "confidence": payload.get("confidence"),
            "predicted_prob": predicted,
            "implied_prob": implied,
        }


# Singleton instance used by FastAPI handlers
prediction_engine = PredictionEngine()
