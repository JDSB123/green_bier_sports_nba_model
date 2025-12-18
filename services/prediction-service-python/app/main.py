"""
NBA Prediction Service v5.0
FastAPI service for NBA betting predictions.
"""
import logging
from datetime import datetime
from typing import Dict, Optional, List
from uuid import UUID

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.config import settings
from app.predictor import prediction_engine

logger = logging.getLogger(__name__)


# -----------------------------
# Pydantic request/response DTOs
# -----------------------------

class GameFeaturesInput(BaseModel):
    """Input features for a game."""

    # Core scoring / projection inputs
    home_ppg: float
    home_papg: float
    home_total_ppg: Optional[float] = None
    away_ppg: float
    away_papg: float
    away_total_ppg: Optional[float] = None
    predicted_margin: float
    predicted_total: float
    predicted_margin_adj: Optional[float] = None
    predicted_total_adj: Optional[float] = None

    # Team form + situational factors
    home_win_pct: Optional[float] = None
    home_avg_margin: Optional[float] = None
    away_win_pct: Optional[float] = None
    away_avg_margin: Optional[float] = None
    home_rest_days: int = 1
    away_rest_days: int = 1
    rest_advantage: Optional[float] = None
    home_b2b: Optional[int] = None
    away_b2b: Optional[int] = None
    dynamic_hca: Optional[float] = None

    # Matchup history
    h2h_win_pct: Optional[float] = None
    h2h_avg_margin: Optional[float] = None

    # Derived diffs + spreads
    win_pct_diff: Optional[float] = None
    ppg_diff: Optional[float] = None

    # Line-derived features (FG)
    spread_line: Optional[float] = None
    spread_vs_predicted: Optional[float] = None
    spread_opening_line: Optional[float] = None
    spread_movement: Optional[float] = None
    spread_line_std: Optional[float] = None
    total_line: Optional[float] = None
    total_vs_predicted: Optional[float] = None
    total_opening_line: Optional[float] = None
    total_movement: Optional[float] = None
    total_line_std: Optional[float] = None

    # Betting splits / ATS
    home_ats_pct: Optional[float] = None
    away_ats_pct: Optional[float] = None
    home_over_pct: Optional[float] = None
    away_over_pct: Optional[float] = None
    spread_public_home_pct: Optional[float] = None
    spread_ticket_money_diff: Optional[float] = None
    over_public_pct: Optional[float] = None
    total_ticket_money_diff: Optional[float] = None

    # RLM + sharp signals
    is_rlm_spread: Optional[int] = None
    sharp_side_spread: Optional[float] = None
    is_rlm_total: Optional[int] = None
    sharp_side_total: Optional[float] = None

    # Injuries
    home_injury_spread_impact: Optional[float] = None
    away_injury_spread_impact: Optional[float] = None
    injury_spread_diff: Optional[float] = None
    home_injury_total_impact: Optional[float] = None
    away_injury_total_impact: Optional[float] = None
    injury_total_diff: Optional[float] = None
    home_star_out: Optional[int] = None
    away_star_out: Optional[int] = None

    # Custom overrides
    extra_features: Dict[str, float] = Field(default_factory=dict)

    def as_feature_dict(self) -> Dict[str, float]:
        """
        Convert the structured payload into the feature dict required by v4.
        """
        raw = self.model_dump(exclude={"extra_features"})
        features = {k: float(v) for k, v in raw.items() if v is not None}
        features.update({k: float(v) for k, v in self.extra_features.items()})

        # Derive fields commonly expected by the v4 pipeline if missing.
        if "rest_advantage" not in features and {"home_rest_days", "away_rest_days"} <= features.keys():
            features["rest_advantage"] = float(features["home_rest_days"] - features["away_rest_days"])

        if "win_pct_diff" not in features and {"home_win_pct", "away_win_pct"} <= features.keys():
            features["win_pct_diff"] = float(features["home_win_pct"] - features["away_win_pct"])

        if "ppg_diff" not in features:
            features["ppg_diff"] = float(features["home_ppg"] - features["away_ppg"])

        features.setdefault("predicted_margin_adj", features.get("predicted_margin", 0.0))
        features.setdefault("predicted_total_adj", features.get("predicted_total", 0.0))

        return features


class MarketOddsInput(BaseModel):
    """Market odds for a game."""

    spread: Optional[float] = None
    spread_price: int = -110
    total: Optional[float] = None
    over_price: int = -110
    under_price: int = -110
    home_ml: Optional[int] = None
    away_ml: Optional[int] = None

    # First half
    spread_1h: Optional[float] = None
    total_1h: Optional[float] = None
    home_ml_1h: Optional[int] = None
    away_ml_1h: Optional[int] = None


class PredictRequest(BaseModel):
    """Request for predictions."""
    game_id: UUID
    home_team: str
    away_team: str
    commence_time: datetime
    features: GameFeaturesInput
    market_odds: Optional[MarketOddsInput] = None


class PredictionResponse(BaseModel):
    """Response with predictions."""
    game_id: UUID
    home_team: str
    away_team: str
    predictions: dict
    recommendations: List[dict] = Field(default_factory=list)


# -----------------------------
# FastAPI app
# -----------------------------

app = FastAPI(
    title="NBA Prediction Service",
    version=settings.service_version,
    description="ML-based NBA betting predictions for spreads, totals, and moneyline"
)

# CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "service": settings.service_name,
        "version": settings.service_version,
        "status": "ok",
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(req: PredictRequest):
    """
    Generate predictions for a game.
    
    Returns predictions for all markets (FG + 1H spreads, totals, moneyline)
    along with betting recommendations if market odds are provided.
    """
    try:
        features_dict = req.features.as_feature_dict()

        # Inject market-derived features when available.
        if req.market_odds:
            if req.market_odds.spread is not None:
                features_dict.setdefault("spread_line", req.market_odds.spread)
                features_dict.setdefault(
                    "spread_vs_predicted",
                    features_dict.get("predicted_margin", 0.0) - req.market_odds.spread,
                )

            if req.market_odds.total is not None:
                features_dict.setdefault("total_line", req.market_odds.total)
                features_dict.setdefault(
                    "total_vs_predicted",
                    features_dict.get("predicted_total", 0.0) - req.market_odds.total,
                )

        predictions = prediction_engine.predict_full_game(
            features=features_dict,
            spread_line=req.market_odds.spread if req.market_odds else None,
            total_line=req.market_odds.total if req.market_odds else None,
            home_ml_odds=req.market_odds.home_ml if req.market_odds else None,
            away_ml_odds=req.market_odds.away_ml if req.market_odds else None,
        )

        predictions_1h = prediction_engine.predict_first_half(
            features=features_dict,
            spread_line_1h=req.market_odds.spread_1h if req.market_odds else None,
            total_line_1h=req.market_odds.total_1h if req.market_odds else None,
            home_ml_odds_1h=req.market_odds.home_ml_1h if req.market_odds else None,
            away_ml_odds_1h=req.market_odds.away_ml_1h if req.market_odds else None,
        )

        combined_predictions = {
            "full_game": predictions,
            "first_half": predictions_1h,
        }

        recommendations = prediction_engine.build_recommendations(
            full_game=predictions,
            first_half=predictions_1h,
        )

        return PredictionResponse(
            game_id=req.game_id,
            home_team=req.home_team,
            away_team=req.away_team,
            predictions=combined_predictions,
            recommendations=recommendations,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction failure for game %s", req.game_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# Local run helper
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8082, reload=True)
