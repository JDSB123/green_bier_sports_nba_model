"""
NBA Prediction Service v5.0 - STRICT MODE
FastAPI service for NBA betting predictions.

STRICT MODE: All inputs required. No fallbacks. No silent failures.
"""
import logging
from datetime import datetime
from typing import Dict, List
from uuid import UUID

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.config import settings
from app.predictor import prediction_service

logger = logging.getLogger(__name__)


# -----------------------------
# Pydantic request/response DTOs - STRICT MODE
# -----------------------------

class GameFeaturesInput(BaseModel):
    """
    Input features for a game.
    
    STRICT MODE: Core prediction fields are REQUIRED.
    """
    # Core scoring / projection inputs - REQUIRED
    home_ppg: float
    home_papg: float
    away_ppg: float
    away_papg: float
    predicted_margin: float  # REQUIRED for FG spread
    predicted_total: float   # REQUIRED for FG total
    predicted_margin_1h: float  # REQUIRED for 1H spread
    predicted_total_1h: float   # REQUIRED for 1H total
    predicted_margin_q1: float  # REQUIRED for Q1 spread
    predicted_total_q1: float   # REQUIRED for Q1 total

    # Team form + situational factors - optional but recommended
    home_win_pct: float = 0.5
    home_avg_margin: float = 0.0
    away_win_pct: float = 0.5
    away_avg_margin: float = 0.0
    home_rest_days: int = 1
    away_rest_days: int = 1
    home_b2b: int = 0
    away_b2b: int = 0
    dynamic_hca: float = 3.0

    # Matchup history
    h2h_win_pct: float = 0.5
    h2h_avg_margin: float = 0.0

    # Custom overrides
    extra_features: Dict[str, float] = Field(default_factory=dict)

    def as_feature_dict(self) -> Dict[str, float]:
        """Convert to feature dict for prediction engine."""
        raw = self.model_dump(exclude={"extra_features"})
        features = {k: float(v) for k, v in raw.items()}
        features.update({k: float(v) for k, v in self.extra_features.items()})

        # Derive common fields
        features["rest_advantage"] = float(features["home_rest_days"] - features["away_rest_days"])
        features["win_pct_diff"] = float(features["home_win_pct"] - features["away_win_pct"])
        features["ppg_diff"] = float(features["home_ppg"] - features["away_ppg"])

        return features


class MarketOddsInput(BaseModel):
    """
    Market odds for a game.
    
    STRICT MODE: All lines and odds are REQUIRED.
    """
    # Full game - ALL REQUIRED
    fg_spread: float
    fg_total: float
    fg_home_ml: int
    fg_away_ml: int

    # First half - ALL REQUIRED
    fh_spread: float
    fh_total: float
    fh_home_ml: int
    fh_away_ml: int

    # First quarter - ALL REQUIRED
    q1_spread: float
    q1_total: float
    q1_home_ml: int
    q1_away_ml: int


class PredictRequest(BaseModel):
    """
    Request for predictions.
    
    STRICT MODE: All fields required.
    """
    game_id: UUID
    home_team: str
    away_team: str
    commence_time: datetime
    features: GameFeaturesInput
    market_odds: MarketOddsInput  # REQUIRED - not Optional


class PredictionResponse(BaseModel):
    """Response with predictions for all 9 markets."""
    game_id: UUID
    home_team: str
    away_team: str
    predictions: dict
    recommendations: List[dict] = Field(default_factory=list)


# -----------------------------
# FastAPI app - STRICT MODE
# -----------------------------

app = FastAPI(
    title="NBA Prediction Service - STRICT MODE",
    version=settings.service_version,
    description="ML-based NBA betting predictions. STRICT MODE: All inputs required, no fallbacks."
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
        "mode": "STRICT",
        "status": "ok",
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(req: PredictRequest):
    """
    Generate predictions for ALL 9 markets.
    
    STRICT MODE: All inputs required. Missing input = 400 error.
    
    Markets:
    - Full Game: Spread, Total, Moneyline
    - First Half: Spread, Total, Moneyline  
    - First Quarter: Spread, Total, Moneyline
    """
    try:
        features_dict = req.features.as_feature_dict()
        odds = req.market_odds

        # Inject market-derived features
        features_dict["spread_line"] = odds.fg_spread
        features_dict["spread_vs_predicted"] = features_dict["predicted_margin"] - odds.fg_spread
        features_dict["total_line"] = odds.fg_total
        features_dict["total_vs_predicted"] = features_dict["predicted_total"] - odds.fg_total

        # Generate predictions for ALL 9 markets
        predictions = prediction_service.predict_all_markets(
            features=features_dict,
            # Full game
            fg_spread_line=odds.fg_spread,
            fg_total_line=odds.fg_total,
            fg_home_ml_odds=odds.fg_home_ml,
            fg_away_ml_odds=odds.fg_away_ml,
            # First half
            fh_spread_line=odds.fh_spread,
            fh_total_line=odds.fh_total,
            fh_home_ml_odds=odds.fh_home_ml,
            fh_away_ml_odds=odds.fh_away_ml,
            # First quarter
            q1_spread_line=odds.q1_spread,
            q1_total_line=odds.q1_total,
            q1_home_ml_odds=odds.q1_home_ml,
            q1_away_ml_odds=odds.q1_away_ml,
        )

        recommendations = prediction_service.build_recommendations(predictions)

        return PredictionResponse(
            game_id=req.game_id,
            home_team=req.home_team,
            away_team=req.away_team,
            predictions=predictions,
            recommendations=recommendations,
        )
    except ValueError as exc:
        # Missing required input
        raise HTTPException(status_code=400, detail=f"STRICT MODE: {exc}") from exc
    except Exception as exc:
        logger.exception("Prediction failure for game %s", req.game_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# Local run helper
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8082, reload=True)
