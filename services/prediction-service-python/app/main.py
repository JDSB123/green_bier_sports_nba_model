"""
NBA Prediction Service v5.0
FastAPI service for NBA betting predictions.
"""
from datetime import datetime
from typing import Optional, List
from uuid import UUID

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.config import settings
from app.predictor import prediction_engine


# -----------------------------
# Pydantic request/response DTOs
# -----------------------------

class GameFeaturesInput(BaseModel):
    """Input features for a game."""
    home_ppg: float
    home_papg: float
    away_ppg: float
    away_papg: float
    home_rest_days: int = 1
    away_rest_days: int = 1
    predicted_margin: float
    predicted_total: float
    # Add more features as needed


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
        # Convert features to dict
        features_dict = req.features.dict()
        
        # Generate predictions
        predictions = prediction_engine.predict_full_game(
            features=features_dict,
            spread_line=req.market_odds.spread if req.market_odds else None,
            total_line=req.market_odds.total if req.market_odds else None,
            home_ml_odds=req.market_odds.home_ml if req.market_odds else None,
            away_ml_odds=req.market_odds.away_ml if req.market_odds else None,
        )
        
        # Generate first half predictions if 1H odds provided
        predictions_1h = None
        if req.market_odds and (req.market_odds.spread_1h or req.market_odds.total_1h):
            predictions_1h = prediction_engine.predict_first_half(
                features=features_dict,
                spread_line_1h=req.market_odds.spread_1h,
                total_line_1h=req.market_odds.total_1h,
                home_ml_odds_1h=req.market_odds.home_ml_1h,
                away_ml_odds_1h=req.market_odds.away_ml_1h,
            )
        
        # Combine predictions
        combined_predictions = {
            "full_game": predictions,
            "first_half": predictions_1h,
        }
        
        # TODO: Generate recommendations based on edges
        
        return PredictionResponse(
            game_id=req.game_id,
            home_team=req.home_team,
            away_team=req.away_team,
            predictions=combined_predictions,
            recommendations=[],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Local run helper
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8082, reload=True)
