"""Commercial-grade FastAPI model serving for NBA v4.0.

Exposes the full Unified Prediction Engine for all markets (FG + 1H).
"""
from __future__ import annotations
import os
import json
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path as PathLib
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query, Path
from pydantic import BaseModel, Field

from src.config import settings
from src.prediction import UnifiedPredictionEngine
from src.ingestion import the_odds
from src.ingestion.betting_splits import fetch_public_betting_splits
from scripts.build_rich_features import RichFeatureBuilder
from src.utils.logging import get_logger

logger = get_logger(__name__)

# --- Request/Response Models ---

class GamePredictionRequest(BaseModel):
    home_team: str = Field(..., example="Cleveland Cavaliers")
    away_team: str = Field(..., example="Chicago Bulls")
    fg_spread_line: Optional[float] = None
    fg_total_line: Optional[float] = None
    fh_spread_line: Optional[float] = None
    fh_total_line: Optional[float] = None

class MarketPrediction(BaseModel):
    side: Optional[str] = None
    confidence: float
    edge: Optional[float] = None
    passes_filter: bool
    filter_reason: Optional[str] = None

class GamePredictions(BaseModel):
    full_game: Dict[str, Any]
    first_half: Dict[str, Any]

class SlateResponse(BaseModel):
    date: str
    predictions: List[Dict[str, Any]]
    total_plays: int

# --- API Setup ---

app = FastAPI(
    title="NBA v5.0 BETA Commercial Prediction API",
    description="Full-market NBA betting predictions (FG + 1H) with Premium API integration.",
    version="5.0.0-beta"
)

def _models_dir() -> PathLib:
    return PathLib(settings.data_processed_dir) / "models"

@app.on_event("startup")
def startup_event():
    """Initialize the engine and feature builder on startup."""
    models_dir = _models_dir()
    try:
        logger.info(f"Initializing Unified Prediction Engine from {models_dir}")
        app.state.engine = UnifiedPredictionEngine(models_dir=models_dir)
        app.state.feature_builder = RichFeatureBuilder(season=settings.current_season)
        logger.info("NBA Prediction Engine initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize engine: {e}", exc_info=True)
        app.state.engine = None

# --- Endpoints ---

@app.get("/health")
def health():
    """Check API and Engine health."""
    return {
        "status": "ok",
        "engine_loaded": app.state.engine is not None,
        "season": settings.current_season,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/slate/{date}", response_model=SlateResponse)
async def get_slate_predictions(
    date: str = Path(..., description="Date in YYYY-MM-DD format, 'today', or 'tomorrow'"),
    use_splits: bool = Query(True, description="Whether to fetch and use betting splits")
):
    """
    Get all predictions for a full day's slate.
    This is the primary endpoint for commercial consumers.
    """
    if app.state.engine is None:
        raise HTTPException(status_code=503, detail="Prediction engine not loaded")

    # Resolve date
    from scripts.predict import get_target_date, filter_games_for_date
    try:
        target_date = get_target_date(date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    # Fetch games
    try:
        raw_games = await the_odds.fetch_odds()
        games = filter_games_for_date(raw_games, target_date)
    except Exception as e:
        logger.error(f"Error fetching odds: {e}")
        raise HTTPException(status_code=502, detail="Failed to fetch data from Odds API")

    if not games:
        return SlateResponse(date=str(target_date), predictions=[], total_plays=0)

    # Fetch splits
    splits_dict = {}
    if use_splits:
        try:
            splits_dict = await fetch_public_betting_splits(games, source="auto")
        except Exception as e:
            logger.warning(f"Could not fetch splits: {e}")

    # Process slate
    results = []
    total_plays = 0

    for game in games:
        home_team = game["home_team"]
        away_team = game["away_team"]
        
        try:
            # Build features
            game_key = f"{away_team}@{home_team}"
            features = await app.state.feature_builder.build_game_features(
                home_team, away_team, betting_splits=splits_dict.get(game_key)
            )

            # Extract lines
            from scripts.predict import extract_lines
            lines = extract_lines(game, home_team)

            # Predict
            preds = app.state.engine.predict_all_markets(
                features,
                fg_spread_line=lines["fg_spread"],
                fg_total_line=lines["fg_total"],
                fg_home_ml_odds=lines["fg_home_ml"],
                fg_away_ml_odds=lines["fg_away_ml"],
                fh_spread_line=lines["fh_spread"],
                fh_total_line=lines["fh_total"],
                fh_home_ml_odds=lines["fh_home_ml"],
                fh_away_ml_odds=lines["fh_away_ml"],
            )

            # Count plays
            game_plays = 0
            for period in ["full_game", "first_half"]:
                for market in ["spread", "total", "moneyline"]:
                    if preds[period][market].get("passes_filter"):
                        game_plays += 1
            
            total_plays += game_plays

            results.append({
                "matchup": f"{away_team} @ {home_team}",
                "home_team": home_team,
                "away_team": away_team,
                "commence_time": game.get("commence_time"),
                "predictions": preds,
                "has_plays": game_plays > 0
            })

        except Exception as e:
            logger.error(f"Error processing game {home_team} vs {away_team}: {e}")
            continue

    return SlateResponse(
        date=str(target_date),
        predictions=results,
        total_plays=total_plays
    )

@app.post("/predict/game", response_model=GamePredictions)
async def predict_single_game(req: GamePredictionRequest):
    """
    Generate predictions for a specific matchup with custom lines.
    Useful for 'What If' scenarios or checking specific sportsbook lines.
    """
    if app.state.engine is None:
        raise HTTPException(status_code=503, detail="Prediction engine not loaded")

    try:
        # Build features
        features = await app.state.feature_builder.build_game_features(
            req.home_team, req.away_team
        )

        # Predict
        preds = app.state.engine.predict_all_markets(
            features,
            fg_spread_line=req.fg_spread_line,
            fg_total_line=req.fg_total_line,
            fh_spread_line=req.fh_spread_line,
            fh_total_line=req.fh_total_line,
        )
        return preds
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
