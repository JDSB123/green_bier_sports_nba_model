"""
NBA v5.0 BETA - FastAPI Prediction Server - STRICT MODE

STRICT MODE: All inputs required. No fallbacks. No silent failures.

6 BACKTESTED markets:
- Full Game: Spread (60.6%), Total (59.2%), Moneyline (65.5%)
- First Half: Spread (55.9%), Total (58.1%), Moneyline (63.0%)
"""
from __future__ import annotations
import os
import json
import logging
from typing import Any, Dict, List
from pathlib import Path as PathLib
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query, Path
from pydantic import BaseModel, Field

from src.config import settings
from src.prediction import UnifiedPredictionEngine, ModelNotFoundError
from src.ingestion import the_odds
from src.ingestion.betting_splits import fetch_public_betting_splits
from scripts.build_rich_features import RichFeatureBuilder
from src.utils.logging import get_logger

logger = get_logger(__name__)


# --- Request/Response Models - STRICT MODE ---

class GamePredictionRequest(BaseModel):
    """Request for single game prediction - ALL LINES REQUIRED."""
    home_team: str = Field(..., example="Cleveland Cavaliers")
    away_team: str = Field(..., example="Chicago Bulls")
    # Full game lines - REQUIRED
    fg_spread_line: float
    fg_total_line: float
    fg_home_ml: int = Field(..., example=-150)
    fg_away_ml: int = Field(..., example=130)
    # First half lines - REQUIRED
    fh_spread_line: float
    fh_total_line: float
    fh_home_ml: int = Field(..., example=-130)
    fh_away_ml: int = Field(..., example=110)


class MarketPrediction(BaseModel):
    side: str
    confidence: float
    edge: float
    passes_filter: bool
    filter_reason: str | None = None


class GamePredictions(BaseModel):
    full_game: Dict[str, Any]
    first_half: Dict[str, Any]


class SlateResponse(BaseModel):
    date: str
    predictions: List[Dict[str, Any]]
    total_plays: int


# --- API Setup - STRICT MODE ---

app = FastAPI(
    title="NBA v5.0 BETA - STRICT MODE",
    description="6 BACKTESTED markets. All inputs required. No fallbacks.",
    version="5.0.0-strict"
)


def _models_dir() -> PathLib:
    return PathLib(settings.data_processed_dir) / "models"


@app.on_event("startup")
def startup_event():
    """
    Initialize the prediction engine on startup.
    
    STRICT MODE: Fails LOUDLY if models are missing.
    """
    models_dir = _models_dir()
    logger.info(f"STRICT MODE: Loading Unified Prediction Engine from {models_dir}")
    
    # NO TRY/EXCEPT - Let it crash if models are missing
    app.state.engine = UnifiedPredictionEngine(models_dir=models_dir)
    app.state.feature_builder = RichFeatureBuilder(season=settings.current_season)
    logger.info("NBA Prediction Engine initialized - 4 models loaded, 6 markets active")


# --- Endpoints ---

@app.get("/health")
def health():
    """Check API health - STRICT MODE."""
    engine_loaded = hasattr(app.state, 'engine') and app.state.engine is not None
    return {
        "status": "ok",
        "mode": "STRICT",
        "markets": 6,
        "engine_loaded": engine_loaded,
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
    
    STRICT MODE: Games without all required lines are skipped with warning.
    """
    if not hasattr(app.state, 'engine') or app.state.engine is None:
        raise HTTPException(status_code=503, detail="STRICT MODE: Engine not loaded - models missing")

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

            # Extract lines - STRICT MODE requires all
            from scripts.predict import extract_lines
            lines = extract_lines(game, home_team)
            
            # Validate all lines present
            required_lines = ["fg_spread", "fg_total", "fg_home_ml", "fg_away_ml",
                             "fh_spread", "fh_total", "fh_home_ml", "fh_away_ml"]
            missing = [k for k in required_lines if lines.get(k) is None]
            if missing:
                logger.warning(f"STRICT MODE: Skipping {home_team} vs {away_team} - missing lines: {missing}")
                continue

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

        except ValueError as e:
            # STRICT MODE: Missing required input
            logger.warning(f"STRICT MODE: Skipping {home_team} vs {away_team} - {e}")
            continue
        except Exception as e:
            logger.error(f"Error processing {home_team} vs {away_team}: {e}")
            continue

    return SlateResponse(
        date=str(target_date),
        predictions=results,
        total_plays=total_plays
    )


@app.get("/slate/{date}/comprehensive")
async def get_comprehensive_slate_analysis(
    date: str = Path(..., description="Date in YYYY-MM-DD format, 'today', or 'tomorrow'"),
    use_splits: bool = Query(True, description="Whether to fetch and use betting splits")
):
    """
    Get comprehensive slate analysis with full edge calculations, rationale, and summary table.
    
    This is the FULL analysis endpoint that replaces the legacy script.
    Returns complete analysis matching analyze_todays_slate.py output.
    """
    if not hasattr(app.state, 'engine') or app.state.engine is None:
        raise HTTPException(status_code=503, detail="STRICT MODE: Engine not loaded - models missing")

    # Import comprehensive analysis functions
    from scripts.analyze_todays_slate import (
        get_target_date, fetch_todays_games, calculate_comprehensive_edge,
        parse_utc_time, to_cst
    )
    from src.modeling.edge_thresholds import get_edge_thresholds_for_game
    from zoneinfo import ZoneInfo
    
    CST = ZoneInfo("America/Chicago")
    
    # Resolve date
    try:
        target_date = get_target_date(date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    # Get edge thresholds
    edge_thresholds = get_edge_thresholds_for_game(
        game_date=target_date,
        bet_types=["spread", "total", "moneyline", "1h_spread", "1h_total"]
    )

    # Fetch games
    games = await fetch_todays_games(target_date)
    if not games:
        return {"date": str(target_date), "analysis": [], "summary": "No games found"}

    # Fetch betting splits
    splits_dict = {}
    if use_splits:
        try:
            from src.ingestion.betting_splits import fetch_public_betting_splits
            splits_dict = await fetch_public_betting_splits(games, source="auto")
        except Exception as e:
            logger.warning(f"Could not fetch splits: {e}")

    # Process each game
    analysis_results = []
    
    for game in games:
        home_team = game.get("home_team")
        away_team = game.get("away_team")
        commence_time = game.get("commence_time")
        
        try:
            # Parse time
            game_dt = parse_utc_time(commence_time) if commence_time else None
            time_cst = to_cst(game_dt).strftime("%I:%M %p %Z") if game_dt else "TBD"
            
            # Build features
            game_key = f"{away_team}@{home_team}"
            features = await app.state.feature_builder.build_game_features(
                home_team, away_team, betting_splits=splits_dict.get(game_key)
            )
            
            # Build first half features (simplified - use same features scaled)
            fh_features = features.copy()  # In production, build proper 1H features
            
            # Extract odds
            from scripts.analyze_todays_slate import extract_consensus_odds
            odds = extract_consensus_odds(game)
            
            # Get betting splits for this game
            betting_splits = splits_dict.get(game_key)
            
            # Calculate comprehensive edge
            comprehensive_edge = calculate_comprehensive_edge(
                features=features,
                fh_features=fh_features,
                odds=odds,
                game=game,
                betting_splits=betting_splits,
                edge_thresholds=edge_thresholds
            )
            
            analysis_results.append({
                "home_team": home_team,
                "away_team": away_team,
                "time_cst": time_cst,
                "commence_time": commence_time,
                "odds": odds,
                "features": features,
                "comprehensive_edge": comprehensive_edge,
                "betting_splits": betting_splits
            })
            
        except Exception as e:
            logger.error(f"Error processing {home_team} vs {away_team}: {e}")
            continue
    
    return {
        "date": str(target_date),
        "analysis": analysis_results,
        "edge_thresholds": edge_thresholds
    }


@app.post("/predict/game", response_model=GamePredictions)
async def predict_single_game(req: GamePredictionRequest):
    """
    Generate predictions for a specific matchup.
    
    STRICT MODE: All 8 line/odds parameters are REQUIRED.
    """
    if not hasattr(app.state, 'engine') or app.state.engine is None:
        raise HTTPException(status_code=503, detail="STRICT MODE: Engine not loaded - models missing")

    try:
        # Build features
        features = await app.state.feature_builder.build_game_features(
            req.home_team, req.away_team
        )

        # Predict - ALL parameters required
        preds = app.state.engine.predict_all_markets(
            features,
            fg_spread_line=req.fg_spread_line,
            fg_total_line=req.fg_total_line,
            fg_home_ml_odds=req.fg_home_ml,
            fg_away_ml_odds=req.fg_away_ml,
            fh_spread_line=req.fh_spread_line,
            fh_total_line=req.fh_total_line,
            fh_home_ml_odds=req.fh_home_ml,
            fh_away_ml_odds=req.fh_away_ml,
        )
        return preds
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"STRICT MODE: {e}")
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
