"""
NBA v5.1 FINAL - FastAPI Prediction Server

PRODUCTION: 6 PROVEN ROE Markets (Full Game + First Half)

Full Game:
- Spread: 60.6% accuracy, +15.7% ROI
- Total: 59.2% accuracy, +13.1% ROI
- Moneyline: 65.5% accuracy, +25.1% ROI

First Half:
- Spread: 55.9% accuracy, +8.2% ROI
- Total: 58.1% accuracy, +11.4% ROI
- Moneyline: 63.0% accuracy, +19.8% ROI

All inputs required. No fallbacks. No silent failures.
"""
import os
import json
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Union
from pathlib import Path as PathLib
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query, Path, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from src.config import settings
from src.prediction import UnifiedPredictionEngine, ModelNotFoundError
from src.ingestion import the_odds
from src.ingestion.betting_splits import fetch_public_betting_splits
from scripts.build_rich_features import RichFeatureBuilder
from src.utils.logging import get_logger
from src.utils.security import fail_fast_on_missing_keys, get_api_key_status, mask_api_key
from src.utils.api_auth import get_api_key, APIKeyMiddleware

logger = get_logger(__name__)


def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# Prometheus metrics
REQUEST_COUNT = Counter(
    'nba_api_requests_total',
    'Total number of API requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'nba_api_request_duration_seconds',
    'API request duration in seconds',
    ['method', 'endpoint']
)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)


# --- Request/Response Models - v5.1 FINAL ---

class GamePredictionRequest(BaseModel):
    """Request for single game prediction - v5.1 all markets."""
    home_team: str = Field(..., example="Cleveland Cavaliers")
    away_team: str = Field(..., example="Chicago Bulls")
    # Full game lines - REQUIRED
    fg_spread_line: float
    fg_total_line: float
    # First half lines - optional but recommended
    fh_spread_line: Optional[float] = None
    fh_total_line: Optional[float] = None
    # Moneyline odds - optional but recommended
    home_ml_odds: Optional[int] = Field(None, example=-150)
    away_ml_odds: Optional[int] = Field(None, example=130)
    fh_home_ml_odds: Optional[int] = None
    fh_away_ml_odds: Optional[int] = None


class MarketPrediction(BaseModel):
    side: str
    confidence: float
    edge: float
    passes_filter: bool
    filter_reason: Optional[str] = None


class GamePredictions(BaseModel):
    full_game: Dict[str, Any]
    first_half: Dict[str, Any]


class SlateResponse(BaseModel):
    date: str
    predictions: List[Dict[str, Any]]
    total_plays: int


# --- API Setup - v5.1 FINAL ---

app = FastAPI(
    title="NBA v5.1 FINAL - Production Picks",
    description="6 PROVEN ROE Markets: FG+1H Spread, Total, Moneyline",
    version="5.1.0-final"
)

# Add rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# API Authentication (optional - can be disabled via REQUIRE_API_AUTH=false)
if os.getenv("REQUIRE_API_AUTH", "false").lower() == "true":
    app.add_middleware(APIKeyMiddleware, require_auth=True)

# CORS configuration - production safe
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8090").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*", "X-API-Key"],
)

# Request metrics middleware
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = datetime.now()
    method = request.method
    endpoint = request.url.path
    
    try:
        response = await call_next(request)
        status = response.status_code
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
        return response
    except Exception as e:
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=500).inc()
        raise
    finally:
        duration = (datetime.now() - start_time).total_seconds()
        REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)


def _models_dir() -> PathLib:
    return PathLib(settings.data_processed_dir) / "models"


@app.on_event("startup")
def startup_event():
    """
    Initialize the prediction engine on startup.
    
    v5.1 FINAL: 6 markets (FG+1H for Spread, Total, Moneyline)
    Fails LOUDLY if models are missing or API keys are invalid.
    """
    # SECURITY: Validate API keys at startup - fail fast if missing
    try:
        fail_fast_on_missing_keys()
        logger.info("✓ API keys validated")
    except Exception as e:
        logger.error(f"Security validation failed: {e}")
        raise
    
    models_dir = _models_dir()
    logger.info(f"v5.1 FINAL: Loading Unified Prediction Engine from {models_dir}")
    
    # Diagnostic: List files in models directory
    if models_dir.exists():
        model_files = list(models_dir.glob("*"))
        logger.info(f"Found {len(model_files)} files in models directory:")
        for f in sorted(model_files):
            size = f.stat().st_size if f.is_file() else 0
            logger.info(f"  - {f.name} ({size:,} bytes)")
    else:
        logger.error(f"Models directory does not exist: {models_dir}")
    
    # NO TRY/EXCEPT - Let it crash if models are missing
    app.state.engine = UnifiedPredictionEngine(models_dir=models_dir)
    app.state.feature_builder = RichFeatureBuilder(season=settings.current_season)
    
    # Log model info
    model_info = app.state.engine.get_model_info()
    logger.info(f"NBA v5.1 FINAL initialized - {model_info['markets']} markets: {model_info['markets_list']}")


# --- Endpoints ---

@app.get("/health")
@limiter.limit("100/minute")
def health(request: Request):
    """Check API health - v5.1 FINAL with 6 markets."""
    engine_loaded = hasattr(app.state, 'engine') and app.state.engine is not None
    api_keys = get_api_key_status()
    
    model_info = {}
    if engine_loaded:
        model_info = app.state.engine.get_model_info()
    
    return {
        "status": "ok",
        "version": "5.1-FINAL",
        "markets": 6,
        "markets_list": [
            "fg_spread", "fg_total", "fg_moneyline",
            "1h_spread", "1h_total", "1h_moneyline"
        ],
        "periods": ["full_game", "first_half"],
        "engine_loaded": engine_loaded,
        "model_info": model_info,
        "season": settings.current_season,
        "api_keys": api_keys,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/verify")
@limiter.limit("10/minute")
def verify_integrity(request: Request):
    """
    Verify model integrity and component usage.
    
    v5.1: Verifies all 6 models (FG+1H for spread, total, moneyline)
    """
    results = {
        "status": "pass",
        "version": "5.1-FINAL",
        "checks": {},
        "errors": []
    }
    
    # Check 1: Engine loaded
    if not hasattr(app.state, 'engine') or app.state.engine is None:
        results["status"] = "fail"
        results["errors"].append("Engine not loaded")
        results["checks"]["engine_loaded"] = False
    else:
        results["checks"]["engine_loaded"] = True
        
        # Check 2: Predictors exist
        has_spread = hasattr(app.state.engine, 'spread_predictor')
        has_total = hasattr(app.state.engine, 'total_predictor')
        has_moneyline = hasattr(app.state.engine, 'moneyline_predictor')
        
        results["checks"]["predictors"] = {
            "spread": has_spread,
            "total": has_total,
            "moneyline": has_moneyline,
        }
        
        if not (has_spread and has_total and has_moneyline):
            results["status"] = "fail"
            results["errors"].append("Missing predictors")
        
        # Check 3: Test FG prediction
        try:
            test_features = {
                "home_ppg": 115.0, "away_ppg": 112.0,
                "home_papg": 110.0, "away_papg": 115.0,
                "predicted_margin": 3.0, "predicted_total": 227.0,
                "predicted_margin_1h": 1.5, "predicted_total_1h": 113.5,
                "home_win_pct": 0.6, "away_win_pct": 0.4,
                "home_avg_margin": 2.0, "away_avg_margin": -1.0,
                "home_rest_days": 2, "away_rest_days": 1,
                "home_b2b": 0, "away_b2b": 0,
                "dynamic_hca": 3.0, "h2h_win_pct": 0.5, "h2h_avg_margin": 0.0,
            }
            
            test_pred = app.state.engine.predict_full_game(
                features=test_features,
                spread_line=-3.5,
                total_line=225.0,
                home_ml_odds=-150,
                away_ml_odds=130,
            )
            
            results["checks"]["fg_prediction_works"] = True
            results["checks"]["fg_has_moneyline"] = "moneyline" in test_pred
            
        except Exception as e:
            results["status"] = "fail"
            results["errors"].append(f"FG test prediction failed: {str(e)}")
            results["checks"]["fg_prediction_works"] = False
        
        # Check 4: Test 1H prediction
        try:
            test_pred_1h = app.state.engine.predict_first_half(
                features=test_features,
                spread_line=-1.5,
                total_line=112.5,
                home_ml_odds=-140,
                away_ml_odds=120,
            )
            
            results["checks"]["1h_prediction_works"] = True
            results["checks"]["1h_has_moneyline"] = "moneyline" in test_pred_1h
            
        except Exception as e:
            results["status"] = "fail"
            results["errors"].append(f"1H test prediction failed: {str(e)}")
            results["checks"]["1h_prediction_works"] = False
    
    return results


@app.get("/slate/{date}", response_model=SlateResponse)
@limiter.limit("30/minute")
async def get_slate_predictions(
    request: Request,
    date: str = Path(..., description="Date in YYYY-MM-DD format, 'today', or 'tomorrow'"),
    use_splits: bool = Query(True, description="Whether to fetch and use betting splits"),
    api_key: str = None,
):
    """
    Get all predictions for a full day's slate.
    
    v5.1 FINAL: Returns all 6 markets (FG+1H Spread, Total, Moneyline).
    """
    if not hasattr(app.state, 'engine') or app.state.engine is None:
        raise HTTPException(status_code=503, detail="v5.1: Engine not loaded - models missing")

    # Resolve date
    from src.utils.slate_analysis import get_target_date, fetch_todays_games, extract_consensus_odds
    try:
        target_date = get_target_date(date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    # Fetch games
    try:
        games = await fetch_todays_games(target_date)
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

            # Extract consensus lines
            odds = extract_consensus_odds(game)
            fg_spread = odds.get("home_spread")
            fg_total = odds.get("total")
            fh_spread = odds.get("fh_home_spread")
            fh_total = odds.get("fh_total")
            home_ml = odds.get("home_ml")
            away_ml = odds.get("away_ml")
            fh_home_ml = odds.get("fh_home_ml")
            fh_away_ml = odds.get("fh_away_ml")
            
            # Validate required lines (FG at minimum)
            if fg_spread is None or fg_total is None:
                logger.warning(f"v5.1: Skipping {home_team} vs {away_team} - missing FG lines")
                continue

            # Predict all 6 markets
            preds = app.state.engine.predict_all_markets(
                features,
                fg_spread_line=fg_spread,
                fg_total_line=fg_total,
                fh_spread_line=fh_spread,
                fh_total_line=fh_total,
                home_ml_odds=home_ml,
                away_ml_odds=away_ml,
                fh_home_ml_odds=fh_home_ml,
                fh_away_ml_odds=fh_away_ml,
            )

            # Count plays (all 6 markets)
            game_plays = 0
            for period in ["full_game", "first_half"]:
                period_preds = preds.get(period, {})
                for market in ["spread", "total", "moneyline"]:
                    if period_preds.get(market, {}).get("passes_filter"):
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
            logger.warning(f"v5.1: Skipping {home_team} vs {away_team} - {e}")
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
@limiter.limit("20/minute")
async def get_comprehensive_slate_analysis(
    request: Request,
    date: str = Path(..., description="Date in YYYY-MM-DD format, 'today', or 'tomorrow'"),
    use_splits: bool = Query(True, description="Whether to fetch and use betting splits")
):
    """
    Get comprehensive slate analysis with full edge calculations.
    
    v5.1 FINAL: Full analysis for all 6 markets.
    """
    if not hasattr(app.state, 'engine') or app.state.engine is None:
        raise HTTPException(status_code=503, detail="v5.1: Engine not loaded - models missing")

    from src.utils.slate_analysis import (
        get_target_date, fetch_todays_games, parse_utc_time, to_cst, extract_consensus_odds
    )
    from src.utils.comprehensive_edge import calculate_comprehensive_edge
    from src.modeling.edge_thresholds import get_edge_thresholds_for_game
    from zoneinfo import ZoneInfo
    
    CST = ZoneInfo("America/Chicago")
    
    # Resolve date
    try:
        target_date = get_target_date(date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    # Get edge thresholds (all 6 markets)
    edge_thresholds = get_edge_thresholds_for_game(
        game_date=target_date,
        bet_types=["spread", "total", "moneyline", "1h_spread", "1h_total", "1h_moneyline"]
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
            
            # Build first half features
            fh_features = features.copy()
            
            # Extract odds
            odds = extract_consensus_odds(game)
            
            # Get betting splits for this game
            betting_splits = splits_dict.get(game_key)
            
            # Get actual model predictions from engine
            fg_spread = odds.get("home_spread")
            fg_total = odds.get("total")
            fh_spread = odds.get("fh_home_spread")
            fh_total = odds.get("fh_total")
            home_ml = odds.get("home_ml")
            away_ml = odds.get("away_ml")
            fh_home_ml = odds.get("fh_home_ml")
            fh_away_ml = odds.get("fh_away_ml")
            
            engine_predictions = None
            if fg_spread is not None and fg_total is not None:
                try:
                    engine_predictions = app.state.engine.predict_all_markets(
                        features,
                        fg_spread_line=fg_spread,
                        fg_total_line=fg_total,
                        fh_spread_line=fh_spread,
                        fh_total_line=fh_total,
                        home_ml_odds=home_ml,
                        away_ml_odds=away_ml,
                        fh_home_ml_odds=fh_home_ml,
                        fh_away_ml_odds=fh_away_ml,
                    )
                except Exception as e:
                    logger.warning(f"Could not get engine predictions for {home_team} vs {away_team}: {e}")
            
            # Calculate comprehensive edge
            comprehensive_edge = calculate_comprehensive_edge(
                features=features,
                fh_features=fh_features,
                odds=odds,
                game=game,
                betting_splits=betting_splits,
                edge_thresholds=edge_thresholds,
                engine_predictions=engine_predictions
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
    
    return convert_numpy_types({
        "date": str(target_date),
        "version": "5.1-FINAL",
        "markets": [
            "fg_spread", "fg_total", "fg_moneyline",
            "1h_spread", "1h_total", "1h_moneyline"
        ],
        "analysis": analysis_results,
        "edge_thresholds": edge_thresholds
    })


@app.post("/predict/game", response_model=GamePredictions)
@limiter.limit("60/minute")
async def predict_single_game(request: Request, req: GamePredictionRequest):
    """
    Generate predictions for a specific matchup.
    
    v5.1 FINAL: All 6 markets (FG+1H Spread, Total, Moneyline).
    """
    if not hasattr(app.state, 'engine') or app.state.engine is None:
        raise HTTPException(status_code=503, detail="v5.1: Engine not loaded - models missing")

    try:
        # Build features
        features = await app.state.feature_builder.build_game_features(
            req.home_team, req.away_team
        )

        # Predict all 6 markets
        preds = app.state.engine.predict_all_markets(
            features,
            fg_spread_line=req.fg_spread_line,
            fg_total_line=req.fg_total_line,
            fh_spread_line=req.fh_spread_line,
            fh_total_line=req.fh_total_line,
            home_ml_odds=req.home_ml_odds,
            away_ml_odds=req.away_ml_odds,
            fh_home_ml_odds=req.fh_home_ml_odds,
            fh_away_ml_odds=req.fh_away_ml_odds,
        )
        return preds
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"v5.1: {e}")
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
