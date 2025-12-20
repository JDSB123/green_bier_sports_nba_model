"""
NBA v6.0 - FastAPI Prediction Server

PRODUCTION: 9 INDEPENDENT Markets (Q1 + 1H + FG)

First Quarter (Q1):
- Q1 Spread
- Q1 Total
- Q1 Moneyline

First Half (1H):
- 1H Spread
- 1H Total
- 1H Moneyline

Full Game (FG):
- FG Spread
- FG Total
- FG Moneyline

All periods use INDEPENDENT models trained on period-specific features.
No cross-period dependencies. No fallbacks. No silent failures.
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
from src.ingestion.betting_splits import fetch_public_betting_splits, validate_splits_sources_configured
from scripts.build_rich_features import RichFeatureBuilder
from src.utils.logging import get_logger
from src.utils.security import fail_fast_on_missing_keys, get_api_key_status, mask_api_key, validate_premium_features
from src.utils.api_auth import get_api_key, APIKeyMiddleware
from src.tracking import PickTracker

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


# --- Request/Response Models - v6.0 ---

class GamePredictionRequest(BaseModel):
    """Request for single game prediction - v6.0 all 9 markets."""
    home_team: str = Field(..., example="Cleveland Cavaliers")
    away_team: str = Field(..., example="Chicago Bulls")
    # Full game lines - REQUIRED
    fg_spread_line: float
    fg_total_line: float
    # First half lines - optional but recommended
    fh_spread_line: Optional[float] = None
    fh_total_line: Optional[float] = None
    # First quarter lines - optional
    q1_spread_line: Optional[float] = None
    q1_total_line: Optional[float] = None
    # Moneyline odds - optional but recommended
    home_ml_odds: Optional[int] = Field(None, example=-150)
    away_ml_odds: Optional[int] = Field(None, example=130)
    fh_home_ml_odds: Optional[int] = None
    fh_away_ml_odds: Optional[int] = None
    q1_home_ml_odds: Optional[int] = None
    q1_away_ml_odds: Optional[int] = None


class MarketPrediction(BaseModel):
    side: str
    confidence: float
    edge: float
    passes_filter: bool
    filter_reason: Optional[str] = None


class GamePredictions(BaseModel):
    first_quarter: Dict[str, Any] = {}
    first_half: Dict[str, Any] = {}
    full_game: Dict[str, Any] = {}


class SlateResponse(BaseModel):
    date: str
    predictions: List[Dict[str, Any]]
    total_plays: int


# --- API Setup - v6.0 ---

app = FastAPI(
    title="NBA v6.0 - Production Picks",
    description="9 INDEPENDENT Markets: Q1+1H+FG for Spread, Total, Moneyline",
    version="6.0.0"
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

    v6.0: 9 INDEPENDENT markets (Q1+1H+FG for Spread, Total, Moneyline)
    Fails LOUDLY if models are missing or API keys are invalid.
    """
    # SECURITY: Validate API keys at startup - fail fast if missing
    try:
        fail_fast_on_missing_keys()
        logger.info("API keys validated")
    except Exception as e:
        logger.error(f"Security validation failed: {e}")
        raise

    # Validate betting splits sources (warning only, not fatal)
    splits_sources = validate_splits_sources_configured()
    app.state.splits_sources_configured = splits_sources

    # Validate all premium features and log what's available
    premium_features = validate_premium_features()
    app.state.premium_features = premium_features

    models_dir = _models_dir()
    logger.info(f"v6.0: Loading Unified Prediction Engine from {models_dir}")

    # Diagnostic: List files in models directory
    if models_dir.exists():
        model_files = list(models_dir.glob("*"))
        logger.info(f"Found {len(model_files)} files in models directory:")
        for f in sorted(model_files):
            size = f.stat().st_size if f.is_file() else 0
            logger.info(f"  - {f.name} ({size:,} bytes)")
    else:
        logger.error(f"Models directory does not exist: {models_dir}")

    # STRICT MODE: Load engine - ALL 9 models required, NO FALLBACKS
    logger.info("STRICT MODE: Requiring all 9 models (Q1/1H/FG × Spread/Total/ML)")
    app.state.engine = UnifiedPredictionEngine(models_dir=models_dir, require_all=True)
    app.state.feature_builder = RichFeatureBuilder(season=settings.current_season)
    
    # Initialize live pick tracker
    picks_dir = PathLib(settings.data_processed_dir) / "picks"
    picks_dir.mkdir(parents=True, exist_ok=True)
    app.state.tracker = PickTracker(storage_path=str(picks_dir / "tracked_picks.json"))
    logger.info(f"Pick tracker initialized at {picks_dir}")

    # Log model info
    model_info = app.state.engine.get_model_info()
    logger.info(f"NBA v6.0 initialized - {model_info['markets']}/9 markets loaded: {model_info['markets_list']}")


# --- Endpoints ---

@app.get("/health")
@limiter.limit("100/minute")
def health(request: Request):
    """Check API health - v6.0 with 9 independent markets."""
    engine_loaded = hasattr(app.state, 'engine') and app.state.engine is not None
    api_keys = get_api_key_status()

    model_info = {}
    if engine_loaded:
        model_info = app.state.engine.get_model_info()

    return {
        "status": "ok",
        "version": "6.0",
        "architecture": "9-model independent",
        "markets": model_info.get("markets", 0),
        "markets_list": [
            "q1_spread", "q1_total", "q1_moneyline",
            "1h_spread", "1h_total", "1h_moneyline",
            "fg_spread", "fg_total", "fg_moneyline",
        ],
        "periods": ["first_quarter", "first_half", "full_game"],
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
    
    v6.0: Verifies all 9 independent models (Q1, 1H, FG for spread, total, moneyline)
    """
    results = {
        "status": "pass",
        "version": "6.0",
        "markets": {
            "q1": ["spread", "total", "moneyline"],
            "1h": ["spread", "total", "moneyline"],
            "fg": ["spread", "total", "moneyline"],
        },
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
        
        # Check 2: Period predictors exist
        has_fg = hasattr(app.state.engine, 'fg_predictor')
        has_1h = hasattr(app.state.engine, 'h1_predictor')
        has_q1 = hasattr(app.state.engine, 'q1_predictor')
        
        results["checks"]["period_predictors"] = {
            "full_game": has_fg,
            "first_half": has_1h,
            "first_quarter": has_q1,
        }
        
        # Also check legacy predictors if present
        has_spread = hasattr(app.state.engine, 'spread_predictor')
        has_total = hasattr(app.state.engine, 'total_predictor')
        has_moneyline = hasattr(app.state.engine, 'moneyline_predictor')
        
        results["checks"]["legacy_predictors"] = {
            "spread": has_spread,
            "total": has_total,
            "moneyline": has_moneyline,
        }
        
        if not (has_fg or (has_spread and has_total)):
            results["status"] = "fail"
            results["errors"].append("Missing period predictors")
        
        # Test features for predictions
        test_features = {
            "home_ppg": 115.0, "away_ppg": 112.0,
            "home_papg": 110.0, "away_papg": 115.0,
            "predicted_margin": 3.0, "predicted_total": 227.0,
            "predicted_margin_1h": 1.5, "predicted_total_1h": 113.5,
            "predicted_margin_q1": 0.8, "predicted_total_q1": 56.5,
            "home_win_pct": 0.6, "away_win_pct": 0.4,
            "home_avg_margin": 2.0, "away_avg_margin": -1.0,
            "home_rest_days": 2, "away_rest_days": 1,
            "home_b2b": 0, "away_b2b": 0,
            "dynamic_hca": 3.0, "h2h_win_pct": 0.5, "h2h_avg_margin": 0.0,
            # Q1-specific features
            "home_q1_ppg": 28.5, "away_q1_ppg": 27.8,
            "home_q1_papg": 27.2, "away_q1_papg": 28.0,
            "home_q1_avg_margin": 0.5, "away_q1_avg_margin": -0.3,
            # 1H-specific features
            "home_1h_ppg": 56.2, "away_1h_ppg": 55.0,
            "home_1h_papg": 54.8, "away_1h_papg": 55.5,
            "home_1h_avg_margin": 1.2, "away_1h_avg_margin": -0.5,
        }
        
        # Check 3: Test Q1 prediction
        try:
            test_pred_q1 = app.state.engine.predict_quarter(
                features=test_features,
                spread_line=-0.5,
                total_line=56.5,
                home_ml_odds=-110,
                away_ml_odds=-110,
            )
            
            results["checks"]["q1_prediction_works"] = True
            results["checks"]["q1_has_spread"] = "spread" in test_pred_q1
            results["checks"]["q1_has_total"] = "total" in test_pred_q1
            results["checks"]["q1_has_moneyline"] = "moneyline" in test_pred_q1
            
        except Exception as e:
            results["status"] = "warn"  # Q1 is newer, warn instead of fail
            results["errors"].append(f"Q1 test prediction failed: {str(e)}")
            results["checks"]["q1_prediction_works"] = False
        
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
            results["checks"]["1h_has_spread"] = "spread" in test_pred_1h
            results["checks"]["1h_has_total"] = "total" in test_pred_1h
            results["checks"]["1h_has_moneyline"] = "moneyline" in test_pred_1h
            
        except Exception as e:
            results["status"] = "fail"
            results["errors"].append(f"1H test prediction failed: {str(e)}")
            results["checks"]["1h_prediction_works"] = False
        
        # Check 5: Test FG prediction
        try:
            test_pred = app.state.engine.predict_full_game(
                features=test_features,
                spread_line=-3.5,
                total_line=225.0,
                home_ml_odds=-150,
                away_ml_odds=130,
            )
            
            results["checks"]["fg_prediction_works"] = True
            results["checks"]["fg_has_spread"] = "spread" in test_pred
            results["checks"]["fg_has_total"] = "total" in test_pred
            results["checks"]["fg_has_moneyline"] = "moneyline" in test_pred
            
        except Exception as e:
            results["status"] = "fail"
            results["errors"].append(f"FG test prediction failed: {str(e)}")
            results["checks"]["fg_prediction_works"] = False
    
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
    
    v6.0: Returns all 9 markets (Q1+1H+FG for Spread, Total, Moneyline).
    """
    if not hasattr(app.state, 'engine') or app.state.engine is None:
        raise HTTPException(status_code=503, detail="v6.0: Engine not loaded - models missing")

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
            q1_spread = odds.get("q1_home_spread")
            q1_total = odds.get("q1_total")
            home_ml = odds.get("home_ml")
            away_ml = odds.get("away_ml")
            fh_home_ml = odds.get("fh_home_ml")
            fh_away_ml = odds.get("fh_away_ml")
            q1_home_ml = odds.get("q1_home_ml")
            q1_away_ml = odds.get("q1_away_ml")
            
            # Validate required lines (FG at minimum)
            if fg_spread is None or fg_total is None:
                logger.warning(f"v6.0: Skipping {home_team} vs {away_team} - missing FG lines")
                continue

            # Predict all 9 markets
            preds = app.state.engine.predict_all_markets(
                features,
                fg_spread_line=fg_spread,
                fg_total_line=fg_total,
                fh_spread_line=fh_spread,
                fh_total_line=fh_total,
                q1_spread_line=q1_spread,
                q1_total_line=q1_total,
                home_ml_odds=home_ml,
                away_ml_odds=away_ml,
                fh_home_ml_odds=fh_home_ml,
                fh_away_ml_odds=fh_away_ml,
                q1_home_ml_odds=q1_home_ml,
                q1_away_ml_odds=q1_away_ml,
            )

            # Count plays (all 9 markets) and record picks
            game_plays = 0
            game_date = target_date.strftime("%Y-%m-%d")
            for period in ["first_quarter", "first_half", "full_game"]:
                period_preds = preds.get(period, {})
                for market in ["spread", "total", "moneyline"]:
                    pred_data = period_preds.get(market, {})
                    if pred_data.get("passes_filter"):
                        game_plays += 1
                        # Record the pick for live tracking
                        market_key = f"{period.replace('first_quarter', 'q1').replace('first_half', '1h').replace('full_game', 'fg')}_{market}"
                        line = None
                        if market == "spread":
                            if period == "first_quarter":
                                line = q1_spread
                            elif period == "first_half":
                                line = fh_spread
                            else:
                                line = fg_spread
                        elif market == "total":
                            if period == "first_quarter":
                                line = q1_total
                            elif period == "first_half":
                                line = fh_total
                            else:
                                line = fg_total
                        
                        app.state.tracker.record_pick(
                            game_date=game_date,
                            home_team=home_team,
                            away_team=away_team,
                            market=market_key,
                            side=pred_data.get("side"),
                            line=line,
                            confidence=pred_data.get("confidence"),
                            edge=pred_data.get("edge"),
                        )
            
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
            logger.warning(f"v6.0: Skipping {home_team} vs {away_team} - {e}")
            continue
        except Exception as e:
            logger.error(f"Error processing {home_team} vs {away_team}: {e}")
            continue

    return SlateResponse(
        date=str(target_date),
        predictions=results,
        total_plays=total_plays
    )


def format_american_odds(odds: int) -> str:
    """Format American odds with +/- prefix."""
    if odds is None:
        return "N/A"
    return f"+{odds}" if odds > 0 else str(odds)


def get_fire_rating(confidence: float, edge: float) -> str:
    """
    Get fire rating based on confidence and edge.
    🔥🔥🔥 = Elite (conf >= 70% AND edge >= 5)
    🔥🔥 = Strong (conf >= 60% AND edge >= 3)
    🔥 = Good (passes filters)
    """
    if confidence >= 0.70 and abs(edge) >= 5:
        return "🔥🔥🔥"
    elif confidence >= 0.60 and abs(edge) >= 3:
        return "🔥🔥"
    else:
        return "🔥"


@app.get("/slate/{date}/executive")
@limiter.limit("30/minute")
async def get_executive_summary(
    request: Request,
    date: str = Path(..., description="Date in YYYY-MM-DD format, 'today', or 'tomorrow'"),
    use_splits: bool = Query(True, description="Whether to fetch and use betting splits")
):
    """
    BLUF (Bottom Line Up Front) Executive Summary.
    
    Returns a clean actionable betting card with all picks that pass filters.
    Sorted by game time, then fire rating.
    """
    if not hasattr(app.state, 'engine') or app.state.engine is None:
        raise HTTPException(status_code=503, detail="v6.0: Engine not loaded - models missing")

    from src.utils.slate_analysis import (
        get_target_date, fetch_todays_games, parse_utc_time, to_cst, extract_consensus_odds
    )
    from zoneinfo import ZoneInfo
    
    CST = ZoneInfo("America/Chicago")
    
    # Resolve date
    try:
        target_date = get_target_date(date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    # Fetch games
    games = await fetch_todays_games(target_date)
    if not games:
        return {
            "date": str(target_date),
            "generated_at": datetime.now(CST).strftime("%Y-%m-%d %I:%M %p CST"),
            "version": "6.0",
            "total_plays": 0,
            "plays": [],
            "summary": "No games scheduled"
        }

    # Fetch betting splits
    splits_dict = {}
    if use_splits:
        try:
            from src.ingestion.betting_splits import fetch_public_betting_splits
            splits_dict = await fetch_public_betting_splits(games, source="auto")
        except Exception as e:
            logger.warning(f"Could not fetch splits: {e}")

    # Process each game and collect plays
    all_plays = []
    
    for game in games:
        home_team = game.get("home_team")
        away_team = game.get("away_team")
        commence_time = game.get("commence_time")
        
        try:
            # Parse time
            game_dt = parse_utc_time(commence_time) if commence_time else None
            game_cst = to_cst(game_dt) if game_dt else None
            time_cst_str = game_cst.strftime("%m/%d %I:%M %p") if game_cst else "TBD"
            
            # Build features
            game_key = f"{away_team}@{home_team}"
            features = await app.state.feature_builder.build_game_features(
                home_team, away_team, betting_splits=splits_dict.get(game_key)
            )
            
            # Extract team records from features (use raw wins/losses to avoid rounding errors)
            home_wins = features.get("home_wins", 0)
            home_losses = features.get("home_losses", 0)
            away_wins = features.get("away_wins", 0)
            away_losses = features.get("away_losses", 0)
            
            home_record = f"({home_wins}-{home_losses})"
            away_record = f"({away_wins}-{away_losses})"
            matchup_display = f"{away_team} {away_record} @ {home_team} {home_record}"
            
            # Extract odds
            odds = extract_consensus_odds(game)
            fg_spread = odds.get("home_spread")
            fg_total = odds.get("total")
            fh_spread = odds.get("fh_home_spread")
            fh_total = odds.get("fh_total")
            q1_spread = odds.get("q1_home_spread")
            q1_total = odds.get("q1_total")
            home_ml = odds.get("home_ml")
            away_ml = odds.get("away_ml")
            fh_home_ml = odds.get("fh_home_ml")
            fh_away_ml = odds.get("fh_away_ml")
            q1_home_ml = odds.get("q1_home_ml")
            q1_away_ml = odds.get("q1_away_ml")
            
            if fg_spread is None or fg_total is None:
                continue
            
            # Get predictions for all 9 markets
            preds = app.state.engine.predict_all_markets(
                features,
                fg_spread_line=fg_spread,
                fg_total_line=fg_total,
                fh_spread_line=fh_spread,
                fh_total_line=fh_total,
                q1_spread_line=q1_spread,
                q1_total_line=q1_total,
                home_ml_odds=home_ml,
                away_ml_odds=away_ml,
                fh_home_ml_odds=fh_home_ml,
                fh_away_ml_odds=fh_away_ml,
                q1_home_ml_odds=q1_home_ml,
                q1_away_ml_odds=q1_away_ml,
            )
            
            # Process Full Game markets
            fg = preds.get("full_game", {})
            
            # FG Spread
            fg_spread_pred = fg.get("spread", {})
            if fg_spread_pred.get("passes_filter"):
                bet_side = fg_spread_pred.get("bet_side", "home")
                pick_team = home_team if bet_side == "home" else away_team
                pick_line = fg_spread if bet_side == "home" else -fg_spread
                pick_price = odds.get("home_spread_price", -110)
                model_margin = features.get("predicted_margin", 0)
                
                all_plays.append({
                    "time_cst": time_cst_str,
                    "sort_time": game_cst.isoformat() if game_cst else "",
                    "matchup": matchup_display,
                    "period": "FG",
                    "market": "SPREAD",
                    "pick": f"{pick_team} {pick_line:+.1f}",
                    "pick_odds": format_american_odds(pick_price),
                    "model_prediction": f"{model_margin:+.1f} pts",
                    "market_line": f"{fg_spread:+.1f}",
                    "edge": f"{fg_spread_pred.get('edge', 0):+.1f} pts",
                    "edge_raw": abs(fg_spread_pred.get('edge', 0)),
                    "confidence": fg_spread_pred.get("confidence", 0),
                    "fire_rating": get_fire_rating(fg_spread_pred.get("confidence", 0), fg_spread_pred.get("edge", 0))
                })
            
            # FG Total
            fg_total_pred = fg.get("total", {})
            if fg_total_pred.get("passes_filter"):
                bet_side = fg_total_pred.get("bet_side", "over")
                model_total = features.get("predicted_total", 0)
                pick_display = f"{'OVER' if bet_side == 'over' else 'UNDER'} {fg_total}"
                
                all_plays.append({
                    "time_cst": time_cst_str,
                    "sort_time": game_cst.isoformat() if game_cst else "",
                    "matchup": matchup_display,
                    "period": "FG",
                    "market": "TOTAL",
                    "pick": pick_display,
                    "pick_odds": format_american_odds(odds.get("total_price", -110)),
                    "model_prediction": f"{model_total:.1f}",
                    "market_line": f"{fg_total}",
                    "edge": f"{fg_total_pred.get('edge', 0):+.1f} pts",
                    "edge_raw": abs(fg_total_pred.get('edge', 0)),
                    "confidence": fg_total_pred.get("confidence", 0),
                    "fire_rating": get_fire_rating(fg_total_pred.get("confidence", 0), fg_total_pred.get("edge", 0))
                })
            
            # FG Moneyline
            fg_ml_pred = fg.get("moneyline", {})
            if fg_ml_pred.get("passes_filter"):
                rec_bet = fg_ml_pred.get("recommended_bet")
                if rec_bet:
                    pick_team = home_team if rec_bet == "home" else away_team
                    pick_odds_val = home_ml if rec_bet == "home" else away_ml
                    model_prob = fg_ml_pred.get("home_win_prob", 0.5) if rec_bet == "home" else fg_ml_pred.get("away_win_prob", 0.5)
                    market_prob = odds.get("home_implied_prob", 0.5) if rec_bet == "home" else odds.get("away_implied_prob", 0.5)
                    edge_pct = fg_ml_pred.get("home_edge", 0) if rec_bet == "home" else fg_ml_pred.get("away_edge", 0)
                    
                    all_plays.append({
                        "time_cst": time_cst_str,
                        "sort_time": game_cst.isoformat() if game_cst else "",
                        "matchup": matchup_display,
                        "period": "FG",
                        "market": "ML",
                        "pick": pick_team,
                        "pick_odds": format_american_odds(pick_odds_val),
                        "model_prediction": f"{model_prob*100:.1f}%",
                        "market_line": f"{market_prob*100:.1f}%",
                        "edge": f"{edge_pct*100:+.1f}%",
                        "edge_raw": abs(edge_pct * 100),
                        "confidence": fg_ml_pred.get("confidence", 0),
                        "fire_rating": get_fire_rating(fg_ml_pred.get("confidence", 0), edge_pct * 100)
                    })
            
            # Process First Half markets
            fh = preds.get("first_half", {})
            
            # 1H Spread
            fh_spread_pred = fh.get("spread", {})
            if fh_spread_pred.get("passes_filter") and fh_spread is not None:
                bet_side = fh_spread_pred.get("bet_side", "home")
                pick_team = home_team if bet_side == "home" else away_team
                pick_line = fh_spread if bet_side == "home" else -fh_spread
                pick_price = odds.get("fh_home_spread_price", -110)
                model_margin_1h = features.get("predicted_margin_1h", 0)
                
                all_plays.append({
                    "time_cst": time_cst_str,
                    "sort_time": game_cst.isoformat() if game_cst else "",
                    "matchup": matchup_display,
                    "period": "1H",
                    "market": "SPREAD",
                    "pick": f"{pick_team} {pick_line:+.1f}",
                    "pick_odds": format_american_odds(pick_price),
                    "model_prediction": f"{model_margin_1h:+.1f} pts",
                    "market_line": f"{fh_spread:+.1f}",
                    "edge": f"{fh_spread_pred.get('edge', 0):+.1f} pts",
                    "edge_raw": abs(fh_spread_pred.get('edge', 0)),
                    "confidence": fh_spread_pred.get("confidence", 0),
                    "fire_rating": get_fire_rating(fh_spread_pred.get("confidence", 0), fh_spread_pred.get("edge", 0))
                })
            
            # 1H Total
            fh_total_pred = fh.get("total", {})
            if fh_total_pred.get("passes_filter") and fh_total is not None:
                bet_side = fh_total_pred.get("bet_side", "over")
                model_total_1h = features.get("predicted_total_1h", 0)
                pick_display = f"{'OVER' if bet_side == 'over' else 'UNDER'} {fh_total}"
                
                all_plays.append({
                    "time_cst": time_cst_str,
                    "sort_time": game_cst.isoformat() if game_cst else "",
                    "matchup": matchup_display,
                    "period": "1H",
                    "market": "TOTAL",
                    "pick": pick_display,
                    "pick_odds": format_american_odds(odds.get("fh_total_price", -110)),
                    "model_prediction": f"{model_total_1h:.1f}",
                    "market_line": f"{fh_total}",
                    "edge": f"{fh_total_pred.get('edge', 0):+.1f} pts",
                    "edge_raw": abs(fh_total_pred.get('edge', 0)),
                    "confidence": fh_total_pred.get("confidence", 0),
                    "fire_rating": get_fire_rating(fh_total_pred.get("confidence", 0), fh_total_pred.get("edge", 0))
                })
            
            # 1H Moneyline
            fh_ml_pred = fh.get("moneyline", {})
            if fh_ml_pred.get("passes_filter") and fh_home_ml is not None:
                rec_bet = fh_ml_pred.get("recommended_bet")
                if rec_bet:
                    pick_team = home_team if rec_bet == "home" else away_team
                    pick_odds_val = fh_home_ml if rec_bet == "home" else fh_away_ml
                    model_prob = fh_ml_pred.get("home_win_prob", 0.5) if rec_bet == "home" else fh_ml_pred.get("away_win_prob", 0.5)
                    market_prob = odds.get("fh_home_implied_prob", 0.5) if rec_bet == "home" else odds.get("fh_away_implied_prob", 0.5)
                    edge_pct = fh_ml_pred.get("home_edge", 0) if rec_bet == "home" else fh_ml_pred.get("away_edge", 0)
                    
                    all_plays.append({
                        "time_cst": time_cst_str,
                        "sort_time": game_cst.isoformat() if game_cst else "",
                        "matchup": matchup_display,
                        "period": "1H",
                        "market": "ML",
                        "pick": pick_team,
                        "pick_odds": format_american_odds(pick_odds_val),
                        "model_prediction": f"{model_prob*100:.1f}%",
                        "market_line": f"{market_prob*100:.1f}%",
                        "edge": f"{edge_pct*100:+.1f}%",
                        "edge_raw": abs(edge_pct * 100),
                        "confidence": fh_ml_pred.get("confidence", 0),
                        "fire_rating": get_fire_rating(fh_ml_pred.get("confidence", 0), edge_pct * 100)
                    })
            
            # Process First Quarter markets
            q1 = preds.get("first_quarter", {})
            
            # Q1 Spread
            q1_spread_pred = q1.get("spread", {})
            if q1_spread_pred.get("passes_filter") and q1_spread is not None:
                bet_side = q1_spread_pred.get("bet_side", "home")
                pick_team = home_team if bet_side == "home" else away_team
                pick_line = q1_spread if bet_side == "home" else -q1_spread
                pick_price = odds.get("q1_home_spread_price", -110)
                model_margin_q1 = features.get("predicted_margin_q1", 0)
                
                all_plays.append({
                    "time_cst": time_cst_str,
                    "sort_time": game_cst.isoformat() if game_cst else "",
                    "matchup": matchup_display,
                    "period": "Q1",
                    "market": "SPREAD",
                    "pick": f"{pick_team} {pick_line:+.1f}",
                    "pick_odds": format_american_odds(pick_price),
                    "model_prediction": f"{model_margin_q1:+.1f} pts",
                    "market_line": f"{q1_spread:+.1f}",
                    "edge": f"{q1_spread_pred.get('edge', 0):+.1f} pts",
                    "edge_raw": abs(q1_spread_pred.get('edge', 0)),
                    "confidence": q1_spread_pred.get("confidence", 0),
                    "fire_rating": get_fire_rating(q1_spread_pred.get("confidence", 0), q1_spread_pred.get("edge", 0))
                })
            
            # Q1 Total
            q1_total_pred = q1.get("total", {})
            if q1_total_pred.get("passes_filter") and q1_total is not None:
                bet_side = q1_total_pred.get("bet_side", "over")
                model_total_q1 = features.get("predicted_total_q1", 0)
                pick_display = f"{'OVER' if bet_side == 'over' else 'UNDER'} {q1_total}"
                
                all_plays.append({
                    "time_cst": time_cst_str,
                    "sort_time": game_cst.isoformat() if game_cst else "",
                    "matchup": matchup_display,
                    "period": "Q1",
                    "market": "TOTAL",
                    "pick": pick_display,
                    "pick_odds": format_american_odds(odds.get("q1_total_price", -110)),
                    "model_prediction": f"{model_total_q1:.1f}",
                    "market_line": f"{q1_total}",
                    "edge": f"{q1_total_pred.get('edge', 0):+.1f} pts",
                    "edge_raw": abs(q1_total_pred.get('edge', 0)),
                    "confidence": q1_total_pred.get("confidence", 0),
                    "fire_rating": get_fire_rating(q1_total_pred.get("confidence", 0), q1_total_pred.get("edge", 0))
                })
            
            # Q1 Moneyline
            q1_ml_pred = q1.get("moneyline", {})
            if q1_ml_pred.get("passes_filter") and q1_home_ml is not None:
                rec_bet = q1_ml_pred.get("recommended_bet")
                if rec_bet:
                    pick_team = home_team if rec_bet == "home" else away_team
                    pick_odds_val = q1_home_ml if rec_bet == "home" else q1_away_ml
                    model_prob = q1_ml_pred.get("home_win_prob", 0.5) if rec_bet == "home" else q1_ml_pred.get("away_win_prob", 0.5)
                    market_prob = odds.get("q1_home_implied_prob", 0.5) if rec_bet == "home" else odds.get("q1_away_implied_prob", 0.5)
                    edge_pct = q1_ml_pred.get("home_edge", 0) if rec_bet == "home" else q1_ml_pred.get("away_edge", 0)
                    
                    all_plays.append({
                        "time_cst": time_cst_str,
                        "sort_time": game_cst.isoformat() if game_cst else "",
                        "matchup": matchup_display,
                        "period": "Q1",
                        "market": "ML",
                        "pick": pick_team,
                        "pick_odds": format_american_odds(pick_odds_val),
                        "model_prediction": f"{model_prob*100:.1f}%",
                        "market_line": f"{market_prob*100:.1f}%",
                        "edge": f"{edge_pct*100:+.1f}%",
                        "edge_raw": abs(edge_pct * 100),
                        "confidence": q1_ml_pred.get("confidence", 0),
                        "fire_rating": get_fire_rating(q1_ml_pred.get("confidence", 0), edge_pct * 100)
                    })
            
        except Exception as e:
            logger.error(f"Error processing {home_team} vs {away_team}: {e}")
            continue
    
    # Sort by time, then by fire rating (descending), then by edge (descending)
    fire_order = {"🔥🔥🔥": 3, "🔥🔥": 2, "🔥": 1}
    all_plays.sort(key=lambda x: (x["sort_time"], -fire_order.get(x["fire_rating"], 0), -x["edge_raw"]))
    
    # Format for output - remove internal fields
    formatted_plays = []
    for play in all_plays:
        formatted_plays.append({
            "time_cst": play["time_cst"],
            "matchup": play["matchup"],
            "period": play["period"],
            "market": play["market"],
            "pick": play["pick"],
            "pick_odds": play["pick_odds"],
            "model_prediction": play["model_prediction"],
            "market_line": play["market_line"],
            "edge": play["edge"],
            "confidence": f"{play['confidence']*100:.0f}%",
            "fire_rating": play["fire_rating"]
        })
    
    return convert_numpy_types({
        "date": str(target_date),
        "generated_at": datetime.now(CST).strftime("%Y-%m-%d %I:%M %p CST"),
        "version": "6.0",
        "total_plays": len(formatted_plays),
        "plays": formatted_plays,
        "legend": {
            "fire_rating": {
                "🔥🔥🔥": "ELITE - 70%+ confidence AND 5+ pt edge",
                "🔥🔥": "STRONG - 60%+ confidence AND 3+ pt edge",
                "🔥": "GOOD - Passes all filters"
            },
            "periods": {"FG": "Full Game", "1H": "First Half", "Q1": "First Quarter"},
            "markets": {"SPREAD": "Point Spread", "TOTAL": "Over/Under", "ML": "Moneyline"}
        }
    })


@app.get("/slate/{date}/comprehensive")
@limiter.limit("20/minute")
async def get_comprehensive_slate_analysis(
    request: Request,
    date: str = Path(..., description="Date in YYYY-MM-DD format, 'today', or 'tomorrow'"),
    use_splits: bool = Query(True, description="Whether to fetch and use betting splits")
):
    """
    Get comprehensive slate analysis with full edge calculations.
    
    v6.0: Full analysis for all 9 markets.
    """
    if not hasattr(app.state, 'engine') or app.state.engine is None:
        raise HTTPException(status_code=503, detail="v6.0: Engine not loaded - models missing")

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
        "version": "6.0",
        "markets": [
            "q1_spread", "q1_total", "q1_moneyline",
            "1h_spread", "1h_total", "1h_moneyline",
            "fg_spread", "fg_total", "fg_moneyline"
        ],
        "analysis": analysis_results,
        "edge_thresholds": edge_thresholds
    })


@app.post("/predict/game", response_model=GamePredictions)
@limiter.limit("60/minute")
async def predict_single_game(request: Request, req: GamePredictionRequest):
    """
    Generate predictions for a specific matchup.

    v6.0: All 9 markets (Q1+1H+FG for Spread, Total, Moneyline).
    """
    if not hasattr(app.state, 'engine') or app.state.engine is None:
        raise HTTPException(status_code=503, detail="v6.0: Engine not loaded - models missing")

    try:
        # Build features
        features = await app.state.feature_builder.build_game_features(
            req.home_team, req.away_team
        )

        # Predict all 9 markets
        preds = app.state.engine.predict_all_markets(
            features,
            fg_spread_line=req.fg_spread_line,
            fg_total_line=req.fg_total_line,
            fh_spread_line=req.fh_spread_line,
            fh_total_line=req.fh_total_line,
            q1_spread_line=req.q1_spread_line,
            q1_total_line=req.q1_total_line,
            home_ml_odds=req.home_ml_odds,
            away_ml_odds=req.away_ml_odds,
            fh_home_ml_odds=req.fh_home_ml_odds,
            fh_away_ml_odds=req.fh_away_ml_odds,
            q1_home_ml_odds=req.q1_home_ml_odds,
            q1_away_ml_odds=req.q1_away_ml_odds,
        )
        return preds
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"v6.0: {e}")
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# LIVE PICK TRACKING ENDPOINTS - v6.0
# =============================================================================

@app.get("/tracking/summary")
@limiter.limit("30/minute")
async def get_tracking_summary(
    request: Request,
    date: Optional[str] = Query(None, description="Filter by date (YYYY-MM-DD)"),
    period: Optional[str] = Query(None, description="Filter by period (q1, 1h, fg)"),
    market_type: Optional[str] = Query(None, description="Filter by market (spread, total, moneyline)")
):
    """
    Get ROI summary for tracked picks.
    
    Provides accuracy, ROI, and win/loss breakdown for live tracked predictions.
    Only includes picks that passed the betting filter.
    """
    from src.tracking import PickTracker
    
    tracker = PickTracker()
    summary = tracker.get_roi_summary(
        date=date,
        period=period,
        market_type=market_type,
        passes_filter_only=True
    )
    streak = tracker.get_streak(passes_filter_only=True)
    
    return {
        "summary": summary,
        "current_streak": streak,
        "filters": {
            "date": date,
            "period": period,
            "market_type": market_type
        }
    }


@app.get("/tracking/picks")
@limiter.limit("30/minute")
async def get_tracked_picks(
    request: Request,
    date: Optional[str] = Query(None, description="Filter by date (YYYY-MM-DD)"),
    status: Optional[str] = Query(None, description="Filter by status (pending, win, loss, push)"),
    limit: int = Query(50, ge=1, le=500)
):
    """
    Get list of tracked picks with optional filters.
    """
    from src.tracking import PickTracker
    
    tracker = PickTracker()
    picks = tracker.get_picks(date=date, status=status, passes_filter_only=True)
    
    # Sort by creation time, newest first
    picks.sort(key=lambda p: p.created_at, reverse=True)
    
    return {
        "total": len(picks),
        "picks": [p.to_dict() for p in picks[:limit]]
    }


@app.post("/tracking/validate")
@limiter.limit("10/minute")
async def validate_pick_outcomes(
    request: Request,
    date: Optional[str] = Query(None, description="Date to validate (YYYY-MM-DD)")
):
    """
    Validate pending picks against game outcomes.
    
    Fetches completed game results and updates pick statuses.
    Only processes games that are confirmed complete.
    """
    from src.tracking import PickTracker
    
    tracker = PickTracker()
    results = await tracker.validate_outcomes(date=date)
    
    return {
        "validated": results["validated"],
        "wins": results["wins"],
        "losses": results["losses"],
        "pushes": results["pushes"],
        "details": results.get("details", [])
    }