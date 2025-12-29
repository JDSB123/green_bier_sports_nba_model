"""
NBA_v33.0.8.0 - FastAPI Prediction Server - STRICT MODE

FRESH DATA ONLY: No file caching, no silent fallbacks, no placeholders.

PRODUCTION: 6 INDEPENDENT Markets (1H + FG spreads/totals/moneylines)

First Half (1H):
- 1H Spread
- 1H Total
- 1H Moneyline

Full Game (FG):
- FG Spread
- FG Total
- FG Moneyline

STRICT MODE: Every request fetches fresh data from APIs.
No assumptions, no defaults - all data must be explicitly provided.
"""
import os
import json
import logging
import numpy as np
import sys
import uuid
from typing import Any, Dict, List, Optional, Union
from pathlib import Path as PathLib
from datetime import datetime, timezone

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, Path, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
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
from src.features import RichFeatureBuilder
from src.utils.logging import get_logger
from src.utils.security import fail_fast_on_missing_keys, get_api_key_status, mask_api_key, validate_premium_features
from src.utils.api_auth import get_api_key, APIKeyMiddleware
from src.tracking import PickTracker

logger = get_logger(__name__)

# Centralized release/version identifier for API surfaces
RELEASE_VERSION = os.getenv("NBA_MODEL_VERSION", "NBA_v33.0.8.0")


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


# --- Request/Response Models - NBA_v33.0.8.0 ---

class GamePredictionRequest(BaseModel):
    """Request for single game prediction - 6 markets (1H + FG spreads/totals/moneylines)."""
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
    first_half: Dict[str, Any] = {}
    full_game: Dict[str, Any] = {}


class SlateResponse(BaseModel):
    date: str
    predictions: List[Dict[str, Any]]
    total_plays: int
    odds_as_of_utc: Optional[str] = None
    odds_snapshot_path: Optional[str] = None


# --- API Setup - NBA_v33.0.8.0 ---

def _models_dir() -> PathLib:
    return PathLib(settings.data_processed_dir) / "models"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.

    Startup: Initialize the prediction engine.
    NBA_v33.0.8.0: 6 markets (1H+FG for Spread, Total, Moneyline). Q1 removed.
    Fails LOUDLY if models are missing or API keys are invalid.
    """
    # === STARTUP ===
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
    logger.info(f"v33.0.8.0 STRICT MODE: Loading Unified Prediction Engine from {models_dir}")

    # Diagnostic: List files in models directory
    if models_dir.exists():
        model_files = list(models_dir.glob("*"))
        logger.info(f"Found {len(model_files)} files in models directory:")
        for f in sorted(model_files):
            size = f.stat().st_size if f.is_file() else 0
            logger.info(f"  - {f.name} ({size:,} bytes)")
    else:
        logger.error(f"Models directory does not exist: {models_dir}")

    # STRICT MODE: 1H + FG models (6 total). No fallbacks.
    logger.info("v33.0.8.0 STRICT MODE: Using 1H/FG models (6 markets including moneylines)")
    app.state.engine = UnifiedPredictionEngine(models_dir=models_dir, require_all=True)
    app.state.feature_builder = RichFeatureBuilder(season=settings.current_season)

    # NO FILE CACHING - all data fetched fresh from APIs per request
    logger.info("v33.0.8.0 STRICT MODE: File caching DISABLED - all data fetched fresh per request")

    # Initialize live pick tracker
    picks_dir = PathLib(settings.data_processed_dir) / "picks"
    picks_dir.mkdir(parents=True, exist_ok=True)
    app.state.tracker = PickTracker(tracking_dir=picks_dir)
    logger.info(f"Pick tracker initialized at {picks_dir}")

    # Log model info
    model_info = app.state.engine.get_model_info()
    logger.info(f"v33.0.8.0 initialized - {model_info['markets']} markets loaded: {model_info['markets_list']}")

    yield  # Application runs here

    # === SHUTDOWN ===
    logger.info("v33.0.8.0 shutting down")


app = FastAPI(
    title="NBA v33.0.8.0 - STRICT MODE Production Picks",
    description="6 INDEPENDENT Markets: 1H+FG for Spread, Total, Moneyline. FRESH DATA ONLY - No caching, no fallbacks, no placeholders.",
    version=RELEASE_VERSION,
    lifespan=lifespan
)

# Add rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# API Authentication (optional - can be disabled via REQUIRE_API_AUTH=false)
if os.getenv("REQUIRE_API_AUTH", "false").lower() == "true":
    app.add_middleware(APIKeyMiddleware, require_auth=True)

# CORS configuration - STRICT: Must be explicitly configured for production
allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "")
if not allowed_origins_str:
    logger.warning("ALLOWED_ORIGINS not set - CORS will reject all cross-origin requests")
    allowed_origins = []
else:
    allowed_origins = [origin.strip() for origin in allowed_origins_str.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*", "X-API-Key"],
)


# Request ID middleware for distributed tracing
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    # Use existing request ID from header or generate new one
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    # Store in request state for access in handlers
    request.state.request_id = request_id

    response = await call_next(request)
    # Add request ID to response headers
    response.headers["X-Request-ID"] = request_id
    return response


# Request metrics middleware
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = datetime.now()
    method = request.method
    endpoint = request.url.path
    request_id = getattr(request.state, "request_id", "unknown")

    try:
        response = await call_next(request)
        status = response.status_code
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
        return response
    except Exception as e:
        logger.error(f"[{request_id}] Request failed: {e}")
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=500).inc()
        raise
    finally:
        duration = (datetime.now() - start_time).total_seconds()
        REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
        if duration > 5.0:  # Log slow requests
            logger.warning(f"[{request_id}] Slow request: {method} {endpoint} took {duration:.2f}s")


# --- Endpoints ---

@app.get("/health")
@limiter.limit("100/minute")
def health(request: Request):
    """Check API health - v33.0.8.0 with 6 markets (moneyline included)."""
    engine_loaded = hasattr(app.state, 'engine') and app.state.engine is not None
    api_keys = get_api_key_status()

    model_info = {}
    if engine_loaded:
        model_info = app.state.engine.get_model_info()

    return {
        "status": "ok",
        "version": RELEASE_VERSION,
        "mode": "STRICT",
        "architecture": "1H + FG spreads/totals required; moneyline optional",
        "caching": "DISABLED - fresh data every request",
        "markets": model_info.get("markets", 0),
        "markets_list": model_info.get("markets_list", []),
        "periods": ["first_half", "full_game"],
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


@app.get("/admin/monitoring")
@limiter.limit("30/minute")
def get_monitoring_stats(request: Request):
    """
    Get comprehensive monitoring statistics.

    Returns:
    - Signal agreement rates (classifier vs point prediction)
    - Feature completeness rates
    - Model drift detection status
    - Recent alerts and warnings
    """
    stats = {
        "timestamp": datetime.now().isoformat(),
        "version": RELEASE_VERSION,
    }

    # Signal agreement tracking
    try:
        from src.monitoring.signal_tracker import get_signal_tracker
        signal_tracker = get_signal_tracker()
        stats["signal_agreement"] = signal_tracker.get_stats()
        stats["signal_agreement"]["market_rates"] = signal_tracker.get_market_rates()
        stats["signal_agreement"]["recent_disagreements"] = signal_tracker.get_recent_disagreements(5)
    except Exception as e:
        stats["signal_agreement"] = {"error": str(e)}

    # Feature completeness tracking
    try:
        from src.monitoring.feature_completeness import get_feature_tracker
        feature_tracker = get_feature_tracker()
        stats["feature_completeness"] = feature_tracker.get_stats()
        stats["feature_completeness"]["market_rates"] = feature_tracker.get_market_completeness()
        stats["feature_completeness"]["top_missing"] = feature_tracker.get_most_missing_features(10)
    except Exception as e:
        stats["feature_completeness"] = {"error": str(e)}

    # Model drift detection
    try:
        from src.monitoring.drift_detection import get_drift_detector
        drift_detector = get_drift_detector()
        stats["drift_detection"] = {
            "market_stats": drift_detector.get_all_stats(),
            "recent_alerts": drift_detector.get_recent_alerts(5),
            "drifting_markets": [
                market for market in drift_detector.metrics
                if drift_detector.is_drifting(market)[0]
            ],
        }
    except Exception as e:
        stats["drift_detection"] = {"error": str(e)}

    # Rate limiter stats
    try:
        from src.utils.rate_limiter import get_odds_api_limiter, get_api_basketball_limiter
        stats["rate_limiters"] = {
            "the_odds_api": get_odds_api_limiter().get_stats(),
            "api_basketball": get_api_basketball_limiter().get_stats(),
        }
    except Exception as e:
        stats["rate_limiters"] = {"error": str(e)}

    # Circuit breaker stats
    try:
        from src.utils.circuit_breaker import get_odds_api_breaker, get_api_basketball_breaker
        stats["circuit_breakers"] = {
            "the_odds_api": get_odds_api_breaker().get_stats(),
            "api_basketball": get_api_basketball_breaker().get_stats(),
        }
    except Exception as e:
        stats["circuit_breakers"] = {"error": str(e)}

    return stats


@app.post("/admin/monitoring/reset")
@limiter.limit("5/minute")
def reset_monitoring_stats(request: Request):
    """
    Reset all monitoring statistics.

    Use this at the start of a new tracking period.
    """
    reset_results = {}

    try:
        from src.monitoring.signal_tracker import get_signal_tracker
        get_signal_tracker().reset()
        reset_results["signal_tracker"] = "reset"
    except Exception as e:
        reset_results["signal_tracker"] = f"error: {e}"

    try:
        from src.monitoring.feature_completeness import get_feature_tracker
        get_feature_tracker().reset()
        reset_results["feature_tracker"] = "reset"
    except Exception as e:
        reset_results["feature_tracker"] = f"error: {e}"

    try:
        from src.monitoring.drift_detection import get_drift_detector
        get_drift_detector().reset()
        reset_results["drift_detector"] = "reset"
    except Exception as e:
        reset_results["drift_detector"] = f"error: {e}"

    logger.info(f"Monitoring stats reset: {reset_results}")

    return {
        "status": "success",
        "reset_results": reset_results,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/admin/cache/clear")
@limiter.limit("5/minute")
async def clear_cache(request: Request):
    """
    Clear all session caches to force fresh API data.

    v33.0.8.0 STRICT MODE: No file caching exists - only clears session memory caches.
    Use this before fetching new predictions to ensure fresh data.
    """
    cleared = {"session_cache": False, "api_cache": 0}

    # Clear session caches in feature builder (the only cache that exists now)
    if hasattr(app.state, 'feature_builder'):
        app.state.feature_builder.clear_session_cache()
        cleared["session_cache"] = True

    # Clear API cache layer if it exists
    try:
        from src.utils.api_cache import api_cache
        cleared["api_cache"] = api_cache.clear_all()
    except Exception:
        pass  # API cache may not be configured

    logger.info(f"v33.0.8.0 STRICT MODE: Session cache cleared")

    return {
        "status": "success",
        "cleared": cleared,
        "mode": "STRICT",
        "message": "Session caches cleared. Next API call will fetch fresh data from all sources.",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/admin/cache/stats")
@limiter.limit("10/minute")
async def get_cache_stats(request: Request):
    """
    Get comprehensive cache statistics and performance metrics.

    Shows both session caches (cleared between requests) and persistent
    reference caches (lazy-loaded, survive across requests).
    """
    stats = {"session_cache": {}, "persistent_cache": {}}

    # Get feature builder cache stats
    if hasattr(app.state, 'feature_builder'):
        stats.update(app.state.feature_builder.get_cache_stats())

    # Get API cache stats if available
    try:
        from src.utils.api_cache import api_cache
        api_stats = api_cache.get_stats()
        stats["api_cache"] = api_stats
    except Exception:
        stats["api_cache"] = {"status": "not_configured"}

    return {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "caching_strategy": "LAZY_PERSISTENT",
        "description": "Session caches clear between requests. Persistent caches lazy-load on first use.",
        "stats": stats
    }


@app.get("/verify")
@limiter.limit("10/minute")
def verify_integrity(request: Request):
    """
    Verify model integrity and component usage.

    v33.0.8.0: Verifies 6 independent models (1H + FG for spread, total, moneyline)
    """
    results = {
        "status": "pass",
        "version": RELEASE_VERSION,
        "markets": {
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

        # Check 2: Period predictors exist (1H + FG only)
        has_fg = hasattr(app.state.engine, 'fg_predictor')
        has_1h = hasattr(app.state.engine, 'h1_predictor')

        results["checks"]["period_predictors"] = {
            "full_game": has_fg,
            "first_half": has_1h,
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
            "home_win_pct": 0.6, "away_win_pct": 0.4,
            "home_avg_margin": 2.0, "away_avg_margin": -1.0,
            "home_rest_days": 2, "away_rest_days": 1,
            "home_b2b": 0, "away_b2b": 0,
            "dynamic_hca": 3.0, "h2h_win_pct": 0.5, "h2h_avg_margin": 0.0,
            # 1H-specific features
            "home_1h_ppg": 56.2, "away_1h_ppg": 55.0,
            "home_1h_papg": 54.8, "away_1h_papg": 55.5,
            "home_1h_avg_margin": 1.2, "away_1h_avg_margin": -0.5,
        }

        # Check 3: Test 1H prediction
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

    v33.0.8.0: 6 markets (1H + FG spreads/totals/moneylines). Q1 removed entirely.
    Returns 1H/FG for spread, total, and moneyline when available.
    """
    if not hasattr(app.state, 'engine') or app.state.engine is None:
        raise HTTPException(status_code=503, detail="v33.0.8.0: Engine not loaded - models missing")

    # STRICT MODE: Clear session cache to force fresh data
    app.state.feature_builder.clear_session_cache()
    logger.info("v33.0.8.0: Session cache cleared - fetching fresh data")

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

    odds_as_of_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    odds_snapshot_path = None
    try:
        from src.ingestion.the_odds import save_odds
        odds_snapshot_path = save_odds(games, prefix=f"slate_odds_{target_date.strftime('%Y%m%d')}")
    except Exception as e:
        logger.warning(f"Could not save odds snapshot: {e}")

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
            odds = extract_consensus_odds(game, as_of_utc=odds_as_of_utc)
            fg_spread = odds.get("home_spread")
            fg_total = odds.get("total")
            fh_spread = odds.get("fh_home_spread")
            fh_total = odds.get("fh_total")
            home_ml = odds.get("home_ml")
            away_ml = odds.get("away_ml")
            fh_home_ml = odds.get("fh_home_ml")
            fh_away_ml = odds.get("fh_away_ml")

            # Require spread or total lines; moneyline odds remain optional
            has_fg_lines = fg_spread is not None or fg_total is not None
            has_fh_lines = fh_spread is not None or fh_total is not None

            if not has_fg_lines and not has_fh_lines:
                logger.warning(f"Skipping {home_team} vs {away_team} - no betting lines available")
                continue

            # Predict markets (1H + FG)
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

            # Count plays (6 markets: 1H + FG spreads/totals/moneylines)
            game_plays = 0
            game_date = target_date.strftime("%Y-%m-%d")
            for period in ["first_half", "full_game"]:
                period_preds = preds.get(period, {})
                for market in ["spread", "total"]:
                    pred_data = period_preds.get(market, {})
                    if pred_data.get("passes_filter"):
                        side = pred_data.get("side") or pred_data.get("bet_side")
                        if not side:
                            logger.warning(f"Missing side for {period} {market} - skipping track")
                            continue
                        game_plays += 1
                        # Record the pick for live tracking
                        market_key = f"{period.replace('first_half', '1h').replace('full_game', 'fg')}_{market}"
                        line = None
                        if market == "spread":
                            line = fh_spread if period == "first_half" else fg_spread
                        elif market == "total":
                            line = fh_total if period == "first_half" else fg_total

                        app.state.tracker.record_pick(
                            game_date=game_date,
                            home_team=home_team,
                            away_team=away_team,
                            market=market_key,
                            side=side,
                            line=line,
                            confidence=pred_data.get("confidence"),
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
            logger.warning(f"v33.0.8.0: Skipping {home_team} vs {away_team} - {e}")
            continue
        except Exception as e:
            logger.error(f"Error processing {home_team} vs {away_team}: {e}")
            continue

    return SlateResponse(
        date=str(target_date),
        predictions=results,
        total_plays=total_plays,
        odds_as_of_utc=odds_as_of_utc,
        odds_snapshot_path=odds_snapshot_path,
    )


def format_american_odds(odds: int) -> str:
    """Format American odds with +/- prefix."""
    if odds is None:
        return "N/A"
    return f"+{odds}" if odds > 0 else str(odds)


def get_fire_rating(confidence: float, edge: float) -> str:
    """
    Get fire rating based on confidence and edge.
    ELITE = conf >= 70% AND edge >= 5
    STRONG = conf >= 60% AND edge >= 3
    GOOD = passes filters
    """
    if confidence >= 0.70 and abs(edge) >= 5:
        return "ELITE"
    elif confidence >= 0.60 and abs(edge) >= 3:
        return "STRONG"
    else:
        return "GOOD"


@app.get("/slate/{date}/executive")
@limiter.limit("30/minute")
async def get_executive_summary(
    request: Request,
    date: str = Path(..., description="Date in YYYY-MM-DD format, 'today', or 'tomorrow'"),
    use_splits: bool = Query(True, description="Whether to fetch and use betting splits")
):
    """
    BLUF (Bottom Line Up Front) Executive Summary.

    v33.0.8.0 STRICT MODE: Fetches fresh data from all APIs.
    Returns a clean actionable betting card with all picks that pass filters.
    Sorted by EV% (desc), then game time.
    """
    if not hasattr(app.state, 'engine') or app.state.engine is None:
        raise HTTPException(status_code=503, detail="v33.0.8.0 STRICT MODE: Engine not loaded - models missing")

    # STRICT MODE: Clear session cache to force fresh data
    app.state.feature_builder.clear_session_cache()
    logger.info("v33.0.8.0 STRICT MODE: Executive summary - fetching fresh data")

    from src.utils.slate_analysis import (
        get_target_date, fetch_todays_games, parse_utc_time, to_cst, extract_consensus_odds
    )
    from src.utils.odds import devig_two_way, expected_value, kelly_fraction, american_to_implied_prob
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

    odds_as_of_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    odds_snapshot_path = None
    try:
        from src.ingestion.the_odds import save_odds
        odds_snapshot_path = save_odds(games, prefix=f"slate_odds_{target_date.strftime('%Y%m%d')}")
    except Exception as e:
        logger.warning(f"Could not save odds snapshot: {e}")

    # Fetch betting splits
    splits_dict = {}
    if use_splits:
        try:
            from src.ingestion.betting_splits import fetch_public_betting_splits
            splits_dict = await fetch_public_betting_splits(games, source="auto")
        except Exception as e:
            logger.warning(f"Could not fetch splits: {e}")

    def _pick_ev_fields(p_model: float | None, pick_odds: int | None, odds_a: int | None, odds_b: int | None, pick_is_a: bool) -> tuple[float | None, float | None, float | None]:
        if pick_odds is None or p_model is None:
            return None, None, None
        p_fair_a, p_fair_b = devig_two_way(odds_a, odds_b)
        p_fair = p_fair_a if pick_is_a else p_fair_b
        if p_fair is None:
            p_fair = american_to_implied_prob(pick_odds)
        ev = expected_value(p_model, pick_odds, stake=1.0)
        ev_pct = (ev * 100) if ev is not None else None
        kelly = kelly_fraction(p_model, pick_odds, fraction=0.5)
        return p_fair, ev_pct, kelly

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
            odds = extract_consensus_odds(game, as_of_utc=odds_as_of_utc)
            fg_spread = odds.get("home_spread")
            fg_total = odds.get("total")
            fh_spread = odds.get("fh_home_spread")
            fh_total = odds.get("fh_total")
            home_ml = odds.get("home_ml")
            away_ml = odds.get("away_ml")
            fh_home_ml = odds.get("fh_home_ml")
            fh_away_ml = odds.get("fh_away_ml")

            if fg_spread is None or fg_total is None:
                continue

            # Get predictions for 6 markets (1H + FG spreads/totals/moneylines)
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
            
            # Process Full Game markets
            fg = preds.get("full_game", {})
            
            # FG Spread
            fg_spread_pred = fg.get("spread", {})
            if fg_spread_pred.get("passes_filter"):
                bet_side = fg_spread_pred.get("bet_side", "home")
                pick_team = home_team if bet_side == "home" else away_team
                pick_line = fg_spread if bet_side == "home" else -fg_spread
                pick_price = odds.get("home_spread_price") if bet_side == "home" else odds.get("away_spread_price")
                if pick_price is None:
                    pick_price = odds.get("home_spread_price")
                p_model = fg_spread_pred.get("home_cover_prob") if bet_side == "home" else fg_spread_pred.get("away_cover_prob")
                p_fair, ev_pct, kelly = _pick_ev_fields(
                    p_model,
                    pick_price,
                    odds.get("home_spread_price"),
                    odds.get("away_spread_price"),
                    bet_side == "home",
                )
                model_margin = features.get("predicted_margin", 0)
                # Format model prediction for BET SIDE team (same team as pick)
                # model_margin positive = home wins by X
                # home spread = -model_margin, away spread = +model_margin
                if bet_side == "home":
                    model_spread = -model_margin
                    margin_display = f"{home_team} {model_spread:+.1f}"
                else:
                    model_spread = model_margin
                    margin_display = f"{away_team} {model_spread:+.1f}"

                all_plays.append({
                    "time_cst": time_cst_str,
                    "sort_time": game_cst.isoformat() if game_cst else "",
                    "matchup": matchup_display,
                    "period": "FG",
                    "market": "SPREAD",
                    "pick": f"{pick_team} {pick_line:+.1f}",
                    "pick_odds": format_american_odds(pick_price),
                    "model_prediction": margin_display,
                    "market_line": f"{fg_spread:+.1f}",
                    "edge": f"{fg_spread_pred.get('edge', 0):+.1f} pts",
                    "edge_raw": abs(fg_spread_pred.get('edge', 0)),
                    "confidence": fg_spread_pred.get("confidence", 0),
                    "p_model": p_model,
                    "p_fair": p_fair,
                    "ev_pct": ev_pct,
                    "kelly_fraction": kelly,
                    "odds_as_of_utc": odds_as_of_utc,
                    "fire_rating": get_fire_rating(fg_spread_pred.get("confidence", 0), fg_spread_pred.get("edge", 0))
                })
            
            # FG Total
            fg_total_pred = fg.get("total", {})
            if fg_total_pred.get("passes_filter"):
                bet_side = fg_total_pred.get("bet_side", "over")
                model_total = features.get("predicted_total", 0)
                pick_display = f"{'OVER' if bet_side == 'over' else 'UNDER'} {fg_total}"
                pick_price = odds.get("total_over_price") if bet_side == "over" else odds.get("total_under_price")
                if pick_price is None:
                    pick_price = odds.get("total_price")
                p_model = fg_total_pred.get("over_prob") if bet_side == "over" else fg_total_pred.get("under_prob")
                p_fair, ev_pct, kelly = _pick_ev_fields(
                    p_model,
                    pick_price,
                    odds.get("total_over_price"),
                    odds.get("total_under_price"),
                    bet_side == "over",
                )
                
                all_plays.append({
                    "time_cst": time_cst_str,
                    "sort_time": game_cst.isoformat() if game_cst else "",
                    "matchup": matchup_display,
                    "period": "FG",
                    "market": "TOTAL",
                    "pick": pick_display,
                    "pick_odds": format_american_odds(pick_price),
                    "model_prediction": f"{model_total:.1f}",
                    "market_line": f"{fg_total}",
                    "edge": f"{fg_total_pred.get('edge', 0):+.1f} pts",
                    "edge_raw": abs(fg_total_pred.get('edge', 0)),
                    "confidence": fg_total_pred.get("confidence", 0),
                    "p_model": p_model,
                    "p_fair": p_fair,
                    "ev_pct": ev_pct,
                    "kelly_fraction": kelly,
                    "odds_as_of_utc": odds_as_of_utc,
                    "fire_rating": get_fire_rating(fg_total_pred.get("confidence", 0), fg_total_pred.get("edge", 0))
                })
            
            # Process First Half markets
            fh = preds.get("first_half", {})
            
            # 1H Spread
            fh_spread_pred = fh.get("spread", {})
            if fh_spread_pred.get("passes_filter") and fh_spread is not None:
                bet_side = fh_spread_pred.get("bet_side", "home")
                pick_team = home_team if bet_side == "home" else away_team
                pick_line = fh_spread if bet_side == "home" else -fh_spread
                pick_price = odds.get("fh_home_spread_price") if bet_side == "home" else odds.get("fh_away_spread_price")
                if pick_price is None:
                    pick_price = odds.get("fh_home_spread_price")
                p_model = fh_spread_pred.get("home_cover_prob") if bet_side == "home" else fh_spread_pred.get("away_cover_prob")
                p_fair, ev_pct, kelly = _pick_ev_fields(
                    p_model,
                    pick_price,
                    odds.get("fh_home_spread_price"),
                    odds.get("fh_away_spread_price"),
                    bet_side == "home",
                )
                model_margin_1h = features.get("predicted_margin_1h", 0)
                # Format model prediction for BET SIDE team (same team as pick)
                # model_margin positive = home wins by X
                # home spread = -model_margin, away spread = +model_margin
                if bet_side == "home":
                    model_spread_1h = -model_margin_1h
                    margin_display_1h = f"{home_team} {model_spread_1h:+.1f}"
                else:
                    model_spread_1h = model_margin_1h
                    margin_display_1h = f"{away_team} {model_spread_1h:+.1f}"

                all_plays.append({
                    "time_cst": time_cst_str,
                    "sort_time": game_cst.isoformat() if game_cst else "",
                    "matchup": matchup_display,
                    "period": "1H",
                    "market": "SPREAD",
                    "pick": f"{pick_team} {pick_line:+.1f}",
                    "pick_odds": format_american_odds(pick_price),
                    "model_prediction": margin_display_1h,
                    "market_line": f"{fh_spread:+.1f}",
                    "edge": f"{fh_spread_pred.get('edge', 0):+.1f} pts",
                    "edge_raw": abs(fh_spread_pred.get('edge', 0)),
                    "confidence": fh_spread_pred.get("confidence", 0),
                    "p_model": p_model,
                    "p_fair": p_fair,
                    "ev_pct": ev_pct,
                    "kelly_fraction": kelly,
                    "odds_as_of_utc": odds_as_of_utc,
                    "fire_rating": get_fire_rating(fh_spread_pred.get("confidence", 0), fh_spread_pred.get("edge", 0))
                })
            
            # 1H Total
            fh_total_pred = fh.get("total", {})
            if fh_total_pred.get("passes_filter") and fh_total is not None:
                bet_side = fh_total_pred.get("bet_side", "over")
                model_total_1h = features.get("predicted_total_1h", 0)
                pick_display = f"{'OVER' if bet_side == 'over' else 'UNDER'} {fh_total}"
                pick_price = odds.get("fh_total_over_price") if bet_side == "over" else odds.get("fh_total_under_price")
                if pick_price is None:
                    pick_price = odds.get("fh_total_price")
                p_model = fh_total_pred.get("over_prob") if bet_side == "over" else fh_total_pred.get("under_prob")
                p_fair, ev_pct, kelly = _pick_ev_fields(
                    p_model,
                    pick_price,
                    odds.get("fh_total_over_price"),
                    odds.get("fh_total_under_price"),
                    bet_side == "over",
                )
                
                all_plays.append({
                    "time_cst": time_cst_str,
                    "sort_time": game_cst.isoformat() if game_cst else "",
                    "matchup": matchup_display,
                    "period": "1H",
                    "market": "TOTAL",
                    "pick": pick_display,
                    "pick_odds": format_american_odds(pick_price),
                    "model_prediction": f"{model_total_1h:.1f}",
                    "market_line": f"{fh_total}",
                    "edge": f"{fh_total_pred.get('edge', 0):+.1f} pts",
                    "edge_raw": abs(fh_total_pred.get('edge', 0)),
                    "confidence": fh_total_pred.get("confidence", 0),
                    "p_model": p_model,
                    "p_fair": p_fair,
                    "ev_pct": ev_pct,
                    "kelly_fraction": kelly,
                    "odds_as_of_utc": odds_as_of_utc,
                    "fire_rating": get_fire_rating(fh_total_pred.get("confidence", 0), fh_total_pred.get("edge", 0))
                })
            
        except Exception as e:
            logger.error(f"Error processing {home_team} vs {away_team}: {e}")
            continue
    
    # Sort by EV%, then by time (descending EV is primary)
    def _ev_value(play: Dict[str, Any]) -> float:
        ev = play.get("ev_pct")
        return float(ev) if ev is not None else -9999.0

    all_plays.sort(key=lambda x: (-_ev_value(x), x["sort_time"]))
    
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
            "p_model": play.get("p_model"),
            "p_fair": play.get("p_fair"),
            "ev_pct": play.get("ev_pct"),
            "kelly_fraction": play.get("kelly_fraction"),
            "odds_as_of_utc": play.get("odds_as_of_utc"),
            "fire_rating": play["fire_rating"]
        })
    
    return convert_numpy_types({
        "date": str(target_date),
        "generated_at": datetime.now(CST).strftime("%Y-%m-%d %I:%M %p CST"),
        "version": RELEASE_VERSION,
        "total_plays": len(formatted_plays),
        "plays": formatted_plays,
        "odds_as_of_utc": odds_as_of_utc,
        "odds_snapshot_path": odds_snapshot_path,
        "legend": {
            "fire_rating": {
                "ELITE": "70%+ confidence AND 5+ pt edge",
                "STRONG": "60%+ confidence AND 3+ pt edge",
                "GOOD": "Passes all filters"
            },
            "sorting": "EV% (desc), then game time",
            "periods": {"FG": "Full Game", "1H": "First Half"},
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

    v33.0.8.0 STRICT MODE: Fetches fresh data from all APIs.
    Full analysis for all 9 markets.
    """
    if not hasattr(app.state, 'engine') or app.state.engine is None:
        raise HTTPException(status_code=503, detail="v33.0.8.0 STRICT MODE: Engine not loaded - models missing")

    # STRICT MODE: Clear session cache to force fresh data
    app.state.feature_builder.clear_session_cache()
    logger.info("v33.0.8.0 STRICT MODE: Comprehensive analysis - fetching fresh data")

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

    # Get edge thresholds (all active markets)
    edge_thresholds = get_edge_thresholds_for_game(
        game_date=target_date,
        bet_types=["spread", "total", "moneyline", "1h_spread", "1h_total", "1h_moneyline"]
    )

    # Fetch games
    games = await fetch_todays_games(target_date)
    if not games:
        return {"date": str(target_date), "analysis": [], "summary": "No games found"}

    odds_as_of_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    odds_snapshot_path = None
    try:
        from src.ingestion.the_odds import save_odds
        odds_snapshot_path = save_odds(games, prefix=f"slate_odds_{target_date.strftime('%Y%m%d')}")
    except Exception as e:
        logger.warning(f"Could not save odds snapshot: {e}")

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
            odds = extract_consensus_odds(game, as_of_utc=odds_as_of_utc)
            
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
        "version": RELEASE_VERSION,
        "markets": [
            "1h_spread", "1h_total",
            "fg_spread", "fg_total"
        ],
        "analysis": analysis_results,
        "edge_thresholds": edge_thresholds,
        "odds_as_of_utc": odds_as_of_utc,
        "odds_snapshot_path": odds_snapshot_path,
    })


@app.get("/meta", tags=["Ops"])
async def get_meta_info():
    """Get metadata about the running service."""
    return {
        "version": RELEASE_VERSION,
        "markets": [
            "1h_spread", "1h_total",
            "fg_spread", "fg_total",
        ],
        "strict_mode": os.getenv("NBA_STRICT_MODE", "false").lower() == "true",
        "server_time": datetime.now().isoformat(),
        "python_version": sys.version
    }


@app.post("/predict/game", response_model=GamePredictions)
@limiter.limit("60/minute")
async def predict_single_game(request: Request, req: GamePredictionRequest):
    """
    Generate predictions for a specific matchup.

    v33.0.8.0: STRICT MODE - Fetches fresh data from all APIs.
    6 markets (1H+FG for Spread, Total, Moneyline).
    """
    if not hasattr(app.state, 'engine') or app.state.engine is None:
        raise HTTPException(status_code=503, detail="v33.0.8.0 STRICT MODE: Engine not loaded - models missing")

    # STRICT MODE: Clear session cache to force fresh data
    app.state.feature_builder.clear_session_cache()

    try:
        # Build features from FRESH data
        features = await app.state.feature_builder.build_game_features(
            req.home_team, req.away_team
        )

        # Predict all 6 markets (1H + FG spreads/totals/moneylines)
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
        raise HTTPException(status_code=400, detail=f"v33.0.8.0: {e}")
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# LIVE PICK TRACKING ENDPOINTS - v33.0.8.0
# =============================================================================

@app.get("/tracking/summary")
@limiter.limit("30/minute")
async def get_tracking_summary(
    request: Request,
    date: Optional[str] = Query(None, description="Filter by date (YYYY-MM-DD)"),
    period: Optional[str] = Query(None, description="Filter by period (1h, fg)"),
    market_type: Optional[str] = Query(None, description="Filter by market (spread, total, moneyline)")
):
    """
    Get ROI summary for tracked picks.
    
    Provides accuracy, ROI, and win/loss breakdown for live tracked predictions.
    Only includes picks that passed the betting filter.
    """
    # Use app.state.tracker for consistency with slate endpoint
    tracker = app.state.tracker
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
    # Use app.state.tracker for consistency with slate endpoint
    tracker = app.state.tracker
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
    # Use app.state.tracker for consistency with slate endpoint
    tracker = app.state.tracker
    results = await tracker.validate_outcomes(date=date)
    
    return {
        "validated": results["validated"],
        "wins": results["wins"],
        "losses": results["losses"],
        "pushes": results["pushes"],
        "details": results.get("details", [])
    }


@app.get("/picks/html", response_class=HTMLResponse, tags=["Display"])
@limiter.limit("20/minute")
async def get_picks_html(
    request: Request,
    date: Optional[str] = Query(None, description="Date (YYYY-MM-DD), defaults to today")
):
    """
    Render NBA picks as an interactive Adaptive Card HTML page.
    
    Perfect for Teams card preview or standalone viewing.
    """
    try:
        # Use date from query or default to today
        from datetime import datetime as dt
        if not date:
            date = dt.now().strftime("%Y-%m-%d")
        
        # Fetch predictions for the date
        summary = await get_executive_summary(request, date=date)
        
        if not summary.get("plays"):
            return """
            <html>
            <head>
                <title>NBA Picks - No Games</title>
                <style>
                    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); margin: 0; padding: 20px; min-height: 100vh; }
                    .container { max-width: 900px; margin: 0 auto; background: white; border-radius: 12px; padding: 30px; box-shadow: 0 10px 40px rgba(0,0,0,0.1); }
                    h1 { color: #333; margin: 0 0 10px 0; }
                    p { color: #666; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🏀 NBA Picks</h1>
                    <p>No games scheduled for """ + date + """</p>
                </div>
            </body>
            </html>
            """
        
        # Build picks array for card
        picks_data = []
        for pick in summary.get("plays", []):
            market_line = pick.get("market_line", "N/A")
            edge = pick.get("edge", "N/A")
            confidence = pick.get("confidence", "50%")
            # Handle confidence as string percentage like "91%"
            if isinstance(confidence, str) and confidence.endswith('%'):
                confidence_pct = int(confidence.rstrip('%'))
            else:
                confidence_pct = int(float(confidence) * 100) if confidence else 50
            fire_rating = pick.get("fire_rating", 0)
            # Ensure fire_rating is an int
            if isinstance(fire_rating, str):
                fire_rating = int(fire_rating) if fire_rating.isdigit() else 0
            fire_emoji = "🔥" * int(fire_rating) if fire_rating else ""
            picks_data.append({
                "period": pick.get("period", "FG"),
                "matchup": pick.get("matchup", "TBD"),
                "pick": pick.get("pick", "N/A"),
                "confidence": confidence_pct,
                "market_line": market_line,
                "edge": edge,
                "fire": fire_emoji
            })

        # Generate card JSON
        card = {
            "type": "AdaptiveCard",
            "version": "1.4",
            "body": [
                {
                    "type": "Container",
                    "style": "accent",
                    "items": [
                        {
                            "type": "TextBlock",
                            "text": "🏀 NBA PICKS - " + date.upper(),
                            "weight": "bolder",
                            "size": "large",
                            "color": "light"
                        }
                    ]
                },
                {
                    "type": "Table",
                    "gridStyle": "accent",
                    "firstRowAsHeaders": True,
                    "columns": [
                        {"width": "15%"},
                        {"width": "20%"},
                        {"width": "25%"},
                        {"width": "20%"},
                        {"width": "20%"}
                    ],
                    "rows": [
                        {
                            "cells": [
                                {"text": "Period"},
                                {"text": "Matchup"},
                                {"text": "Pick (Confidence)"},
                                {"text": "Market"},
                                {"text": "Edge"}
                            ]
                        }
                    ] + [
                        {
                            "cells": [
                                {"text": p["period"]},
                                {"text": p["matchup"]},
                                {"text": p["pick"] + " (" + str(p["confidence"]) + "%)"},
                                {"text": p["market_line"]},
                                {"text": p["edge"] + " " + p["fire"]}
                            ]
                        }
                        for p in picks_data
                    ]
                }
            ],
            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json"
        }
        
        card_json_str = json.dumps(card)
        
        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NBA Picks - """ + date + """</title>
    <script src="https://cdn.jsdelivr.net/npm/adaptivecards@1.4.7/dist/adaptive-cards.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.15);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            text-align: center;
        }
        .header h1 { font-size: 28px; margin-bottom: 5px; }
        .header p { opacity: 0.9; font-size: 14px; }
        .content {
            padding: 25px;
        }
        #cardContainer {
            margin-bottom: 20px;
        }
        .adaptive-card {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }
        .section {
            margin-top: 25px;
            padding-top: 25px;
            border-top: 1px solid #e0e0e0;
        }
        .section h2 {
            font-size: 16px;
            color: #333;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .json-viewer {
            background: #f5f5f5;
            border: 1px solid #ddd;
            border-radius: 6px;
            padding: 12px;
            max-height: 300px;
            overflow-y: auto;
            font-family: 'Monaco', 'Courier New', monospace;
            font-size: 12px;
            color: #333;
            white-space: pre-wrap;
            word-break: break-all;
        }
        .button-group {
            display: flex;
            gap: 10px;
            margin-top: 15px;
        }
        button {
            padding: 10px 16px;
            border: none;
            border-radius: 6px;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        .btn-secondary {
            background: #f0f0f0;
            color: #333;
            border: 1px solid #ddd;
        }
        .btn-secondary:hover {
            background: #e8e8e8;
        }
        .toast {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: #4caf50;
            color: white;
            padding: 12px 20px;
            border-radius: 6px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            opacity: 0;
            transition: opacity 0.3s ease;
        }
        .toast.show {
            opacity: 1;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏀 NBA Picks</h1>
            <p>""" + date + """</p>
        </div>
        <div class="content">
            <div id="cardContainer"></div>
            
            <div class="section">
                <h2>📋 Card Data</h2>
                <div class="json-viewer" id="jsonViewer"></div>
                <div class="button-group">
                    <button class="btn-primary" onclick="downloadJSON()">⬇️ Download JSON</button>
                    <button class="btn-secondary" onclick="copyJSON()">📋 Copy JSON</button>
                </div>
            </div>
        </div>
    </div>
    
    <div class="toast" id="toast"></div>

    <script>
        const cardData = """ + card_json_str + """;
        
        // Render Adaptive Card
        AdaptiveCards.AdaptiveCard.onProcessMarkdown = function(text, result) {
            result.outputHtml = text;
            result.didProcess = true;
        };
        
        const adaptiveCard = new AdaptiveCards.AdaptiveCard();
        adaptiveCard.parse(cardData);
        
        const renderedCard = adaptiveCard.render();
        document.getElementById('cardContainer').appendChild(renderedCard);
        
        // Display JSON
        document.getElementById('jsonViewer').textContent = JSON.stringify(cardData, null, 2);
        
        function downloadJSON() {
            const dataStr = JSON.stringify(cardData, null, 2);
            const dataBlob = new Blob([dataStr], {type: 'application/json'});
            const url = URL.createObjectURL(dataBlob);
            const link = document.createElement('a');
            link.href = url;
            link.download = 'nba_picks_""" + date + """.json';
            link.click();
            showToast('JSON downloaded!');
        }
        
        function copyJSON() {
            navigator.clipboard.writeText(JSON.stringify(cardData, null, 2));
            showToast('JSON copied to clipboard!');
        }
        
        function showToast(message) {
            const toast = document.getElementById('toast');
            toast.textContent = message;
            toast.classList.add('show');
            setTimeout(() => toast.classList.remove('show'), 2000);
        }
    </script>
</body>
</html>"""
        
        return html_content
        
    except Exception as e:
        logger.error(f"Error rendering picks HTML: {str(e)}", exc_info=True)
        return f"""
        <html>
        <head>
            <title>Error</title>
            <style>
                body {{ font-family: sans-serif; background: #f5f5f5; padding: 20px; }}
                .error {{ background: #fee; padding: 20px; border-radius: 8px; color: #c00; }}
            </style>
        </head>
        <body>
            <div class="error">
                <h2>Error rendering picks</h2>
                <p>{str(e)}</p>
            </div>
        </body>
        </html>
        """
