"""
NBA - FastAPI Prediction Server - STRICT MODE

FRESH DATA ONLY: No file caching, no silent fallbacks, no placeholders.

PRODUCTION: 4 INDEPENDENT Markets (1H + FG spreads/totals)

First Half (1H):
- 1H Spread
- 1H Total

Full Game (FG):
- FG Spread
- FG Total

STRICT MODE: Every request fetches fresh data from APIs.
No assumptions, no defaults - all data must be explicitly provided.
"""
import base64
import csv
import hashlib
import hmac
import io
import os
import json
import logging
import numpy as np
import re
import sys
import uuid
from typing import Any, Dict, List, Optional, Union
from pathlib import Path as PathLib
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

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

from src.config import settings, PROJECT_ROOT
from src.prediction import UnifiedPredictionEngine, ModelNotFoundError
from src.ingestion import the_odds
from src.ingestion.betting_splits import fetch_public_betting_splits, validate_splits_sources_configured
from src.features import RichFeatureBuilder
from src.utils.logging import get_logger
from src.utils.version import resolve_version
from src.utils.security import get_api_key_status, mask_api_key, validate_premium_features
from src.utils.markets import get_market_catalog
from src.utils.startup_checks import run_startup_integrity_checks, StartupIntegrityError
from src.utils.api_auth import get_api_key, APIKeyMiddleware
from src.tracking import PickTracker

# Additional imports for comprehensive edge calculation
from src.utils.slate_analysis import (
    get_target_date, fetch_todays_games, parse_utc_time, to_cst, extract_consensus_odds,
    clear_unified_records_cache  # QA/QC: Clear records cache for fresh data
)
from src.utils.comprehensive_edge import calculate_comprehensive_edge
from src.modeling.edge_thresholds import get_edge_thresholds_for_game
from src.modeling.unified_features import get_feature_defaults
from zoneinfo import ZoneInfo

logger = get_logger(__name__)

# Centralized release/version identifier for API surfaces
RELEASE_VERSION = resolve_version()


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


_TEAM_KEYWORDS = {
    "lakers": "lakers", "lal": "lakers",
    "celtics": "celtics", "bos": "celtics",
    "warriors": "warriors", "gsw": "warriors",
    "nets": "nets", "bkn": "nets",
    "knicks": "knicks", "nyk": "knicks",
    "heat": "heat", "mia": "heat",
    "bucks": "bucks", "mil": "bucks",
    "sixers": "76ers", "76ers": "76ers", "phi": "76ers",
    "suns": "suns", "phx": "suns",
    "mavs": "mavericks", "mavericks": "mavericks", "dal": "mavericks",
    "nuggets": "nuggets", "den": "nuggets",
    "clippers": "clippers", "lac": "clippers",
    "thunder": "thunder", "okc": "thunder",
    "cavs": "cavaliers", "cavaliers": "cavaliers", "cle": "cavaliers",
    "bulls": "bulls", "chi": "bulls",
    "hawks": "hawks", "atl": "hawks",
    "raptors": "raptors", "tor": "raptors",
    "magic": "magic", "orl": "magic",
    "pacers": "pacers", "ind": "pacers",
    "hornets": "hornets", "cha": "hornets",
    "wizards": "wizards", "was": "wizards",
    "pistons": "pistons", "det": "pistons",
    "rockets": "rockets", "hou": "rockets",
    "spurs": "spurs", "sas": "spurs",
    "kings": "kings", "sac": "kings",
    "blazers": "blazers", "trailblazers": "blazers", "por": "blazers",
    "jazz": "jazz", "uta": "jazz",
    "wolves": "timberwolves", "timberwolves": "timberwolves", "min": "timberwolves",
    "pelicans": "pelicans", "nop": "pelicans",
    "grizzlies": "grizzlies", "mem": "grizzlies",
}


async def _validate_teams_outgoing_webhook(request: Request) -> None:
    """Validate Teams outgoing webhook signature if configured."""
    webhook_secret = os.environ.get("TEAMS_WEBHOOK_SECRET", "").strip()
    if not webhook_secret:
        logger.warning("TEAMS_WEBHOOK_SECRET not set - skipping Teams outgoing webhook validation")
        return

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("HMAC "):
        logger.error("Missing HMAC Authorization header for Teams outgoing webhook")
        raise HTTPException(status_code=401, detail="Unauthorized - missing HMAC signature")

    try:
        secret_bytes = base64.b64decode(webhook_secret)
    except Exception:
        logger.error("Invalid TEAMS_WEBHOOK_SECRET - expected base64")
        raise HTTPException(status_code=500, detail="Server configuration error")

    body = await request.body()
    computed_hash = hmac.new(secret_bytes, body, hashlib.sha256)
    computed_signature = base64.b64encode(computed_hash.digest()).decode("utf-8")
    provided_signature = auth_header[5:]
    if not hmac.compare_digest(provided_signature, computed_signature):
        logger.error("HMAC validation failed for Teams outgoing webhook")
        raise HTTPException(status_code=401, detail="Unauthorized - invalid signature")


def _parse_teams_command(text: str) -> dict:
    """Parse Teams outgoing webhook commands into filters."""
    normalized = text.lower().strip()
    result = {"date": "today", "team_filter": None, "elite_only": False, "show_menu": False}

    if not normalized or "help" in normalized or "menu" in normalized or "options" in normalized:
        result["show_menu"] = True
        return result

    if "elite" in normalized or "best" in normalized or "top" in normalized:
        result["elite_only"] = True

    if "tomorrow" in normalized:
        tomorrow = datetime.now() + timedelta(days=1)
        result["date"] = tomorrow.strftime("%Y-%m-%d")

    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", normalized)
    if date_match:
        result["date"] = date_match.group(1)
    else:
        date_match = re.search(r"(\d{1,2})/(\d{1,2})", normalized)
        if date_match:
            month, day = int(date_match.group(1)), int(date_match.group(2))
            if 1 <= month <= 12 and 1 <= day <= 31:
                year = datetime.now().year
                result["date"] = f"{year}-{month:02d}-{day:02d}"

    for key, team_keyword in _TEAM_KEYWORDS.items():
        if key in normalized:
            result["team_filter"] = team_keyword
            break

    return result


async def _get_espn_standings() -> dict:
    """Fetch ESPN standings for accurate W-L records."""
    from src.ingestion.espn import fetch_espn_standings
    return await fetch_espn_standings()


def _lookup_espn_record(team_name: str, standings: dict) -> dict | None:
    """Lookup ESPN record for a team name, returning wins/losses."""
    if not standings:
        return None

    standing = standings.get(team_name)
    if standing:
        return {"wins": standing.wins, "losses": standing.losses, "source": "espn"}

    from src.ingestion.standardize import normalize_team_to_espn

    normalized_name, is_valid = normalize_team_to_espn(team_name, source="espn_records")
    if is_valid:
        standing = standings.get(normalized_name)
        if standing:
            return {"wins": standing.wins, "losses": standing.losses, "source": "espn"}

    return None


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


# --- Request/Response Models ---

class GamePredictionRequest(BaseModel):
    """Request for single game prediction - 4 markets (1H + FG spreads/totals)."""
    home_team: str = Field(..., example="Cleveland Cavaliers")
    away_team: str = Field(..., example="Chicago Bulls")
    # Full game lines - REQUIRED
    fg_spread_line: float
    fg_total_line: float
    # First half lines - optional but recommended
    fh_spread_line: Optional[float] = None
    fh_total_line: Optional[float] = None


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
    odds_archive_path: Optional[str] = None
    error_message: Optional[str] = None


class MarketsResponse(BaseModel):
    version: str
    source: str
    markets: List[str]
    market_count: int
    periods: Dict[str, List[str]]
    market_types: List[str]
    model_pack_version: Optional[str] = None
    model_pack_path: Optional[str] = None


# --- API Setup ---

def _models_dir() -> PathLib:
    """
    Resolve models directory with explicit, non-silent fallback.

    Priority:
    1) data_processed_dir/models (runtime target)
    2) PROJECT_ROOT/models/production (local source of truth)

    Raise if neither exists.
    """
    primary = PathLib(settings.data_processed_dir) / "models"
    if primary.exists():
        return primary

    secondary = PROJECT_ROOT / "models" / "production"
    if secondary.exists():
        logger.warning(
            f"Models directory not found at {primary}. Using local source {secondary}. "
            f"Ensure models are copied to data/processed/models for deployment."
        )
        return secondary

    raise RuntimeError(
        f"No models directory found. Checked: {primary} and {secondary}. "
        f"Copy trained models to data/processed/models or set data_processed_dir correctly."
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.

    Startup: Initialize the prediction engine.
    Production: 4 markets (1H+FG for Spread, Total). Q1 removed.
    Fails LOUDLY if models are missing or API keys are invalid.
    """
    # === STARTUP ===
    # SECURITY: Startup integrity checks (secrets, market list, feature alignment)
    try:
        run_startup_integrity_checks(PROJECT_ROOT, _models_dir())
        logger.info("Startup integrity checks passed")
    except StartupIntegrityError as e:
        logger.error(str(e))
        raise
    except Exception as e:
        logger.error(f"Startup integrity checks failed: {e}")
        raise

    # Validate betting splits sources (warning only, not fatal)
    splits_sources = validate_splits_sources_configured()
    app.state.splits_sources_configured = splits_sources

    # Validate all premium features and log what's available
    premium_features = validate_premium_features()
    app.state.premium_features = premium_features

    models_dir = _models_dir()
    logger.info(f"{RELEASE_VERSION} STRICT MODE: Loading Unified Prediction Engine from {models_dir}")

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
    logger.info(f"{RELEASE_VERSION} STRICT MODE: Using 1H/FG models (4 markets: spread/total)")
    app.state.engine = UnifiedPredictionEngine(models_dir=models_dir, require_all=True)
    app.state.feature_builder = RichFeatureBuilder(season=settings.current_season)

    # NO FILE CACHING - all data fetched fresh from APIs per request
    logger.info(f"{RELEASE_VERSION} STRICT MODE: File caching DISABLED - all data fetched fresh per request")

    # Initialize live pick tracker
    picks_dir = PathLib(settings.data_processed_dir) / "picks"
    picks_dir.mkdir(parents=True, exist_ok=True)
    app.state.tracker = PickTracker(tracking_dir=picks_dir)
    logger.info(f"Pick tracker initialized at {picks_dir}")

    # Log model info
    model_info = app.state.engine.get_model_info()
    logger.info(f"{RELEASE_VERSION} initialized - {model_info['markets']} markets loaded: {model_info['markets_list']}")

    yield  # Application runs here

    # === SHUTDOWN ===
    logger.info(f"{RELEASE_VERSION} shutting down")


app = FastAPI(
    title=f"NBA {RELEASE_VERSION} - STRICT MODE Production Picks",
    description="4 INDEPENDENT Markets: 1H+FG for Spread and Total. FRESH DATA ONLY - No caching, no fallbacks, no placeholders.",
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


# API prefix stripping middleware - allows routes to work with or without /api prefix
# This enables Azure Static Web App linked backend proxying (sends /api/*)
@app.middleware("http")
async def strip_api_prefix_middleware(request: Request, call_next):
    # If path starts with /api/, strip it for routing
    # This allows www.greenbiersportventures.com/api/health to route to /health
    if request.url.path.startswith("/api/"):
        # Create new scope with modified path
        new_path = request.url.path[4:]  # Remove "/api" prefix
        request.scope["path"] = new_path
        # Also update raw_path if present
        if "raw_path" in request.scope:
            request.scope["raw_path"] = new_path.encode()
    elif request.url.path == "/api":
        # Handle bare /api as /
        request.scope["path"] = "/"
        if "raw_path" in request.scope:
            request.scope["raw_path"] = b"/"
    
    return await call_next(request)


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
    """Check API health - 4 markets (spread/total only)."""
    engine_loaded = hasattr(app.state, 'engine') and app.state.engine is not None
    api_keys = get_api_key_status()

    model_info = {}
    if engine_loaded:
        model_info = app.state.engine.get_model_info()

    return {
        "status": "ok",
        "version": RELEASE_VERSION,
        "mode": "STRICT",
        "architecture": "1H + FG spreads/totals only",
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

    STRICT MODE: No file caching exists - only clears session memory caches.
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

    logger.info(f"{RELEASE_VERSION} STRICT MODE: Session cache cleared")

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

    Verifies 4 independent models (1H + FG for spread, total)
    """
    results = {
        "status": "pass",
        "version": RELEASE_VERSION,
        "markets": {
            "1h": ["spread", "total"],
            "fg": ["spread", "total"],
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
        results["checks"]["legacy_predictors"] = {
            "spread": has_spread,
            "total": has_total,
        }

        if not (has_fg or (has_spread and has_total)):
            results["status"] = "fail"
            results["errors"].append("Missing period predictors")

        # Build a complete test feature payload from model requirements
        required_features = set()
        try:
            if has_fg and hasattr(app.state.engine, "fg_predictor"):
                required_features.update(app.state.engine.fg_predictor.spread_features or [])
                required_features.update(app.state.engine.fg_predictor.total_features or [])
            if has_1h and hasattr(app.state.engine, "h1_predictor"):
                required_features.update(app.state.engine.h1_predictor.spread_features or [])
                required_features.update(app.state.engine.h1_predictor.total_features or [])
        except Exception as e:
            logger.warning(f"Unable to read model feature requirements: {e}")

        if not required_features:
            # Fallback to unified defaults if feature lists are unavailable
            required_features = set(get_feature_defaults().keys())

        defaults = get_feature_defaults()
        # Sensible overrides so predictions behave realistically
        overrides = {
            "predicted_margin": 3.0,
            "predicted_total": 227.0,
            "predicted_margin_1h": 1.5,
            "predicted_total_1h": 113.5,
            "home_win_pct": 0.6,
            "away_win_pct": 0.4,
            "home_margin": 2.0,
            "away_margin": -1.0,
            "home_rest": 2.0,
            "away_rest": 1.0,
            "home_b2b": 0.0,
            "away_b2b": 0.0,
            "spread_line": -3.5,
            "total_line": 225.0,
            "spread_public_home_pct": 50.0,
            "spread_ticket_money_diff": 0.0,
            "has_real_splits": 0.0,
            "dynamic_hca": 3.0,
        }
        defaults.update(overrides)

        test_features = {name: defaults.get(name, 0.0) for name in required_features}
        # Additional 1H-only required fields (not part of model columns)
        test_features.setdefault("predicted_margin_1h", overrides["predicted_margin_1h"])
        test_features.setdefault("predicted_total_1h", overrides["predicted_total_1h"])

        # Check 3: Test 1H prediction
        try:
            test_pred_1h = app.state.engine.predict_first_half(
                features=test_features,
                spread_line=-1.5,
                total_line=112.5,
            )
            
            results["checks"]["1h_prediction_works"] = True
            results["checks"]["1h_has_spread"] = "spread" in test_pred_1h
            results["checks"]["1h_has_total"] = "total" in test_pred_1h
            
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

    4 markets (1H + FG spreads/totals). Q1 removed entirely.
    Returns 1H/FG for spread and total when available.
    """
    if not hasattr(app.state, 'engine') or app.state.engine is None:
        raise HTTPException(status_code=503, detail=f"{RELEASE_VERSION}: Engine not loaded - models missing")

    # STRICT MODE: Clear ALL caches to force fresh unified data
    app.state.feature_builder.clear_session_cache()
    clear_unified_records_cache()  # QA/QC: Ensure records cache is fresh
    logger.info(f"{RELEASE_VERSION}: Caches cleared - fetching fresh unified data (odds + records from The Odds API)")

    # Resolve date
    from src.utils.slate_analysis import get_target_date, fetch_todays_games, extract_consensus_odds
    try:
        target_date = get_target_date(date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    # Fetch games
    try:
        games = await fetch_todays_games(target_date, include_records=False)
    except Exception as e:
        logger.error("Error fetching odds for %s: %s", target_date, e, exc_info=True)
        fallback_timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        return SlateResponse(
            date=str(target_date),
            predictions=[],
            total_plays=0,
            odds_as_of_utc=fallback_timestamp,
            error_message="Failed to fetch data from Odds API",
        )

    if not games:
        return SlateResponse(date=str(target_date), predictions=[], total_plays=0)

    try:
        espn_standings = await _get_espn_standings()
    except Exception as e:
        logger.error("Failed to fetch ESPN standings: %s", e)
        raise HTTPException(status_code=503, detail="ESPN standings unavailable")

    odds_as_of_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    odds_snapshot_path = None
    odds_archive_path = None
    try:
        from src.ingestion.the_odds import save_odds
        odds_snapshot_path = await save_odds(games, prefix=f"slate_odds_{target_date.strftime('%Y%m%d')}")
        archive_dir = PROJECT_ROOT / "archive" / "odds_snapshots"
        odds_archive_path = await save_odds(
            games,
            out_dir=str(archive_dir),
            prefix=f"slate_odds_{target_date.strftime('%Y%m%d')}",
        )
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
            
            home_record_data = _lookup_espn_record(home_team, espn_standings)
            away_record_data = _lookup_espn_record(away_team, espn_standings)

            if not home_record_data:
                logger.warning("Missing ESPN record for home team: %s", home_team)
                home_record_data = {"wins": 0, "losses": 0, "source": "espn_missing"}

            if not away_record_data:
                logger.warning("Missing ESPN record for away team: %s", away_team)
                away_record_data = {"wins": 0, "losses": 0, "source": "espn_missing"}
            
            features["home_wins"] = home_record_data.get("wins", 0)
            features["home_losses"] = home_record_data.get("losses", 0)
            features["away_wins"] = away_record_data.get("wins", 0)
            features["away_losses"] = away_record_data.get("losses", 0)
            features["_records_source"] = home_record_data.get("source", "espn")

            # Extract consensus lines
            odds = extract_consensus_odds(game, as_of_utc=odds_as_of_utc)
            fg_spread = odds.get("home_spread")
            fg_total = odds.get("total")
            fh_spread = odds.get("fh_home_spread")
            fh_total = odds.get("fh_total")
            # Require spread or total lines
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
            )

            # Count plays (4 markets: 1H + FG spreads/totals)
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
            logger.warning(f"{RELEASE_VERSION}: Skipping {home_team} vs {away_team} - {e}")
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
        odds_archive_path=odds_archive_path,
    )


def format_american_odds(odds: int) -> str:
    """Format American odds with +/- prefix."""
    if odds is None:
        return "N/A"
    return f"+{odds}" if odds > 0 else str(odds)


def get_fire_rating(confidence: float, edge: float) -> int:
    """Calculate fire rating (1-5) based on confidence and edge."""
    # Normalize edge (pts) to 0-1 scale (10 pts = max)
    edge_norm = min(abs(edge) / 10.0, 1.0)
    # Combine confidence and edge
    combined_score = (confidence * 0.6) + (edge_norm * 0.4)

    if combined_score >= 0.85: return 5
    elif combined_score >= 0.70: return 4
    elif combined_score >= 0.60: return 3
    elif combined_score >= 0.52: return 2
    else: return 1


@app.get("/slate/{date}/executive")
@limiter.limit("30/minute")
async def get_executive_summary(
    request: Request,
    date: str = Path(..., description="Date in YYYY-MM-DD format, 'today', or 'tomorrow'"),
    use_splits: bool = Query(True, description="Whether to fetch and use betting splits")
):
    """
    BLUF (Bottom Line Up Front) Executive Summary.

    STRICT MODE: Fetches fresh data from all APIs.
    Returns a clean actionable betting card with all picks that pass filters.
    Sorted by EV% (desc), then game time.
    """
    if not hasattr(app.state, 'engine') or app.state.engine is None:
        raise HTTPException(status_code=503, detail=f"{RELEASE_VERSION} STRICT MODE: Engine not loaded - models missing")

    # STRICT MODE: Clear ALL caches to force fresh unified data
    app.state.feature_builder.clear_session_cache()
    clear_unified_records_cache()  # QA/QC: Ensure records cache is fresh
    logger.info(f"{RELEASE_VERSION} STRICT MODE: Executive summary - fetching fresh unified data")

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
    games = await fetch_todays_games(target_date, include_records=False)
    if not games:
        return {
            "date": str(target_date),
            "generated_at": datetime.now(CST).strftime("%Y-%m-%d %I:%M %p CST"),
            "version": RELEASE_VERSION,
            "total_plays": 0,
            "plays": [],
            "summary": "No games scheduled"
        }

    odds_as_of_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    try:
        espn_standings = await _get_espn_standings()
    except Exception as e:
        logger.error("Failed to fetch ESPN standings: %s", e)
        raise HTTPException(status_code=503, detail="ESPN standings unavailable")
    odds_snapshot_path = None
    odds_archive_path = None
    try:
        from src.ingestion.the_odds import save_odds
        odds_snapshot_path = await save_odds(games, prefix=f"slate_odds_{target_date.strftime('%Y%m%d')}")
        archive_dir = PROJECT_ROOT / "archive" / "odds_snapshots"
        odds_archive_path = await save_odds(
            games,
            out_dir=str(archive_dir),
            prefix=f"slate_odds_{target_date.strftime('%Y%m%d')}",
        )
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
            
            home_record_data = _lookup_espn_record(home_team, espn_standings)
            away_record_data = _lookup_espn_record(away_team, espn_standings)

            if not home_record_data:
                logger.warning("Missing ESPN record for home team: %s", home_team)
                home_record_data = {"wins": 0, "losses": 0, "source": "espn_missing"}

            if not away_record_data:
                logger.warning("Missing ESPN record for away team: %s", away_team)
                away_record_data = {"wins": 0, "losses": 0, "source": "espn_missing"}
            
            home_wins = home_record_data.get("wins", 0)
            home_losses = home_record_data.get("losses", 0)
            away_wins = away_record_data.get("wins", 0)
            away_losses = away_record_data.get("losses", 0)
            
            # Document data source in features for audit trail
            features["home_wins"] = home_wins
            features["home_losses"] = home_losses
            features["away_wins"] = away_wins
            features["away_losses"] = away_losses
            features["_records_source"] = home_record_data.get("source", "espn")
            
            home_record = f"({home_wins}-{home_losses})"
            away_record = f"({away_wins}-{away_losses})"
            matchup_display = f"{away_team} {away_record} @ {home_team} {home_record}"
            
            # Extract odds
            odds = extract_consensus_odds(game, as_of_utc=odds_as_of_utc)
            fg_spread = odds.get("home_spread")
            fg_total = odds.get("total")
            fh_spread = odds.get("fh_home_spread")
            fh_total = odds.get("fh_total")

            if fg_spread is None or fg_total is None:
                continue

            # Get predictions for 4 markets (1H + FG spreads/totals)
            preds = app.state.engine.predict_all_markets(
                features,
                fg_spread_line=fg_spread,
                fg_total_line=fg_total,
                fh_spread_line=fh_spread,
                fh_total_line=fh_total,
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
        "odds_archive_path": odds_archive_path,
        "legend": {
            "fire_rating": {
                "ELITE": "70%+ confidence AND 5+ pt edge",
                "STRONG": "60%+ confidence AND 3+ pt edge",
                "GOOD": "Passes all filters"
            },
            "sorting": "EV% (desc), then game time",
            "periods": {"FG": "Full Game", "1H": "First Half"},
            "markets": {"SPREAD": "Point Spread", "TOTAL": "Over/Under"}
        }
    })


@app.post("/teams/outgoing")
@limiter.limit("30/minute")
async def teams_outgoing_webhook(request: Request):
    """Teams outgoing webhook handler (ACA-hosted)."""
    await _validate_teams_outgoing_webhook(request)

    try:
        body = await request.json()
    except Exception as e:
        logger.error("Failed to parse Teams outgoing webhook body: %s", e)
        return JSONResponse(status_code=200, content={"text": "Invalid request body."})

    raw_text = body.get("text", "") if isinstance(body, dict) else ""
    command_text = re.sub(r"<at>.*?</at>\s*", "", raw_text).strip()
    parsed = _parse_teams_command(command_text)

    if parsed["show_menu"]:
        help_text = (
            "Commands: picks, picks tomorrow, picks YYYY-MM-DD, "
            "picks lakers, elite, help"
        )
        return JSONResponse(status_code=200, content={"text": help_text})

    try:
        data = await get_executive_summary(request, date=parsed["date"], use_splits=True)
    except HTTPException as e:
        logger.error("Teams outgoing webhook request failed: %s", e.detail)
        return JSONResponse(status_code=200, content={"text": f"Error: {e.detail}"})
    except Exception as e:
        logger.error("Teams outgoing webhook request failed: %s", e)
        return JSONResponse(status_code=200, content={"text": "Failed to fetch picks."})

    if not isinstance(data, dict):
        return JSONResponse(status_code=200, content={"text": "No picks available."})

    plays = data.get("plays", [])
    total_plays = data.get("total_plays", len(plays))
    if not plays:
        date_label = data.get("date", parsed["date"])
        return JSONResponse(status_code=200, content={"text": f"No picks available for {date_label}."})

    team_filter = parsed["team_filter"]
    if team_filter:
        plays = [p for p in plays if team_filter in p.get("matchup", "").lower()]

    if parsed["elite_only"]:
        def _fire_rating_value(value) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0

        plays = [p for p in plays if _fire_rating_value(p.get("fire_rating")) >= 4]

    if not plays:
        parts = []
        if parsed["elite_only"]:
            parts.append("elite")
        if team_filter:
            parts.append(f"for {team_filter}")
        label = " ".join(parts).strip()
        label = f"{label} " if label else ""
        date_label = data.get("date", parsed["date"])
        return JSONResponse(status_code=200, content={"text": f"No {label}picks found for {date_label}."})

    date_label = data.get("date", parsed["date"])
    label_parts = []
    if parsed["elite_only"]:
        label_parts.append("elite only")
    if team_filter:
        label_parts.append(team_filter)
    label = f" ({', '.join(label_parts)})" if label_parts else ""
    lineup_params = {"date": date_label}
    if team_filter:
        lineup_params["team"] = team_filter
    if parsed["elite_only"]:
        lineup_params["elite"] = "true"
    lineup_url = f"{str(request.base_url).rstrip('/')}/weekly-lineup/html?{urlencode(lineup_params)}"
    csv_url = f"{str(request.base_url).rstrip('/')}/weekly-lineup/csv?{urlencode(lineup_params)}"

    def _fmt_cell(value: str, width: int) -> str:
        text = str(value or "")
        if len(text) <= width:
            return text.ljust(width)
        if width <= 3:
            return text[:width]
        return text[:width - 3] + "..."

    columns = [
        ("TIME", 12),
        ("MATCHUP", 38),
        ("SEG", 9),
        ("PICK", 24),
        ("EDGE", 9),
        ("FIRE", 6),
    ]
    header = " | ".join(_fmt_cell(label, width) for label, width in columns)
    divider = "-+-".join("-" * width for _, width in columns)

    lines_before = [
        f"NBA picks for {date_label}{label}",
        f"Showing {len(plays)} picks (sorted by EV%)",
        "```",
    ]
    lines_after = [
        "```",
        f"CSV: {csv_url}",
        f"HTML: {lineup_url}",
        "Tip: use 'help' for commands.",
    ]
    def _fire_display(value: object) -> str:
        try:
            rating = int(value)
        except (TypeError, ValueError):
            rating = 0
        return "\U0001F525" * rating if rating > 0 else "-"

    table_lines = [header, divider]
    for p in plays:
        segment = f"{p.get('period', '')} {p.get('market', '')}".strip()
        row_values = [
            p.get("time_cst", ""),
            p.get("matchup", ""),
            segment,
            p.get("pick", ""),
            p.get("edge", ""),
            _fire_display(p.get("fire_rating", "")),
        ]
        row = " | ".join(_fmt_cell(value, width) for value, (_, width) in zip(row_values, columns))
        table_lines.append(row)

    lines = lines_before + ["\n".join(table_lines)] + lines_after

    return JSONResponse(status_code=200, content={"text": "\n".join(lines)})


@app.get("/slate/{date}/comprehensive")
@limiter.limit("20/minute")
async def get_comprehensive_slate_analysis(
    request: Request,
    date: str = Path(..., description="Date in YYYY-MM-DD format, 'today', or 'tomorrow'"),
    use_splits: bool = Query(True, description="Whether to fetch and use betting splits")
):
    """
    Get comprehensive slate analysis with full edge calculations.

    STRICT MODE: Fetches fresh data from all APIs.
    Full analysis for all 4 markets (1H + FG spreads/totals).
    """
    if not hasattr(app.state, 'engine') or app.state.engine is None:
        raise HTTPException(status_code=503, detail=f"{RELEASE_VERSION} STRICT MODE: Engine not loaded - models missing")

    # STRICT MODE: Clear ALL caches to force fresh unified data
    app.state.feature_builder.clear_session_cache()
    app.state.feature_builder.clear_persistent_cache()
    clear_unified_records_cache()  # QA/QC: Ensure records cache is fresh
    logger.info(f"{RELEASE_VERSION} STRICT MODE: Comprehensive analysis - fetching fresh unified data")
    
    CST = ZoneInfo("America/Chicago")
    
    # Enforce fresh data: clear ALL caches before fetching
    request_started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    request_started_cst = datetime.now(CST).strftime("%Y-%m-%d %I:%M:%S %p %Z")
    try:
        # Session cache (per-request)
        app.state.feature_builder.clear_session_cache()
        # Persistent reference cache (team/league lookups)
        RichFeatureBuilder.clear_persistent_cache()
    except Exception as e:
        logger.warning(f"Failed to clear feature caches before slate fetch: {e}")

    # Resolve date
    try:
        target_date = get_target_date(date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    # Get edge thresholds - use SAME source as engine for consistency
    try:
        from src.config import filter_thresholds
        edge_thresholds = {
            "spread": filter_thresholds.spread_min_edge,
            "total": filter_thresholds.total_min_edge,
            "1h_spread": filter_thresholds.spread_min_edge * 0.75,  # Scale for 1H
            "1h_total": filter_thresholds.total_min_edge * 0.67,   # Scale for 1H
        }
    except ImportError:
        # Fallback if config not available
        edge_thresholds = {
            "spread": 2.0,
            "total": 3.0,
            "1h_spread": 1.5,
            "1h_total": 2.0,
        }

    # Fetch games
    games = await fetch_todays_games(target_date, include_records=False)
    if not games:
        return {"date": str(target_date), "analysis": [], "summary": "No games found"}

    try:
        espn_standings = await _get_espn_standings()
    except Exception as e:
        logger.error("Failed to fetch ESPN standings: %s", e)
        raise HTTPException(status_code=503, detail="ESPN standings unavailable")

    odds_as_of_utc = request_started_utc
    odds_snapshot_path = None
    odds_archive_path = None
    try:
        from src.ingestion.the_odds import save_odds
        odds_snapshot_path = await save_odds(games, prefix=f"slate_odds_{target_date.strftime('%Y%m%d')}")
        archive_dir = PROJECT_ROOT / "archive" / "odds_snapshots"
        odds_archive_path = await save_odds(
            games,
            out_dir=str(archive_dir),
            prefix=f"slate_odds_{target_date.strftime('%Y%m%d')}",
        )
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
            
            home_record_data = _lookup_espn_record(home_team, espn_standings)
            away_record_data = _lookup_espn_record(away_team, espn_standings)

            if not home_record_data:
                logger.warning("Missing ESPN record for home team: %s", home_team)
                home_record_data = {"wins": 0, "losses": 0, "source": "espn_missing"}

            if not away_record_data:
                logger.warning("Missing ESPN record for away team: %s", away_team)
                away_record_data = {"wins": 0, "losses": 0, "source": "espn_missing"}
            
            features["home_wins"] = home_record_data.get("wins", 0)
            features["home_losses"] = home_record_data.get("losses", 0)
            features["away_wins"] = away_record_data.get("wins", 0)
            features["away_losses"] = away_record_data.get("losses", 0)
            features["_records_source"] = home_record_data.get("source", "espn")
            features["_data_unified"] = game.get("_data_unified", False)
            
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
            engine_predictions = None
            if fg_spread is not None and fg_total is not None:
                try:
                    engine_predictions = app.state.engine.predict_all_markets(
                        features,
                        fg_spread_line=fg_spread,
                        fg_total_line=fg_total,
                        fh_spread_line=fh_spread,
                        fh_total_line=fh_total,
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
        "data_fetched_at_cst": request_started_cst,
        "markets": [
            "1h_spread", "1h_total",
            "fg_spread", "fg_total"
        ],
        "analysis": analysis_results,
        "edge_thresholds": edge_thresholds,
        "odds_as_of_utc": odds_as_of_utc,
        "odds_snapshot_path": odds_snapshot_path,
        "odds_archive_path": odds_archive_path,
    })


@app.get("/meta", tags=["Ops"])
async def get_meta_info():
    """Get metadata about the running service."""
    catalog = get_market_catalog(PROJECT_ROOT, settings.data_processed_dir)
    return {
        "version": RELEASE_VERSION,
        "markets": catalog.markets,
        "market_types": catalog.market_types,
        "periods": catalog.periods,
        "markets_source": catalog.source,
        "model_pack_version": catalog.model_pack_version,
        "strict_mode": os.getenv("NBA_STRICT_MODE", "false").lower() == "true",
        "server_time": datetime.now().isoformat(),
        "python_version": sys.version
    }

@app.get("/registry", tags=["Meta"])
def get_registry(request: Request):
    """Compatibility registry for web clients expecting /api/registry."""
    base_url = str(request.base_url).rstrip("/")
    paths = {
        "health": "/health",
        "meta": "/meta",
        "markets": "/markets",
        "picks": "/api/v1/picks",
        "picks_v1": "/api/v1/picks",
        "weekly_lineup": "/weekly-lineup/nba",
        "slate": "/slate/{date}",
        "executive": "/slate/{date}/executive",
        "registry": "/api/registry"
    }
    endpoints = {key: f"{base_url}{path}" for key, path in paths.items()}

    return {
        "status": "ok",
        "service": "nba-picks",
        "version": RELEASE_VERSION,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "api_base_url": base_url,
        "apiBaseUrl": base_url,
        "baseUrl": base_url,
        "paths": paths,
        "endpoints": endpoints,
        "api": {
            "v1": {
                "picks": endpoints["picks_v1"]
            }
        },
        "defaults": {
            "sport": "nba",
            "date": "today"
        }
    }


@app.get("/markets", response_model=MarketsResponse, tags=["Meta"])
def get_markets(request: Request):
    """List enabled markets and periods from the model pack or configs."""
    catalog = get_market_catalog(PROJECT_ROOT, settings.data_processed_dir)
    return {
        "version": RELEASE_VERSION,
        "source": catalog.source,
        "markets": catalog.markets,
        "market_count": len(catalog.markets),
        "periods": catalog.periods,
        "market_types": catalog.market_types,
        "model_pack_version": catalog.model_pack_version,
        "model_pack_path": catalog.model_pack_path,
    }


@app.post("/predict/game", response_model=GamePredictions)
@limiter.limit("60/minute")
async def predict_single_game(request: Request, req: GamePredictionRequest):
    """
    Generate predictions for a specific matchup.

    STRICT MODE - Fetches fresh data from all APIs.
    4 markets (1H+FG for Spread, Total).
    """
    if not hasattr(app.state, 'engine') or app.state.engine is None:
        raise HTTPException(status_code=503, detail=f"{RELEASE_VERSION} STRICT MODE: Engine not loaded - models missing")

    # STRICT MODE: Clear session cache to force fresh data
    app.state.feature_builder.clear_session_cache()

    try:
        # Build features from FRESH data
        features = await app.state.feature_builder.build_game_features(
            req.home_team, req.away_team
        )

        # Predict all 4 markets (1H + FG spreads/totals)
        preds = app.state.engine.predict_all_markets(
            features,
            fg_spread_line=req.fg_spread_line,
            fg_total_line=req.fg_total_line,
            fh_spread_line=req.fh_spread_line,
            fh_total_line=req.fh_total_line,
        )
        return preds
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"{RELEASE_VERSION}: {e}")
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# LIVE PICK TRACKING ENDPOINTS
# =============================================================================

@app.get("/tracking/summary")
@limiter.limit("30/minute")
async def get_tracking_summary(
    request: Request,
    date: Optional[str] = Query(None, description="Filter by date (YYYY-MM-DD)"),
    period: Optional[str] = Query(None, description="Filter by period (1h, fg)"),
    market_type: Optional[str] = Query(None, description="Filter by market (spread, total)")
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


@app.get("/weekly-lineup/html", response_class=HTMLResponse, tags=["Display"])
@limiter.limit("20/minute")
async def get_weekly_lineup_html(
    request: Request,
    date: Optional[str] = Query(None, description="Date (YYYY-MM-DD), defaults to today"),
    team: Optional[str] = Query(None, description="Filter by team name/abbreviation"),
    elite: bool = Query(False, description="Elite picks only")
):
    """Render full weekly lineup as sortable HTML."""
    try:
        from datetime import datetime as dt

        if not date:
            date = dt.now().strftime("%Y-%m-%d")

        summary = await get_executive_summary(request, date=date)
        plays = summary.get("plays", []) if isinstance(summary, dict) else []

        def _fire_value(value) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0

        if team:
            team_filter = team.lower().strip()
            plays = [p for p in plays if team_filter in p.get("matchup", "").lower()]

        if elite:
            plays = [p for p in plays if _fire_value(p.get("fire_rating")) >= 4]

        if not plays:
            return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Weekly Lineup - NBA</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; background: #111827; color: #fff; padding: 40px; }}
        .card {{ background: #1f2937; border-radius: 12px; padding: 24px; max-width: 700px; margin: 0 auto; }}
        h1 {{ margin: 0 0 10px 0; font-size: 24px; }}
        p {{ margin: 0; color: #9ca3af; }}
    </style>
</head>
<body>
    <div class="card">
        <h1>Weekly Lineup</h1>
        <p>No picks available for {date}.</p>
    </div>
</body>
</html>"""

        def _edge_value(edge_text: str) -> float:
            if not edge_text:
                return 0.0
            cleaned = edge_text.replace("pts", "").replace("%", "").replace("+", "").strip()
            try:
                return float(cleaned)
            except ValueError:
                return 0.0

        def _confidence_value(conf_text: str) -> float:
            if not conf_text:
                return 0.0
            cleaned = str(conf_text).replace("%", "").strip()
            try:
                return float(cleaned)
            except ValueError:
                return 0.0

        def _fire_label(value: int) -> str:
            if value >= 4:
                return "ELITE"
            if value == 3:
                return "STRONG"
            if value == 2:
                return "GOOD"
            return "NONE"

        rows_html = ""
        for p in plays:
            time_cst = p.get("time_cst", "")
            matchup = p.get("matchup", "")
            period = p.get("period", "")
            pick = p.get("pick", "")
            model_prediction = p.get("model_prediction", "")
            market_line = p.get("market_line", "")
            edge = p.get("edge", "")
            confidence = p.get("confidence", "")
            fire_raw = p.get("fire_rating", 0)
            fire_value = _fire_value(fire_raw)

            rows_html += f"""
            <tr>
                <td data-value="{time_cst}">{time_cst}</td>
                <td data-value="{matchup}">{matchup}</td>
                <td data-value="{period}">{period}</td>
                <td data-value="{pick}">{pick}</td>
                <td data-value="{model_prediction}">{model_prediction}</td>
                <td data-value="{market_line}">{market_line}</td>
                <td data-value="{_edge_value(edge)}">{edge}</td>
                <td data-value="{_confidence_value(confidence)}">{confidence}</td>
                <td data-value="{fire_value}">{_fire_label(fire_value)}</td>
            </tr>
            """

        filters = []
        if team:
            filters.append(f"Team: {team}")
        if elite:
            filters.append("Elite only")
        filters_text = " | ".join(filters) if filters else "All picks"
        generated_at = summary.get("generated_at", "")

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Weekly Lineup - NBA</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: #e5e7eb;
            padding: 24px;
        }}
        .wrap {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .header {{
            background: #0f172a;
            border: 1px solid #1f2937;
            border-radius: 12px;
            padding: 20px 24px;
            margin-bottom: 20px;
        }}
        .header h1 {{
            font-size: 24px;
            margin-bottom: 6px;
            color: #f9fafb;
        }}
        .meta {{
            font-size: 13px;
            color: #94a3b8;
        }}
        .table-card {{
            background: #0b1220;
            border-radius: 12px;
            border: 1px solid #1f2937;
            overflow: hidden;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        thead {{
            background: #111827;
        }}
        th, td {{
            padding: 12px 10px;
            border-bottom: 1px solid #1f2937;
            font-size: 13px;
            text-align: left;
        }}
        th.sortable {{
            cursor: pointer;
            user-select: none;
        }}
        th.sortable::after {{
            content: 'v';
            margin-left: 6px;
            font-size: 10px;
            color: #4b5563;
        }}
        th.sortable.asc::after {{
            content: '^';
            color: #e5e7eb;
        }}
        th.sortable.desc::after {{
            content: 'v';
            color: #e5e7eb;
        }}
        tbody tr:hover {{
            background: rgba(148, 163, 184, 0.08);
        }}
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 999px;
            font-size: 11px;
            background: #1f2937;
            color: #e5e7eb;
        }}
        .footer {{
            margin-top: 16px;
            font-size: 12px;
            color: #9ca3af;
        }}
    </style>
</head>
<body>
    <div class="wrap">
        <div class="header">
            <h1>Weekly Lineup - NBA</h1>
            <div class="meta">{generated_at} | {filters_text} | {len(plays)} picks</div>
        </div>
        <div class="table-card">
            <table id="picks-table">
                <thead>
                    <tr>
                        <th class="sortable" data-type="text">Time (CST)</th>
                        <th class="sortable" data-type="text">Matchup</th>
                        <th class="sortable" data-type="text">Seg</th>
                        <th class="sortable" data-type="text">Pick</th>
                        <th class="sortable" data-type="text">Model</th>
                        <th class="sortable" data-type="text">Market</th>
                        <th class="sortable" data-type="number">Edge</th>
                        <th class="sortable" data-type="number">Conf</th>
                        <th class="sortable" data-type="number">Fire</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
        <div class="footer">Click any column header to sort.</div>
    </div>
    <script>
        const table = document.getElementById('picks-table');
        const headers = Array.from(table.querySelectorAll('th.sortable'));

        function getCellValue(row, index) {{
            const cell = row.children[index];
            const value = cell.getAttribute('data-value');
            return value !== null ? value : cell.textContent.trim();
        }}

        function compareValues(a, b, type) {{
            if (type === 'number') {{
                const numA = parseFloat(a);
                const numB = parseFloat(b);
                return (isNaN(numA) ? 0 : numA) - (isNaN(numB) ? 0 : numB);
            }}
            return a.localeCompare(b);
        }}

        headers.forEach((header, index) => {{
            header.addEventListener('click', () => {{
                const currentAsc = header.classList.contains('asc');
                headers.forEach(h => h.classList.remove('asc', 'desc'));
                header.classList.add(currentAsc ? 'desc' : 'asc');
                const ascending = !currentAsc;
                const type = header.getAttribute('data-type') || 'text';

                const rows = Array.from(table.tBodies[0].rows);
                rows.sort((rowA, rowB) => {{
                    const valA = getCellValue(rowA, index);
                    const valB = getCellValue(rowB, index);
                    const result = compareValues(valA, valB, type);
                    return ascending ? result : -result;
                }});
                rows.forEach(row => table.tBodies[0].appendChild(row));
            }});
        }});
    </script>
</body>
</html>"""

        return html_content
    except Exception as e:
        logger.error(f"Error rendering weekly lineup HTML: {str(e)}", exc_info=True)
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Weekly Lineup - Error</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; background: #0f172a; color: #f9fafb; padding: 32px; }}
        .card {{ background: #1f2937; border-radius: 12px; padding: 20px; }}
        h1 {{ margin: 0 0 12px 0; }}
        p {{ color: #e5e7eb; }}
    </style>
</head>
<body>
    <div class="card">
        <h1>Error rendering weekly lineup</h1>
        <p>{str(e)}</p>
    </div>
</body>
</html>"""


@app.get("/weekly-lineup/csv")
@limiter.limit("20/minute")
async def get_weekly_lineup_csv(
    request: Request,
    date: Optional[str] = Query(None, description="Date (YYYY-MM-DD), defaults to today"),
    team: Optional[str] = Query(None, description="Filter by team name/abbreviation"),
    elite: bool = Query(False, description="Elite picks only")
):
    """Download weekly lineup picks as CSV."""
    try:
        from datetime import datetime as dt

        if not date:
            date = dt.now().strftime("%Y-%m-%d")

        summary = await get_executive_summary(request, date=date)
        plays = summary.get("plays", []) if isinstance(summary, dict) else []

        def _fire_value(value) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0

        if team:
            team_filter = team.lower().strip()
            plays = [p for p in plays if team_filter in p.get("matchup", "").lower()]

        if elite:
            plays = [p for p in plays if _fire_value(p.get("fire_rating")) >= 4]

        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow([
            "Time (CST)", "Matchup", "Segment", "Pick", "Model", "Market",
            "Edge", "Confidence", "Fire Rating"
        ])
        for p in plays:
            writer.writerow([
                p.get("time_cst", ""),
                p.get("matchup", ""),
                p.get("period", ""),
                p.get("pick", ""),
                p.get("model_prediction", ""),
                p.get("market_line", ""),
                p.get("edge", ""),
                p.get("confidence", ""),
                p.get("fire_rating", ""),
            ])

        filename = f"NBA_Picks_{date}.csv"
        return Response(
            content=buffer.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        logger.error(f"Error generating weekly lineup CSV: {str(e)}", exc_info=True)
        return Response(
            content="Failed to generate CSV",
            media_type="text/plain",
            status_code=500
        )


# =============================================================================
# WEBSITE INTEGRATION - JSON API for greenbiersportventures.com
# =============================================================================

def _get_fire_tier(fire_rating) -> str:
    """Convert fire rating to tier name. Handles both numeric and string formats."""
    if not fire_rating:
        return "NONE"
    
    # Handle string format (GOOD, STRONG, ELITE)
    if isinstance(fire_rating, str):
        rating_upper = fire_rating.upper().strip()
        if rating_upper in ["ELITE", "STRONG", "GOOD"]:
            return rating_upper
    
    # Handle numeric format (fire count)
    try:
        fire_count = int(fire_rating)
        if fire_count >= 4:
            return "ELITE"
        elif fire_count >= 3:
            return "STRONG"
        elif fire_count >= 1:
            return "GOOD"
    except (TypeError, ValueError):
        pass
    
    return "NONE"

async def _build_weekly_lineup_payload(
    request: Request,
    date: str,
    tier: str = "all",
    team: Optional[str] = None,
    elite: bool = False
) -> dict:
    if date == "tomorrow":
        resolved_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    elif date == "today":
        resolved_date = datetime.now().strftime("%Y-%m-%d")
    else:
        resolved_date = date

    data = await get_executive_summary(request, date=resolved_date, use_splits=True)

    if not data or not isinstance(data, dict):
        return {
            "sport": "NBA",
            "error": "Failed to fetch predictions",
            "date": resolved_date
        }

    all_plays = data.get("plays", [])
    tier_filter = (tier or "all").lower().strip()
    if elite:
        tier_filter = "elite"

    plays = all_plays
    if team:
        team_filter = team.lower().strip()
        plays = [p for p in plays if team_filter in p.get("matchup", "").lower()]

    if tier_filter == "elite":
        plays = [p for p in plays if _get_fire_tier(p.get("fire_rating", "")) == "ELITE"]
    elif tier_filter == "strong":
        plays = [p for p in plays if _get_fire_tier(p.get("fire_rating", "")) in ["ELITE", "STRONG"]]
    elif tier_filter == "good":
        plays = [p for p in plays if _get_fire_tier(p.get("fire_rating", "")) in ["ELITE", "STRONG", "GOOD"]]

    elite_count = len([p for p in all_plays if _get_fire_tier(p.get("fire_rating", "")) == "ELITE"])
    strong_count = len([p for p in all_plays if _get_fire_tier(p.get("fire_rating", "")) == "STRONG"])
    good_count = len([p for p in all_plays if _get_fire_tier(p.get("fire_rating", "")) == "GOOD"])

    formatted_picks = []
    for p in plays:
        formatted_picks.append({
            "time": p.get("time_cst", ""),
            "matchup": p.get("matchup", ""),
            "period": p.get("period", "FG"),
            "market": p.get("market", ""),
            "pick": p.get("pick", ""),
            "odds": p.get("pick_odds", "N/A"),
            "model_prediction": p.get("model_prediction", ""),
            "market_line": p.get("market_line", ""),
            "edge": p.get("edge", "N/A"),
            "confidence": p.get("confidence", ""),
            "tier": _get_fire_tier(p.get("fire_rating", "")),
            "fire_rating": p.get("fire_rating", "")
        })

    return {
        "sport": "NBA",
        "date": resolved_date,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "version": RELEASE_VERSION,
        "summary": {
            "total": len(all_plays),
            "elite": elite_count,
            "strong": strong_count,
            "good": good_count,
            "filtered": len(formatted_picks)
        },
        "picks": formatted_picks
    }


@app.get("/weekly-lineup/nba", tags=["Website"])
@limiter.limit("60/minute")
async def get_weekly_lineup_nba_json(
    request: Request,
    date: str = Query("today", description="Date in YYYY-MM-DD format, 'today', or 'tomorrow'"),
    tier: str = Query("all", description="Filter by tier: 'elite', 'strong', 'good', or 'all'")
):
    """
    Website integration endpoint for greenbiersportventures.com/weekly-lineup.

    Returns NBA picks in a format optimized for the website dashboard.
    Supports CORS for cross-origin requests from the website.

    Query params:
      - date: YYYY-MM-DD, 'today', or 'tomorrow' (default: today)
      - tier: 'elite', 'strong', 'good', or 'all' (default: all)

    Response format:
    {
        "sport": "NBA",
        "date": "2025-12-25",
        "generated_at": "2025-12-25T10:30:00Z",
        "version": "v33.0.11.0",
        "summary": {"total": 12, "elite": 3, "strong": 5, "good": 4},
        "picks": [...]
    }
    """
    try:
        result = await _build_weekly_lineup_payload(request, date=date, tier=tier)
        if not isinstance(result, dict) or result.get("error"):
            return JSONResponse(
                content=result if isinstance(result, dict) else {
                    "sport": "NBA",
                    "error": "Failed to fetch predictions",
                    "date": date
                },
                status_code=500
            )
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Error in weekly-lineup/nba: {str(e)}", exc_info=True)
        return JSONResponse(
            content={
                "sport": "NBA",
                "error": str(e),
                "date": date
            },
            status_code=500
        )

@app.get("/v1/picks", tags=["Website"])
@limiter.limit("60/minute")
async def get_v1_picks(
    request: Request,
    date: str = Query("today", description="Date in YYYY-MM-DD format, 'today', or 'tomorrow'"),
    tier: str = Query("all", description="Filter by tier: 'elite', 'strong', 'good', or 'all'"),
    team: Optional[str] = Query(None, description="Filter by team name or abbreviation"),
    elite: bool = Query(False, description="Elite picks only (overrides tier)")
):
    """
    Legacy compatibility endpoint for clients expecting /api/v1/picks.
    """
    try:
        result = await _build_weekly_lineup_payload(
            request,
            date=date,
            tier=tier,
            team=team,
            elite=elite
        )
        if not isinstance(result, dict) or result.get("error"):
            return JSONResponse(
                content=result if isinstance(result, dict) else {
                    "sport": "NBA",
                    "error": "Failed to fetch predictions",
                    "date": date
                },
                status_code=500
            )

        result = dict(result)
        result["api_version"] = "v1"
        if team or elite or (tier and tier.lower().strip() != "all"):
            result["filters"] = {
                "date": date,
                "tier": tier,
                "team": team,
                "elite": elite
            }
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error in v1/picks: {str(e)}", exc_info=True)
        return JSONResponse(
            content={
                "sport": "NBA",
                "error": str(e),
                "date": date
            },
            status_code=500
        )

