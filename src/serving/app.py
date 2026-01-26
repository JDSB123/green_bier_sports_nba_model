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

import csv
import io
import json
import logging
import os
import re
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path as PathLib
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import numpy as np
from fastapi import FastAPI, HTTPException, Path, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from prometheus_client import Counter, Histogram
from pydantic import BaseModel, Field
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from starlette.responses import Response

from src.config import PROJECT_ROOT, settings
from src.features import RichFeatureBuilder
from src.ingestion.betting_splits import validate_splits_sources_configured
from src.prediction import UnifiedPredictionEngine
from src.prediction.feature_validation import MissingFeaturesError
from src.serving.dependencies import (
    RELEASE_VERSION,
    canonical_game_key,
    convert_numpy_types,
    fetch_required_splits,
    format_american_odds,
    get_fire_rating,
    limiter,
    missing_market_lines,
    models_dir,
    require_splits_if_strict,
)
from src.serving.routes.admin import meta_router
from src.serving.routes.admin import router as admin_router

# Import route modules for incremental migration
from src.serving.routes.health import router as health_router
from src.serving.routes.teams import set_executive_summary_func, teams_router
from src.serving.routes.tracking import router as tracking_router
from src.tracking import PickTracker
from src.utils.api_auth import APIKeyMiddleware
from src.utils.comprehensive_edge import calculate_comprehensive_edge
from src.utils.logging import get_logger
from src.utils.security import validate_premium_features

# Additional imports for comprehensive edge calculation
from src.utils.slate_analysis import (
    clear_unified_records_cache,  # QA/QC: Clear records cache for fresh data
)
from src.utils.slate_analysis import (
    extract_consensus_odds,
    fetch_todays_games,
    parse_utc_time,
    to_cst,
)
from src.utils.startup_checks import StartupIntegrityError, run_startup_integrity_checks

logger = get_logger(__name__)


# Prometheus metrics
REQUEST_COUNT = Counter(
    "nba_api_requests_total", "Total number of API requests", ["method", "endpoint", "status"]
)

REQUEST_DURATION = Histogram(
    "nba_api_request_duration_seconds", "API request duration in seconds", ["method", "endpoint"]
)

# Rate limiter - use shared instance from dependencies

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
    errors: Optional[List[str]] = None


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
# Helper functions (models_dir, canonical_game_key, require_splits_if_strict, fetch_required_splits)
# are now imported from src.serving.dependencies


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
        run_startup_integrity_checks(PROJECT_ROOT, models_dir())
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

    model_path = models_dir()
    logger.info(
        f"{RELEASE_VERSION} STRICT MODE: Loading Unified Prediction Engine from {model_path}"
    )

    # Diagnostic: List files in models directory
    if model_path.exists():
        model_files = list(model_path.glob("*"))
        logger.info(f"Found {len(model_files)} files in models directory:")
        for f in sorted(model_files):
            size = f.stat().st_size if f.is_file() else 0
            logger.info(f"  - {f.name} ({size:,} bytes)")
    else:
        logger.error(f"Models directory does not exist: {model_path}")

    # STRICT MODE: 1H + FG models (6 total). No fallbacks.
    logger.info(f"{RELEASE_VERSION} STRICT MODE: Using 1H/FG models (4 markets: spread/total)")
    app.state.engine = UnifiedPredictionEngine(models_dir=model_path, require_all=True)
    app.state.feature_builder = RichFeatureBuilder(season=settings.current_season)
    # NO FILE CACHING - all data fetched fresh from APIs per request
    logger.info(
        f"{RELEASE_VERSION} STRICT MODE: File caching DISABLED - all data fetched fresh per request"
    )

    # Initialize live pick tracker
    picks_dir = PathLib(settings.data_processed_dir) / "picks"
    picks_dir.mkdir(parents=True, exist_ok=True)
    app.state.tracker = PickTracker(tracking_dir=picks_dir)
    logger.info(f"Pick tracker initialized at {picks_dir}")

    # Log model info
    model_info = app.state.engine.get_model_info()
    logger.info(
        f"{RELEASE_VERSION} initialized - {model_info['markets']} markets loaded: {model_info['markets_list']}"
    )

    yield  # Application runs here

    # === SHUTDOWN ===
    logger.info(f"{RELEASE_VERSION} shutting down")


app = FastAPI(
    title=f"NBA {RELEASE_VERSION} - STRICT MODE Production Picks",
    description="4 INDEPENDENT Markets: 1H+FG for Spread and Total. FRESH DATA ONLY - No caching, no fallbacks, no placeholders.",
    version=RELEASE_VERSION,
    lifespan=lifespan,
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
    # Hard fail to avoid silently running without CORS configuration
    raise RuntimeError(
        "ALLOWED_ORIGINS is required but not set; set ALLOWED_ORIGINS to a comma-separated list of allowed origins."
    )
allowed_origins = [origin.strip() for origin in allowed_origins_str.split(",") if origin.strip()]
if not allowed_origins:
    raise RuntimeError("ALLOWED_ORIGINS resolved to an empty list; provide at least one origin.")
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


# --- Include Route Modules (incremental migration) ---
# NOTE: Routes are duplicated during migration. New routes take precedence.
# Once verified, remove the inline endpoint definitions below.
app.include_router(health_router)  # ✓ ENABLED - health, metrics, verify
app.include_router(admin_router)  # ✓ ENABLED - admin/monitoring, admin/cache
app.include_router(meta_router)  # ✓ ENABLED - meta, registry, markets
app.include_router(
    tracking_router
)  # ✓ ENABLED - tracking/summary, tracking/picks, tracking/validate
app.include_router(teams_router)  # ✓ ENABLED - teams/workflow, teams/outgoing


# --- Endpoints (legacy - will be migrated to routes/) ---

# /health, /metrics, /verify -> MIGRATED to src/serving/routes/health.py


@app.get("/", tags=["Website"])
@limiter.limit("60/minute")
async def root_picks(
    request: Request,
    date: str = Query("today", description="Date in YYYY-MM-DD format, 'today', or 'tomorrow'"),
    tier: str = Query("all", description="Filter by tier: 'elite', 'strong', 'good', or 'all'"),
):
    """
    Root endpoint returning weekly NBA picks.

    Alias for /weekly-lineup/nba - designed for simplified URL convention:
    www.greenbiersportventures.com/prediction/nba

    Query params:
      - date: YYYY-MM-DD, 'today', or 'tomorrow' (default: today)
      - tier: 'elite', 'strong', 'good', or 'all' (default: all)
    """
    try:
        result = await _build_weekly_lineup_payload(request, date=date, tier=tier)
        if not isinstance(result, dict) or result.get("error"):
            return JSONResponse(
                content=(
                    result
                    if isinstance(result, dict)
                    else {"sport": "NBA", "error": "Failed to fetch predictions", "date": date}
                ),
                status_code=500,
            )
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error in root picks endpoint: {str(e)}", exc_info=True)
        return JSONResponse(
            content={"sport": "NBA", "error": str(e), "date": date}, status_code=500
        )


# /metrics -> MIGRATED to src/serving/routes/health.py

# /admin/monitoring -> MIGRATED to src/serving/routes/admin.py
# /admin/monitoring/reset -> MIGRATED to src/serving/routes/admin.py
# /admin/cache/clear -> MIGRATED to src/serving/routes/admin.py
# /admin/cache/stats -> MIGRATED to src/serving/routes/admin.py

# /verify -> MIGRATED to src/serving/routes/health.py


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

    DATE ALIGNMENT: All data sources (The Odds API, ESPN, API-Basketball, Action Network)
    are queried with the same target date to ensure consistency.
    """
    if not hasattr(app.state, "engine") or app.state.engine is None:
        raise HTTPException(
            status_code=503, detail=f"{RELEASE_VERSION}: Engine not loaded - models missing"
        )
    use_splits = require_splits_if_strict(use_splits)

    # STRICT MODE: Clear ALL caches to force fresh unified data
    app.state.feature_builder.clear_session_cache()
    clear_unified_records_cache()  # QA/QC: Ensure records cache is fresh
    logger.info(
        f"{RELEASE_VERSION}: Caches cleared - fetching fresh data (odds from The Odds API, records from ESPN when available)"
    )

    # =========================================================================
    # DATE VALIDATION GATE - Single source of truth for date across all sources
    # =========================================================================
    from src.utils.slate_analysis import (
        DateContext,
        extract_consensus_odds,
        fetch_todays_games,
        parse_utc_time,
        to_cst,
        validate_date_alignment,
    )

    try:
        date_ctx = DateContext.from_request(date)
        target_date = date_ctx.target_date
        logger.info(
            f"{RELEASE_VERSION}: Date context created: {date_ctx} "
            f"(ISO={date_ctx.as_iso}, ESPN={date_ctx.as_espn}, UTC_start={date_ctx.as_utc_start})"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Fetch games with DATE ALIGNMENT - pass DateContext for UTC window filtering
    try:
        # Records are sourced from ESPN standings when available (odds remain from The Odds API)
        games = await fetch_todays_games(target_date, include_records=True, date_context=date_ctx)
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

    # Validate date alignment (log warning if games don't match target date)
    date_validation = validate_date_alignment(date_ctx, games)
    if not date_validation["is_valid"]:
        logger.warning(
            f"{RELEASE_VERSION}: DATE ALIGNMENT: {date_validation['games_off_date']} games "
            f"not on target date {date_ctx.as_iso}"
        )

    odds_as_of_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    odds_snapshot_path = None
    odds_archive_path = None
    try:
        from src.ingestion.the_odds import save_odds

        odds_snapshot_path = await save_odds(
            games, prefix=f"slate_odds_{target_date.strftime('%Y%m%d')}"
        )
        archive_dir = PROJECT_ROOT / "archive" / "odds_snapshots"
        odds_archive_path = await save_odds(
            games,
            out_dir=str(archive_dir),
            prefix=f"slate_odds_{target_date.strftime('%Y%m%d')}",
        )
    except Exception as e:
        logger.warning(f"Could not save odds snapshot: {e}")

    # Fetch splits with DATE ALIGNMENT - ensure splits match the target slate date
    splits_dict = {}
    if use_splits:
        splits_dict = await fetch_required_splits(games, target_date=date_ctx.as_iso)

    # Process slate
    results = []
    errors: List[str] = []
    total_plays = 0

    for game in games:
        home_team = game["home_team"]
        away_team = game["away_team"]

        try:
            # Build features
            game_key = canonical_game_key(home_team, away_team, source="odds")
            features = await app.state.feature_builder.build_game_features(
                home_team,
                away_team,
                betting_splits=splits_dict.get(game_key),
            )

            # UNIFIED SOURCE: records are attached to the game object by fetch_todays_games(include_records=True)
            home_record_data = game.get("home_team_record") or {}
            away_record_data = game.get("away_team_record") or {}

            features["home_wins"] = int(home_record_data.get("wins", 0) or 0)
            features["home_losses"] = int(home_record_data.get("losses", 0) or 0)
            features["away_wins"] = int(away_record_data.get("wins", 0) or 0)
            features["away_losses"] = int(away_record_data.get("losses", 0) or 0)
            features["_records_source"] = home_record_data.get("source", "unknown")
            features["_data_unified"] = bool(game.get("_data_unified", False))

            # Extract consensus lines
            odds = extract_consensus_odds(game, as_of_utc=odds_as_of_utc)
            fg_spread = odds.get("home_spread")
            fg_total = odds.get("total")
            fh_spread = odds.get("fh_home_spread")
            fh_total = odds.get("fh_total")
            missing_lines = missing_market_lines(fg_spread, fg_total, fh_spread, fh_total)
            if missing_lines:
                msg = (
                    f"{home_team} vs {away_team}: missing required market lines " f"{missing_lines}"
                )
                logger.warning(f"{RELEASE_VERSION}: {msg}")
                errors.append(msg)
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

            commence_time = game.get("commence_time")
            commence_time_cst = game.get("commence_time_cst")
            if not commence_time_cst and commence_time:
                try:
                    dt_utc = parse_utc_time(commence_time)
                    dt_cst = to_cst(dt_utc) if dt_utc else None
                    commence_time_cst = dt_cst.isoformat() if dt_cst else None
                except Exception:
                    commence_time_cst = None

            results.append(
                {
                    "matchup": f"{away_team} @ {home_team}",
                    "home_team": home_team,
                    "away_team": away_team,
                    "commence_time": commence_time,
                    "commence_time_cst": commence_time_cst,
                    "date": game.get("date"),
                    "predictions": preds,
                    "has_plays": game_plays > 0,
                }
            )

        except MissingFeaturesError as e:
            msg = f"{home_team} vs {away_team}: {e}"
            logger.warning(f"{RELEASE_VERSION}: {msg}")
            errors.append(msg)
            continue
        except ValueError as e:
            msg = f"{home_team} vs {away_team}: {e}"
            logger.warning(f"{RELEASE_VERSION}: Skipping {msg}")
            errors.append(msg)
            continue
        except Exception as e:
            msg = f"{home_team} vs {away_team}: {e}"
            logger.error(f"Error processing {msg}")
            errors.append(msg)
            continue

    return SlateResponse(
        date=str(target_date),
        predictions=results,
        total_plays=total_plays,
        odds_as_of_utc=odds_as_of_utc,
        odds_snapshot_path=odds_snapshot_path,
        odds_archive_path=odds_archive_path,
        errors=errors or None,
    )


# format_american_odds is imported from src.serving.dependencies
# missing_market_lines is imported from src.serving.dependencies
# get_fire_rating is imported from src.serving.dependencies


@app.get("/slate/{date}/executive")
@limiter.limit("30/minute")
async def get_executive_summary(
    request: Request,
    date: str = Path(..., description="Date in YYYY-MM-DD format, 'today', or 'tomorrow'"),
    use_splits: bool = Query(True, description="Whether to fetch and use betting splits"),
):
    """
    BLUF (Bottom Line Up Front) Executive Summary.

    STRICT MODE: Fetches fresh data from all APIs.
    Returns a clean actionable betting card with all picks that pass filters.
    Sorted by EV% (desc), then game time.
    """
    if not hasattr(app.state, "engine") or app.state.engine is None:
        raise HTTPException(
            status_code=503,
            detail=f"{RELEASE_VERSION} STRICT MODE: Engine not loaded - models missing",
        )
    use_splits = require_splits_if_strict(use_splits)

    # STRICT MODE: Clear ALL caches to force fresh unified data
    app.state.feature_builder.clear_session_cache()
    clear_unified_records_cache()  # QA/QC: Ensure records cache is fresh
    logger.info(f"{RELEASE_VERSION} STRICT MODE: Executive summary - fetching fresh unified data")

    from zoneinfo import ZoneInfo

    from src.utils.odds import (
        american_to_implied_prob,
        devig_two_way,
        expected_value,
        kelly_fraction,
    )
    from src.utils.slate_analysis import (
        DateContext,
        extract_consensus_odds,
        fetch_todays_games,
        parse_utc_time,
        to_cst,
        validate_date_alignment,
    )

    CST = ZoneInfo("America/Chicago")

    # DATE VALIDATION GATE - Single source of truth for date across all sources
    try:
        date_ctx = DateContext.from_request(date)
        target_date = date_ctx.target_date
        logger.info(
            f"{RELEASE_VERSION}: Executive - date context: {date_ctx} "
            f"(ISO={date_ctx.as_iso}, ESPN={date_ctx.as_espn})"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Fetch games with DATE ALIGNMENT - pass DateContext for UTC window filtering
    games = await fetch_todays_games(target_date, include_records=True, date_context=date_ctx)
    if not games:
        return {
            "date": str(target_date),
            "generated_at": datetime.now(CST).strftime("%Y-%m-%d %I:%M %p CST"),
            "version": RELEASE_VERSION,
            "total_plays": 0,
            "plays": [],
            "summary": "No games scheduled",
        }

    # Validate date alignment
    date_validation = validate_date_alignment(date_ctx, games)
    if not date_validation["is_valid"]:
        logger.warning(
            f"{RELEASE_VERSION}: Executive DATE ALIGNMENT: {date_validation['games_off_date']} games "
            f"not on target date {date_ctx.as_iso}"
        )

    odds_as_of_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    odds_snapshot_path = None
    odds_archive_path = None
    try:
        from src.ingestion.the_odds import save_odds

        odds_snapshot_path = await save_odds(
            games, prefix=f"slate_odds_{target_date.strftime('%Y%m%d')}"
        )
        archive_dir = PROJECT_ROOT / "archive" / "odds_snapshots"
        odds_archive_path = await save_odds(
            games,
            out_dir=str(archive_dir),
            prefix=f"slate_odds_{target_date.strftime('%Y%m%d')}",
        )
    except Exception as e:
        logger.warning(f"Could not save odds snapshot: {e}")

    # Fetch betting splits with DATE ALIGNMENT
    splits_dict = {}
    if use_splits:
        try:
            splits_dict = await fetch_required_splits(games, target_date=date_ctx.as_iso)
        except Exception as e:
            logger.warning(f"Could not fetch splits: {e}")

    def _pick_ev_fields(
        p_model: float | None,
        pick_odds: int | None,
        odds_a: int | None,
        odds_b: int | None,
        pick_is_a: bool,
    ) -> tuple[float | None, float | None, float | None]:
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
    errors: List[str] = []

    for game in games:
        home_team = game.get("home_team")
        away_team = game.get("away_team")
        commence_time = game.get("commence_time")
        commence_time_cst = game.get("commence_time_cst")

        try:
            # Parse time (prefer standardized CST time if present)
            game_cst = None
            if commence_time_cst:
                try:
                    game_cst = datetime.fromisoformat(commence_time_cst)
                    if game_cst.tzinfo is None:
                        game_cst = game_cst.replace(tzinfo=ZoneInfo("America/Chicago"))
                except Exception:
                    game_cst = None
            if game_cst is None:
                game_dt = parse_utc_time(commence_time) if commence_time else None
                game_cst = to_cst(game_dt) if game_dt else None
            time_cst_str = game_cst.strftime("%m/%d %I:%M %p") if game_cst else "TBD"

            # Build features
            game_key = canonical_game_key(home_team, away_team, source="odds")
            features = await app.state.feature_builder.build_game_features(
                home_team,
                away_team,
                betting_splits=splits_dict.get(game_key),
            )

            home_record_data = game.get("home_team_record") or {}
            away_record_data = game.get("away_team_record") or {}

            home_wins = int(home_record_data.get("wins", 0) or 0)
            home_losses = int(home_record_data.get("losses", 0) or 0)
            away_wins = int(away_record_data.get("wins", 0) or 0)
            away_losses = int(away_record_data.get("losses", 0) or 0)

            # Document data source in features for audit trail
            features["home_wins"] = home_wins
            features["home_losses"] = home_losses
            features["away_wins"] = away_wins
            features["away_losses"] = away_losses
            features["_records_source"] = home_record_data.get("source", "unknown")
            features["_data_unified"] = bool(game.get("_data_unified", False))

            home_record = f"({home_wins}-{home_losses})"
            away_record = f"({away_wins}-{away_losses})"
            matchup_display = f"{away_team} {away_record} @ {home_team} {home_record}"

            # Extract odds
            odds = extract_consensus_odds(game, as_of_utc=odds_as_of_utc)
            fg_spread = odds.get("home_spread")
            fg_total = odds.get("total")
            fh_spread = odds.get("fh_home_spread")
            fh_total = odds.get("fh_total")

            missing_lines = missing_market_lines(fg_spread, fg_total, fh_spread, fh_total)
            if missing_lines:
                msg = (
                    f"{home_team} vs {away_team}: missing required market lines " f"{missing_lines}"
                )
                logger.warning(f"{RELEASE_VERSION}: {msg}")
                errors.append(msg)
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
            if fg_spread is not None and fg_spread_pred.get("passes_filter"):
                bet_side = fg_spread_pred.get("bet_side", "home")
                pick_team = home_team if bet_side == "home" else away_team
                pick_line = fg_spread if bet_side == "home" else -fg_spread
                pick_price = (
                    odds.get("home_spread_price")
                    if bet_side == "home"
                    else odds.get("away_spread_price")
                )
                if pick_price is None:
                    pick_price = odds.get("home_spread_price")
                p_model = (
                    fg_spread_pred.get("home_cover_prob")
                    if bet_side == "home"
                    else fg_spread_pred.get("away_cover_prob")
                )
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

                all_plays.append(
                    {
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
                        "edge_raw": abs(fg_spread_pred.get("edge", 0)),
                        "confidence": fg_spread_pred.get("confidence", 0),
                        "p_model": p_model,
                        "p_fair": p_fair,
                        "ev_pct": ev_pct,
                        "kelly_fraction": kelly,
                        "odds_as_of_utc": odds_as_of_utc,
                        "fire_rating": get_fire_rating(
                            fg_spread_pred.get("confidence", 0), fg_spread_pred.get("edge", 0)
                        ),
                    }
                )

            # FG Total
            fg_total_pred = fg.get("total", {})
            if fg_total is not None and fg_total_pred.get("passes_filter"):
                bet_side = fg_total_pred.get("bet_side", "over")
                model_total = features.get("predicted_total", 0)
                pick_display = f"{'OVER' if bet_side == 'over' else 'UNDER'} {fg_total}"
                pick_price = (
                    odds.get("total_over_price")
                    if bet_side == "over"
                    else odds.get("total_under_price")
                )
                if pick_price is None:
                    pick_price = odds.get("total_price")
                p_model = (
                    fg_total_pred.get("over_prob")
                    if bet_side == "over"
                    else fg_total_pred.get("under_prob")
                )
                p_fair, ev_pct, kelly = _pick_ev_fields(
                    p_model,
                    pick_price,
                    odds.get("total_over_price"),
                    odds.get("total_under_price"),
                    bet_side == "over",
                )

                all_plays.append(
                    {
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
                        "edge_raw": abs(fg_total_pred.get("edge", 0)),
                        "confidence": fg_total_pred.get("confidence", 0),
                        "p_model": p_model,
                        "p_fair": p_fair,
                        "ev_pct": ev_pct,
                        "kelly_fraction": kelly,
                        "odds_as_of_utc": odds_as_of_utc,
                        "fire_rating": get_fire_rating(
                            fg_total_pred.get("confidence", 0), fg_total_pred.get("edge", 0)
                        ),
                    }
                )

            # Process First Half markets
            fh = preds.get("first_half", {})

            # 1H Spread
            fh_spread_pred = fh.get("spread", {})
            if fh_spread_pred.get("passes_filter") and fh_spread is not None:
                bet_side = fh_spread_pred.get("bet_side", "home")
                pick_team = home_team if bet_side == "home" else away_team
                pick_line = fh_spread if bet_side == "home" else -fh_spread
                pick_price = (
                    odds.get("fh_home_spread_price")
                    if bet_side == "home"
                    else odds.get("fh_away_spread_price")
                )
                if pick_price is None:
                    pick_price = odds.get("fh_home_spread_price")
                p_model = (
                    fh_spread_pred.get("home_cover_prob")
                    if bet_side == "home"
                    else fh_spread_pred.get("away_cover_prob")
                )
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

                all_plays.append(
                    {
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
                        "edge_raw": abs(fh_spread_pred.get("edge", 0)),
                        "confidence": fh_spread_pred.get("confidence", 0),
                        "p_model": p_model,
                        "p_fair": p_fair,
                        "ev_pct": ev_pct,
                        "kelly_fraction": kelly,
                        "odds_as_of_utc": odds_as_of_utc,
                        "fire_rating": get_fire_rating(
                            fh_spread_pred.get("confidence", 0), fh_spread_pred.get("edge", 0)
                        ),
                    }
                )

            # 1H Total
            fh_total_pred = fh.get("total", {})
            if fh_total_pred.get("passes_filter") and fh_total is not None:
                bet_side = fh_total_pred.get("bet_side", "over")
                model_total_1h = features.get("predicted_total_1h", 0)
                pick_display = f"{'OVER' if bet_side == 'over' else 'UNDER'} {fh_total}"
                pick_price = (
                    odds.get("fh_total_over_price")
                    if bet_side == "over"
                    else odds.get("fh_total_under_price")
                )
                if pick_price is None:
                    pick_price = odds.get("fh_total_price")
                p_model = (
                    fh_total_pred.get("over_prob")
                    if bet_side == "over"
                    else fh_total_pred.get("under_prob")
                )
                p_fair, ev_pct, kelly = _pick_ev_fields(
                    p_model,
                    pick_price,
                    odds.get("fh_total_over_price"),
                    odds.get("fh_total_under_price"),
                    bet_side == "over",
                )

                all_plays.append(
                    {
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
                        "edge_raw": abs(fh_total_pred.get("edge", 0)),
                        "confidence": fh_total_pred.get("confidence", 0),
                        "p_model": p_model,
                        "p_fair": p_fair,
                        "ev_pct": ev_pct,
                        "kelly_fraction": kelly,
                        "odds_as_of_utc": odds_as_of_utc,
                        "fire_rating": get_fire_rating(
                            fh_total_pred.get("confidence", 0), fh_total_pred.get("edge", 0)
                        ),
                    }
                )

        except MissingFeaturesError as e:
            msg = f"{home_team} vs {away_team}: {e}"
            logger.warning(f"{RELEASE_VERSION}: {msg}")
            errors.append(msg)
            continue
        except ValueError as e:
            msg = f"{home_team} vs {away_team}: {e}"
            logger.warning(f"{RELEASE_VERSION}: Skipping {msg}")
            errors.append(msg)
            continue
        except Exception as e:
            msg = f"{home_team} vs {away_team}: {e}"
            logger.error(f"Error processing {msg}")
            errors.append(msg)
            continue

    # Sort by EV%, then by time (descending EV is primary)
    def _ev_value(play: Dict[str, Any]) -> float:
        ev = play.get("ev_pct")
        return float(ev) if ev is not None else -9999.0

    all_plays.sort(key=lambda x: (-_ev_value(x), x["sort_time"]))

    # Format for output - remove internal fields
    formatted_plays = []
    for play in all_plays:
        formatted_plays.append(
            {
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
                "fire_rating": play["fire_rating"],
            }
        )

    return convert_numpy_types(
        {
            "date": str(target_date),
            "generated_at": datetime.now(CST).strftime("%Y-%m-%d %I:%M %p CST"),
            "version": RELEASE_VERSION,
            "total_plays": len(formatted_plays),
            "plays": formatted_plays,
            "errors": errors or None,
            "odds_as_of_utc": odds_as_of_utc,
            "odds_snapshot_path": odds_snapshot_path,
            "odds_archive_path": odds_archive_path,
            "legend": {
                "fire_rating": {
                    "ELITE": "70%+ confidence AND 5+ pt edge",
                    "STRONG": "60%+ confidence AND 3+ pt edge",
                    "GOOD": "Passes all filters",
                },
                "sorting": "EV% (desc), then game time",
                "periods": {"FG": "Full Game", "1H": "First Half"},
                "markets": {"SPREAD": "Point Spread", "TOTAL": "Over/Under"},
            },
        }
    )


# =============================================================================
# TEAMS ROUTES - MIGRATED to src/serving/routes/teams.py
# =============================================================================
# Wire up the executive summary function for Teams routes to avoid circular imports
set_executive_summary_func(get_executive_summary)

# /teams/workflow, /teams/outgoing -> MIGRATED to src/serving/routes/teams.py


@app.get("/slate/{date}/comprehensive")
@limiter.limit("20/minute")
async def get_comprehensive_slate_analysis(
    request: Request,
    date: str = Path(..., description="Date in YYYY-MM-DD format, 'today', or 'tomorrow'"),
    use_splits: bool = Query(True, description="Whether to fetch and use betting splits"),
):
    """
    Get comprehensive slate analysis with full edge calculations.

    STRICT MODE: Fetches fresh data from all APIs.
    Full analysis for all 4 markets (1H + FG spreads/totals).
    """
    if not hasattr(app.state, "engine") or app.state.engine is None:
        raise HTTPException(
            status_code=503,
            detail=f"{RELEASE_VERSION} STRICT MODE: Engine not loaded - models missing",
        )
    use_splits = require_splits_if_strict(use_splits)

    # STRICT MODE: Clear ALL caches to force fresh unified data
    app.state.feature_builder.clear_session_cache()
    app.state.feature_builder.clear_persistent_cache()
    clear_unified_records_cache()  # QA/QC: Ensure records cache is fresh
    logger.info(
        f"{RELEASE_VERSION} STRICT MODE: Comprehensive analysis - fetching fresh unified data"
    )

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

    # DATE VALIDATION GATE - Single source of truth for date across all sources
    from src.utils.slate_analysis import DateContext, validate_date_alignment

    try:
        date_ctx = DateContext.from_request(date)
        target_date = date_ctx.target_date
        logger.info(
            f"{RELEASE_VERSION}: Comprehensive - date context: {date_ctx} "
            f"(ISO={date_ctx.as_iso}, ESPN={date_ctx.as_espn})"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Get edge thresholds - use SAME source as engine for consistency
    try:
        from src.config import filter_thresholds

        edge_thresholds = {
            "spread": filter_thresholds.spread_min_edge,
            "total": filter_thresholds.total_min_edge,
            "1h_spread": filter_thresholds.spread_min_edge * 0.75,  # Scale for 1H
            "1h_total": filter_thresholds.total_min_edge * 0.67,  # Scale for 1H
        }
    except ImportError:
        # Fallback if config not available
        edge_thresholds = {
            "spread": 2.0,
            "total": 3.0,
            "1h_spread": 1.5,
            "1h_total": 2.0,
        }

    # Fetch games with DATE ALIGNMENT - pass DateContext for UTC window filtering
    games = await fetch_todays_games(target_date, include_records=True, date_context=date_ctx)
    if not games:
        return {"date": str(target_date), "analysis": [], "summary": "No games found"}

    # Validate date alignment
    date_validation = validate_date_alignment(date_ctx, games)
    if not date_validation["is_valid"]:
        logger.warning(
            f"{RELEASE_VERSION}: Comprehensive DATE ALIGNMENT: {date_validation['games_off_date']} games "
            f"not on target date {date_ctx.as_iso}"
        )

    odds_as_of_utc = request_started_utc
    odds_snapshot_path = None
    odds_archive_path = None
    try:
        from src.ingestion.the_odds import save_odds

        odds_snapshot_path = await save_odds(
            games, prefix=f"slate_odds_{target_date.strftime('%Y%m%d')}"
        )
        archive_dir = PROJECT_ROOT / "archive" / "odds_snapshots"
        odds_archive_path = await save_odds(
            games,
            out_dir=str(archive_dir),
            prefix=f"slate_odds_{target_date.strftime('%Y%m%d')}",
        )
    except Exception as e:
        logger.warning(f"Could not save odds snapshot: {e}")

    # Fetch betting splits with DATE ALIGNMENT
    splits_dict = {}
    if use_splits:
        try:
            splits_dict = await fetch_required_splits(games, target_date=date_ctx.as_iso)
        except Exception as e:
            logger.warning(f"Could not fetch splits: {e}")

    # Process each game
    analysis_results = []

    for game in games:
        home_team = game.get("home_team")
        away_team = game.get("away_team")
        commence_time = game.get("commence_time")
        commence_time_cst = game.get("commence_time_cst")

        try:
            # Parse time (prefer standardized CST time if present)
            game_cst = None
            if commence_time_cst:
                try:
                    game_cst = datetime.fromisoformat(commence_time_cst)
                    if game_cst.tzinfo is None:
                        game_cst = game_cst.replace(tzinfo=ZoneInfo("America/Chicago"))
                except Exception:
                    game_cst = None
            if game_cst is None:
                game_dt = parse_utc_time(commence_time) if commence_time else None
                game_cst = to_cst(game_dt) if game_dt else None
            time_cst = game_cst.strftime("%I:%M %p %Z") if game_cst else "TBD"

            # Build features
            game_key = canonical_game_key(home_team, away_team, source="odds")
            features = await app.state.feature_builder.build_game_features(
                home_team, away_team, betting_splits=splits_dict.get(game_key)
            )

            home_record_data = game.get("home_team_record") or {}
            away_record_data = game.get("away_team_record") or {}

            features["home_wins"] = int(home_record_data.get("wins", 0) or 0)
            features["home_losses"] = int(home_record_data.get("losses", 0) or 0)
            features["away_wins"] = int(away_record_data.get("wins", 0) or 0)
            features["away_losses"] = int(away_record_data.get("losses", 0) or 0)
            features["_records_source"] = home_record_data.get("source", "unknown")
            features["_data_unified"] = bool(game.get("_data_unified", False))

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
            missing_lines = missing_market_lines(fg_spread, fg_total, fh_spread, fh_total)
            if missing_lines:
                logger.warning(
                    f"{RELEASE_VERSION}: {home_team} vs {away_team} missing required market lines "
                    f"{missing_lines} - skipping engine predictions"
                )
            else:
                try:
                    engine_predictions = app.state.engine.predict_all_markets(
                        features,
                        fg_spread_line=fg_spread,
                        fg_total_line=fg_total,
                        fh_spread_line=fh_spread,
                        fh_total_line=fh_total,
                    )
                except Exception as e:
                    logger.warning(
                        f"Could not get engine predictions for {home_team} vs {away_team}: {e}"
                    )

            # Calculate comprehensive edge
            comprehensive_edge = calculate_comprehensive_edge(
                features=features,
                fh_features=fh_features,
                odds=odds,
                game=game,
                betting_splits=betting_splits,
                edge_thresholds=edge_thresholds,
                engine_predictions=engine_predictions,
            )

            analysis_results.append(
                {
                    "home_team": home_team,
                    "away_team": away_team,
                    "time_cst": time_cst,
                    "commence_time": commence_time,
                    "odds": odds,
                    "features": features,
                    "comprehensive_edge": comprehensive_edge,
                    "betting_splits": betting_splits,
                }
            )

        except Exception as e:
            logger.error(f"Error processing {home_team} vs {away_team}: {e}")
            continue

    return convert_numpy_types(
        {
            "date": str(target_date),
            "version": RELEASE_VERSION,
            "data_fetched_at_cst": request_started_cst,
            "markets": ["1h_spread", "1h_total", "fg_spread", "fg_total"],
            "analysis": analysis_results,
            "edge_thresholds": edge_thresholds,
            "odds_as_of_utc": odds_as_of_utc,
            "odds_snapshot_path": odds_snapshot_path,
            "odds_archive_path": odds_archive_path,
        }
    )


# /meta -> MIGRATED to src/serving/routes/admin.py (meta_router)
# /registry -> MIGRATED to src/serving/routes/admin.py (meta_router)
# /markets -> MIGRATED to src/serving/routes/admin.py (meta_router)


@app.post("/predict/game", response_model=GamePredictions)
@limiter.limit("60/minute")
async def predict_single_game(request: Request, req: GamePredictionRequest):
    """
    Generate predictions for a specific matchup.

    STRICT MODE - Fetches fresh data from all APIs.
    4 markets (1H+FG for Spread, Total).
    """
    if not hasattr(app.state, "engine") or app.state.engine is None:
        raise HTTPException(
            status_code=503,
            detail=f"{RELEASE_VERSION} STRICT MODE: Engine not loaded - models missing",
        )

    # STRICT MODE: Clear session cache to force fresh data
    app.state.feature_builder.clear_session_cache()

    try:
        betting_splits = None
        if bool(getattr(settings, "require_action_network_splits", False)) or bool(
            getattr(settings, "require_real_splits", False)
        ):
            # Use today's date for single game prediction
            from src.utils.slate_analysis import DateContext

            date_ctx = DateContext.from_request("today")
            games = [{"home_team": req.home_team, "away_team": req.away_team}]
            splits_dict = await fetch_required_splits(games, target_date=date_ctx.as_iso)
            game_key = canonical_game_key(req.home_team, req.away_team, source="odds")
            betting_splits = splits_dict.get(game_key)

        # Build features from FRESH data
        features = await app.state.feature_builder.build_game_features(
            req.home_team,
            req.away_team,
            betting_splits=betting_splits,
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


# /tracking/summary, /tracking/picks, /tracking/validate -> MIGRATED to src/serving/routes/tracking.py


@app.get("/picks/html", response_class=HTMLResponse, tags=["Display"])
@limiter.limit("20/minute")
async def get_picks_html(
    request: Request,
    date: Optional[str] = Query(None, description="Date (YYYY-MM-DD), defaults to today"),
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
            return (
                """
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
                    <p>No games scheduled for """
                + date
                + """</p>
                </div>
            </body>
            </html>
            """
            )

        # Build picks array for card
        picks_data = []
        for pick in summary.get("plays", []):
            market_line = pick.get("market_line", "N/A")
            edge = pick.get("edge", "N/A")
            confidence = pick.get("confidence", "50%")
            # Handle confidence as string percentage like "91%"
            if isinstance(confidence, str) and confidence.endswith("%"):
                confidence_pct = int(confidence.rstrip("%"))
            else:
                confidence_pct = int(float(confidence) * 100) if confidence else 50
            fire_rating = pick.get("fire_rating", 0)
            # Ensure fire_rating is an int
            if isinstance(fire_rating, str):
                fire_rating = int(fire_rating) if fire_rating.isdigit() else 0
            fire_emoji = "🔥" * int(fire_rating) if fire_rating else ""
            picks_data.append(
                {
                    "period": pick.get("period", "FG"),
                    "matchup": pick.get("matchup", "TBD"),
                    "pick": pick.get("pick", "N/A"),
                    "confidence": confidence_pct,
                    "market_line": market_line,
                    "edge": edge,
                    "fire": fire_emoji,
                }
            )

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
                            "color": "light",
                        }
                    ],
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
                        {"width": "20%"},
                    ],
                    "rows": [
                        {
                            "cells": [
                                {"text": "Period"},
                                {"text": "Matchup"},
                                {"text": "Pick (Confidence)"},
                                {"text": "Market"},
                                {"text": "Edge"},
                            ]
                        }
                    ]
                    + [
                        {
                            "cells": [
                                {"text": p["period"]},
                                {"text": p["matchup"]},
                                {"text": p["pick"] + " (" + str(p["confidence"]) + "%)"},
                                {"text": p["market_line"]},
                                {"text": p["edge"] + " " + p["fire"]},
                            ]
                        }
                        for p in picks_data
                    ],
                },
            ],
            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        }

        card_json_str = json.dumps(card)

        html_content = (
            """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NBA Picks - """
            + date
            + """</title>
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
            <h1>🏀 NBA PICKS</h1>
            <p>"""
            + date
            + """</p>
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
        const cardData = """
            + card_json_str
            + """;

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
            link.download = 'nba_picks_"""
            + date
            + """.json';
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
        )

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
    elite: bool = Query(False, description="Elite picks only"),
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
    elite: bool = Query(False, description="Elite picks only"),
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
        writer.writerow(
            [
                "Time (CST)",
                "Matchup",
                "Segment",
                "Pick",
                "Model",
                "Market",
                "Edge",
                "Confidence",
                "Fire Rating",
            ]
        )
        for p in plays:
            writer.writerow(
                [
                    p.get("time_cst", ""),
                    p.get("matchup", ""),
                    p.get("period", ""),
                    p.get("pick", ""),
                    p.get("model_prediction", ""),
                    p.get("market_line", ""),
                    p.get("edge", ""),
                    p.get("confidence", ""),
                    p.get("fire_rating", ""),
                ]
            )

        filename = f"NBA_Picks_{date}.csv"
        return Response(
            content=buffer.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )
    except Exception as e:
        logger.error(f"Error generating weekly lineup CSV: {str(e)}", exc_info=True)
        return Response(content="Failed to generate CSV", media_type="text/plain", status_code=500)


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
    request: Request, date: str, tier: str = "all", team: Optional[str] = None, elite: bool = False
) -> dict:
    if date == "tomorrow":
        resolved_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    elif date == "today":
        resolved_date = datetime.now().strftime("%Y-%m-%d")
    else:
        resolved_date = date

    data = await get_executive_summary(request, date=resolved_date, use_splits=True)

    if not data or not isinstance(data, dict):
        return {"sport": "NBA", "error": "Failed to fetch predictions", "date": resolved_date}

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
        plays = [
            p for p in plays if _get_fire_tier(p.get("fire_rating", "")) in ["ELITE", "STRONG"]
        ]
    elif tier_filter == "good":
        plays = [
            p
            for p in plays
            if _get_fire_tier(p.get("fire_rating", "")) in ["ELITE", "STRONG", "GOOD"]
        ]

    elite_count = len([p for p in all_plays if _get_fire_tier(p.get("fire_rating", "")) == "ELITE"])
    strong_count = len(
        [p for p in all_plays if _get_fire_tier(p.get("fire_rating", "")) == "STRONG"]
    )
    good_count = len([p for p in all_plays if _get_fire_tier(p.get("fire_rating", "")) == "GOOD"])

    formatted_picks = []
    for p in plays:
        formatted_picks.append(
            {
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
                "fire_rating": p.get("fire_rating", ""),
            }
        )

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
            "filtered": len(formatted_picks),
        },
        "picks": formatted_picks,
    }


@app.get("/weekly-lineup/nba", tags=["Website"])
@limiter.limit("60/minute")
async def get_weekly_lineup_nba_json(
    request: Request,
    date: str = Query("today", description="Date in YYYY-MM-DD format, 'today', or 'tomorrow'"),
    tier: str = Query("all", description="Filter by tier: 'elite', 'strong', 'good', or 'all'"),
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
        "version": "v33.0.15.0",
        "summary": {"total": 12, "elite": 3, "strong": 5, "good": 4},
        "picks": [...]
    }
    """
    try:
        result = await _build_weekly_lineup_payload(request, date=date, tier=tier)
        if not isinstance(result, dict) or result.get("error"):
            return JSONResponse(
                content=(
                    result
                    if isinstance(result, dict)
                    else {"sport": "NBA", "error": "Failed to fetch predictions", "date": date}
                ),
                status_code=500,
            )
        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Error in weekly-lineup/nba: {str(e)}", exc_info=True)
        return JSONResponse(
            content={"sport": "NBA", "error": str(e), "date": date}, status_code=500
        )


@app.get("/v1/picks", tags=["Website"])
@limiter.limit("60/minute")
async def get_v1_picks(
    request: Request,
    date: str = Query("today", description="Date in YYYY-MM-DD format, 'today', or 'tomorrow'"),
    tier: str = Query("all", description="Filter by tier: 'elite', 'strong', 'good', or 'all'"),
    team: Optional[str] = Query(None, description="Filter by team name or abbreviation"),
    elite: bool = Query(False, description="Elite picks only (overrides tier)"),
):
    """
    Legacy compatibility endpoint for clients expecting /api/v1/picks.
    """
    try:
        result = await _build_weekly_lineup_payload(
            request, date=date, tier=tier, team=team, elite=elite
        )
        if not isinstance(result, dict) or result.get("error"):
            return JSONResponse(
                content=(
                    result
                    if isinstance(result, dict)
                    else {"sport": "NBA", "error": "Failed to fetch predictions", "date": date}
                ),
                status_code=500,
            )

        result = dict(result)
        result["api_version"] = "v1"
        if team or elite or (tier and tier.lower().strip() != "all"):
            result["filters"] = {"date": date, "tier": tier, "team": team, "elite": elite}
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error in v1/picks: {str(e)}", exc_info=True)
        return JSONResponse(
            content={"sport": "NBA", "error": str(e), "date": date}, status_code=500
        )
