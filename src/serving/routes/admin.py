"""
Admin routes for monitoring, cache management, and metadata.
"""

import os
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Request

from src.config import settings
from src.serving.dependencies import RELEASE_VERSION, logger
from src.utils.markets import get_market_catalog

# Get project root for market catalog
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

router = APIRouter(prefix="/admin", tags=["Admin"])


@router.get("/monitoring")
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
        stats["signal_agreement"]["recent_disagreements"] = signal_tracker.get_recent_disagreements(
            5
        )
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
                market for market in drift_detector.metrics if drift_detector.is_drifting(market)[0]
            ],
        }
    except Exception as e:
        stats["drift_detection"] = {"error": str(e)}

    # Rate limiter stats
    try:
        from src.utils.rate_limiter import get_api_basketball_limiter, get_odds_api_limiter

        stats["rate_limiters"] = {
            "the_odds_api": get_odds_api_limiter().get_stats(),
            "api_basketball": get_api_basketball_limiter().get_stats(),
        }
    except Exception as e:
        stats["rate_limiters"] = {"error": str(e)}

    # Circuit breaker stats
    try:
        from src.utils.circuit_breaker import get_api_basketball_breaker, get_odds_api_breaker

        stats["circuit_breakers"] = {
            "the_odds_api": get_odds_api_breaker().get_stats(),
            "api_basketball": get_api_basketball_breaker().get_stats(),
        }
    except Exception as e:
        stats["circuit_breakers"] = {"error": str(e)}

    return stats


@router.post("/monitoring/reset")
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


@router.post("/cache/clear")
async def clear_cache(request: Request):
    """
    Clear all session caches to force fresh API data.

    STRICT MODE: No file caching exists - only clears session memory caches.
    Use this before fetching new predictions to ensure fresh data.
    """
    cleared = {"session_cache": False, "api_cache": 0}

    if hasattr(request.app.state, "feature_builder"):
        request.app.state.feature_builder.clear_session_cache()
        cleared["session_cache"] = True

    try:
        from src.utils.api_cache import api_cache

        cleared["api_cache"] = api_cache.clear_all()
    except Exception:
        pass

    logger.info(f"{RELEASE_VERSION} STRICT MODE: Session cache cleared")

    return {
        "status": "success",
        "cleared": cleared,
        "mode": "STRICT",
        "message": "Session caches cleared. Next API call will fetch fresh data from all sources.",
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/cache/stats")
async def get_cache_stats(request: Request):
    """
    Get comprehensive cache statistics and performance metrics.
    """
    stats = {"session_cache": {}, "persistent_cache": {}}

    if hasattr(request.app.state, "feature_builder"):
        stats.update(request.app.state.feature_builder.get_cache_stats())

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
        "stats": stats,
    }


# Meta endpoints (not under /admin prefix, but related)
meta_router = APIRouter(tags=["Meta"])


@meta_router.get("/meta")
def get_meta(request: Request):
    """Get API metadata."""
    import sys

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
        "python_version": sys.version,
    }


@meta_router.get("/registry")
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
        "registry": "/api/registry",
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
        "api": {"v1": {"picks": endpoints["picks_v1"]}},
        "defaults": {"sport": "nba", "date": "today"},
    }


@meta_router.get("/markets")
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
