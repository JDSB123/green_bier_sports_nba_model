"""
Health, metrics, and verification routes.
"""

import os
from datetime import datetime

from fastapi import APIRouter, Request
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from src.config import settings
from src.ingestion.betting_splits import validate_splits_sources_configured
from src.modeling.unified_features import get_feature_defaults
from src.serving.dependencies import RELEASE_VERSION, logger
from src.utils.security import get_api_key_status

router = APIRouter()


@router.get("/health")
def health(request: Request):
    """Check API health - 4 markets (spread/total only)."""
    engine_loaded = hasattr(request.app.state, "engine") and request.app.state.engine is not None
    api_keys = get_api_key_status()
    try:
        splits_sources = validate_splits_sources_configured()
    except Exception as e:
        splits_sources = {"error": str(e)}

    model_info = {}
    if engine_loaded:
        model_info = request.app.state.engine.get_model_info()

    return {
        "status": "ok",
        "version": RELEASE_VERSION,
        "build": {
            "image_tag": os.getenv("NBA_IMAGE_TAG") or os.getenv("GITHUB_SHA") or "unknown",
            "hostname": os.getenv("HOSTNAME") or "unknown",
            "container_app_name": os.getenv("CONTAINER_APP_NAME") or "unknown",
            "container_app_revision": os.getenv("CONTAINER_APP_REVISION") or "unknown",
        },
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
        "betting_splits_sources": splits_sources,
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/verify")
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
        "errors": [],
    }

    # Check 1: Engine loaded
    if not hasattr(request.app.state, "engine") or request.app.state.engine is None:
        results["status"] = "fail"
        results["errors"].append("Engine not loaded")
        results["checks"]["engine_loaded"] = False
    else:
        results["checks"]["engine_loaded"] = True

        # Check 2: Period predictors exist (1H + FG only)
        has_fg = hasattr(request.app.state.engine, "fg_predictor")
        has_1h = hasattr(request.app.state.engine, "h1_predictor")

        results["checks"]["period_predictors"] = {
            "full_game": has_fg,
            "first_half": has_1h,
        }

        has_spread = hasattr(request.app.state.engine, "spread_predictor")
        has_total = hasattr(request.app.state.engine, "total_predictor")
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
            if has_fg and hasattr(request.app.state.engine, "fg_predictor"):
                required_features.update(
                    request.app.state.engine.fg_predictor.spread_features or []
                )
                required_features.update(request.app.state.engine.fg_predictor.total_features or [])
            if has_1h and hasattr(request.app.state.engine, "h1_predictor"):
                required_features.update(
                    request.app.state.engine.h1_predictor.spread_features or []
                )
                required_features.update(request.app.state.engine.h1_predictor.total_features or [])
        except Exception as e:
            logger.warning(f"Unable to read model feature requirements: {e}")

        if not required_features:
            required_features = set(get_feature_defaults().keys())

        defaults = get_feature_defaults()
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
        test_features.setdefault("predicted_margin", overrides["predicted_margin"])
        test_features.setdefault("predicted_total", overrides["predicted_total"])
        test_features.setdefault("predicted_margin_1h", overrides["predicted_margin_1h"])
        test_features.setdefault("predicted_total_1h", overrides["predicted_total_1h"])

        # Check 3: Test 1H prediction
        try:
            test_pred_1h = request.app.state.engine.predict_first_half(
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
            test_pred = request.app.state.engine.predict_full_game(
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
