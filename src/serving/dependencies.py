"""
Shared dependencies for API routes.

This module provides common utilities, state accessors, and helper functions
used across all route modules.
"""

import os
from datetime import datetime, timezone
from pathlib import Path as PathLib
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import HTTPException, Request

from src.config import settings
from src.ingestion.standardize import normalize_team_to_espn
from src.modeling.unified_features import get_feature_defaults
from src.utils.logging import get_logger
from src.utils.version import resolve_version

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


def models_dir() -> PathLib:
    """
    Resolve models directory from a single configured path.

    Single source of truth:
    - settings.models_dir (env: MODELS_DIR)

    Raise if the configured directory does not exist.
    """
    configured = PathLib(getattr(settings, "models_dir", "")).expanduser()
    if not str(configured).strip():
        raise RuntimeError(
            "MODELS_DIR is not configured (settings.models_dir empty). "
            "Set MODELS_DIR to the production model directory."
        )
    if not configured.exists():
        raise RuntimeError(
            f"Models directory does not exist: {configured}. "
            "Ensure the container image includes the model pack at MODELS_DIR."
        )
    return configured


def canonical_game_key(home_team: str | None, away_team: str | None, source: str = "odds") -> str:
    """Build a canonical game key using ESPN-normalized team names."""
    if not home_team or not away_team:
        return ""
    try:
        home_norm, home_valid = normalize_team_to_espn(str(home_team), source=source)
        away_norm, away_valid = normalize_team_to_espn(str(away_team), source=source)
        if home_valid and away_valid:
            return f"{away_norm}@{home_norm}"
    except Exception:
        pass
    return f"{away_team}@{home_team}"


def require_splits_if_strict(use_splits: bool) -> bool:
    """Enforce betting splits when strict live-data flags are enabled."""
    strict = bool(getattr(settings, "require_action_network_splits", False)) or bool(
        getattr(settings, "require_real_splits", False)
    )
    if strict and not use_splits:
        raise HTTPException(
            status_code=400,
            detail="STRICT MODE: Betting splits are required; cannot disable use_splits.",
        )
    return True if strict else use_splits


async def fetch_required_splits(
    games: List[Dict[str, Any]], target_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Fetch betting splits, enforcing strict Action Network requirements when configured.

    Args:
        games: List of game dicts
        target_date: Target date in YYYY-MM-DD format

    Returns:
        Dict mapping game_key to GameSplits
    """
    from src.ingestion.betting_splits import fetch_public_betting_splits

    require_action_network = bool(getattr(settings, "require_action_network_splits", False))
    require_real = bool(getattr(settings, "require_real_splits", False))

    if not (require_action_network or require_real):
        try:
            return await fetch_public_betting_splits(games, source="auto", target_date=target_date)
        except Exception as e:
            logger.warning(f"Could not fetch splits: {e}")
            return {}

    # Strict mode
    try:
        splits = await fetch_public_betting_splits(
            games,
            source="auto",
            require_action_network=require_action_network,
            require_non_empty=True,
            target_date=target_date,
        )
    except Exception as e:
        logger.error(f"STRICT MODE: Failed to fetch required betting splits: {e}")
        raise HTTPException(
            status_code=502,
            detail=f"STRICT MODE: Action Network betting splits unavailable: {e}",
        )

    missing_keys = []
    for g in games:
        home_team = g.get("home_team")
        away_team = g.get("away_team")
        if home_team and away_team:
            game_key = canonical_game_key(home_team, away_team, source="odds")
            if game_key not in splits:
                missing_keys.append(game_key)

    if missing_keys:
        raise HTTPException(
            status_code=502,
            detail=(
                "STRICT MODE: Action Network did not provide betting splits for all games. "
                f"Missing: {missing_keys[:10]}"
            ),
        )

    return splits


def missing_market_lines(
    fg_spread: Optional[float],
    fg_total: Optional[float],
    fh_spread: Optional[float],
    fh_total: Optional[float],
) -> List[str]:
    """Check which market lines are missing."""
    missing = []
    if fg_spread is None:
        missing.append("fg_spread_line")
    if fg_total is None:
        missing.append("fg_total_line")
    if fh_spread is None:
        missing.append("1h_spread_line")
    if fh_total is None:
        missing.append("1h_total_line")
    return missing


def get_fire_rating(confidence: float, edge: float) -> int:
    """Calculate fire rating (1-5) based on confidence and edge."""
    edge_norm = min(abs(edge) / 10.0, 1.0)
    combined_score = (confidence * 0.6) + (edge_norm * 0.4)

    if combined_score >= 0.85:
        return 5
    elif combined_score >= 0.70:
        return 4
    elif combined_score >= 0.60:
        return 3
    elif combined_score >= 0.52:
        return 2
    else:
        return 1


def format_american_odds(odds: int) -> str:
    """Format American odds with +/- prefix."""
    if odds is None:
        return "N/A"
    return f"+{odds}" if odds > 0 else str(odds)


def get_engine(request: Request):
    """Get the prediction engine from app state."""
    if not hasattr(request.app.state, "engine") or request.app.state.engine is None:
        raise HTTPException(
            status_code=503, detail=f"{RELEASE_VERSION}: Engine not loaded - models missing"
        )
    return request.app.state.engine


def get_feature_builder(request: Request):
    """Get the feature builder from app state."""
    if not hasattr(request.app.state, "feature_builder"):
        raise HTTPException(status_code=503, detail="Feature builder not initialized")
    return request.app.state.feature_builder


def get_tracker(request: Request):
    """Get the pick tracker from app state."""
    if not hasattr(request.app.state, "tracker"):
        raise HTTPException(status_code=503, detail="Tracker not initialized")
    return request.app.state.tracker
