"""
Pydantic models for API responses.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    """Request payload for single-game prediction."""

    home_team: str
    away_team: str
    fg_spread_line: float
    fg_total_line: float
    fh_spread_line: Optional[float] = None
    fh_total_line: Optional[float] = None


class MarketPrediction(BaseModel):
    """Single market prediction result."""

    side: str
    confidence: float
    edge: float
    passes_filter: bool
    filter_reason: Optional[str] = None


class GamePredictions(BaseModel):
    """Predictions for a single game across all markets."""

    first_half: Dict[str, Any] = {}
    full_game: Dict[str, Any] = {}


class SlateResponse(BaseModel):
    """Response for slate endpoints."""

    date: str
    predictions: List[Dict[str, Any]]
    total_plays: int
    odds_as_of_utc: Optional[str] = None
    odds_snapshot_path: Optional[str] = None
    odds_archive_path: Optional[str] = None
    error_message: Optional[str] = None
    errors: Optional[List[str]] = None


class MarketsResponse(BaseModel):
    """Response for markets endpoint."""

    model_config = ConfigDict(protected_namespaces=())

    version: str
    source: str
    markets: List[str]
    market_count: int
    periods: Dict[str, List[str]]
    market_types: List[str]
    model_pack_version: Optional[str] = None
    model_pack_path: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    mode: str
    markets: int
    engine_loaded: bool
