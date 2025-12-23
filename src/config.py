from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional

# Anchor paths to the repository root even when scripts are executed elsewhere
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# NOTE: Secrets are managed via src.utils.secrets
# Priority: Env Vars > Docker Secrets > Baked-in Secrets > Local Files
# For production (Azure), use Environment Variables.

from src.utils.secrets import read_secret_strict

def get_nba_season(d: date | None = None) -> str:
    """Get the NBA season string for a given date.

    NBA seasons span two calendar years (Oct-Apr).
    - Oct 2025 - Apr 2026 = "2025-2026" season
    - Oct 2024 - Apr 2025 = "2024-2025" season

    Args:
        d: Date to check. Defaults to today.

    Returns:
        Season string like "2025-2026"
    """
    if d is None:
        d = date.today()

    # NBA season starts in October
    # If we're in Jan-Sep, we're in the previous year's season
    # If we're in Oct-Dec, we're in the current year's season
    if d.month >= 10:  # Oct-Dec
        start_year = d.year
    else:  # Jan-Sep
        start_year = d.year - 1

    return f"{start_year}-{start_year + 1}"


def get_current_nba_season() -> str:
    """Get the current NBA season based on today's date."""
    return get_nba_season(date.today())


def _env_required(key: str) -> str:
    """Resolve required environment variable - raises if not set."""
    value = os.getenv(key)
    if not value:
        raise ValueError(f"Required environment variable not set: {key}")
    return value


def _env_optional(key: str) -> Optional[str]:
    """Resolve optional environment variable - returns None if not set."""
    return os.getenv(key)


def _current_season() -> str:
    """Resolve the current season from env (required) or raise."""
    return _env_required("CURRENT_SEASON") or get_current_nba_season()  # Fallback only if not set, but raise per strict


@dataclass(frozen=True)
class FilterThresholds:
    """
    Configurable filter thresholds for betting predictions.

    These thresholds determine whether a prediction passes the betting filter.
    A prediction must meet BOTH confidence AND edge thresholds to pass.

    Thresholds MUST be set via environment variables - no defaults:
    - FILTER_SPREAD_MIN_CONFIDENCE
    - FILTER_SPREAD_MIN_EDGE
    - FILTER_TOTAL_MIN_CONFIDENCE
    - FILTER_TOTAL_MIN_EDGE
    - FILTER_MONEYLINE_MIN_CONFIDENCE
    - FILTER_MONEYLINE_MIN_EDGE_PCT

    Q1 thresholds (DEPRECATED in v6.6 - Q1 markets disabled):
    - FILTER_Q1_MIN_CONFIDENCE (optional, no longer used)
    - FILTER_Q1_MIN_EDGE_PCT (optional, no longer used)
    """
    # Spread thresholds
    spread_min_confidence: float = field(
        default_factory=lambda: float(_env_required("FILTER_SPREAD_MIN_CONFIDENCE"))
    )
    spread_min_edge: float = field(
        default_factory=lambda: float(_env_required("FILTER_SPREAD_MIN_EDGE"))
    )

    # Total thresholds
    total_min_confidence: float = field(
        default_factory=lambda: float(_env_required("FILTER_TOTAL_MIN_CONFIDENCE"))
    )
    total_min_edge: float = field(
        default_factory=lambda: float(_env_required("FILTER_TOTAL_MIN_EDGE"))
    )

    # Moneyline thresholds (FG/1H)
    moneyline_min_confidence: float = field(
        default_factory=lambda: float(_env_required("FILTER_MONEYLINE_MIN_CONFIDENCE"))
    )
    moneyline_min_edge_pct: float = field(
        default_factory=lambda: float(_env_required("FILTER_MONEYLINE_MIN_EDGE_PCT"))
    )

    # Q1-specific thresholds (DISABLED in v6.6 - Q1 markets removed)
    # Kept for backward compatibility with .env but not used
    q1_min_confidence: float = field(
        default_factory=lambda: float(os.getenv("FILTER_Q1_MIN_CONFIDENCE", "0.65"))
    )
    q1_min_edge_pct: float = field(
        default_factory=lambda: float(os.getenv("FILTER_Q1_MIN_EDGE_PCT", "15.0"))
    )


# Global filter thresholds instance
filter_thresholds = FilterThresholds()


@dataclass(frozen=True)
class Settings:
    # Core API Keys (Required) - STRICT MODE: Env only, raise if missing
    the_odds_api_key: str = field(default_factory=lambda: read_secret_strict("THE_ODDS_API_KEY"))
    api_basketball_key: str = field(default_factory=lambda: read_secret_strict("API_BASKETBALL_KEY"))
    
    # Optional API Keys (None if not set)
    betsapi_key: Optional[str] = field(default_factory=lambda: _env_optional("BETSAPI_KEY"))
    action_network_username: Optional[str] = field(default_factory=lambda: _env_optional("ACTION_NETWORK_USERNAME"))
    action_network_password: Optional[str] = field(default_factory=lambda: _env_optional("ACTION_NETWORK_PASSWORD"))
    kaggle_api_token: Optional[str] = field(default_factory=lambda: _env_optional("KAGGLE_API_TOKEN"))

    # API base URLs (required - raise if not set)
    the_odds_base_url: str = field(
        default_factory=lambda: _env_required("THE_ODDS_BASE_URL")
    )
    api_basketball_base_url: str = field(
        default_factory=lambda: _env_required("API_BASKETBALL_BASE_URL")
    )

    # Season Configuration (required)
    current_season: str = field(default_factory=lambda: _env_required("CURRENT_SEASON"))
    seasons_to_process: list[str] = field(
        default_factory=lambda: _env_required("SEASONS_TO_PROCESS").split(",")
    )

    # Data directories (required)
    data_raw_dir: str = field(
        default_factory=lambda: _env_required("DATA_RAW_DIR")
    )
    data_processed_dir: str = field(
        default_factory=lambda: _env_required("DATA_PROCESSED_DIR")
    )


settings = Settings()
