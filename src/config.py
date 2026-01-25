from __future__ import annotations
from src.utils.secrets import read_secret_strict, read_secret_lax

import os
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional

# Anchor paths to the repository root even when scripts are executed elsewhere
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load repo-root .env for local development (do not override shell env)
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    load_dotenv(PROJECT_ROOT / ".env.local", override=False)
except Exception:
    pass

# NOTE: Secrets are managed via src.utils.secrets
# Priority: Env Vars > Docker Secrets > Baked-in Secrets > Local Files
# For production (Azure), use Environment Variables.


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


def _env_optional(key: str, default: Optional[str] = None) -> Optional[str]:
    """Resolve optional environment variable - returns default if not set."""
    return os.getenv(key, default)


def _env_bool_optional(key: str, default: bool = False) -> bool:
    """Resolve optional boolean environment variable."""
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float_required(key: str, default: float = None) -> float:
    """Resolve required float environment variable with fallback."""
    value = os.getenv(key)
    if value:
        try:
            return float(value)
        except ValueError:
            pass
    if default is not None:
        return default
    raise ValueError(
        f"Required environment variable not set or invalid: {key}")


def _env_float_optional(key: str, default: float = 0.0) -> float:
    """Resolve optional float environment variable with fallback."""
    value = os.getenv(key)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _current_season() -> str:
    """Resolve the current season from env (required) or raise."""
    return _env_required("CURRENT_SEASON") or get_current_nba_season()  # Fallback only if not set, but raise per strict


@dataclass(frozen=True)
class FilterThresholds:
    """
    Configurable filter thresholds for betting predictions.

    These thresholds determine whether a prediction passes the betting filter.
    EDGE-ONLY filtering: confidence thresholds disabled (set to 0.0)
    A prediction only needs to meet the edge threshold to pass.

    UPDATED 2026-01-22: Switched to edge-only filtering.
    Confidence is still calculated and reported but not used for filtering.

    Thresholds can be set via environment variables. Defaults provided for local development:
    - FILTER_SPREAD_MIN_CONFIDENCE (default: 0.0) - DISABLED (edge-only)
    - FILTER_SPREAD_MIN_EDGE (default: 5.0) - Primary filter
    - FILTER_TOTAL_MIN_CONFIDENCE (default: 0.0) - DISABLED (edge-only)
    - FILTER_TOTAL_MIN_EDGE (default: 3.0) - Primary filter
    - FILTER_1H_SPREAD_MIN_CONFIDENCE (default: 0.0) - DISABLED (edge-only)
    - FILTER_1H_SPREAD_MIN_EDGE (default: 2.5) - Primary filter
    - FILTER_1H_TOTAL_MIN_CONFIDENCE (default: 0.0) - DISABLED (edge-only)
    - FILTER_1H_TOTAL_MIN_EDGE (default: 2.5) - Primary filter

    Edge Thresholds (tuned per market):
    - spread_min_edge: 5.0 pts (high-conviction only)
    - total_min_edge: 3.0 pts
    - 1H_spread_min_edge: 2.5 pts
    - 1H_total_min_edge: 2.5 pts

    See: OPTIMIZATION_RESULTS_SUMMARY.md for full details
    """
    # Spread thresholds - EDGE-ONLY
    spread_min_confidence: float = field(
        default_factory=lambda: _env_float_required(
            "FILTER_SPREAD_MIN_CONFIDENCE", 0.0)
    )
    spread_min_edge: float = field(
        default_factory=lambda: _env_float_required(
            "FILTER_SPREAD_MIN_EDGE", 5.0)
    )

    # Total thresholds - EDGE-ONLY
    total_min_confidence: float = field(
        default_factory=lambda: _env_float_required(
            "FILTER_TOTAL_MIN_CONFIDENCE", 0.0)
    )
    total_min_edge: float = field(
        default_factory=lambda: _env_float_required(
            "FILTER_TOTAL_MIN_EDGE", 3.0)
    )

    # 1H Spread thresholds - EDGE-ONLY
    fh_spread_min_confidence: float = field(
        default_factory=lambda: _env_float_required(
            "FILTER_1H_SPREAD_MIN_CONFIDENCE", 0.0)
    )
    fh_spread_min_edge: float = field(
        default_factory=lambda: _env_float_required(
            "FILTER_1H_SPREAD_MIN_EDGE", 2.5)
    )

    # 1H Total thresholds - EDGE-ONLY
    fh_total_min_confidence: float = field(
        default_factory=lambda: _env_float_required(
            "FILTER_1H_TOTAL_MIN_CONFIDENCE", 0.0)
    )
    fh_total_min_edge: float = field(
        default_factory=lambda: _env_float_required(
            "FILTER_1H_TOTAL_MIN_EDGE", 2.5)
    )


# Global filter thresholds instance
filter_thresholds = FilterThresholds()


@dataclass(frozen=True)
class Settings:
    # Core API Keys - STRICT MODE: Env only, validated at runtime
    the_odds_api_key: str = field(
        default_factory=lambda: read_secret_lax("THE_ODDS_API_KEY"))
    api_basketball_key: str = field(
        default_factory=lambda: read_secret_lax("API_BASKETBALL_KEY"))

    # Optional API Keys (None if not set)
    betsapi_key: Optional[str] = field(
        default_factory=lambda: _env_optional("BETSAPI_KEY"))
    action_network_username: Optional[str] = field(
        default_factory=lambda: read_secret_lax("ACTION_NETWORK_USERNAME"))
    action_network_password: Optional[str] = field(
        default_factory=lambda: read_secret_lax("ACTION_NETWORK_PASSWORD"))
    kaggle_api_token: Optional[str] = field(
        default_factory=lambda: _env_optional("KAGGLE_API_TOKEN"))

    # API base URLs (with defaults for import compatibility)
    the_odds_base_url: str = field(
        default_factory=lambda: _env_optional(
            "THE_ODDS_BASE_URL", "https://api.the-odds-api.com/v4")
    )
    api_basketball_base_url: str = field(
        default_factory=lambda: _env_optional(
            "API_BASKETBALL_BASE_URL", "https://v1.basketball.api-sports.io")
    )

    # Season Configuration (with defaults for import compatibility)
    current_season: str = field(
        default_factory=lambda: _env_optional("CURRENT_SEASON", "2025-2026"))
    seasons_to_process: list[str] = field(
        default_factory=lambda: _env_optional(
            "SEASONS_TO_PROCESS", "2023-2024,2024-2025,2025-2026").split(",")
    )

    # Data directories (with defaults for import compatibility)
    data_raw_dir: str = field(
        default_factory=lambda: _env_optional("DATA_RAW_DIR", "data/raw")
    )
    data_processed_dir: str = field(
        default_factory=lambda: _env_optional(
            "DATA_PROCESSED_DIR", "data/processed")
    )

    # Models directory (single source of truth)
    # Production container should set MODELS_DIR=/app/models/production
    models_dir: str = field(
        default_factory=lambda: _env_optional(
            "MODELS_DIR", str(PROJECT_ROOT / "models" / "production")
        )
    )

    # STRICT LIVE DATA GUARDS (default off for tests/local)
    require_action_network_splits: bool = field(
        default_factory=lambda: _env_bool_optional(
            "REQUIRE_ACTION_NETWORK_SPLITS", False
        )
    )
    require_real_splits: bool = field(
        default_factory=lambda: _env_bool_optional(
            "REQUIRE_REAL_SPLITS", False
        )
    )
    require_sharp_book_data: bool = field(
        default_factory=lambda: _env_bool_optional(
            "REQUIRE_SHARP_BOOK_DATA", False
        )
    )
    require_injury_fetch_success: bool = field(
        default_factory=lambda: _env_bool_optional(
            "REQUIRE_INJURY_FETCH_SUCCESS", False
        )
    )
    min_feature_completeness: float = field(
        default_factory=lambda: _env_float_optional(
            "MIN_FEATURE_COMPLETENESS", 0.0
        )
    )


# Lazy-loaded settings to avoid import-time failures
_settings = None


def get_settings() -> Settings:
    """Get settings instance, creating it on first access."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

# For backward compatibility, create a settings object that behaves like the old one


class LazySettings:
    """Lazy settings proxy that creates Settings instance on first access."""

    def __init__(self):
        self._settings = None

    def __getattr__(self, name):
        if self._settings is None:
            self._settings = Settings()
        return getattr(self._settings, name)


settings = LazySettings()
