from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

# Anchor paths to the repository root even when scripts are executed elsewhere
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# NOTE: Secrets are managed via src.utils.secrets
# Priority: Env Vars > Docker Secrets > Baked-in Secrets > Local Files
# For production (Azure), use Environment Variables.

# Import secrets utility
try:
    from src.utils.secrets import read_secret_or_env, read_secret_strict
except ImportError:
    # Fallback if secrets module not available
    def read_secret_or_env(secret_name: str, env_name: str = None, default: str = "") -> str:
        if env_name is None:
            env_name = secret_name
        return os.getenv(env_name, default)
    def read_secret_strict(secret_name: str) -> str:
        value = os.getenv(secret_name, "")
        if not value:
            raise RuntimeError(f"Required secret not found: {secret_name}")
        return value


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


def _env_or_default(key: str, default: str) -> str:
    """Resolve an environment variable at instantiation time."""
    return os.getenv(key, default)


def _secret_strict(secret_name: str) -> str:
    """
    STRICT MODE: Read secret from baked-in location (/app/secrets).
    Secrets are baked into container at build time - fully self-contained.
    NO FALLBACKS. FAILS LOUDLY if not found.
    """
    return read_secret_strict(secret_name)


def _secret_or_env(secret_name: str, env_name: str = None, default: str = "") -> str:
    """Resolve a secret from Docker secrets or environment variable (for optional keys only)."""
    return read_secret_or_env(secret_name, env_name, default)


def _current_season_default() -> str:
    """Resolve the current season from env or today's date at instantiation time."""
    return _env_or_default("CURRENT_SEASON", get_current_nba_season())


@dataclass(frozen=True)
class FilterThresholds:
    """
    Configurable filter thresholds for betting predictions.

    These thresholds determine whether a prediction passes the betting filter.
    A prediction must meet BOTH confidence AND edge thresholds to pass.

    Thresholds can be overridden via environment variables:
    - FILTER_SPREAD_MIN_CONFIDENCE (default: 0.55)
    - FILTER_SPREAD_MIN_EDGE (default: 1.0)
    - FILTER_TOTAL_MIN_CONFIDENCE (default: 0.55)
    - FILTER_TOTAL_MIN_EDGE (default: 1.5)
    - FILTER_MONEYLINE_MIN_CONFIDENCE (default: 0.55)
    - FILTER_MONEYLINE_MIN_EDGE_PCT (default: 0.03, i.e., 3%)

    Q1-specific thresholds (STRICTER due to marginal accuracy):
    - FILTER_Q1_MIN_CONFIDENCE (default: 0.60) - Higher than FG/1H
    - FILTER_Q1_MIN_EDGE_PCT (default: 0.05)
    """
    # Spread thresholds
    spread_min_confidence: float = field(
        default_factory=lambda: float(_env_or_default("FILTER_SPREAD_MIN_CONFIDENCE", "0.55"))
    )
    spread_min_edge: float = field(
        default_factory=lambda: float(_env_or_default("FILTER_SPREAD_MIN_EDGE", "1.0"))
    )

    # Total thresholds
    total_min_confidence: float = field(
        default_factory=lambda: float(_env_or_default("FILTER_TOTAL_MIN_CONFIDENCE", "0.55"))
    )
    total_min_edge: float = field(
        default_factory=lambda: float(_env_or_default("FILTER_TOTAL_MIN_EDGE", "1.5"))
    )

    # Moneyline thresholds (FG/1H)
    moneyline_min_confidence: float = field(
        default_factory=lambda: float(_env_or_default("FILTER_MONEYLINE_MIN_CONFIDENCE", "0.55"))
    )
    moneyline_min_edge_pct: float = field(
        default_factory=lambda: float(_env_or_default("FILTER_MONEYLINE_MIN_EDGE_PCT", "0.03"))
    )

    # Q1-specific thresholds (STRICTER - backtest shows 53% overall, 58.8% at >=60% conf)
    q1_min_confidence: float = field(
        default_factory=lambda: float(_env_or_default("FILTER_Q1_MIN_CONFIDENCE", "0.60"))
    )
    q1_min_edge_pct: float = field(
        default_factory=lambda: float(_env_or_default("FILTER_Q1_MIN_EDGE_PCT", "0.05"))
    )


# Global filter thresholds instance
filter_thresholds = FilterThresholds()


@dataclass(frozen=True)
class Settings:
    # Core API Keys (Required) - STRICT MODE: Docker secrets only, NO FALLBACKS
    the_odds_api_key: str = field(default_factory=lambda: _secret_strict("THE_ODDS_API_KEY"))
    api_basketball_key: str = field(default_factory=lambda: _secret_strict("API_BASKETBALL_KEY"))
    
    # Optional API Keys (for enhanced features)
    betsapi_key: str = field(default_factory=lambda: _secret_or_env("BETSAPI_KEY", "BETSAPI_KEY", ""))
    action_network_username: str = field(default_factory=lambda: _secret_or_env("ACTION_NETWORK_USERNAME", "ACTION_NETWORK_USERNAME", ""))
    action_network_password: str = field(default_factory=lambda: _secret_or_env("ACTION_NETWORK_PASSWORD", "ACTION_NETWORK_PASSWORD", ""))
    kaggle_api_token: str = field(default_factory=lambda: _secret_or_env("KAGGLE_API_TOKEN", "KAGGLE_API_TOKEN", ""))

    # Optional: allow overriding API base URLs from environment
    the_odds_base_url: str = field(
        default_factory=lambda: _env_or_default(
            "THE_ODDS_BASE_URL", "https://api.the-odds-api.com/v4"
        )
    )
    api_basketball_base_url: str = field(
        default_factory=lambda: _env_or_default(
            "API_BASKETBALL_BASE_URL", "https://v1.basketball.api-sports.io"
        )
    )

    # Season Configuration
    current_season: str = field(default_factory=_current_season_default)
    seasons_to_process: list[str] = field(
        default_factory=lambda: os.getenv(
            "SEASONS_TO_PROCESS", "2023-2024,2024-2025,2025-2026"
        ).split(",")
    )

    data_raw_dir: str = os.getenv(
        "DATA_RAW_DIR", str(PROJECT_ROOT / "data" / "raw")
    )
    data_processed_dir: str = os.getenv(
        "DATA_PROCESSED_DIR", str(PROJECT_ROOT / "data" / "processed")
    )


settings = Settings()
