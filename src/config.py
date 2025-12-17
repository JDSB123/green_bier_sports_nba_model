from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from dotenv import load_dotenv

# Anchor paths to the repository root even when scripts are executed elsewhere
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env from project root
load_dotenv(PROJECT_ROOT / ".env")


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


def _current_season_default() -> str:
    """Resolve the current season from env or today's date at instantiation time."""
    return _env_or_default("CURRENT_SEASON", get_current_nba_season())


@dataclass(frozen=True)
class Settings:
    # Core API Keys (Required)
    the_odds_api_key: str = field(default_factory=lambda: _env_or_default("THE_ODDS_API_KEY", ""))
    api_basketball_key: str = field(default_factory=lambda: _env_or_default("API_BASKETBALL_KEY", ""))
    
    # Optional API Keys (for enhanced features)
    betsapi_key: str = field(default_factory=lambda: _env_or_default("BETSAPI_KEY", ""))
    action_network_username: str = field(default_factory=lambda: _env_or_default("ACTION_NETWORK_USERNAME", ""))
    action_network_password: str = field(default_factory=lambda: _env_or_default("ACTION_NETWORK_PASSWORD", ""))
    kaggle_api_token: str = field(default_factory=lambda: _env_or_default("KAGGLE_API_TOKEN", ""))

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
