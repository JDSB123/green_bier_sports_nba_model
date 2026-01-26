"""
NBA Season Utilities.

Defines NBA season boundaries and provides helper functions for
season-aware data processing.
"""

from datetime import date, datetime
from typing import Optional, Tuple

import pandas as pd

# NBA Season Definitions
# Regular season typically starts mid-late October and ends mid-April
# Playoffs run April-June
# Offseason is June-October

NBA_SEASONS = {
    # Format: "YYYY-YYYY": (start_date, end_date)
    "2022-2023": (date(2022, 10, 18), date(2023, 6, 12)),  # Finals ended June 12
    "2023-2024": (date(2023, 10, 24), date(2024, 6, 17)),  # Finals ended June 17
    "2024-2025": (date(2024, 10, 22), date(2025, 6, 22)),  # Projected
    "2025-2026": (date(2025, 10, 21), date(2026, 6, 21)),  # Projected
}

# Maximum rest days before we cap (start of season)
MAX_REST_DAYS_SEASON_START = 7

# Offseason duration (roughly June to October = ~120-150 days)
TYPICAL_OFFSEASON_DAYS = 130


def get_season_for_date(game_date: date) -> Optional[str]:
    """
    Get the NBA season string for a given date.

    Args:
        game_date: Date to check

    Returns:
        Season string like "2024-2025" or None if date is in offseason
    """
    if isinstance(game_date, datetime):
        game_date = game_date.date()
    elif isinstance(game_date, pd.Timestamp):
        game_date = game_date.date()

    for season, (start, end) in NBA_SEASONS.items():
        if start <= game_date <= end:
            return season

    # If not in a defined season, infer from date
    # NBA season runs Oct-Jun, so:
    # Oct-Dec -> current year to next year
    # Jan-Jun -> previous year to current year
    year = game_date.year
    month = game_date.month

    if month >= 10:  # Oct-Dec
        return f"{year}-{year + 1}"
    elif month <= 6:  # Jan-Jun
        return f"{year - 1}-{year}"
    else:  # Jul-Sep (offseason)
        return None


def get_season_boundaries(season: str) -> Tuple[date, date]:
    """
    Get the start and end dates for a season.

    Args:
        season: Season string like "2024-2025"

    Returns:
        Tuple of (start_date, end_date)
    """
    if season in NBA_SEASONS:
        return NBA_SEASONS[season]

    # Infer boundaries for unknown seasons
    years = season.split("-")
    start_year = int(years[0])
    end_year = int(years[1])

    # Typical season: late October to mid-June
    return (date(start_year, 10, 22), date(end_year, 6, 15))


def is_season_start(game_date: date, lookback_days: int = 14) -> bool:
    """
    Check if a date is near the start of an NBA season.

    Args:
        game_date: Date to check
        lookback_days: How many days from season start to consider "early"

    Returns:
        True if date is within first `lookback_days` of a season
    """
    if isinstance(game_date, datetime):
        game_date = game_date.date()
    elif isinstance(game_date, pd.Timestamp):
        game_date = game_date.date()

    season = get_season_for_date(game_date)
    if not season:
        return False

    start, _ = get_season_boundaries(season)
    days_since_start = (game_date - start).days

    return 0 <= days_since_start <= lookback_days


def is_crossing_offseason(last_game_date: date, current_game_date: date) -> bool:
    """
    Check if the gap between two dates crosses an NBA offseason.

    Args:
        last_game_date: Date of previous game
        current_game_date: Date of current game

    Returns:
        True if the games are in different seasons
    """
    if isinstance(last_game_date, datetime):
        last_game_date = last_game_date.date()
    elif isinstance(last_game_date, pd.Timestamp):
        last_game_date = last_game_date.date()

    if isinstance(current_game_date, datetime):
        current_game_date = current_game_date.date()
    elif isinstance(current_game_date, pd.Timestamp):
        current_game_date = current_game_date.date()

    last_season = get_season_for_date(last_game_date)
    current_season = get_season_for_date(current_game_date)

    # If either is in offseason, or they're in different seasons
    return last_season != current_season


def compute_effective_rest_days(
    last_game_date: date,
    current_game_date: date,
    cap_at_season_start: int = MAX_REST_DAYS_SEASON_START,
) -> int:
    """
    Compute effective rest days, handling offseason appropriately.

    If crossing an offseason boundary, caps rest at a reasonable value
    rather than reporting 100+ days.

    Args:
        last_game_date: Date of previous game
        current_game_date: Date of current game
        cap_at_season_start: Maximum rest days to return if crossing offseason

    Returns:
        Effective rest days (capped at season start)
    """
    if isinstance(last_game_date, pd.Timestamp):
        last_game_date = last_game_date.to_pydatetime().date()
    elif isinstance(last_game_date, datetime):
        last_game_date = last_game_date.date()

    if isinstance(current_game_date, pd.Timestamp):
        current_game_date = current_game_date.to_pydatetime().date()
    elif isinstance(current_game_date, datetime):
        current_game_date = current_game_date.date()

    actual_days = (current_game_date - last_game_date).days - 1
    actual_days = max(0, actual_days)

    # If crossing offseason, cap the rest days
    if is_crossing_offseason(last_game_date, current_game_date):
        return min(actual_days, cap_at_season_start)

    return actual_days
