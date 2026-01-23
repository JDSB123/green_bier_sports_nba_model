"""
Data standardization and ingestion module.

Re-exports from src.ingestion.standardize for backwards compatibility.
The canonical standardization module is src/ingestion/standardize.py.
"""
from src.ingestion.standardize import (
    # Core functions
    normalize_team_to_espn,
    is_valid_espn_team_name as is_valid_team,
    # Timezone
    to_cst,
    CST,
    UTC,
    # Match keys
    generate_match_key,
    # Record standardization
    standardize_game_data as standardize_game_record,
)


def standardize_team_name(team: str) -> str:
    """
    Convert any team name variant to canonical full name.

    Wrapper for normalize_team_to_espn that returns just the name string
    for backwards compatibility.
    """
    normalized, _ = normalize_team_to_espn(
        team, source="standardize_team_name")
    return normalized


# Additional convenience functions
def get_all_team_names() -> list[str]:
    """Get list of all valid team names."""
    from src.utils.team_names import CANONICAL_NAMES
    return list(CANONICAL_NAMES.values())


def to_cst_date(dt, source_is_utc: bool = True):
    """Convert datetime to CST date string (YYYY-MM-DD)."""
    cst_dt = to_cst(dt)
    return cst_dt.strftime("%Y-%m-%d") if cst_dt else None


def to_cst_local(dt):
    """Convert local US datetime to CST (alias for to_cst)."""
    return to_cst(dt)


def to_cst_date_local(dt):
    """Convert local US datetime to CST date string."""
    return to_cst_date(dt)


__all__ = [
    "standardize_team_name",
    "get_all_team_names",
    "is_valid_team",
    "to_cst",
    "to_cst_date",
    "to_cst_local",
    "to_cst_date_local",
    "CST",
    "UTC",
    "generate_match_key",
    "standardize_game_record",
]
