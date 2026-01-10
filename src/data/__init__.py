"""
Data standardization and ingestion module.

SINGLE SOURCE OF TRUTH for all data transformations.
"""
from src.data.standardization import (
    # Team names
    standardize_team_name,
    get_all_team_names,
    is_valid_team,
    # Timezone (UTC source)
    to_cst,
    to_cst_date,
    # Timezone (local US source - for Kaggle etc.)
    to_cst_local,
    to_cst_date_local,
    CST,
    UTC,
    # Match keys
    generate_match_key,
    # Record standardization
    standardize_game_record,
)

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
