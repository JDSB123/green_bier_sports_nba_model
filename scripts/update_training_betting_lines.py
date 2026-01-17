#!/usr/bin/env python3
"""
Update training_data.csv with improved betting lines from theodds_lines.csv.

This script merges the regenerated derived betting lines (which now include
proper 1H data from period_odds) into the existing training data.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging import get_logger

logger = get_logger(__name__)


def normalize_team_name(name: str) -> str:
    """Normalize team names for matching."""
    # Common variations
    mappings = {
        "LA Clippers": "Los Angeles Clippers",
        "LA Lakers": "Los Angeles Lakers",
    }
    return mappings.get(name, name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Update training data with improved betting lines")
    parser.add_argument(
        "--training-data",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "training_data.csv",
        help="Path to training_data.csv",
    )
    parser.add_argument(
        "--derived-lines",
        type=Path,
        default=PROJECT_ROOT / "data" / "historical" / "derived" / "theodds_lines.csv",
        help="Path to theodds_lines.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (defaults to overwriting training_data.csv)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show changes without saving",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = args.training_data

    # Load training data
    logger.info(f"Loading training data from {args.training_data}")
    train = pd.read_csv(args.training_data, low_memory=False)
    train["game_date"] = pd.to_datetime(train["game_date"])
    logger.info(f"Loaded {len(train)} games from training data")

    # Load derived lines
    logger.info(f"Loading derived lines from {args.derived_lines}")
    derived = pd.read_csv(args.derived_lines)
    derived["commence_time"] = pd.to_datetime(derived["commence_time"], utc=True)
    # Convert to local date (the commence_time is in UTC, training data uses local dates)
    derived["game_date"] = derived["commence_time"].dt.tz_convert("America/Chicago").dt.date
    derived["game_date"] = pd.to_datetime(derived["game_date"])

    # Normalize team names
    derived["home_team"] = derived["home_team"].apply(normalize_team_name)
    derived["away_team"] = derived["away_team"].apply(normalize_team_name)

    logger.info(f"Loaded {len(derived)} games from derived lines")

    # Map column names from derived to training data
    column_mapping = {
        "fg_ml_home": "fg_ml_home",
        "fg_ml_away": "fg_ml_away",
        "fg_spread_line": "fg_spread_line",
        "fg_total_line": "fg_total_line",
        "fh_ml_home": "1h_ml_home",
        "fh_ml_away": "1h_ml_away",
        "fh_spread_line": "1h_spread_line",
        "fh_total_line": "1h_total_line",
    }

    # Create merge keys
    train["_merge_key"] = train["game_date"].dt.strftime("%Y-%m-%d") + "_" + train["home_team"].astype(str)
    derived["_merge_key"] = derived["game_date"].dt.strftime("%Y-%m-%d") + "_" + derived["home_team"].astype(str)

    # Check overlap
    train_keys = set(train["_merge_key"].dropna())
    derived_keys = set(derived["_merge_key"].dropna())
    overlap = train_keys & derived_keys
    logger.info(f"Merge key overlap: {len(overlap)} games")

    # Create lookup dict from derived
    derived_lookup = derived.set_index("_merge_key")

    # Track updates
    updates = {col: 0 for col in column_mapping.values()}
    total_fills = 0

    # Update training data with derived values
    for idx, row in train.iterrows():
        merge_key = row["_merge_key"]
        if pd.isna(merge_key) or merge_key not in derived_keys:
            continue

        derived_row = derived_lookup.loc[merge_key]
        if isinstance(derived_row, pd.DataFrame):
            # Multiple matches - take first
            derived_row = derived_row.iloc[0]

        for src_col, dst_col in column_mapping.items():
            src_val = derived_row.get(src_col)
            if pd.notna(src_val):
                dst_val = row.get(dst_col)
                if pd.isna(dst_val):
                    # Fill missing value
                    train.at[idx, dst_col] = src_val
                    updates[dst_col] += 1
                    total_fills += 1

    # Report updates
    logger.info("Column fill counts:")
    for col, count in updates.items():
        logger.info(f"  {col}: {count} values filled")
    logger.info(f"Total values filled: {total_fills}")

    # Report new coverage
    logger.info("\nUpdated coverage:")
    for col in column_mapping.values():
        if col in train.columns:
            pct = train[col].notna().mean() * 100
            logger.info(f"  {col}: {pct:.1f}%")

    # Save
    if args.dry_run:
        logger.info("DRY RUN - not saving")
    else:
        train = train.drop(columns=["_merge_key"])
        train.to_csv(args.output, index=False)
        logger.info(f"Saved updated training data to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
