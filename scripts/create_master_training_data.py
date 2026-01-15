#!/usr/bin/env python3
"""
Create master training data file with complete coverage for seasons 2023-24, 2024-25, 2025-26.

This script:
1. Loads the current training_data.csv
2. Filters to only include seasons with complete data (2023-24, 2024-25, 2025-26)
3. Validates coverage meets quality thresholds
4. Outputs a master file ready for production backtesting
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Seasons to include in master file
TARGET_SEASONS = ["2023-24", "2024-25", "2025-26"]

# Minimum coverage thresholds for key columns
COVERAGE_THRESHOLDS = {
    # Full game markets - should be near 100%
    "fg_spread_line": 95.0,
    "fg_total_line": 95.0,
    "fg_ml_home": 95.0,
    # 1H markets - slightly lower due to period_odds availability
    "1h_spread_line": 90.0,
    "1h_total_line": 90.0,
    "1h_ml_home": 90.0,
    # Features
    "home_elo": 95.0,
    "away_elo": 95.0,
    # Actuals
    "home_score": 95.0,
    "away_score": 95.0,
}


def get_season(dt: pd.Timestamp) -> str:
    """Determine NBA season from game date."""
    if pd.isna(dt):
        return "unknown"
    if dt < pd.Timestamp("2023-10-01"):
        return "2022-23"
    elif dt < pd.Timestamp("2024-10-01"):
        return "2023-24"
    elif dt < pd.Timestamp("2025-10-01"):
        return "2024-25"
    else:
        return "2025-26"


def main() -> int:
    parser = argparse.ArgumentParser(description="Create master training data for target seasons")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "training_data.csv",
        help="Input training data file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "master_training_data.csv",
        help="Output master file path",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip coverage validation checks",
    )
    args = parser.parse_args()

    # Load data
    logger.info(f"Loading training data from {args.input}")
    df = pd.read_csv(args.input, low_memory=False)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    logger.info(f"Loaded {len(df)} total games")

    # Add season column
    df["season"] = df["game_date"].apply(get_season)

    # Filter to target seasons
    logger.info(f"Filtering to seasons: {TARGET_SEASONS}")
    master = df[df["season"].isin(TARGET_SEASONS)].copy()
    logger.info(f"Filtered to {len(master)} games")

    # Report by season
    print("\n" + "=" * 80)
    print("MASTER TRAINING DATA - SEASON BREAKDOWN")
    print("=" * 80)
    for season in TARGET_SEASONS:
        count = len(master[master["season"] == season])
        print(f"  {season}: {count:,} games")
    print(f"  TOTAL: {len(master):,} games")

    # Validate coverage
    print("\n" + "-" * 80)
    print("COVERAGE VALIDATION")
    print("-" * 80)

    validation_passed = True
    for col, threshold in COVERAGE_THRESHOLDS.items():
        if col in master.columns:
            coverage = master[col].notna().mean() * 100
            status = "✓" if coverage >= threshold else "✗"
            if coverage < threshold:
                validation_passed = False
            print(f"  {col}: {coverage:.1f}% (threshold: {threshold}%) {status}")
        else:
            print(f"  {col}: MISSING COLUMN ✗")
            validation_passed = False

    # Additional coverage report for all key columns
    print("\n" + "-" * 80)
    print("FULL COVERAGE REPORT")
    print("-" * 80)

    key_columns = {
        "Betting Lines": ["fg_spread_line", "fg_total_line", "fg_ml_home", "fg_ml_away",
                          "1h_spread_line", "1h_total_line", "1h_ml_home", "1h_ml_away"],
        "Labels": ["fg_spread_covered", "fg_total_over", "1h_spread_covered", "1h_total_over"],
        "ELO": ["home_elo", "away_elo", "elo_diff", "elo_prob_home"],
        "Injury": ["away_injury_spread_impact", "home_injury_spread_impact", "injury_spread_diff"],
        "Travel": ["away_travel_distance", "home_rest_days", "away_rest_days", "rest_adv"],
        "Actuals": ["home_score", "away_score", "home_1h", "away_1h"],
    }

    for category, cols in key_columns.items():
        print(f"\n{category}:")
        for col in cols:
            if col in master.columns:
                coverage = master[col].notna().mean() * 100
                print(f"  {col}: {coverage:.1f}%")
            else:
                print(f"  {col}: MISSING")

    # Report per-season coverage for key betting columns
    print("\n" + "-" * 80)
    print("PER-SEASON COVERAGE (Key Betting Columns)")
    print("-" * 80)

    betting_cols = ["fg_ml_home", "1h_spread_line", "1h_total_line", "1h_ml_home"]
    for col in betting_cols:
        if col in master.columns:
            print(f"\n{col}:")
            for season in TARGET_SEASONS:
                season_data = master[master["season"] == season]
                coverage = season_data[col].notna().mean() * 100
                print(f"  {season}: {coverage:.1f}%")

    # Save
    if not args.skip_validation and not validation_passed:
        logger.warning("Coverage validation FAILED - saving anyway but review recommended")

    # Remove temporary season column before saving (or keep it for reference)
    # master = master.drop(columns=["season"])  # Uncomment to remove season column

    args.output.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(args.output, index=False)

    print("\n" + "=" * 80)
    print(f"SAVED: {args.output}")
    print(f"Games: {len(master):,}")
    print(f"Columns: {len(master.columns)}")
    print(f"Date Range: {master['game_date'].min().date()} to {master['game_date'].max().date()}")
    print("=" * 80)

    logger.info(f"Master training data saved to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
