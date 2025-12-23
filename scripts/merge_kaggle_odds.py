#!/usr/bin/env python3
"""
Merge Kaggle NBA betting data with training data.

This script:
1. Loads the Kaggle NBA betting dataset (2008-2025)
2. Maps abbreviated team names to ESPN full names
3. Merges betting lines (spread, total, moneyline) into training data
4. Recalculates outcome labels (spread_covered, total_over, etc.)

Usage:
    python scripts/merge_kaggle_odds.py
    python scripts/merge_kaggle_odds.py --kaggle-file data/external/kaggle/nba_2008-2025.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Team abbreviation to ESPN full name mapping
TEAM_ABBREV_MAP = {
    # Current teams
    "atl": "Atlanta Hawks",
    "bos": "Boston Celtics",
    "bkn": "Brooklyn Nets",
    "cha": "Charlotte Hornets",
    "chi": "Chicago Bulls",
    "cle": "Cleveland Cavaliers",
    "dal": "Dallas Mavericks",
    "den": "Denver Nuggets",
    "det": "Detroit Pistons",
    "gs": "Golden State Warriors",
    "gsw": "Golden State Warriors",
    "hou": "Houston Rockets",
    "ind": "Indiana Pacers",
    "lac": "Los Angeles Clippers",
    "lal": "Los Angeles Lakers",
    "mem": "Memphis Grizzlies",
    "mia": "Miami Heat",
    "mil": "Milwaukee Bucks",
    "min": "Minnesota Timberwolves",
    "no": "New Orleans Pelicans",
    "nop": "New Orleans Pelicans",
    "nyk": "New York Knicks",
    "ny": "New York Knicks",
    "okc": "Oklahoma City Thunder",
    "orl": "Orlando Magic",
    "phi": "Philadelphia 76ers",
    "phx": "Phoenix Suns",
    "por": "Portland Trail Blazers",
    "sac": "Sacramento Kings",
    "sa": "San Antonio Spurs",
    "sas": "San Antonio Spurs",
    "tor": "Toronto Raptors",
    "utah": "Utah Jazz",
    "uta": "Utah Jazz",
    "was": "Washington Wizards",
    "wsh": "Washington Wizards",
    # Historical names
    "nj": "Brooklyn Nets",  # New Jersey Nets -> Brooklyn Nets
    "njn": "Brooklyn Nets",
    "sea": "Oklahoma City Thunder",  # Seattle SuperSonics -> OKC
    "noh": "New Orleans Pelicans",  # New Orleans Hornets
    "nok": "New Orleans Pelicans",  # New Orleans/OKC Hornets
    "cha_old": "Charlotte Hornets",  # Charlotte Bobcats
    "chb": "Charlotte Hornets",
}


def map_team_name(abbrev: str) -> str:
    """Map team abbreviation to ESPN full name."""
    abbrev_lower = abbrev.lower().strip()
    return TEAM_ABBREV_MAP.get(abbrev_lower, abbrev)


def load_kaggle_data(filepath: Path) -> pd.DataFrame:
    """Load and preprocess Kaggle betting data."""
    df = pd.read_csv(filepath)

    # Map team names
    df["home_team"] = df["home"].apply(map_team_name)
    df["away_team"] = df["away"].apply(map_team_name)

    # Parse date
    df["date"] = pd.to_datetime(df["date"])

    # Create spread_line (from home perspective - negate if away is favored)
    # In Kaggle data: spread is always positive, whos_favored indicates direction
    df["spread_line"] = df.apply(
        lambda r: -r["spread"] if r["whos_favored"] == "home" else r["spread"],
        axis=1
    )

    # Total line
    df["total_line"] = df["total"]

    # Moneylines
    df["home_ml"] = df["moneyline_home"]
    df["away_ml"] = df["moneyline_away"]

    # First half lines (h2 appears to be second half, so estimate 1H)
    # Using standard NBA scaling: 1H ~ 50% of FG
    df["fh_spread_line"] = df["spread_line"] / 2
    df["fh_total_line"] = df["total_line"] / 2

    # Q1 lines (estimate: Q1 ~ 25% of FG)
    df["q1_spread_line"] = df["spread_line"] / 4
    df["q1_total_line"] = df["total_line"] / 4

    return df


def load_training_data(filepath: Path) -> pd.DataFrame:
    """Load existing training data."""
    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def merge_betting_data(training_df: pd.DataFrame, kaggle_df: pd.DataFrame) -> pd.DataFrame:
    """Merge betting lines from Kaggle into training data."""
    from datetime import timedelta

    # Create date-only column for matching (ignore time)
    # Handle UTC timezone in training data - games after midnight UTC might be previous day local
    training_df["date_only"] = training_df["date"].dt.tz_localize(None).dt.date
    kaggle_df["date_only"] = kaggle_df["date"].dt.date

    # Also create alternate date (day before) for timezone edge cases
    training_df["date_alt"] = (training_df["date"].dt.tz_localize(None) - timedelta(days=1)).dt.date

    # Create match key using team names (lowercase, stripped)
    training_df["team_key"] = (
        training_df["home_team"].str.lower().str.strip() + "_" +
        training_df["away_team"].str.lower().str.strip()
    )
    kaggle_df["team_key"] = (
        kaggle_df["home_team"].str.lower().str.strip() + "_" +
        kaggle_df["away_team"].str.lower().str.strip()
    )

    # Primary match key
    training_df["match_key"] = (
        training_df["date_only"].astype(str) + "_" + training_df["team_key"]
    )
    # Alternate match key (day before)
    training_df["match_key_alt"] = (
        training_df["date_alt"].astype(str) + "_" + training_df["team_key"]
    )
    kaggle_df["match_key"] = (
        kaggle_df["date_only"].astype(str) + "_" + kaggle_df["team_key"]
    )

    # Select columns to merge from Kaggle
    betting_cols = [
        "spread_line", "total_line",
        "home_ml", "away_ml",
        "fh_spread_line", "fh_total_line",
        "q1_spread_line", "q1_total_line",
    ]

    # Remove existing betting columns from training (they're all NaN anyway)
    drop_cols = [c for c in betting_cols if c in training_df.columns]
    if drop_cols:
        training_df = training_df.drop(columns=drop_cols)

    # Create kaggle lookup dict for faster matching
    kaggle_lookup = kaggle_df.set_index("match_key")[betting_cols].to_dict("index")

    # Match using primary key first, then alternate key
    matched_data = []
    for idx, row in training_df.iterrows():
        match_key = row["match_key"]
        match_key_alt = row["match_key_alt"]

        if match_key in kaggle_lookup:
            matched_data.append(kaggle_lookup[match_key])
        elif match_key_alt in kaggle_lookup:
            matched_data.append(kaggle_lookup[match_key_alt])
        else:
            matched_data.append({col: None for col in betting_cols})

    # Add matched betting data to training df
    matched_df = pd.DataFrame(matched_data)
    for col in betting_cols:
        training_df[col] = matched_df[col].values

    # Clean up temporary columns
    cleanup_cols = ["date_only", "date_alt", "team_key", "match_key", "match_key_alt"]
    training_df = training_df.drop(columns=[c for c in cleanup_cols if c in training_df.columns])

    return training_df


def recalculate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Recalculate betting outcome labels based on merged lines."""

    # Full game labels
    if "spread_line" in df.columns and "home_score" in df.columns:
        df["actual_margin"] = df["home_score"] - df["away_score"]
        # Spread covered: home margin > -spread_line
        df["spread_covered"] = np.where(
            df["spread_line"].notna(),
            (df["actual_margin"] > -df["spread_line"]).astype(int),
            np.nan
        )

    if "total_line" in df.columns and "home_score" in df.columns:
        df["actual_total"] = df["home_score"] + df["away_score"]
        df["total_over"] = np.where(
            df["total_line"].notna(),
            (df["actual_total"] > df["total_line"]).astype(int),
            np.nan
        )

    # First half labels
    if all(c in df.columns for c in ["home_q1", "home_q2", "away_q1", "away_q2"]):
        df["home_1h_score"] = df["home_q1"].fillna(0) + df["home_q2"].fillna(0)
        df["away_1h_score"] = df["away_q1"].fillna(0) + df["away_q2"].fillna(0)
        df["actual_1h_margin"] = df["home_1h_score"] - df["away_1h_score"]
        df["actual_1h_total"] = df["home_1h_score"] + df["away_1h_score"]
        df["home_1h_win"] = (df["home_1h_score"] > df["away_1h_score"]).astype(int)

        if "fh_spread_line" in df.columns:
            df["1h_spread_line"] = df["fh_spread_line"]
            df["1h_spread_covered"] = np.where(
                df["1h_spread_line"].notna(),
                (df["actual_1h_margin"] > -df["1h_spread_line"]).astype(int),
                np.nan
            )

        if "fh_total_line" in df.columns:
            df["1h_total_line"] = df["fh_total_line"]
            df["1h_total_over"] = np.where(
                df["1h_total_line"].notna(),
                (df["actual_1h_total"] > df["1h_total_line"]).astype(int),
                np.nan
            )

    # Q1 labels
    if all(c in df.columns for c in ["home_q1", "away_q1"]):
        df["actual_q1_margin"] = df["home_q1"].fillna(0) - df["away_q1"].fillna(0)
        df["actual_q1_total"] = df["home_q1"].fillna(0) + df["away_q1"].fillna(0)
        df["home_q1_win"] = (df["home_q1"].fillna(0) > df["away_q1"].fillna(0)).astype(int)

        if "q1_spread_line" in df.columns:
            df["q1_spread_covered"] = np.where(
                df["q1_spread_line"].notna(),
                (df["actual_q1_margin"] > -df["q1_spread_line"]).astype(int),
                np.nan
            )

        if "q1_total_line" in df.columns:
            df["q1_total_over"] = np.where(
                df["q1_total_line"].notna(),
                (df["actual_q1_total"] > df["q1_total_line"]).astype(int),
                np.nan
            )

    return df


def main():
    parser = argparse.ArgumentParser(description="Merge Kaggle betting data with training data")
    parser.add_argument(
        "--kaggle-file",
        type=Path,
        default=PROJECT_ROOT / "data" / "external" / "kaggle" / "nba_2008-2025.csv",
        help="Path to Kaggle betting data CSV"
    )
    parser.add_argument(
        "--training-file",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "training_data.csv",
        help="Path to training data CSV"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: overwrite training data)"
    )
    args = parser.parse_args()

    output_path = args.output or args.training_file

    print("=" * 60)
    print("MERGING KAGGLE BETTING DATA")
    print("=" * 60)

    # Load data
    print(f"\n[1] Loading Kaggle data from {args.kaggle_file}")
    kaggle_df = load_kaggle_data(args.kaggle_file)
    print(f"    Loaded {len(kaggle_df)} games from Kaggle")
    print(f"    Date range: {kaggle_df['date'].min().date()} to {kaggle_df['date'].max().date()}")

    print(f"\n[2] Loading training data from {args.training_file}")
    training_df = load_training_data(args.training_file)
    print(f"    Loaded {len(training_df)} games from training data")

    # Check existing betting data
    existing_spread = training_df["spread_line"].notna().sum() if "spread_line" in training_df.columns else 0
    print(f"    Existing spread_line values: {existing_spread}")

    # Merge
    print("\n[3] Merging betting lines...")
    merged_df = merge_betting_data(training_df, kaggle_df)

    # Stats after merge
    merged_spread = merged_df["spread_line"].notna().sum()
    merge_rate = merged_spread / len(merged_df) * 100
    print(f"    Matched games with betting lines: {merged_spread} ({merge_rate:.1f}%)")

    # Recalculate labels
    print("\n[4] Recalculating betting outcome labels...")
    merged_df = recalculate_labels(merged_df)

    # Verify labels
    spread_covered_valid = merged_df["spread_covered"].notna().sum()
    total_over_valid = merged_df["total_over"].notna().sum()
    print(f"    spread_covered valid: {spread_covered_valid}")
    print(f"    total_over valid: {total_over_valid}")

    # Save
    print(f"\n[5] Saving to {output_path}")
    merged_df.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print("MERGE COMPLETE")
    print("=" * 60)
    print(f"\nTraining data now has {merged_spread} games with betting lines!")
    print("You can now run backtesting:")
    print("  python scripts/backtest.py --markets fg_spread,fg_total")

    return 0


if __name__ == "__main__":
    sys.exit(main())
