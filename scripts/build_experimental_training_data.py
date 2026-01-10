#!/usr/bin/env python3
"""
Build expanded training data from ALL available historical sources.

This script creates training data for experimental model development
WITHOUT affecting production data or models.

Data Sources:
1. Kaggle (2008-2025): 17 seasons, ~23,000 games with outcomes and lines
2. The Odds API (2023-2025): Consensus betting lines with moneyline

Output:
    data/experimental/training_data_full.csv       - All 17 seasons
    data/experimental/training_data_2018_2025.csv  - Last 7 seasons
    data/experimental/training_data_2020_2025.csv  - Last 5 seasons

Usage:
    python scripts/build_experimental_training_data.py
    python scripts/build_experimental_training_data.py --seasons 5  # Last 5 seasons only
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Directories
DATA_DIR = PROJECT_ROOT / "data"
EXTERNAL_DIR = DATA_DIR / "external"
HISTORICAL_DIR = DATA_DIR / "historical"
EXPERIMENTAL_DIR = DATA_DIR / "experimental"

# Team name mapping (Kaggle abbreviations to standardized names)
TEAM_ABBREV_TO_NAME = {
    "atl": "Atlanta Hawks",
    "bkn": "Brooklyn Nets",
    "bos": "Boston Celtics",
    "cha": "Charlotte Hornets",
    "chi": "Chicago Bulls",
    "cle": "Cleveland Cavaliers",
    "dal": "Dallas Mavericks",
    "den": "Denver Nuggets",
    "det": "Detroit Pistons",
    "gs": "Golden State Warriors",
    "hou": "Houston Rockets",
    "ind": "Indiana Pacers",
    "lac": "Los Angeles Clippers",
    "lal": "Los Angeles Lakers",
    "mem": "Memphis Grizzlies",
    "mia": "Miami Heat",
    "mil": "Milwaukee Bucks",
    "min": "Minnesota Timberwolves",
    "no": "New Orleans Pelicans",
    "nyk": "New York Knicks",
    "okc": "Oklahoma City Thunder",
    "orl": "Orlando Magic",
    "phi": "Philadelphia 76ers",
    "phx": "Phoenix Suns",
    "por": "Portland Trail Blazers",
    "sa": "San Antonio Spurs",
    "sac": "Sacramento Kings",
    "tor": "Toronto Raptors",
    "utah": "Utah Jazz",
    "wsh": "Washington Wizards",
    # Historical team names
    "nj": "Brooklyn Nets",  # New Jersey Nets -> Brooklyn
    "sea": "Oklahoma City Thunder",  # Seattle Supersonics -> OKC
    "cha_old": "Charlotte Hornets",  # Bobcats era
    "van": "Memphis Grizzlies",  # Vancouver Grizzlies -> Memphis
    "ny": "New York Knicks",  # Alternative abbreviation
    "nyk": "New York Knicks",
}


def load_kaggle_data() -> pd.DataFrame:
    """Load and preprocess Kaggle NBA dataset."""
    kaggle_path = EXTERNAL_DIR / "kaggle" / "nba_2008-2025.csv"
    
    if not kaggle_path.exists():
        raise FileNotFoundError(f"Kaggle data not found: {kaggle_path}")
    
    logger.info(f"Loading Kaggle data from {kaggle_path}")
    df = pd.read_csv(kaggle_path)
    logger.info(f"  Loaded {len(df):,} games from Kaggle")
    
    # Parse date
    df["game_date"] = pd.to_datetime(df["date"])
    
    # Map team abbreviations to full names
    df["home_team"] = df["home"].map(TEAM_ABBREV_TO_NAME)
    df["away_team"] = df["away"].map(TEAM_ABBREV_TO_NAME)
    
    # Check for unmapped teams
    unmapped_home = df[df["home_team"].isna()]["home"].unique()
    unmapped_away = df[df["away_team"].isna()]["away"].unique()
    if len(unmapped_home) > 0:
        logger.warning(f"Unmapped home teams: {unmapped_home}")
    if len(unmapped_away) > 0:
        logger.warning(f"Unmapped away teams: {unmapped_away}")
    
    # Calculate 1H scores from quarter scores
    df["1h_score_home"] = df["q1_home"] + df["q2_home"]
    df["1h_score_away"] = df["q1_away"] + df["q2_away"]
    df["1h_total_actual"] = df["1h_score_home"] + df["1h_score_away"]
    df["1h_margin"] = df["1h_score_home"] - df["1h_score_away"]  # Home perspective
    
    # Full game actuals
    df["fg_total_actual"] = df["score_home"] + df["score_away"]
    df["fg_margin"] = df["score_home"] - df["score_away"]  # Home perspective
    
    # Rename Kaggle columns to standardized names
    # NOTE: Kaggle "h2_spread" and "h2_total" are FIRST HALF lines (despite confusing name)
    # Verified by: h2_total/total ratio = 0.497 (exactly half)
    # And: home covers h2_spread 50% of time when properly signed
    df = df.rename(columns={
        "spread": "fg_spread_line",
        "total": "fg_total_line",
        "h2_spread": "fh_spread_line",  # First half spread (absolute value)
        "h2_total": "fh_total_line",    # First half total
    })
    
    # Kaggle stores spreads as ABSOLUTE VALUES
    # whos_favored indicates direction: "home" means home is favorite (negative spread)
    # Convert to standard convention: negative = home favored, positive = home underdog
    
    # Full game spread sign adjustment
    df["fg_spread_line"] = df.apply(
        lambda row: -row["fg_spread_line"] if row["whos_favored"] == "home" else row["fg_spread_line"],
        axis=1
    )
    
    # First half spread sign adjustment (same direction as full game)
    df["fh_spread_line"] = df.apply(
        lambda row: -row["fh_spread_line"] if row["whos_favored"] == "home" else row["fh_spread_line"],
        axis=1
    )
    
    # Create outcome labels
    # Spread covered: home team beats the spread
    # If home spread is -5, and home wins by 7, margin=7, spread=-5, 7+(-5)=2>0 = covered
    df["fg_spread_covered"] = (df["fg_margin"] + df["fg_spread_line"]) > 0
    
    # Total over: actual total > line
    df["fg_total_over"] = df["fg_total_actual"] > df["fg_total_line"]
    
    # 1H outcomes (properly signed, with NaN handling)
    # Use np.where to preserve NaN when line is missing
    df["1h_spread_covered"] = np.where(
        df["fh_spread_line"].notna(),
        (df["1h_margin"] + df["fh_spread_line"]) > 0,
        np.nan
    )
    df["1h_total_over"] = np.where(
        df["fh_total_line"].notna(),
        df["1h_total_actual"] > df["fh_total_line"],
        np.nan
    )
    
    return df


def calculate_rolling_features(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Calculate rolling team statistics for feature engineering.
    
    Features calculated (per team):
    - Rolling PPG (points per game)
    - Rolling PAPG (points allowed per game)
    - Rolling win percentage
    - Rolling margin
    - Rest days since last game
    - Back-to-back indicator
    """
    logger.info(f"Calculating rolling features with window={window}")
    
    # Sort by date for proper rolling calculations
    df = df.sort_values("game_date").reset_index(drop=True)
    
    # Create team-game records (each game appears twice: once per team)
    home_records = df[["game_date", "home_team", "score_home", "score_away", "season"]].copy()
    home_records.columns = ["game_date", "team", "points_for", "points_against", "season"]
    home_records["is_home"] = True
    
    away_records = df[["game_date", "away_team", "score_away", "score_home", "season"]].copy()
    away_records.columns = ["game_date", "team", "points_for", "points_against", "season"]
    away_records["is_home"] = False
    
    team_games = pd.concat([home_records, away_records]).sort_values(["team", "game_date"])
    
    # Calculate per-team rolling stats
    team_games["win"] = team_games["points_for"] > team_games["points_against"]
    team_games["margin"] = team_games["points_for"] - team_games["points_against"]
    
    # Rolling calculations per team
    team_games = team_games.sort_values(["team", "game_date"]).reset_index(drop=True)
    
    # Calculate rolling stats using transform (keeps original index)
    for col, source in [
        ("rolling_ppg", "points_for"),
        ("rolling_papg", "points_against"),
        ("rolling_win_pct", "win"),
        ("rolling_margin", "margin"),
    ]:
        team_games[col] = team_games.groupby("team")[source].transform(
            lambda x: x.rolling(window, min_periods=3).mean()
        )
    
    # Shift to avoid data leakage (use only past data)
    for col in ["rolling_ppg", "rolling_papg", "rolling_win_pct", "rolling_margin"]:
        team_games[col] = team_games.groupby("team")[col].shift(1)
    
    team_rolling = team_games
    
    # Calculate rest days
    team_rolling["prev_game_date"] = team_rolling.groupby("team")["game_date"].shift(1)
    team_rolling["rest_days"] = (team_rolling["game_date"] - team_rolling["prev_game_date"]).dt.days
    team_rolling["is_b2b"] = team_rolling["rest_days"] == 1
    
    # Merge back to original games
    # Home team features
    home_features = team_rolling[team_rolling["is_home"]][
        ["game_date", "team", "rolling_ppg", "rolling_papg", "rolling_win_pct", 
         "rolling_margin", "rest_days", "is_b2b"]
    ].rename(columns={
        "team": "home_team",
        "rolling_ppg": "home_rolling_ppg",
        "rolling_papg": "home_rolling_papg",
        "rolling_win_pct": "home_rolling_win_pct",
        "rolling_margin": "home_rolling_margin",
        "rest_days": "home_rest_days",
        "is_b2b": "home_is_b2b",
    })
    
    # Away team features
    away_features = team_rolling[~team_rolling["is_home"]][
        ["game_date", "team", "rolling_ppg", "rolling_papg", "rolling_win_pct",
         "rolling_margin", "rest_days", "is_b2b"]
    ].rename(columns={
        "team": "away_team",
        "rolling_ppg": "away_rolling_ppg",
        "rolling_papg": "away_rolling_papg",
        "rolling_win_pct": "away_rolling_win_pct",
        "rolling_margin": "away_rolling_margin",
        "rest_days": "away_rest_days",
        "is_b2b": "away_is_b2b",
    })
    
    # Merge features into original dataframe
    df = df.merge(home_features, on=["game_date", "home_team"], how="left")
    df = df.merge(away_features, on=["game_date", "away_team"], how="left")
    
    # Derived features
    df["ppg_diff"] = df["home_rolling_ppg"] - df["away_rolling_ppg"]
    df["papg_diff"] = df["home_rolling_papg"] - df["away_rolling_papg"]
    df["win_pct_diff"] = df["home_rolling_win_pct"] - df["away_rolling_win_pct"]
    df["margin_diff"] = df["home_rolling_margin"] - df["away_rolling_margin"]
    df["rest_advantage"] = df["home_rest_days"] - df["away_rest_days"]
    
    return df


def filter_valid_games(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to games with valid betting lines and outcomes."""
    initial_count = len(df)
    
    # Must have FG spread line
    df = df[df["fg_spread_line"].notna()]
    
    # Must have FG total line
    df = df[df["fg_total_line"].notna()]
    
    # Must have valid team names
    df = df[df["home_team"].notna() & df["away_team"].notna()]
    
    # Must have scores
    df = df[df["score_home"].notna() & df["score_away"].notna()]
    
    # Must have rolling features (requires minimum games)
    df = df[df["home_rolling_ppg"].notna() & df["away_rolling_ppg"].notna()]
    
    final_count = len(df)
    logger.info(f"  Filtered: {initial_count:,} -> {final_count:,} valid games")
    
    return df


def select_training_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select and order columns for training data output."""
    
    # Core identification
    id_cols = [
        "game_date", "season", "home_team", "away_team",
    ]
    
    # Betting lines (inputs)
    line_cols = [
        "fg_spread_line", "fg_total_line",
        "fh_spread_line", "fh_total_line",
    ]
    
    # Labels (targets)
    label_cols = [
        "fg_spread_covered", "fg_total_over",
        "1h_spread_covered", "1h_total_over",
    ]
    
    # Rolling features (model inputs)
    feature_cols = [
        "home_rolling_ppg", "home_rolling_papg", "home_rolling_win_pct", "home_rolling_margin",
        "away_rolling_ppg", "away_rolling_papg", "away_rolling_win_pct", "away_rolling_margin",
        "home_rest_days", "away_rest_days", "home_is_b2b", "away_is_b2b",
        "ppg_diff", "papg_diff", "win_pct_diff", "margin_diff", "rest_advantage",
    ]
    
    # Actuals (for backtesting verification)
    actual_cols = [
        "score_home", "score_away", "fg_margin", "fg_total_actual",
        "1h_score_home", "1h_score_away", "1h_margin", "1h_total_actual",
    ]
    
    all_cols = id_cols + line_cols + label_cols + feature_cols + actual_cols
    
    # Filter to existing columns
    available_cols = [c for c in all_cols if c in df.columns]
    missing_cols = [c for c in all_cols if c not in df.columns]
    
    if missing_cols:
        logger.warning(f"Missing columns: {missing_cols}")
    
    return df[available_cols]


def build_training_data(seasons_filter: list[int] | None = None) -> Path:
    """
    Build training data from all sources.
    
    Args:
        seasons_filter: If provided, filter to only these seasons (e.g., [2020, 2021, 2022, 2023, 2024, 2025])
    
    Returns:
        Path to the output CSV file
    """
    logger.info("=" * 60)
    logger.info("Building Experimental Training Data")
    logger.info("=" * 60)
    
    # Create output directory
    EXPERIMENTAL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load Kaggle data
    df = load_kaggle_data()
    
    # Filter seasons if requested
    if seasons_filter:
        df = df[df["season"].isin(seasons_filter)]
        logger.info(f"  Filtered to seasons: {seasons_filter}")
        logger.info(f"  Remaining games: {len(df):,}")
    
    # Calculate rolling features
    df = calculate_rolling_features(df, window=10)
    
    # Filter to valid games
    df = filter_valid_games(df)
    
    # Select training columns
    df = select_training_features(df)
    
    # Determine output filename
    if seasons_filter:
        min_season = min(seasons_filter)
        max_season = max(seasons_filter)
        filename = f"training_data_{min_season}_{max_season}.csv"
    else:
        filename = "training_data_full.csv"
    
    output_path = EXPERIMENTAL_DIR / filename
    
    # Save
    df.to_csv(output_path, index=False)
    logger.info(f"  Saved: {output_path}")
    logger.info(f"  Games: {len(df):,}")
    logger.info(f"  Seasons: {df['season'].unique().tolist()}")
    logger.info(f"  Date range: {df['game_date'].min()} to {df['game_date'].max()}")
    
    # Save manifest
    manifest = {
        "created_at": datetime.now().isoformat(),
        "source": "kaggle/nba_2008-2025.csv",
        "games": len(df),
        "seasons": sorted(df["season"].unique().tolist()),
        "features": list(df.columns),
        "date_range": {
            "start": str(df["game_date"].min()),
            "end": str(df["game_date"].max()),
        }
    }
    
    import json
    manifest_path = EXPERIMENTAL_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Build experimental training data from historical sources"
    )
    parser.add_argument(
        "--seasons",
        type=int,
        default=None,
        help="Number of recent seasons to include (default: all)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: data/experimental)"
    )
    
    args = parser.parse_args()
    
    # Determine seasons filter
    seasons_filter = None
    if args.seasons:
        # Get last N seasons
        current_year = datetime.now().year
        if datetime.now().month < 10:  # Before October
            current_season = current_year
        else:
            current_season = current_year + 1
        
        seasons_filter = list(range(current_season - args.seasons + 1, current_season + 1))
        logger.info(f"Filtering to last {args.seasons} seasons: {seasons_filter}")
    
    # Build all variants
    logger.info("\n" + "=" * 60)
    logger.info("Building FULL dataset (all seasons)")
    logger.info("=" * 60)
    build_training_data(seasons_filter=None)
    
    logger.info("\n" + "=" * 60)
    logger.info("Building 7-SEASON dataset (2018-2025)")
    logger.info("=" * 60)
    build_training_data(seasons_filter=list(range(2018, 2026)))
    
    logger.info("\n" + "=" * 60)
    logger.info("Building 5-SEASON dataset (2020-2025)")
    logger.info("=" * 60)
    build_training_data(seasons_filter=list(range(2020, 2026)))
    
    logger.info("\n" + "=" * 60)
    logger.info("Building 3-SEASON dataset (2023-2025)")
    logger.info("=" * 60)
    build_training_data(seasons_filter=list(range(2023, 2026)))
    
    logger.info("\n✓ Done! Training data saved to data/experimental/")


if __name__ == "__main__":
    main()
