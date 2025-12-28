#!/usr/bin/env python3
"""
Prepare training data from Kaggle historical NBA dataset.

This script takes the Kaggle dataset (which has historical betting lines)
and computes engineered features for model training.

The Kaggle dataset has:
- spread, total, moneyline_home, moneyline_away
- h2_spread, h2_total (first half lines)
- Quarter scores (q1-q4)

Usage:
    python scripts/prepare_kaggle_training_data.py
    python scripts/prepare_kaggle_training_data.py --seasons 2020,2021,2022,2023,2024
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from src.config import settings
from src.modeling.features import FeatureEngineer
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Kaggle team abbreviations to ESPN full names
KAGGLE_TO_ESPN = {
    "atl": "Atlanta Hawks",
    "bos": "Boston Celtics",
    "bkn": "Brooklyn Nets",
    "brk": "Brooklyn Nets",
    "cha": "Charlotte Hornets",
    "cho": "Charlotte Hornets",
    "chi": "Chicago Bulls",
    "cle": "Cleveland Cavaliers",
    "dal": "Dallas Mavericks",
    "den": "Denver Nuggets",
    "det": "Detroit Pistons",
    "gsw": "Golden State Warriors",
    "gs": "Golden State Warriors",
    "hou": "Houston Rockets",
    "ind": "Indiana Pacers",
    "lac": "Los Angeles Clippers",
    "lal": "Los Angeles Lakers",
    "mem": "Memphis Grizzlies",
    "mia": "Miami Heat",
    "mil": "Milwaukee Bucks",
    "min": "Minnesota Timberwolves",
    "nop": "New Orleans Pelicans",
    "no": "New Orleans Pelicans",
    "nyk": "New York Knicks",
    "ny": "New York Knicks",
    "okc": "Oklahoma City Thunder",
    "orl": "Orlando Magic",
    "phi": "Philadelphia 76ers",
    "phx": "Phoenix Suns",
    "pho": "Phoenix Suns",
    "por": "Portland Trail Blazers",
    "sac": "Sacramento Kings",
    "sas": "San Antonio Spurs",
    "sa": "San Antonio Spurs",
    "tor": "Toronto Raptors",
    "uta": "Utah Jazz",
    "was": "Washington Wizards",
    "wsh": "Washington Wizards",
    # Historical
    "njn": "Brooklyn Nets",
    "nj": "Brooklyn Nets",
    "sea": "Seattle Supersonics",
}


def normalize_team(abbr: str) -> Optional[str]:
    """Convert Kaggle team abbreviation to ESPN full name."""
    return KAGGLE_TO_ESPN.get(abbr.lower().strip())


def load_kaggle_data(path: str, seasons: Optional[List[int]] = None) -> pd.DataFrame:
    """Load and clean Kaggle dataset."""
    logger.info(f"Loading Kaggle data from {path}")

    df = pd.read_csv(path)
    logger.info(f"  Loaded {len(df)} games")

    # Filter seasons if specified
    if seasons:
        df = df[df["season"].isin(seasons)]
        logger.info(f"  Filtered to seasons {seasons}: {len(df)} games")

    # Convert team names
    df["home_team"] = df["home"].apply(normalize_team)
    df["away_team"] = df["away"].apply(normalize_team)

    # Drop rows with invalid teams
    invalid_mask = df["home_team"].isna() | df["away_team"].isna()
    if invalid_mask.any():
        logger.warning(f"  Dropping {invalid_mask.sum()} games with invalid team names")
        df = df[~invalid_mask]

    # Parse date
    df["date"] = pd.to_datetime(df["date"])

    # Rename columns to match expected format
    df = df.rename(columns={
        "score_home": "home_score",
        "score_away": "away_score",
        "q1_home": "home_q1",
        "q2_home": "home_q2",
        "q3_home": "home_q3",
        "q4_home": "home_q4",
        "q1_away": "away_q1",
        "q2_away": "away_q2",
        "q3_away": "away_q3",
        "q4_away": "away_q4",
        "total": "total_line",
        "h2_total": "1h_total_line",
    })

    # Convert Kaggle spread format to standard home spread format
    # Kaggle: spread is always positive, whos_favored indicates which team
    # Standard: negative = home favored, positive = home underdog
    # Example: spread=5, whos_favored=home -> home_spread=-5 (home gives 5 points)
    # Example: spread=5, whos_favored=away -> home_spread=+5 (home gets 5 points)
    df["spread_line"] = df.apply(
        lambda r: -r["spread"] if r.get("whos_favored") == "home" else r["spread"],
        axis=1
    )
    df["1h_spread_line"] = df.apply(
        lambda r: -r["h2_spread"] if r.get("whos_favored") == "home" else r["h2_spread"]
        if pd.notna(r.get("h2_spread")) else None,
        axis=1
    )

    # Compute labels
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    df["actual_margin"] = df["home_score"] - df["away_score"]
    df["actual_total"] = df["home_score"] + df["away_score"]

    # Spread covered (home team covers if actual_margin > spread_line)
    # With proper sign: home_spread=-5 means home must win by >5 to cover
    # actual_margin=7, spread_line=-5 -> 7 > -5 = True (home covered)
    # actual_margin=3, spread_line=-5 -> 3 > -5 = True but home didn't cover by 5!
    # WRONG - need: actual_margin > -spread_line for negative spreads
    # OR: actual_margin + spread_line > 0
    df["spread_covered"] = df.apply(
        lambda r: int(r["actual_margin"] + r["spread_line"] > 0)
        if pd.notna(r.get("spread_line")) else None,
        axis=1
    )

    # Total over (actual_total > total_line)
    df["total_over"] = df.apply(
        lambda r: int(r["actual_total"] > r["total_line"])
        if pd.notna(r.get("total_line")) else None,
        axis=1
    )

    # First half labels
    df["home_1h_score"] = df["home_q1"].fillna(0) + df["home_q2"].fillna(0)
    df["away_1h_score"] = df["away_q1"].fillna(0) + df["away_q2"].fillna(0)
    df["home_1h_win"] = (df["home_1h_score"] > df["away_1h_score"]).astype(int)
    df["actual_1h_margin"] = df["home_1h_score"] - df["away_1h_score"]
    df["actual_1h_total"] = df["home_1h_score"] + df["away_1h_score"]

    df["1h_spread_covered"] = df.apply(
        lambda r: int(r["actual_1h_margin"] + r["1h_spread_line"] > 0)
        if pd.notna(r.get("1h_spread_line")) else None,
        axis=1
    )

    df["1h_total_over"] = df.apply(
        lambda r: int(r["actual_1h_total"] > r["1h_total_line"])
        if pd.notna(r.get("1h_total_line")) else None,
        axis=1
    )

    logger.info(f"  Processed {len(df)} games with labels")
    logger.info(f"  Spread coverage: {df['spread_line'].notna().sum()} games ({df['spread_line'].notna().mean()*100:.1f}%)")
    logger.info(f"  Total coverage: {df['total_line'].notna().sum()} games ({df['total_line'].notna().mean()*100:.1f}%)")
    logger.info(f"  Moneyline coverage: {df['moneyline_home'].notna().sum()} games ({df['moneyline_home'].notna().mean()*100:.1f}%)")

    return df


def compute_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute engineered features for training."""
    logger.info("Computing engineered features...")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    fe = FeatureEngineer(lookback=10)

    all_features = []
    total_games = len(df)

    for idx, game in df.iterrows():
        game_date = pd.to_datetime(game["date"])

        # Historical games are all games BEFORE this game
        historical_df = df[df["date"] < game_date].copy()

        # Need at least 10 games of history for reliable stats
        if len(historical_df) < 10:
            all_features.append({})
            continue

        try:
            features = fe.build_game_features(game, historical_df)
            all_features.append(features if features else {})
        except Exception as e:
            all_features.append({})

        if (idx + 1) % 500 == 0:
            logger.info(f"  Processed {idx + 1}/{total_games} games...")

    # Merge features back into dataframe
    if all_features:
        features_df = pd.DataFrame(all_features)

        # Drop columns that would conflict
        overlap_cols = [c for c in features_df.columns if c in df.columns]
        if overlap_cols:
            features_df = features_df.drop(columns=overlap_cols, errors="ignore")

        df = pd.concat([df.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)

        new_feature_count = len(features_df.columns)
        games_with_features = features_df.notna().any(axis=1).sum()
        logger.info(f"✓ Computed {new_feature_count} engineered features for {games_with_features}/{total_games} games")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Prepare training data from Kaggle historical dataset"
    )
    parser.add_argument(
        "--seasons",
        type=str,
        default="2020,2021,2022,2023,2024,2025",
        help="Comma-separated seasons to include (e.g., 2020,2021,2022)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="training_data.csv",
        help="Output filename (in data/processed/)",
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Skip feature engineering (for quick testing)",
    )
    args = parser.parse_args()

    # Parse seasons
    seasons = [int(s.strip()) for s in args.seasons.split(",")]

    # Input/output paths
    kaggle_path = Path(settings.data_raw_dir).parent / "external" / "kaggle" / "nba_2008-2025.csv"
    output_path = Path(settings.data_processed_dir) / args.output

    if not kaggle_path.exists():
        logger.error(f"Kaggle data not found at {kaggle_path}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("PREPARING TRAINING DATA FROM KAGGLE DATASET")
    logger.info("=" * 60)
    logger.info(f"Seasons: {seasons}")
    logger.info(f"Output: {output_path}")

    # Load and clean data
    df = load_kaggle_data(str(kaggle_path), seasons)

    # Compute engineered features
    if not args.skip_features:
        df = compute_engineered_features(df)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"✓ Saved training data to {output_path}")

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total games: {len(df)}")
    logger.info(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    logger.info(f"Games with spread lines: {df['spread_line'].notna().sum()}")
    logger.info(f"Games with total lines: {df['total_line'].notna().sum()}")
    logger.info(f"Games with moneylines: {df['moneyline_home'].notna().sum()}")

    feature_cols = ["home_ppg", "away_ppg", "home_elo", "away_elo", "ppg_diff", "elo_diff"]
    existing = [c for c in feature_cols if c in df.columns]
    logger.info(f"Engineered features: {len(existing)}/{len(feature_cols)} core features present")


if __name__ == "__main__":
    main()
