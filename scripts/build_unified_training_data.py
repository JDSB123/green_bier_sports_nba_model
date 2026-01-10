#!/usr/bin/env python3
"""
Build unified training dataset from 2023 forward.

Data sources:
- Kaggle betting (nba_2008-2025.csv): Q1-Q4 scores + FG betting lines
- The Odds API: Real pre-game 1H lines (spreads_h1, totals_h1, h2h_h1)

Note: Kaggle h2_spread/h2_total are SECOND HALF (halftime) lines, NOT usable.

Usage:
    python scripts/build_unified_training_data.py
    python scripts/build_unified_training_data.py --start-date 2023-01-01
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Paths
DATA_DIR = PROJECT_ROOT / "data"
KAGGLE_FILE = DATA_DIR / "external" / "kaggle" / "nba_2008-2025.csv"
THEODDS_DIR = DATA_DIR / "historical" / "the_odds"
OUTPUT_DIR = DATA_DIR / "processed"

# Team abbreviation to full name mapping
TEAM_ABBREV_MAP = {
    "atl": "Atlanta Hawks", "bos": "Boston Celtics", "bkn": "Brooklyn Nets",
    "cha": "Charlotte Hornets", "chi": "Chicago Bulls", "cle": "Cleveland Cavaliers",
    "dal": "Dallas Mavericks", "den": "Denver Nuggets", "det": "Detroit Pistons",
    "gs": "Golden State Warriors", "gsw": "Golden State Warriors",
    "hou": "Houston Rockets", "ind": "Indiana Pacers",
    "lac": "Los Angeles Clippers", "lal": "Los Angeles Lakers",
    "mem": "Memphis Grizzlies", "mia": "Miami Heat", "mil": "Milwaukee Bucks",
    "min": "Minnesota Timberwolves", "no": "New Orleans Pelicans",
    "nop": "New Orleans Pelicans", "nyk": "New York Knicks", "ny": "New York Knicks",
    "okc": "Oklahoma City Thunder", "orl": "Orlando Magic",
    "phi": "Philadelphia 76ers", "phx": "Phoenix Suns",
    "por": "Portland Trail Blazers", "sac": "Sacramento Kings",
    "sa": "San Antonio Spurs", "sas": "San Antonio Spurs",
    "tor": "Toronto Raptors", "utah": "Utah Jazz", "uta": "Utah Jazz",
    "was": "Washington Wizards", "wsh": "Washington Wizards",
    # Historical
    "nj": "Brooklyn Nets", "njn": "Brooklyn Nets",
    "sea": "Oklahoma City Thunder",
    "noh": "New Orleans Pelicans", "nok": "New Orleans Pelicans",
}


def map_team_name(abbrev: str) -> str:
    """Map team abbreviation to full name."""
    return TEAM_ABBREV_MAP.get(abbrev.lower().strip(), abbrev)


def load_kaggle_data(start_date: str = "2023-01-01") -> pd.DataFrame:
    """Load Kaggle NBA betting data with Q1-Q4 scores."""
    if not KAGGLE_FILE.exists():
        logger.error(f"Kaggle file not found: {KAGGLE_FILE}")
        return pd.DataFrame()
    
    df = pd.read_csv(KAGGLE_FILE)
    df["game_date"] = pd.to_datetime(df["date"])
    
    # Filter to start date
    df = df[df["game_date"] >= start_date].copy()
    
    # Map team names
    df["home_team"] = df["home"].apply(map_team_name)
    df["away_team"] = df["away"].apply(map_team_name)
    
    # Final scores
    df["home_score"] = df["score_home"]
    df["away_score"] = df["score_away"]
    df["fg_margin"] = df["home_score"] - df["away_score"]
    df["fg_total_actual"] = df["home_score"] + df["away_score"]
    
    # Q1-Q4 scores (Kaggle has these!)
    df["home_q1"] = df["q1_home"]
    df["home_q2"] = df["q2_home"]
    df["home_q3"] = df["q3_home"]
    df["home_q4"] = df["q4_home"]
    df["away_q1"] = df["q1_away"]
    df["away_q2"] = df["q2_away"]
    df["away_q3"] = df["q3_away"]
    df["away_q4"] = df["q4_away"]
    
    # Calculate 1H scores
    df["home_1h_score"] = df["home_q1"].fillna(0) + df["home_q2"].fillna(0)
    df["away_1h_score"] = df["away_q1"].fillna(0) + df["away_q2"].fillna(0)
    df["1h_total_actual"] = df["home_1h_score"] + df["away_1h_score"]
    df["1h_margin"] = df["home_1h_score"] - df["away_1h_score"]
    
    # FG betting lines (properly signed)
    df["fg_spread_line"] = df.apply(
        lambda r: -r["spread"] if r["whos_favored"] == "home" else r["spread"],
        axis=1
    )
    df["fg_total_line"] = df["total"]
    df["fg_ml_home"] = df["moneyline_home"]
    df["fg_ml_away"] = df["moneyline_away"]
    
    # Create match key for joining with The Odds API 1H lines
    df["date_str"] = df["game_date"].dt.strftime("%Y-%m-%d")
    df["match_key"] = (
        df["date_str"] + "_" +
        df["home_team"].str.lower().str.strip() + "_" +
        df["away_team"].str.lower().str.strip()
    )
    
    logger.info(f"Loaded {len(df)} NBA games from Kaggle (from {start_date})")
    return df


def load_theodds_1h_lines() -> pd.DataFrame:
    """Load 1H betting lines from The Odds API period_odds."""
    rows = []
    
    period_dir = THEODDS_DIR / "period_odds"
    for season_dir in sorted(period_dir.glob("*")):
        if not season_dir.is_dir():
            continue
        
        for json_file in season_dir.glob("period_odds_1h.json"):
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)
                
                events = data.get("data", [])
                logger.info(f"  Processing {json_file.name} ({len(events)} events)")
                
                for event_wrapper in events:
                    event = event_wrapper.get("data", event_wrapper)
                    if not event:
                        continue
                    
                    home_team = event.get("home_team", "")
                    away_team = event.get("away_team", "")
                    commence = event.get("commence_time", "")
                    
                    if not all([home_team, away_team, commence]):
                        continue
                    
                    # Collect all bookmaker lines
                    spreads_h1 = []
                    totals_h1 = []
                    ml_home_h1 = []
                    ml_away_h1 = []
                    
                    for bm in event.get("bookmakers", []):
                        for mkt in bm.get("markets", []):
                            key = mkt.get("key", "").lower()
                            outcomes = mkt.get("outcomes", [])
                            
                            if key == "spreads_h1":
                                for o in outcomes:
                                    if o.get("name") == home_team and o.get("point") is not None:
                                        spreads_h1.append(o.get("point"))
                            
                            elif key == "totals_h1":
                                for o in outcomes:
                                    if o.get("name") == "Over" and o.get("point") is not None:
                                        totals_h1.append(o.get("point"))
                            
                            elif key == "h2h_h1":
                                for o in outcomes:
                                    if o.get("name") == home_team and o.get("price") is not None:
                                        ml_home_h1.append(o.get("price"))
                                    elif o.get("name") == away_team and o.get("price") is not None:
                                        ml_away_h1.append(o.get("price"))
                    
                    # Calculate median consensus
                    row = {
                        "commence_time": commence,
                        "home_team": home_team,
                        "away_team": away_team,
                        "1h_spread_line": _safe_median(spreads_h1),
                        "1h_total_line": _safe_median(totals_h1),
                        "1h_ml_home": _safe_median(ml_home_h1),
                        "1h_ml_away": _safe_median(ml_away_h1),
                        "1h_bookmakers": len(event.get("bookmakers", [])),
                    }
                    rows.append(row)
            
            except Exception as e:
                logger.warning(f"Error reading {json_file}: {e}")
    
    if not rows:
        logger.warning("No 1H lines found from The Odds API")
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    
    # Parse date and create match key
    df["game_date"] = pd.to_datetime(df["commence_time"]).dt.tz_localize(None)
    df["date_str"] = df["game_date"].dt.strftime("%Y-%m-%d")
    df["match_key"] = (
        df["date_str"] + "_" +
        df["home_team"].str.lower().str.strip() + "_" +
        df["away_team"].str.lower().str.strip()
    )
    
    logger.info(f"Loaded {len(df)} 1H lines from The Odds API")
    return df


def _safe_median(values: list) -> float | None:
    """Calculate median, handling None and empty lists."""
    cleaned = [v for v in values if v is not None]
    if not cleaned:
        return None
    return float(median(cleaned))


def merge_1h_lines(kaggle_df: pd.DataFrame, theodds_1h_df: pd.DataFrame) -> pd.DataFrame:
    """Merge The Odds API 1H lines into Kaggle data."""
    if theodds_1h_df.empty:
        kaggle_df["1h_spread_line"] = np.nan
        kaggle_df["1h_total_line"] = np.nan
        kaggle_df["1h_ml_home"] = np.nan
        kaggle_df["1h_ml_away"] = np.nan
        return kaggle_df
    
    # Deduplicate 1H lines
    line_cols = ["match_key", "1h_spread_line", "1h_total_line", "1h_ml_home", "1h_ml_away"]
    theodds_dedup = theodds_1h_df[line_cols].drop_duplicates(subset=["match_key"])
    
    # Merge
    before = len(kaggle_df)
    df = kaggle_df.merge(theodds_dedup, on="match_key", how="left", suffixes=("", "_theodds"))
    
    matched = df["1h_spread_line"].notna().sum()
    logger.info(f"Merged 1H lines: {matched}/{before} games matched ({matched/before*100:.1f}%)")
    
    return df


def calculate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate betting outcome labels."""
    
    # FG spread covered: home beats the spread
    df["fg_spread_covered"] = np.where(
        df["fg_spread_line"].notna(),
        (df["fg_margin"] + df["fg_spread_line"]) > 0,
        np.nan
    )
    
    # FG total over
    df["fg_total_over"] = np.where(
        df["fg_total_line"].notna(),
        df["fg_total_actual"] > df["fg_total_line"],
        np.nan
    )
    
    # 1H spread covered
    df["1h_spread_covered"] = np.where(
        df["1h_spread_line"].notna(),
        (df["1h_margin"] + df["1h_spread_line"]) > 0,
        np.nan
    )
    
    # 1H total over
    df["1h_total_over"] = np.where(
        df["1h_total_line"].notna(),
        df["1h_total_actual"] > df["1h_total_line"],
        np.nan
    )
    
    # Win labels
    df["fg_home_win"] = df["fg_margin"] > 0
    df["1h_home_win"] = df["1h_margin"] > 0
    
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add basic features."""
    df = df.sort_values("game_date").reset_index(drop=True)
    
    # Season
    df["season"] = df["game_date"].apply(
        lambda d: f"{d.year}-{d.year+1}" if d.month >= 10 else f"{d.year-1}-{d.year}"
    )
    
    # Day of week
    df["day_of_week"] = df["game_date"].dt.dayofweek
    
    # Month
    df["month"] = df["game_date"].dt.month
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Build unified training data from 2023+")
    parser.add_argument(
        "--start-date",
        type=str,
        default="2023-01-01",
        help="Start date (default: 2023-01-01)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "training_data_unified_2023.csv",
        help="Output path"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("BUILDING UNIFIED NBA TRAINING DATA (2023+)")
    print("=" * 60)
    
    # Load Kaggle data (has Q1-Q4 scores + FG lines)
    print("\n[1] Loading Kaggle NBA betting data...")
    kaggle_df = load_kaggle_data(args.start_date)
    
    if kaggle_df.empty:
        logger.error("No Kaggle data loaded")
        return 1
    
    # Load The Odds API 1H lines
    print("\n[2] Loading The Odds API 1H lines...")
    theodds_1h_df = load_theodds_1h_lines()
    
    # Merge 1H lines
    print("\n[3] Merging 1H lines...")
    df = merge_1h_lines(kaggle_df, theodds_1h_df)
    
    # Calculate labels
    print("\n[4] Calculating betting outcome labels...")
    df = calculate_labels(df)
    
    # Add features
    print("\n[5] Adding features...")
    df = add_features(df)
    
    # Select output columns
    output_cols = [
        # Identifiers
        "game_date", "season", "home_team", "away_team",
        # Scores
        "home_score", "away_score", "fg_margin", "fg_total_actual",
        "home_1h_score", "away_1h_score", "1h_margin", "1h_total_actual",
        "home_q1", "home_q2", "home_q3", "home_q4",
        "away_q1", "away_q2", "away_q3", "away_q4",
        # FG betting lines
        "fg_spread_line", "fg_total_line", "fg_ml_home", "fg_ml_away",
        # 1H betting lines (from The Odds API)
        "1h_spread_line", "1h_total_line", "1h_ml_home", "1h_ml_away",
        # Labels
        "fg_spread_covered", "fg_total_over", "fg_home_win",
        "1h_spread_covered", "1h_total_over", "1h_home_win",
        # Features
        "day_of_week", "month",
    ]
    
    output_cols = [c for c in output_cols if c in df.columns]
    df_out = df[output_cols].copy()
    
    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.output, index=False)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nTotal NBA games: {len(df_out):,}")
    print(f"Date range: {df_out['game_date'].min()} to {df_out['game_date'].max()}")
    print(f"\nFG lines available: {df_out['fg_spread_line'].notna().sum():,}")
    print(f"1H lines available: {df_out['1h_spread_line'].notna().sum():,}")
    
    # Label distributions
    for label in ["fg_spread_covered", "fg_total_over", "1h_spread_covered", "1h_total_over"]:
        if label in df_out.columns:
            valid = df_out[label].notna().sum()
            if valid > 0:
                rate = df_out[label].dropna().mean() * 100
                print(f"\n{label}: {valid:,} games, {rate:.1f}% positive")
    
    print(f"\nOutput saved: {args.output}")
    print("\n" + "=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
