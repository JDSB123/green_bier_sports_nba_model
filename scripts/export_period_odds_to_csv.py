#!/usr/bin/env python3
"""
Export Period Odds (1H Markets) from JSON to CSV Format.

This script extracts first-half betting data from the period_odds JSON files
and exports them to analysis-ready CSV format, similar to the featured odds exports.

Markets Exported:
- h2h_h1: First half moneyline
- spreads_h1: First half spread
- totals_h1: First half total

Output:
    data/historical/exports/
    ├── 2023-2024_odds_1h.csv
    ├── 2023-2024_odds_1h.parquet
    ├── 2024-2025_odds_1h.csv
    └── 2024-2025_odds_1h.parquet

Usage:
    python scripts/export_period_odds_to_csv.py
    python scripts/export_period_odds_to_csv.py --season 2024-2025
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Directories
DATA_DIR = PROJECT_ROOT / "data"
PERIOD_ODDS_DIR = DATA_DIR / "historical" / "the_odds" / "period_odds"
EXPORTS_DIR = DATA_DIR / "historical" / "exports"

# Markets to extract
FIRST_HALF_MARKETS = ["h2h_h1", "spreads_h1", "totals_h1"]


def extract_odds_from_event(event_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract odds rows from a single event's data.
    
    Args:
        event_data: Event data from period_odds JSON
        
    Returns:
        List of flattened odds rows
    """
    rows = []
    
    # Get timestamp and event info
    timestamp = event_data.get("timestamp")
    inner_data = event_data.get("data", {})
    
    event_id = inner_data.get("id", "")
    home_team = inner_data.get("home_team", "")
    away_team = inner_data.get("away_team", "")
    commence_time = inner_data.get("commence_time", "")
    
    bookmakers = inner_data.get("bookmakers", [])
    
    for bookmaker in bookmakers:
        bm_key = bookmaker.get("key", "")
        bm_title = bookmaker.get("title", "")
        bm_last_update = bookmaker.get("last_update", "")
        
        markets = bookmaker.get("markets", [])
        
        for market in markets:
            market_key = market.get("key", "")
            market_last_update = market.get("last_update", "")
            
            # Only process first-half markets
            if market_key not in FIRST_HALF_MARKETS:
                continue
            
            outcomes = market.get("outcomes", [])
            
            for outcome in outcomes:
                row = {
                    "snapshot_timestamp": timestamp,
                    "event_id": event_id,
                    "home_team": home_team,
                    "away_team": away_team,
                    "commence_time": commence_time,
                    "bookmaker_key": bm_key,
                    "bookmaker_title": bm_title,
                    "bookmaker_last_update": bm_last_update,
                    "market_key": market_key,
                    "market_last_update": market_last_update,
                    "outcome_name": outcome.get("name", ""),
                    "outcome_price": outcome.get("price"),
                    "outcome_point": outcome.get("point"),
                }
                rows.append(row)
    
    return rows


def export_season(season: str) -> tuple[Path, Path]:
    """
    Export period odds for a season to CSV and Parquet.
    
    Args:
        season: Season string (e.g., "2024-2025")
        
    Returns:
        Tuple of (csv_path, parquet_path)
    """
    period_file = PERIOD_ODDS_DIR / season / "period_odds_1h.json"
    
    if not period_file.exists():
        raise FileNotFoundError(f"Period odds file not found: {period_file}")
    
    logger.info(f"Loading period odds from {period_file}")
    
    with open(period_file) as f:
        data = json.load(f)
    
    events = data.get("data", [])
    logger.info(f"Processing {len(events)} events for {season}")
    
    # Extract all rows
    all_rows = []
    for event in events:
        rows = extract_odds_from_event(event)
        all_rows.extend(rows)
    
    logger.info(f"Extracted {len(all_rows):,} odds rows")
    
    # Create DataFrame
    df = pd.DataFrame(all_rows)
    
    # Add derived columns
    if not df.empty:
        df["game_date"] = pd.to_datetime(df["commence_time"]).dt.date
    
    # Ensure output directory exists
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    csv_path = EXPORTS_DIR / f"{season}_odds_1h.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV: {csv_path}")
    
    # Save to Parquet
    parquet_path = EXPORTS_DIR / f"{season}_odds_1h.parquet"
    df.to_parquet(parquet_path, index=False)
    logger.info(f"Saved Parquet: {parquet_path}")
    
    return csv_path, parquet_path


def print_summary(df: pd.DataFrame, season: str) -> None:
    """Print summary statistics for exported data."""
    print(f"\n{'=' * 60}")
    print(f"EXPORT SUMMARY: {season}")
    print(f"{'=' * 60}")
    print(f"Total rows: {len(df):,}")
    print(f"Unique events: {df['event_id'].nunique():,}")
    print(f"Bookmakers: {df['bookmaker_key'].unique().tolist()}")
    print(f"\nMarkets:")
    for market in df['market_key'].unique():
        count = len(df[df['market_key'] == market])
        print(f"  {market}: {count:,} rows")
    
    if 'game_date' in df.columns:
        print(f"\nDate range: {df['game_date'].min()} to {df['game_date'].max()}")
    
    # Sample h2h_h1 (moneyline) data
    h2h = df[df['market_key'] == 'h2h_h1']
    if not h2h.empty:
        print(f"\nSample h2h_h1 (1H moneyline) - first event:")
        sample = h2h[h2h['event_id'] == h2h['event_id'].iloc[0]]
        print(sample[['home_team', 'away_team', 'outcome_name', 'outcome_price']].to_string())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export period odds (1H markets) to CSV"
    )
    parser.add_argument(
        "--season",
        type=str,
        help="Specific season to export (e.g., 2024-2025)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Export all available seasons",
    )
    
    args = parser.parse_args()
    
    # Find available seasons
    available_seasons = []
    if PERIOD_ODDS_DIR.exists():
        for season_dir in sorted(PERIOD_ODDS_DIR.iterdir()):
            if season_dir.is_dir():
                if (season_dir / "period_odds_1h.json").exists():
                    available_seasons.append(season_dir.name)
    
    logger.info(f"Available seasons with 1H data: {available_seasons}")
    
    if not available_seasons:
        logger.error("No period odds data found")
        return 1
    
    # Determine which seasons to export
    if args.season:
        seasons_to_export = [args.season]
    else:
        seasons_to_export = available_seasons
    
    # Export each season
    for season in seasons_to_export:
        try:
            csv_path, parquet_path = export_season(season)
            
            # Print summary
            df = pd.read_csv(csv_path)
            print_summary(df, season)
            
        except Exception as e:
            logger.error(f"Failed to export {season}: {e}")
            return 1
    
    print(f"\n{'=' * 60}")
    print("EXPORT COMPLETE")
    print(f"{'=' * 60}")
    print(f"Exported seasons: {seasons_to_export}")
    print(f"Output directory: {EXPORTS_DIR}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
