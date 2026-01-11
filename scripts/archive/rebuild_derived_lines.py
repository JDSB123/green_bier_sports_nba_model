#!/usr/bin/env python3
"""
Rebuild Derived Lines CSV with Moneyline Data.

This script rebuilds the theodds_lines.csv file to include:
- Full Game: spread, total, moneyline (h2h)
- First Half: spread, total, moneyline (h2h_h1)

Uses median consensus prices across bookmakers for each game.

Output:
    data/historical/derived/theodds_lines.csv (updated with moneyline)

Usage:
    python scripts/rebuild_derived_lines.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Directories
DATA_DIR = PROJECT_ROOT / "data"
EXPORTS_DIR = DATA_DIR / "historical" / "exports"
DERIVED_DIR = DATA_DIR / "historical" / "derived"


def median_price(values: List[float]) -> Optional[float]:
    """Calculate median of non-null values."""
    cleaned = [v for v in values if v is not None and pd.notna(v)]
    if not cleaned:
        return None
    return float(median(cleaned))


def extract_consensus_lines(odds_df: pd.DataFrame, market_type: str = "fg") -> pd.DataFrame:
    """
    Extract consensus lines (median across bookmakers) for each event.
    
    Args:
        odds_df: DataFrame with odds data from exports
        market_type: "fg" for full game, "1h" for first half
        
    Returns:
        DataFrame with one row per event and consensus lines
    """
    # Determine market keys based on type
    if market_type == "fg":
        ml_market = "h2h"
        spread_market = "spreads"
        total_market = "totals"
        ml_home_col = "fg_ml_home"
        ml_away_col = "fg_ml_away"
        spread_col = "fg_spread_line"
        total_col = "fg_total_line"
    else:  # 1h
        ml_market = "h2h_h1"
        spread_market = "spreads_h1"
        total_market = "totals_h1"
        ml_home_col = "fh_ml_home"
        ml_away_col = "fh_ml_away"
        spread_col = "fh_spread_line"
        total_col = "fh_total_line"
    
    results = []
    
    for event_id in odds_df['event_id'].unique():
        event_data = odds_df[odds_df['event_id'] == event_id]
        
        # Get event metadata from first row
        first_row = event_data.iloc[0]
        
        row = {
            "event_id": event_id,
            "home_team": first_row["home_team"],
            "away_team": first_row["away_team"],
            "commence_time": first_row["commence_time"],
        }
        
        if "snapshot_timestamp" in event_data.columns:
            row["snapshot_timestamp"] = first_row["snapshot_timestamp"]
        
        # Extract moneyline (h2h)
        ml_data = event_data[event_data['market_key'] == ml_market]
        if not ml_data.empty:
            home_ml = ml_data[ml_data['outcome_name'] == first_row["home_team"]]['outcome_price'].tolist()
            away_ml = ml_data[ml_data['outcome_name'] == first_row["away_team"]]['outcome_price'].tolist()
            row[ml_home_col] = median_price(home_ml)
            row[ml_away_col] = median_price(away_ml)
        
        # Extract spread
        spread_data = event_data[event_data['market_key'] == spread_market]
        if not spread_data.empty:
            home_spread = spread_data[spread_data['outcome_name'] == first_row["home_team"]]['outcome_point'].tolist()
            row[spread_col] = median_price(home_spread)
        
        # Extract total
        total_data = event_data[event_data['market_key'] == total_market]
        if not total_data.empty:
            over_total = total_data[total_data['outcome_name'] == "Over"]['outcome_point'].tolist()
            row[total_col] = median_price(over_total)
        
        results.append(row)
    
    return pd.DataFrame(results)


def rebuild_derived_lines() -> Path:
    """
    Rebuild the derived lines CSV with moneyline data.
    
    Returns:
        Path to saved file
    """
    all_rows = []
    
    # Process each season
    for season in ['2023-2024', '2024-2025']:
        logger.info(f"Processing {season}...")
        
        # Load full game odds
        fg_file = EXPORTS_DIR / f"{season}_odds_featured.csv"
        if fg_file.exists():
            fg_odds = pd.read_csv(fg_file)
            logger.info(f"  Loaded {len(fg_odds):,} FG odds rows")
            fg_lines = extract_consensus_lines(fg_odds, market_type="fg")
            logger.info(f"  Extracted {len(fg_lines):,} FG events")
        else:
            logger.warning(f"  FG odds not found: {fg_file}")
            fg_lines = pd.DataFrame()
        
        # Load first half odds
        h1_file = EXPORTS_DIR / f"{season}_odds_1h.csv"
        if h1_file.exists():
            h1_odds = pd.read_csv(h1_file)
            logger.info(f"  Loaded {len(h1_odds):,} 1H odds rows")
            h1_lines = extract_consensus_lines(h1_odds, market_type="1h")
            logger.info(f"  Extracted {len(h1_lines):,} 1H events")
        else:
            logger.warning(f"  1H odds not found: {h1_file}")
            h1_lines = pd.DataFrame()
        
        # Merge FG and 1H data
        if not fg_lines.empty and not h1_lines.empty:
            # Merge on event_id
            merged = fg_lines.merge(
                h1_lines[['event_id', 'fh_ml_home', 'fh_ml_away', 'fh_spread_line', 'fh_total_line']],
                on='event_id',
                how='left'
            )
        elif not fg_lines.empty:
            merged = fg_lines
        elif not h1_lines.empty:
            merged = h1_lines
        else:
            continue
        
        # Add season and line date
        merged['season'] = season
        merged['commence_time'] = pd.to_datetime(merged['commence_time'])
        merged['line_date'] = merged['commence_time'].dt.date
        
        all_rows.append(merged)
    
    # Combine all seasons
    if not all_rows:
        logger.error("No data to combine")
        return None
    
    df = pd.concat(all_rows, ignore_index=True)
    
    # Sort by date
    df = df.sort_values(['commence_time', 'home_team'])
    
    # Ensure output directory exists
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    output_file = DERIVED_DIR / "theodds_lines.csv"
    
    # Select and order columns
    columns = [
        'event_id', 'commence_time', 'home_team', 'away_team', 'snapshot_timestamp',
        'fg_ml_home', 'fg_ml_away', 'fg_spread_line', 'fg_total_line',
        'fh_ml_home', 'fh_ml_away', 'fh_spread_line', 'fh_total_line',
        'line_date', 'season'
    ]
    
    # Only include columns that exist
    existing_cols = [c for c in columns if c in df.columns]
    df = df[existing_cols]
    
    df.to_csv(output_file, index=False)
    logger.info(f"Saved {len(df):,} rows to {output_file}")
    
    return output_file


def print_summary(df: pd.DataFrame) -> None:
    """Print summary of rebuilt data."""
    print("\n" + "=" * 70)
    print("DERIVED LINES SUMMARY (with Moneyline)")
    print("=" * 70)
    print(f"Total games: {len(df):,}")
    print(f"Seasons: {sorted(df['season'].unique().tolist())}")
    
    print("\nColumn coverage:")
    for col in df.columns:
        non_null = df[col].notna().sum()
        pct = 100 * non_null / len(df)
        print(f"  {col}: {non_null:,} ({pct:.1f}%)")
    
    print("\nSample rows (first 3):")
    sample_cols = ['home_team', 'away_team', 'fg_ml_home', 'fg_ml_away', 'fg_spread_line', 'fh_ml_home', 'fh_spread_line']
    existing = [c for c in sample_cols if c in df.columns]
    print(df[existing].head(3).to_string())
    
    # Check moneyline data
    if 'fg_ml_home' in df.columns:
        print(f"\nFG Moneyline coverage: {df['fg_ml_home'].notna().sum():,} games")
    if 'fh_ml_home' in df.columns:
        print(f"1H Moneyline coverage: {df['fh_ml_home'].notna().sum():,} games")


def main() -> int:
    logger.info("Rebuilding derived lines with moneyline data...")
    
    output_file = rebuild_derived_lines()
    
    if output_file is None:
        return 1
    
    # Print summary
    df = pd.read_csv(output_file)
    print_summary(df)
    
    print("\n" + "=" * 70)
    print("REBUILD COMPLETE")
    print("=" * 70)
    print(f"Output: {output_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
