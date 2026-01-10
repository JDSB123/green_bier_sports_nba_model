#!/usr/bin/env python3
"""
Ingest FiveThirtyEight ELO Ratings for NBA Historical Data.

This script fetches FREE historical ELO data from FiveThirtyEight's GitHub
repository and stores it in our historical data directory for use in
feature engineering and backtesting.

Data Sources:
- nbaallelo.csv: Historical ELO ratings (1946-present)
- nba_elo_latest.csv: Latest season ELO forecasts
- nba_elo.csv: NBA forecasts with ELO

Usage:
    python scripts/ingest_elo_ratings.py
    python scripts/ingest_elo_ratings.py --dataset elo_historical
    python scripts/ingest_elo_ratings.py --all

Output:
    data/historical/elo/
    ├── fivethirtyeight_elo_historical.csv
    ├── fivethirtyeight_elo_latest.csv
    └── fivethirtyeight_nba_forecasts.csv
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.github_data import (
    GitHubDataFetcher,
    FIVETHIRTYEIGHT_URLS,
    fetch_fivethirtyeight_elo,
    fetch_fivethirtyeight_all,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Output directory for ELO data
ELO_OUTPUT_DIR = PROJECT_ROOT / "data" / "historical" / "elo"


async def ingest_elo_dataset(dataset: str) -> Path:
    """
    Ingest a single FiveThirtyEight ELO dataset.
    
    Args:
        dataset: Dataset name ('elo_historical', 'elo_latest', 'nba_forecasts')
        
    Returns:
        Path to saved file
    """
    logger.info(f"Ingesting ELO dataset: {dataset}")
    
    # Fetch data
    df = await fetch_fivethirtyeight_elo(dataset, use_cache=False)
    
    # Ensure output directory exists
    ELO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    output_file = ELO_OUTPUT_DIR / f"fivethirtyeight_{dataset}.csv"
    df.to_csv(output_file, index=False)
    
    logger.info(f"Saved {len(df):,} rows to {output_file}")
    return output_file


async def ingest_all_elo() -> dict[str, Path]:
    """
    Ingest all FiveThirtyEight ELO datasets.
    
    Returns:
        Dictionary mapping dataset names to saved file paths
    """
    logger.info("Ingesting all FiveThirtyEight ELO datasets...")
    
    results = {}
    all_data = await fetch_fivethirtyeight_all()
    
    ELO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for dataset_name, df in all_data.items():
        if df.empty:
            logger.warning(f"Skipping empty dataset: {dataset_name}")
            continue
            
        output_file = ELO_OUTPUT_DIR / f"fivethirtyeight_{dataset_name}.csv"
        df.to_csv(output_file, index=False)
        results[dataset_name] = output_file
        logger.info(f"Saved {dataset_name}: {len(df):,} rows to {output_file}")
    
    return results


def analyze_elo_data(df, dataset_name: str) -> None:
    """Print analysis of ELO dataset."""
    print(f"\n{'=' * 60}")
    print(f"DATASET: {dataset_name}")
    print(f"{'=' * 60}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    
    # Check date range
    date_cols = [c for c in df.columns if 'date' in c.lower()]
    if date_cols:
        date_col = date_cols[0]
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        min_date = df[date_col].min()
        max_date = df[date_col].max()
        print(f"Date range: {min_date} to {max_date}")
    
    # Check ELO columns
    elo_cols = [c for c in df.columns if 'elo' in c.lower()]
    print(f"ELO columns: {elo_cols}")
    
    # Sample data
    print(f"\nSample rows (last 5):")
    print(df.tail(5).to_string())


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest FiveThirtyEight ELO ratings for NBA"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(FIVETHIRTYEIGHT_URLS.keys()),
        help="Specific dataset to ingest",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Ingest all available datasets",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Print analysis of ingested data",
    )
    
    args = parser.parse_args()
    
    # Default to all if no specific dataset
    if not args.dataset and not args.all:
        args.all = True
    
    try:
        if args.all:
            results = await ingest_all_elo()
            print(f"\n{'=' * 60}")
            print("INGESTION COMPLETE")
            print(f"{'=' * 60}")
            for name, path in results.items():
                print(f"  {name}: {path}")
                
            if args.analyze:
                import pandas as pd
                for name, path in results.items():
                    df = pd.read_csv(path)
                    analyze_elo_data(df, name)
        else:
            path = await ingest_elo_dataset(args.dataset)
            print(f"\nIngested: {path}")
            
            if args.analyze:
                import pandas as pd
                df = pd.read_csv(path)
                analyze_elo_data(df, args.dataset)
        
        return 0
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        return 1


if __name__ == "__main__":
    import pandas as pd  # Import here for analyze function
    sys.exit(asyncio.run(main()))
