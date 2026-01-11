"""
Merge training data files to create a comprehensive dataset.

Merges:
- training_data_complete_2023.csv (3,979 games, 2023-01-01 to 2026-01-10)
- training_data_2025_26.csv (576 games, 2025-10-22 to 2026-01-19)

Output: data/processed/training_data_all_seasons.csv

Handles:
- Column schema alignment
- Duplicate game removal
- Date parsing consistency
"""

import argparse
import logging
from pathlib import Path
from typing import Set, Tuple

import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_main_file(path: Path) -> pd.DataFrame:
    """Load the main training data file."""
    logger.info(f"Loading main file: {path}")
    df = pd.read_csv(path, low_memory=False)
    logger.info(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def load_2025_26_file(path: Path) -> pd.DataFrame:
    """Load the 2025-26 season file."""
    logger.info(f"Loading 2025-26 file: {path}")
    df = pd.read_csv(path, low_memory=False)
    logger.info(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def normalize_date(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize game_date column to consistent datetime format."""
    df = df.copy()
    
    # Parse game_date with mixed format handling
    df['game_date'] = pd.to_datetime(df['game_date'], format='mixed', errors='coerce')
    
    # Remove any rows with invalid dates
    invalid_dates = df['game_date'].isna().sum()
    if invalid_dates > 0:
        logger.warning(f"  Dropping {invalid_dates} rows with invalid dates")
        df = df.dropna(subset=['game_date'])
    
    return df


def create_match_key(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a unique match key for deduplication.
    
    Format: YYYY-MM-DD_away_vs_home (normalized)
    """
    df = df.copy()
    
    # Ensure we have home_team and away_team
    if 'home_team' not in df.columns and 'home' in df.columns:
        df['home_team'] = df['home']
    if 'away_team' not in df.columns and 'away' in df.columns:
        df['away_team'] = df['away']
    
    # Create date string
    date_str = df['game_date'].dt.strftime('%Y-%m-%d')
    
    # Normalize team names (lowercase, strip whitespace)
    home_norm = df['home_team'].str.lower().str.strip()
    away_norm = df['away_team'].str.lower().str.strip()
    
    # Create dedup key
    df['dedup_key'] = date_str + '_' + away_norm + '_vs_' + home_norm
    
    return df


def identify_duplicates(df_main: pd.DataFrame, df_2526: pd.DataFrame) -> Tuple[Set[str], int]:
    """
    Identify duplicate games between the two datasets.
    
    Returns:
        Tuple of (set of duplicate dedup_keys, count)
    """
    main_keys = set(df_main['dedup_key'].dropna())
    s26_keys = set(df_2526['dedup_key'].dropna())
    
    duplicates = main_keys & s26_keys
    return duplicates, len(duplicates)


def merge_datasets(df_main: pd.DataFrame, df_2526: pd.DataFrame, duplicates: Set[str]) -> pd.DataFrame:
    """
    Merge the two datasets, preferring data from the main file for duplicates.
    
    The main file has 267 columns vs 44 in 2025-26, so we prefer its richer data.
    """
    logger.info("Merging datasets...")
    
    # Remove duplicates from 2025-26 file (prefer main file's richer data)
    df_2526_unique = df_2526[~df_2526['dedup_key'].isin(duplicates)].copy()
    logger.info(f"  After removing duplicates: {len(df_2526_unique)} unique rows from 2025-26")
    
    # Get all columns from both dataframes
    all_columns = list(df_main.columns)
    for col in df_2526_unique.columns:
        if col not in all_columns:
            all_columns.append(col)
    
    # Add missing columns to 2025-26 data (will be NaN) using efficient method
    missing_cols = [col for col in all_columns if col not in df_2526_unique.columns]
    if missing_cols:
        # Create a DataFrame with all missing columns at once (more efficient)
        missing_df = pd.DataFrame(
            np.nan, 
            index=df_2526_unique.index, 
            columns=missing_cols
        )
        df_2526_unique = pd.concat([df_2526_unique, missing_df], axis=1)
    
    # Reorder columns to match main file
    df_2526_unique = df_2526_unique[all_columns].copy()
    
    # Concatenate
    df_merged = pd.concat([df_main, df_2526_unique], ignore_index=True)
    
    # Sort by date
    df_merged = df_merged.sort_values('game_date').reset_index(drop=True)
    
    logger.info(f"  Final merged dataset: {len(df_merged)} rows")
    
    return df_merged


def validate_merged_data(df: pd.DataFrame) -> bool:
    """Validate the merged dataset."""
    logger.info("Validating merged data...")
    
    valid = True
    
    # Check for duplicate dedup_keys
    dup_count = df['dedup_key'].duplicated().sum()
    if dup_count > 0:
        logger.error(f"  [FAIL] Found {dup_count} duplicate games!")
        valid = False
    else:
        logger.info("  [OK] No duplicate games")
    
    # Check essential columns
    essential = ['game_date', 'home_team', 'away_team', 'home_score', 'away_score']
    missing = [col for col in essential if col not in df.columns]
    if missing:
        logger.error(f"  [FAIL] Missing essential columns: {missing}")
        valid = False
    else:
        logger.info("  [OK] All essential columns present")
    
    # Check for valid scores
    score_nulls = df[['home_score', 'away_score']].isna().sum().sum()
    if score_nulls > 0:
        logger.warning(f"  [WARN] {score_nulls} null score values")
    else:
        logger.info("  [OK] All scores valid")
    
    # Report date range
    min_date = df['game_date'].min()
    max_date = df['game_date'].max()
    logger.info(f"  Date range: {min_date} to {max_date}")
    
    # Report by season
    df['season'] = df['game_date'].dt.year.astype(str) + '-' + (df['game_date'].dt.year + 1).astype(str)
    # Adjust for NBA season (Oct-Jun)
    mask = df['game_date'].dt.month >= 10
    df.loc[mask, 'season'] = df.loc[mask, 'game_date'].dt.year.astype(str) + '-' + (df.loc[mask, 'game_date'].dt.year + 1).astype(str)
    df.loc[~mask, 'season'] = (df.loc[~mask, 'game_date'].dt.year - 1).astype(str) + '-' + df.loc[~mask, 'game_date'].dt.year.astype(str)
    
    season_counts = df.groupby('season').size()
    logger.info("  Games by season:")
    for season, count in season_counts.items():
        logger.info(f"    {season}: {count} games")
    
    return valid


def main():
    parser = argparse.ArgumentParser(description='Merge NBA training data files')
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed/training_data_all_seasons.csv',
        help='Output file path'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Analyze files without writing output'
    )
    args = parser.parse_args()
    
    # Define file paths
    main_path = Path('data/processed/training_data_complete_2023.csv')
    s26_path = Path('data/processed/training_data_2025_26.csv')
    output_path = Path(args.output)
    
    # Verify input files exist
    if not main_path.exists():
        logger.error(f"Main file not found: {main_path}")
        return 1
    if not s26_path.exists():
        logger.error(f"2025-26 file not found: {s26_path}")
        return 1
    
    # Load files
    df_main = load_main_file(main_path)
    df_2526 = load_2025_26_file(s26_path)
    
    # Normalize dates
    df_main = normalize_date(df_main)
    df_2526 = normalize_date(df_2526)
    
    # Create match keys for deduplication
    df_main = create_match_key(df_main)
    df_2526 = create_match_key(df_2526)
    
    # Identify duplicates
    duplicates, dup_count = identify_duplicates(df_main, df_2526)
    logger.info(f"Found {dup_count} duplicate games between files")
    
    if args.dry_run:
        logger.info("Dry run - not writing output")
        
        # Show what would happen
        unique_from_2526 = len(df_2526) - dup_count
        total = len(df_main) + unique_from_2526
        logger.info(f"Would merge: {len(df_main)} + {unique_from_2526} = {total} total games")
        return 0
    
    # Merge datasets
    df_merged = merge_datasets(df_main, df_2526, duplicates)
    
    # Validate
    valid = validate_merged_data(df_merged)
    if not valid:
        logger.error("Validation failed - check errors above")
        return 1
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Drop the temporary dedup_key column
    df_merged = df_merged.drop(columns=['dedup_key'])
    
    # Write output
    logger.info(f"Writing merged data to: {output_path}")
    df_merged.to_csv(output_path, index=False)
    logger.info(f"Successfully wrote {len(df_merged)} games to {output_path}")
    
    return 0


if __name__ == '__main__':
    exit(main())
