#!/usr/bin/env python3
"""
Process Kaggle TeamStatistics.csv to generate advanced features:
1. Rest Days (days since last game)
2. Rolling Advanced Stats (Off Rtg, Def Rtg, Pace - if not already present)
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "kaggle" / "TeamStatistics.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "kaggle_features.csv"

def main():
    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found.")
        print("Run scripts/fetch_external_data.py first.")
        return 1

    print(f"Loading {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return 1
        
    print(f"Loaded {len(df):,} rows.")
    
    # Check required columns
    required = ['teamId', 'gameDateTimeEst', 'gameId']
    if not all(col in df.columns for col in required):
        print(f"Missing columns. Found: {df.columns.tolist()}")
        return 1

    # Convert date
    # Handle timezone offsets like "2024-10-04 12:00:00-04:00"
    df['date'] = pd.to_datetime(df['gameDateTimeEst'], format='mixed', utc=True)
    df = df.sort_values(['teamId', 'date'])
    
    # Create full team name for merging
    df['team_full_name'] = df['teamCity'] + ' ' + df['teamName']
    
    # 1. Calculate Rest Days
    print("Calculating Rest Days...")
    df['prev_game_date'] = df.groupby('teamId')['date'].shift(1)
    df['rest_days'] = (df['date'] - df['prev_game_date']).dt.days
    
    # Cap rest days at 7 (anything > 7 is essentially "well rested") and fill NA with 3 (avg rest)
    df['rest_days'] = df['rest_days'].fillna(3).clip(upper=7).astype(int)
    
    # 2. Rolling Stats (e.g. Points, FG%)
    # This dataset has 'fieldGoalsPercentage', 'threePointersPercentage', 'pointsInThePaint', etc.
    cols_to_roll = [
        'fieldGoalsPercentage', 
        'threePointersPercentage', 
        'freeThrowsPercentage', 
        'reboundsTotal',
        'assists', 
        'turnovers', 
        'pointsInThePaint', 
        'pointsFastBreak', 
        'pointsSecondChance'
    ]
    
    # Filter to columns that actually exist
    cols_to_roll = [c for c in cols_to_roll if c in df.columns]
    
    print(f"Calculating rolling averages for: {cols_to_roll}")
    
    for col in cols_to_roll:
        # 5-game rolling average (shifted so we don't include current game in prediction features)
        df[f'rolling_5_{col}'] = df.groupby('teamId')[col].transform(
            lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
        )
        
        # 10-game rolling
        df[f'rolling_10_{col}'] = df.groupby('teamId')[col].transform(
            lambda x: x.shift(1).rolling(window=10, min_periods=3).mean()
        )

    # Select output columns
    output_cols = ['gameId', 'teamId', 'team_full_name', 'date', 'rest_days'] + \
                  [c for c in df.columns if 'rolling_' in c]
    
    out_df = df[output_cols].copy()
    
    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_FILE, index=False)
    
    print("="*60)
    print("KAGGLE FEATURE EXTRACTION COMPLETE")
    print("="*60)
    print(f"Saved to {OUTPUT_FILE}")
    print(f"Features generated for {len(out_df):,} team-games")
    print("\nSample Rest Days:")
    print(out_df[['date', 'teamId', 'rest_days']].head())
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
