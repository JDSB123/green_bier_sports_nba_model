#!/usr/bin/env python3
"""
Merge Kaggle features (Rest Days, Rolling Stats) into the main training dataset.
"""
import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings

TRAINING_DATA = PROJECT_ROOT / "data" / "processed" / "training_data_kaggle.csv"
KAGGLE_FEATURES = PROJECT_ROOT / "data" / "processed" / "kaggle_features.csv"
BACKUP_FILE = PROJECT_ROOT / "data" / "processed" / "training_data_kaggle.csv.bak"

def main():
    if not KAGGLE_FEATURES.exists():
        print(f"Error: {KAGGLE_FEATURES} not found.")
        return 1

    print("Loading datasets...")
    train_df = pd.read_csv(TRAINING_DATA)
    feats_df = pd.read_csv(KAGGLE_FEATURES)

    # Normalize dates to UTC date for merging
    # training_data date format: 2010-11-02 19:00:00-04:00
    train_df['merge_date'] = pd.to_datetime(train_df['date'], utc=True).dt.date
    
    # kaggle_features date format: 2025-10-03 05:30:00+00:00
    feats_df['merge_date'] = pd.to_datetime(feats_df['date'], utc=True).dt.date
    
    # Drop raw date from features to avoid collision
    feats_df = feats_df.drop(columns=['date', 'gameId', 'teamId'])

    print(f"Training data: {len(train_df)} games")
    print(f"Features data: {len(feats_df)} team-games")

    # Merge Home Stats
    print("Merging Home Team features...")
    # Prepare features for home join
    home_feats = feats_df.rename(columns={'team_full_name': 'home_team'})
    # Rename value columns to home_ prefix
    feature_cols = [c for c in home_feats.columns if c not in ['merge_date', 'home_team']]
    rename_map = {c: f"home_{c}" for c in feature_cols}
    home_feats = home_feats.rename(columns=rename_map)
    
    train_df = pd.merge(train_df, home_feats, on=['merge_date', 'home_team'], how='left')

    # Merge Away Stats
    print("Merging Away Team features...")
    # Prepare features for away join
    away_feats = feats_df.rename(columns={'team_full_name': 'away_team'})
    # Rename value columns to away_ prefix
    rename_map = {c: f"away_{c}" for c in feature_cols}
    away_feats = away_feats.rename(columns=rename_map)
    
    train_df = pd.merge(train_df, away_feats, on=['merge_date', 'away_team'], how='left')

    # Drop merge helper
    train_df = train_df.drop(columns=['merge_date'])

    # Fill NaNs
    # Rest days: fill with 3 (average)
    if 'home_rest_days' in train_df.columns:
        train_df['home_rest_days'] = train_df['home_rest_days'].fillna(3)
        train_df['away_rest_days'] = train_df['away_rest_days'].fillna(3)
        
        # Calculate rest advantage (positive = home has more rest)
        train_df['rest_diff'] = train_df['home_rest_days'] - train_df['away_rest_days']

    # Rolling stats: forward fill or fill with mean? 
    # For now, let's fill with 0 or mean, but XGBoost handles NaNs well.
    # However, let's check coverage.
    missing = train_df['home_rest_days'].isna().sum()
    if missing > 0:
        print(f"Warning: {missing} games missing feature data (likely team name mismatch or missing dates)")

    # Backup original
    print(f"Backing up original to {BACKUP_FILE}...")
    shutil.copy(TRAINING_DATA, BACKUP_FILE)

    # Save
    print(f"Saving merged data to {TRAINING_DATA}...")
    train_df.to_csv(TRAINING_DATA, index=False)
    
    print("Merge complete.")
    print("New columns added:")
    new_cols = [c for c in train_df.columns if c not in pd.read_csv(BACKUP_FILE, nrows=0).columns]
    print(new_cols)

    return 0

if __name__ == "__main__":
    sys.exit(main())
