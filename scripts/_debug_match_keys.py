"""Debug match key mismatches between training data and Kaggle."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.data.standardization import standardize_team_name, generate_match_key

# Load data
train = pd.read_csv('data/processed/training_data_complete_2023.csv', nrows=10, low_memory=False)
kaggle = pd.read_csv('data/external/kaggle/nba_2008-2025.csv', nrows=50000)
kaggle['date'] = pd.to_datetime(kaggle['date'])
kaggle = kaggle[kaggle['date'] >= '2023-01-01'].head(10)

print("=" * 60)
print("TRAINING DATA MATCH KEYS (existing):")
print("=" * 60)
for _, r in train.iterrows():
    print(f"  {r['match_key']}")

print()
print("=" * 60)
print("KAGGLE (computed from raw data):")
print("=" * 60)
for _, r in kaggle.iterrows():
    home_std = standardize_team_name(r['home'])
    away_std = standardize_team_name(r['away'])
    mk = generate_match_key(r['date'], home_std, away_std, source_is_utc=False)
    print(f"  {mk}")
    print(f"    raw: {r['date'].strftime('%Y-%m-%d')} | {r['home']} vs {r['away']}")
    print(f"    std: {home_std} vs {away_std}")
    if pd.notna(r.get('home_ml')) and pd.notna(r.get('away_ml')):
        print(f"    ML: home={r['home_ml']}, away={r['away_ml']}")

# Check one exact match attempt
print()
print("=" * 60)
print("MATCH LOOKUP TEST:")
print("=" * 60)
train_keys = set(train['match_key'].values)
kaggle_computed = []
for _, r in kaggle.iterrows():
    home_std = standardize_team_name(r['home'])
    away_std = standardize_team_name(r['away'])
    mk = generate_match_key(r['date'], home_std, away_std, source_is_utc=False)
    kaggle_computed.append(mk)
    if mk in train_keys:
        print(f"  MATCH: {mk}")

kaggle_keys = set(kaggle_computed)
overlap = train_keys & kaggle_keys
print(f"\nOverlap: {len(overlap)} keys match")
print(f"Training keys: {len(train_keys)}")
print(f"Kaggle keys: {len(kaggle_keys)}")
