#!/usr/bin/env python3
"""Debug why backtest produces 0 predictions."""
import sys
from pathlib import Path
import pandas as pd
import traceback

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.features import FeatureEngineer

# Load data
df = pd.read_csv(PROJECT_ROOT / "data/processed/training_data.csv")
df["date"] = pd.to_datetime(df["date"], utc=True)
df = df.sort_values("date").reset_index(drop=True)

# Create labels
df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)

print(f"Loaded {len(df)} games")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

# Try building features for a single game
fe = FeatureEngineer(lookback=10)
test_idx = 200
test_game = df.iloc[test_idx]
historical = df.iloc[:test_idx]

print(f"\nTesting feature building for game {test_idx}:")
print(f"  Home: {test_game['home_team']} vs Away: {test_game['away_team']}")
print(f"  Historical games: {len(historical)}")

try:
    features = fe.build_game_features(test_game, historical)
    if features:
        print(f"  [OK] Features built: {len(features)} features")
        print(f"  Sample features: {list(features.keys())[:10]}")
    else:
        print(f"  [FAIL] build_game_features returned None/empty")
except Exception as e:
    print(f"  [FAIL] Exception: {e}")
    traceback.print_exc()

