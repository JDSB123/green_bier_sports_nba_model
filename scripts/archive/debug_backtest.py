#!/usr/bin/env python3
"""Debug why backtest is producing 0 results."""
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.features import FeatureEngineer
from src.modeling.models import MoneylineModel

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def main():
    print("=" * 60)
    print("DEBUG BACKTEST")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv(PROCESSED_DIR / "training_data.csv")
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date").reset_index(drop=True)
    
    # Add home_win label
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    
    print(f"Loaded {len(df)} games")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Home win rate: {df['home_win'].mean():.1%}")
    
    # Check columns
    print(f"\nColumns: {df.columns.tolist()}")
    
    # Initialize feature engineer
    fe = FeatureEngineer(lookback=10)
    
    # Try building features for game 300
    test_idx = 300
    test_game = df.iloc[test_idx]
    historical = df[df["date"] < test_game["date"]].copy()
    
    print(f"\n--- Testing game {test_idx} ---")
    print(f"Game: {test_game['home_team']} vs {test_game['away_team']}")
    print(f"Date: {test_game['date']}")
    print(f"Historical games available: {len(historical)}")
    
    # Check if home team has enough games
    home_team = test_game["home_team"]
    away_team = test_game["away_team"]
    
    home_games = historical[
        (historical["home_team"] == home_team) | 
        (historical["away_team"] == home_team)
    ]
    away_games = historical[
        (historical["home_team"] == away_team) | 
        (historical["away_team"] == away_team)
    ]
    
    print(f"Home team ({home_team}) prior games: {len(home_games)}")
    print(f"Away team ({away_team}) prior games: {len(away_games)}")
    
    # Try to build features
    try:
        features = fe.build_game_features(test_game, historical)
        if features:
            print(f"\n[OK] Features built: {len(features)} features")
            print("Sample features:")
            for k, v in list(features.items())[:10]:
                print(f"  {k}: {v}")
        else:
            print("\n[WARN] build_game_features returned empty dict")
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    
    # Now try to build features for multiple games and fit model
    print("\n--- Building training features ---")
    train_df = df.iloc[:250].copy()
    
    train_features = []
    for idx, game in train_df.iterrows():
        hist = train_df[train_df["date"] < game["date"]]
        if len(hist) < 30:
            continue
        
        try:
            features = fe.build_game_features(game, hist)
            if features:
                features["home_win"] = game["home_win"]
                train_features.append(features)
        except Exception as e:
            pass
    
    print(f"Built features for {len(train_features)} games")
    
    if len(train_features) > 50:
        train_features_df = pd.DataFrame(train_features)
        print(f"Feature columns: {train_features_df.columns.tolist()[:20]}...")
        
        # Try fitting model
        print("\n--- Fitting model ---")
        try:
            model = MoneylineModel(model_type="logistic", use_calibration=True)
            y_train = train_features_df["home_win"].astype(int)
            model.fit(train_features_df, y_train)
            print(f"[OK] Model fitted with {len(model.feature_columns)} features")
            print(f"Features used: {model.feature_columns}")
        except Exception as e:
            print(f"[ERROR] Model fitting failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("[WARN] Not enough training features generated")


if __name__ == "__main__":
    main()
