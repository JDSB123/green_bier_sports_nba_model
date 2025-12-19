#!/usr/bin/env python3
"""Debug feature building and model fitting for backtest."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.modeling.features import FeatureEngineer
from src.modeling.models import MoneylineModel

def main():
    # Load data
    df = pd.read_csv(PROJECT_ROOT / "data/processed/training_data.csv")
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date").reset_index(drop=True)
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)

    print(f"Loaded {len(df)} games")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    fe = FeatureEngineer(lookback=10)
    
    # Build features for first 200 games for training
    print("\nBuilding training features for 200 games...")
    train_df = df.iloc[:200].copy()
    train_features = []
    
    for idx, game in train_df.iterrows():
        historical = train_df[train_df["date"] < game["date"]]
        if len(historical) < 30:
            continue
        
        try:
            features = fe.build_game_features(game, historical)
            if features:
                features["home_win"] = game["home_win"]
                train_features.append(features)
        except Exception as e:
            pass
    
    print(f"Built features for {len(train_features)} games")
    
    if len(train_features) > 50:
        train_features_df = pd.DataFrame(train_features)
        print(f"Feature columns: {len(train_features_df.columns)}")
        
        # Try fitting model
        print("\nFitting MoneylineModel...")
        try:
            model = MoneylineModel(model_type="logistic", use_calibration=True)
            y_train = train_features_df["home_win"].astype(int)
            model.fit(train_features_df, y_train)
            print(f"[OK] Model fitted with {len(model.feature_columns)} features")
            
            # Test prediction on a new game
            test_game = df.iloc[250]
            historical = df.iloc[:250].copy()
            test_features = fe.build_game_features(test_game, historical)
            if test_features:
                test_df = pd.DataFrame([test_features])
                proba = model.predict_proba(test_df)[0, 1]
                print(f"\n[OK] Prediction for game 250: {proba:.3f} (actual: {test_game['home_win']})")
            
        except Exception as e:
            print(f"[ERROR] Model fitting failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("[WARN] Not enough training features")

if __name__ == "__main__":
    main()
