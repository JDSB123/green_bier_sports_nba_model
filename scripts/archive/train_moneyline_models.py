#!/usr/bin/env python3
"""
Train moneyline models for FG and 1H markets.

This script trains calibrated moneyline models using:
- MoneylineModel (Full Game)
- FirstHalfMoneylineModel (First Half)

Both models include:
- Probability calibration (isotonic regression)
- Moneyline-specific features (Elo, Pythagorean, momentum)
- Strength of schedule features

Usage:
    python scripts/train_moneyline_models.py
    python scripts/train_moneyline_models.py --model-type gradient_boosting
    python scripts/train_moneyline_models.py --no-calibration

Output:
    data/processed/models/moneyline_model.joblib
    data/processed/models/first_half_moneyline_model.joblib
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.models import MoneylineModel, FirstHalfMoneylineModel
from src.modeling.features import FeatureEngineer
from src.modeling import io

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROCESSED_DIR / "models"


def load_training_data() -> pd.DataFrame:
    """Load training data with game outcomes."""
    training_path = PROCESSED_DIR / "training_data.csv"
    
    if not training_path.exists():
        print(f"[ERROR] Training data not found: {training_path}")
        print("Run: python scripts/build_training_dataset.py")
        sys.exit(1)
    
    df = pd.read_csv(training_path)
    print(f"[OK] Loaded {len(df)} games from training data")
    
    # Ensure required columns
    required = ["home_team", "away_team", "date", "home_score", "away_score"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[ERROR] Missing required columns: {missing}")
        sys.exit(1)
    
    return df


def prepare_features(df: pd.DataFrame, include_1h: bool = True) -> pd.DataFrame:
    """Build features for all games."""
    print("\n[INFO] Building features...")
    
    # Ensure date is datetime
    df["date"] = pd.to_datetime(df["date"])
    
    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)
    
    # Initialize feature engineer
    fe = FeatureEngineer(lookback=10)
    
    # Build features for each game
    all_features = []
    
    for idx, game in df.iterrows():
        if idx % 50 == 0:
            print(f"  Processing game {idx + 1}/{len(df)}...")
        
        try:
            # Use games before this date as historical data
            historical = df[df["date"] < game["date"]].copy()
            
            if len(historical) < 30:
                continue
            
            features = fe.build_game_features(game, historical)
            
            if features:
                # Add labels
                features["home_win"] = 1 if game["home_score"] > game["away_score"] else 0
                features["home_team"] = game["home_team"]
                features["away_team"] = game["away_team"]
                features["date"] = game["date"]
                features["home_score"] = game["home_score"]
                features["away_score"] = game["away_score"]
                
                # Add 1H labels if available
                if include_1h and "home_q1" in game and "home_q2" in game:
                    home_1h = (game.get("home_q1", 0) or 0) + (game.get("home_q2", 0) or 0)
                    away_1h = (game.get("away_q1", 0) or 0) + (game.get("away_q2", 0) or 0)
                    if home_1h > 0 and away_1h > 0:
                        features["home_1h_win"] = 1 if home_1h > away_1h else 0
                        features["home_1h_score"] = home_1h
                        features["away_1h_score"] = away_1h
                
                all_features.append(features)
        except Exception as e:
            continue
    
    features_df = pd.DataFrame(all_features)
    print(f"[OK] Built features for {len(features_df)} games")
    
    return features_df


def train_fg_moneyline(
    features_df: pd.DataFrame,
    model_type: str = "logistic",
    use_calibration: bool = True,
) -> tuple:
    """Train full-game moneyline model."""
    print("\n" + "=" * 60)
    print("TRAINING FULL GAME MONEYLINE MODEL")
    print("=" * 60)
    
    # Filter to games with valid labels
    df = features_df[features_df["home_win"].notna()].copy()
    
    # Time-based split (80/20)
    df = df.sort_values("date")
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    print(f"Training set: {len(train_df)} games")
    print(f"Test set: {len(test_df)} games")
    
    # Initialize model
    model = MoneylineModel(
        name="moneyline_model",
        model_type=model_type,
        use_calibration=use_calibration,
    )
    
    # Get target
    y_train = train_df["home_win"].astype(int)
    y_test = test_df["home_win"].astype(int)
    
    # Train
    print(f"\nTraining {model_type} model with calibration={use_calibration}...")
    model.fit(train_df, y_train)
    
    print(f"Features used: {len(model.feature_columns)}")
    
    # Evaluate
    print("\n--- Training Set Performance ---")
    train_metrics = model.evaluate(train_df, y_train)
    print(f"Accuracy: {train_metrics.accuracy:.1%}")
    print(f"ROI: {train_metrics.roi:+.1%}")
    print(f"Brier: {train_metrics.brier:.4f}")
    
    print("\n--- Test Set Performance ---")
    test_metrics = model.evaluate(test_df, y_test)
    print(f"Accuracy: {test_metrics.accuracy:.1%}")
    print(f"ROI: {test_metrics.roi:+.1%}")
    print(f"Brier: {test_metrics.brier:.4f}")
    
    # Calibration analysis
    print("\n--- Calibration Analysis ---")
    probas = model.predict_proba(test_df)[:, 1]
    
    bins = [(0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8)]
    for low, high in bins:
        mask = (probas >= low) & (probas < high)
        if mask.sum() > 0:
            predicted = probas[mask].mean()
            actual = y_test[mask].mean()
            print(f"  {low:.0%}-{high:.0%}: Predicted={predicted:.1%}, Actual={actual:.1%}, N={mask.sum()}")
    
    return model, test_metrics


def train_1h_moneyline(
    features_df: pd.DataFrame,
    model_type: str = "logistic",
    use_calibration: bool = True,
) -> tuple:
    """Train first-half moneyline model."""
    print("\n" + "=" * 60)
    print("TRAINING FIRST HALF MONEYLINE MODEL")
    print("=" * 60)
    
    # Filter to games with 1H labels
    if "home_1h_win" not in features_df.columns:
        print("[WARN] No 1H labels available. Skipping 1H model training.")
        return None, None
    
    df = features_df[features_df["home_1h_win"].notna()].copy()
    
    if len(df) < 100:
        print(f"[WARN] Only {len(df)} games with 1H data. Need 100+. Skipping.")
        return None, None
    
    # Time-based split (80/20)
    df = df.sort_values("date")
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    print(f"Training set: {len(train_df)} games")
    print(f"Test set: {len(test_df)} games")
    
    # Initialize model
    model = FirstHalfMoneylineModel(
        name="first_half_moneyline_model",
        model_type=model_type,
        use_calibration=use_calibration,
    )
    
    # Get target
    y_train = train_df["home_1h_win"].astype(int)
    y_test = test_df["home_1h_win"].astype(int)
    
    # Train
    print(f"\nTraining {model_type} model with calibration={use_calibration}...")
    model.fit(train_df, y_train)
    
    print(f"Features used: {len(model.feature_columns)}")
    
    # Evaluate
    print("\n--- Training Set Performance ---")
    train_metrics = model.evaluate(train_df, y_train)
    print(f"Accuracy: {train_metrics.accuracy:.1%}")
    print(f"ROI: {train_metrics.roi:+.1%}")
    print(f"Brier: {train_metrics.brier:.4f}")
    
    print("\n--- Test Set Performance ---")
    test_metrics = model.evaluate(test_df, y_test)
    print(f"Accuracy: {test_metrics.accuracy:.1%}")
    print(f"ROI: {test_metrics.roi:+.1%}")
    print(f"Brier: {test_metrics.brier:.4f}")
    
    return model, test_metrics


def save_model(model, name: str, metrics) -> str:
    """Save model to disk."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    path = MODELS_DIR / f"{name}.joblib"
    
    payload = {
        "pipeline": model.pipeline,
        "model": model.model,
        "feature_columns": model.feature_columns,
        "name": model.name,
        "meta": {
            "model_type": model.model_type,
            "use_calibration": model.use_calibration,
            "trained_at": datetime.now().isoformat(),
            "accuracy": metrics.accuracy if metrics else None,
            "roi": metrics.roi if metrics else None,
            "brier": metrics.brier if metrics else None,
        },
    }
    
    io.save_model(payload, str(path))
    print(f"[OK] Saved model to {path}")
    
    return str(path)


def main():
    parser = argparse.ArgumentParser(description="Train moneyline models")
    parser.add_argument(
        "--model-type",
        choices=["logistic", "gradient_boosting"],
        default="logistic",
        help="Model type to use",
    )
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Disable probability calibration",
    )
    args = parser.parse_args()
    
    use_calibration = not args.no_calibration
    
    print("=" * 60)
    print("MONEYLINE MODEL TRAINING")
    print("=" * 60)
    print(f"Model type: {args.model_type}")
    print(f"Calibration: {use_calibration}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    df = load_training_data()
    
    # Build features
    features_df = prepare_features(df)
    
    if len(features_df) < 100:
        print(f"[ERROR] Not enough training data: {len(features_df)} games")
        sys.exit(1)
    
    # Train FG model
    fg_model, fg_metrics = train_fg_moneyline(
        features_df,
        model_type=args.model_type,
        use_calibration=use_calibration,
    )
    
    # Train 1H model
    fh_model, fh_metrics = train_1h_moneyline(
        features_df,
        model_type=args.model_type,
        use_calibration=use_calibration,
    )
    
    # Save models
    print("\n" + "=" * 60)
    print("SAVING MODELS")
    print("=" * 60)
    
    save_model(fg_model, "moneyline_model", fg_metrics)
    
    if fh_model is not None:
        save_model(fh_model, "first_half_moneyline_model", fh_metrics)
    
    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nFull Game Moneyline:")
    print(f"  Accuracy: {fg_metrics.accuracy:.1%}")
    print(f"  ROI: {fg_metrics.roi:+.1%}")
    
    if fh_metrics:
        print(f"\nFirst Half Moneyline:")
        print(f"  Accuracy: {fh_metrics.accuracy:.1%}")
        print(f"  ROI: {fh_metrics.roi:+.1%}")
    
    print(f"\nModels saved to: {MODELS_DIR}")


if __name__ == "__main__":
    main()
