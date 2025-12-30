#!/usr/bin/env python3
"""
Verify calibration on all 4 models.

Checks that model probabilities match actual win rates across bins.
A well-calibrated model should have:
- Predicted 60% confidence -> ~60% actual win rate
- Predicted 70% confidence -> ~70% actual win rate

Usage:
    python scripts/verify_calibration.py

Output:
    Calibration analysis for each model
    Reliability diagrams data
"""
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.models import (
    SpreadsModel,
    TotalsModel,
    FirstHalfSpreadsModel,
    FirstHalfTotalsModel,
)
from src.modeling.features import FeatureEngineer
from src.modeling import io

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
# Note: Models are at models/production/ locally, not data/processed/models/
# MODELS_DIR is not used in this script (models are trained fresh for calibration testing)


def load_training_data() -> pd.DataFrame:
    """Load training data with game outcomes."""
    training_path = PROCESSED_DIR / "training_data.csv"
    
    if not training_path.exists():
        print(f"[ERROR] Training data not found: {training_path}")
        sys.exit(1)
    
    df = pd.read_csv(training_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    if "spread_line" in df.columns:
        df["actual_margin"] = df["home_score"] - df["away_score"]
        df["spread_covered"] = (df["actual_margin"] > -df["spread_line"]).astype(int)
    
    if "total_line" in df.columns:
        df["actual_total"] = df["home_score"] + df["away_score"]
        df["total_over"] = (df["actual_total"] > df["total_line"]).astype(int)
    
    if "home_q1" in df.columns and "home_q2" in df.columns:
        df["home_1h_score"] = df["home_q1"].fillna(0) + df["home_q2"].fillna(0)
        df["away_1h_score"] = df["away_q1"].fillna(0) + df["away_q2"].fillna(0)
        if "spread_line" in df.columns:
            df["1h_spread_line"] = df["spread_line"] / 2
            df["actual_1h_margin"] = df["home_1h_score"] - df["away_1h_score"]
            df["1h_spread_covered"] = (df["actual_1h_margin"] > -df["1h_spread_line"]).astype(int)
        
        if "total_line" in df.columns:
            df["1h_total_line"] = df["total_line"] / 2
            df["actual_1h_total"] = df["home_1h_score"] + df["away_1h_score"]
            df["1h_total_over"] = (df["actual_1h_total"] > df["1h_total_line"]).astype(int)
    
    return df


def compute_calibration(
    probas: np.ndarray,
    actuals: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Compute calibration data for reliability diagram.
    
    Returns DataFrame with:
    - bin_center: Center of probability bin
    - predicted: Mean predicted probability in bin
    - actual: Actual win rate in bin
    - count: Number of predictions in bin
    - error: |predicted - actual|
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    
    calibration_data = []
    
    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (probas >= low) & (probas < high)
        
        if mask.sum() > 0:
            predicted = probas[mask].mean()
            actual = actuals[mask].mean()
            count = mask.sum()
            
            calibration_data.append({
                "bin_center": (low + high) / 2,
                "predicted": predicted,
                "actual": actual,
                "count": count,
                "error": abs(predicted - actual),
            })
    
    return pd.DataFrame(calibration_data)


def compute_brier_score(probas: np.ndarray, actuals: np.ndarray) -> float:
    """Compute Brier score (lower is better, perfect = 0)."""
    return np.mean((probas - actuals) ** 2)


def verify_model_calibration(
    df: pd.DataFrame,
    model_name: str,
    ModelClass,
    label_col: str,
    train_ratio: float = 0.8,
) -> dict:
    """
    Verify calibration for a single model.
    """
    print(f"\n{'='*60}")
    print(f"CALIBRATION: {model_name}")
    print(f"{'='*60}")
    
    if label_col not in df.columns:
        print(f"[WARN] Label column {label_col} not found. Skipping.")
        return {}
    
    valid_df = df[df[label_col].notna()].copy()
    
    if len(valid_df) < 200:
        print(f"[WARN] Not enough data: {len(valid_df)} games. Skipping.")
        return {}
    
    # Time-based split
    valid_df = valid_df.sort_values("date")
    split_idx = int(len(valid_df) * train_ratio)
    train_df = valid_df.iloc[:split_idx]
    test_df = valid_df.iloc[split_idx:]
    
    print(f"Training set: {len(train_df)} games")
    print(f"Test set: {len(test_df)} games")
    
    # Build features
    fe = FeatureEngineer(lookback=10)
    
    train_features = []
    for idx, game in train_df.iterrows():
        historical = train_df[train_df["date"] < game["date"]]
        if len(historical) < 30:
            continue
        
        features = fe.build_game_features(game, historical)
        if features:
            features[label_col] = game[label_col]
            train_features.append(features)
    
    test_features = []
    for idx, game in test_df.iterrows():
        historical = valid_df[valid_df["date"] < game["date"]]
        if len(historical) < 30:
            continue
        
        features = fe.build_game_features(game, historical)
        if features:
            features[label_col] = game[label_col]
            test_features.append(features)
    
    if len(train_features) < 50 or len(test_features) < 20:
        print("[WARN] Not enough features generated. Skipping.")
        return {}
    
    train_features_df = pd.DataFrame(train_features)
    test_features_df = pd.DataFrame(test_features)
    
    # Train model
    model = ModelClass(
        model_type="logistic",
        use_calibration=True,
    )
    
    y_train = train_features_df[label_col].astype(int)
    y_test = test_features_df[label_col].astype(int)
    
    print("Training model with calibration...")
    model.fit(train_features_df, y_train)
    
    # Get predictions
    probas = model.predict_proba(test_features_df)[:, 1]
    
    # Compute calibration
    calibration_df = compute_calibration(probas, y_test.values)
    
    print("\nCalibration by Probability Bin:")
    print("-" * 50)
    
    for _, row in calibration_df.iterrows():
        print(
            f"  {row['bin_center']:.0%}: "
            f"Predicted={row['predicted']:.1%}, "
            f"Actual={row['actual']:.1%}, "
            f"Error={row['error']:.1%}, "
            f"N={row['count']:.0f}"
        )
    
    # Compute overall metrics
    brier_score = compute_brier_score(probas, y_test.values)
    mean_calibration_error = calibration_df["error"].mean()
    max_calibration_error = calibration_df["error"].max()
    
    print(f"\nOverall Metrics:")
    print(f"  Brier Score: {brier_score:.4f} (lower is better)")
    print(f"  Mean Calibration Error: {mean_calibration_error:.1%}")
    print(f"  Max Calibration Error: {max_calibration_error:.1%}")
    
    # Check if well-calibrated
    is_well_calibrated = mean_calibration_error < 0.05 and max_calibration_error < 0.15
    
    if is_well_calibrated:
        print("\n✅ Model is WELL CALIBRATED")
    else:
        print("\n⚠️ Model calibration needs improvement")
    
    return {
        "model": model_name,
        "brier_score": brier_score,
        "mean_calibration_error": mean_calibration_error,
        "max_calibration_error": max_calibration_error,
        "is_well_calibrated": is_well_calibrated,
        "test_samples": len(test_features_df),
        "calibration_data": calibration_df,
    }


def main():
    print("=" * 60)
    print("MODEL CALIBRATION VERIFICATION")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    df = load_training_data()
    print(f"[OK] Loaded {len(df)} games")
    
    # Models to verify
    models_to_verify = [
        ("FG Spreads", SpreadsModel, "spread_covered"),
        ("FG Totals", TotalsModel, "total_over"),
        ("1H Spreads", FirstHalfSpreadsModel, "1h_spread_covered"),
        ("1H Totals", FirstHalfTotalsModel, "1h_total_over"),
    ]
    
    all_results = []
    
    for model_name, ModelClass, label_col in models_to_verify:
        result = verify_model_calibration(df, model_name, ModelClass, label_col)
        if result:
            all_results.append(result)
    
    # Summary
    print("\n" + "=" * 60)
    print("CALIBRATION SUMMARY")
    print("=" * 60)
    
    print("\n| Model | Brier | Mean Error | Max Error | Status |")
    print("|-------|-------|------------|-----------|--------|")
    
    for result in all_results:
        status = "✅" if result["is_well_calibrated"] else "⚠️"
        print(
            f"| {result['model']} | "
            f"{result['brier_score']:.4f} | "
            f"{result['mean_calibration_error']:.1%} | "
            f"{result['max_calibration_error']:.1%} | "
            f"{status} |"
        )
    
    # Save calibration data
    all_calibration = []
    for result in all_results:
        if "calibration_data" in result:
            cal_df = result["calibration_data"].copy()
            cal_df["model"] = result["model"]
            all_calibration.append(cal_df)
    
    if all_calibration:
        combined_cal = pd.concat(all_calibration, ignore_index=True)
        output_path = PROCESSED_DIR / "calibration_analysis.csv"
        combined_cal.to_csv(output_path, index=False)
        print(f"\n[OK] Calibration data saved to {output_path}")
    
    print(f"\n{'='*60}")
    print("CALIBRATION VERIFICATION COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
