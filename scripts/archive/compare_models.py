#!/usr/bin/env python3
"""
Compare model types (classifier vs regressor) for NBA betting.

Runs both approaches on the same data and produces a comparison report.

Usage:
    python scripts/compare_models.py
    python scripts/compare_models.py --target spreads
"""
from __future__ import annotations
import argparse
import os
import sys
from typing import Dict, List, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from src.config import settings
from src.modeling.models import SpreadsModel, TotalsModel
from src.modeling.feature_config import (
    get_spreads_features,
    get_totals_features,
    filter_available_features,
)


def calculate_roi(preds: np.ndarray, actuals: np.ndarray, odds: float = -110) -> float:
    """Calculate ROI assuming standard odds."""
    correct = (preds == actuals).sum()
    total = len(preds)
    if total == 0:
        return 0.0
    profit = correct * (100 / abs(odds)) - (total - correct)
    return profit / total


def evaluate_model_approach(
    model_class,
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_columns: List[str],
    is_regression: bool,
    line_column: str,
    score_column: str,
) -> Dict[str, Any]:
    """Evaluate a single model approach."""
    available_features = [f for f in feature_columns if f in X_train.columns]
    
    model = model_class(
        name=f"comparison_{model_type}",
        model_type=model_type,
        feature_columns=available_features,
    )
    
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        return {"error": str(e)}
    
    # Predictions
    y_pred_raw = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Convert to binary outcomes
    if is_regression and line_column in X_test.columns:
        # For regression, compare prediction to line
        if "spread" in line_column.lower():
            # Spread: covered if margin > -spread_line
            y_pred_binary = (y_pred_raw > -X_test[line_column]).astype(int)
            y_actual_binary = (y_test > -X_test[line_column]).astype(int)
        else:
            # Total: over if score > line
            y_pred_binary = (y_pred_raw > X_test[line_column]).astype(int)
            y_actual_binary = (y_test > X_test[line_column]).astype(int)
    else:
        y_pred_binary = y_pred_raw.astype(int)
        y_actual_binary = y_test.values.astype(int)
    
    # Metrics
    accuracy = accuracy_score(y_actual_binary, y_pred_binary)
    roi = calculate_roi(y_pred_binary, y_actual_binary)
    
    # Calibration
    try:
        brier = brier_score_loss(y_actual_binary, y_prob)
    except Exception:
        brier = None
    
    try:
        logloss = log_loss(y_actual_binary, y_prob)
    except Exception:
        logloss = None
    
    # Confidence analysis
    high_conf_mask = (y_prob > 0.6) | (y_prob < 0.4)
    if high_conf_mask.sum() > 0:
        high_conf_acc = accuracy_score(
            y_actual_binary[high_conf_mask], 
            y_pred_binary[high_conf_mask]
        )
        high_conf_roi = calculate_roi(
            y_pred_binary[high_conf_mask], 
            y_actual_binary[high_conf_mask]
        )
        high_conf_n = high_conf_mask.sum()
    else:
        high_conf_acc = high_conf_roi = None
        high_conf_n = 0
    
    return {
        "model_type": model_type,
        "accuracy": accuracy,
        "roi": roi,
        "brier_score": brier,
        "log_loss": logloss,
        "high_conf_accuracy": high_conf_acc,
        "high_conf_roi": high_conf_roi,
        "high_conf_n": high_conf_n,
        "n_samples": len(y_test),
        "n_features": len(model.feature_columns),
    }


def run_comparison(target: str = "spreads", n_splits: int = 5) -> pd.DataFrame:
    """Run full comparison of model types."""
    print("=" * 70)
    print(f"MODEL TYPE COMPARISON: {target.upper()}")
    print("=" * 70)
    
    # Load training data
    training_path = os.path.join(settings.data_processed_dir, "training_data.csv")
    if not os.path.exists(training_path):
        print(f"Training data not found: {training_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(training_path)
    date_col = "date" if "date" in df.columns else "game_date"
    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.sort_values("date")
    
    print(f"Loaded {len(df)} games")
    
    # Setup for target
    if target == "spreads":
        model_class = SpreadsModel
        feature_columns = get_spreads_features()
        line_column = "spread_line"
        score_column = "home_margin"
        binary_target = "spread_covered"
    else:
        model_class = TotalsModel
        feature_columns = get_totals_features()
        line_column = "total_line"
        score_column = "total_score"
        binary_target = "went_over"
    
    # Filter to valid games
    df = df[df[line_column].notna()].copy()
    if binary_target not in df.columns:
        print(f"Target column '{binary_target}' not found")
        return pd.DataFrame()
    
    print(f"Using {len(df)} games with {target} data")
    
    # Model types to compare
    model_types = ["logistic", "gradient_boosting", "regression"]
    
    # Run TimeSeriesSplit CV for each model type
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    all_results = []
    
    for model_type in model_types:
        print(f"\n--- Evaluating: {model_type.upper()} ---")
        
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(df), 1):
            X_train = df.iloc[train_idx]
            X_test = df.iloc[test_idx]
            
            if model_type == "regression":
                y_train = X_train[score_column]
                y_test = X_test[score_column]
                is_regression = True
            else:
                y_train = X_train[binary_target]
                y_test = X_test[binary_target]
                is_regression = False
            
            result = evaluate_model_approach(
                model_class,
                model_type,
                X_train,
                y_train,
                X_test,
                y_test,
                feature_columns,
                is_regression,
                line_column,
                score_column,
            )
            
            if "error" not in result:
                fold_results.append(result)
        
        if fold_results:
            # Aggregate fold results
            avg_result = {
                "model_type": model_type,
                "accuracy_mean": np.mean([r["accuracy"] for r in fold_results]),
                "accuracy_std": np.std([r["accuracy"] for r in fold_results]),
                "roi_mean": np.mean([r["roi"] for r in fold_results]),
                "roi_std": np.std([r["roi"] for r in fold_results]),
                "brier_mean": np.mean([r["brier_score"] for r in fold_results if r["brier_score"]]),
                "high_conf_acc_mean": np.mean([r["high_conf_accuracy"] for r in fold_results if r["high_conf_accuracy"]]),
                "high_conf_roi_mean": np.mean([r["high_conf_roi"] for r in fold_results if r["high_conf_roi"]]),
                "total_samples": sum([r["n_samples"] for r in fold_results]),
            }
            all_results.append(avg_result)
            
            print(f"  Accuracy: {avg_result['accuracy_mean']:.1%} ± {avg_result['accuracy_std']:.1%}")
            print(f"  ROI:      {avg_result['roi_mean']:+.1%} ± {avg_result['roi_std']:.1%}")
            if avg_result.get("brier_mean"):
                print(f"  Brier:    {avg_result['brier_mean']:.4f}")
            if avg_result.get("high_conf_acc_mean"):
                print(f"  High-Conf Acc: {avg_result['high_conf_acc_mean']:.1%}")
                print(f"  High-Conf ROI: {avg_result['high_conf_roi_mean']:+.1%}")
    
    if not all_results:
        print("No results to compare")
        return pd.DataFrame()
    
    # Summary comparison
    results_df = pd.DataFrame(all_results)
    
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(results_df.to_string(index=False))
    
    # Recommendation
    best_by_roi = results_df.loc[results_df["roi_mean"].idxmax()]
    best_by_acc = results_df.loc[results_df["accuracy_mean"].idxmax()]
    
    print("\n" + "-" * 70)
    print("RECOMMENDATIONS:")
    print(f"  Best by ROI:      {best_by_roi['model_type']} ({best_by_roi['roi_mean']:+.1%})")
    print(f"  Best by Accuracy: {best_by_acc['model_type']} ({best_by_acc['accuracy_mean']:.1%})")
    
    if best_by_roi["model_type"] == best_by_acc["model_type"]:
        print(f"\n  [WINNER] Clear winner: {best_by_roi['model_type']}")
    else:
        print(f"\n  [TRADEOFF] Consider {best_by_roi['model_type']} for profit, "
              f"{best_by_acc['model_type']} for consistency")
    
    print("-" * 70)
    
    # Save results
    output_path = os.path.join(settings.data_processed_dir, f"model_comparison_{target}.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(description="Compare model types")
    parser.add_argument(
        "--target",
        choices=["spreads", "totals"],
        default="spreads",
        help="Prediction target",
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=5,
        help="Number of CV splits",
    )
    args = parser.parse_args()
    
    run_comparison(target=args.target, n_splits=args.splits)


if __name__ == "__main__":
    main()

