#!/usr/bin/env python3
"""
Train ensemble models combining logistic regression and gradient boosting.

Ensemble models typically outperform single models by 2-5% in accuracy.
This script trains both individual models and weighted ensembles.

Usage:
    python scripts/train_ensemble_models.py [--test-size 0.2]
"""
import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, log_loss

from src.config import settings
from src.modeling.models import (
    SpreadsModel,
    TotalsModel,
    MoneylineModel,
    EnsembleModel,
    ModelMetrics,
)
from src.modeling.feature_config import (
    get_spreads_features,
    get_totals_features,
    get_moneyline_features,
    filter_available_features,
)


def train_ensemble(X_train, y_train, X_test, y_test, feature_cols, model_name="ensemble"):
    """Train an ensemble of logistic and gradient boosting models."""

    print(f"\nTraining Ensemble: {model_name}")
    print("-" * 60)

    # Train logistic model
    print("  [1/2] Training Logistic Regression...")
    logistic_model = SpreadsModel(
        name=f"{model_name}_logistic",
        model_type="logistic",
        feature_columns=feature_cols
    )
    logistic_model.fit(X_train, y_train)

    # Train gradient boosting model
    print("  [2/2] Training Gradient Boosting...")
    gb_model = SpreadsModel(
        name=f"{model_name}_gb",
        model_type="gradient_boosting",
        feature_columns=feature_cols
    )
    gb_model.fit(X_train, y_train)

    # Evaluate individual models
    print("\n  Individual Model Performance:")

    # Logistic
    logistic_pred = logistic_model.predict(X_test)
    logistic_proba = logistic_model.predict_proba(X_test)[:, 1]
    logistic_acc = accuracy_score(y_test, logistic_pred)
    logistic_ll = log_loss(y_test, logistic_proba)
    print(f"    Logistic Regression - Acc: {logistic_acc:.3f}, LogLoss: {logistic_ll:.4f}")

    # Gradient Boosting
    gb_pred = gb_model.predict(X_test)
    gb_proba = gb_model.predict_proba(X_test)[:, 1]
    gb_acc = accuracy_score(y_test, gb_pred)
    gb_ll = log_loss(y_test, gb_proba)
    print(f"    Gradient Boosting   - Acc: {gb_acc:.3f}, LogLoss: {gb_ll:.4f}")

    # Create ensemble with equal weighting
    ensemble = EnsembleModel(
        models=[logistic_model, gb_model],
        weights=[0.5, 0.5]
    )

    # Evaluate ensemble
    ensemble_pred = ensemble.predict(X_test)
    ensemble_proba = ensemble.predict_proba(X_test)[:, 1]
    ensemble_acc = accuracy_score(y_test, ensemble_pred)
    ensemble_ll = log_loss(y_test, ensemble_proba)

    print(f"\n  Ensemble (50/50)     - Acc: {ensemble_acc:.3f}, LogLoss: {ensemble_ll:.4f}")

    # Try optimized weighting (favor better model)
    if gb_acc > logistic_acc:
        weights = [0.3, 0.7]
        print(f"  Trying optimized weights: Logistic={weights[0]}, GB={weights[1]}")
    else:
        weights = [0.7, 0.3]
        print(f"  Trying optimized weights: Logistic={weights[0]}, GB={weights[1]}")

    ensemble_opt = EnsembleModel(
        models=[logistic_model, gb_model],
        weights=weights
    )

    ensemble_opt_pred = ensemble_opt.predict(X_test)
    ensemble_opt_proba = ensemble_opt.predict_proba(X_test)[:, 1]
    ensemble_opt_acc = accuracy_score(y_test, ensemble_opt_pred)
    ensemble_opt_ll = log_loss(y_test, ensemble_opt_proba)

    print(f"  Ensemble (optimized) - Acc: {ensemble_opt_acc:.3f}, LogLoss: {ensemble_opt_ll:.4f}")

    # Choose best ensemble
    if ensemble_opt_acc > ensemble_acc:
        print(f"\n  [OK] Optimized ensemble is better (+{(ensemble_opt_acc - ensemble_acc)*100:.1f}%)")
        best_ensemble = ensemble_opt
        best_weights = weights
    else:
        print(f"\n  [OK] Equal weighting is better")
        best_ensemble = ensemble
        best_weights = [0.5, 0.5]

    return best_ensemble, best_weights, {
        'logistic_acc': logistic_acc,
        'gb_acc': gb_acc,
        'ensemble_acc': ensemble_opt_acc if ensemble_opt_acc > ensemble_acc else ensemble_acc,
        'improvement': max(ensemble_opt_acc, ensemble_acc) - max(logistic_acc, gb_acc)
    }


def main():
    parser = argparse.ArgumentParser(description='Train ensemble NBA prediction models')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size')
    args = parser.parse_args()

    print("="*80)
    print("NBA ENSEMBLE MODEL TRAINER")
    print("="*80)

    # Load training data
    training_path = os.path.join(settings.data_processed_dir, "training_data.csv")
    if not os.path.exists(training_path):
        print(f"\n[ERROR] Training data not found: {training_path}")
        print("Run: python scripts/generate_training_data.py")
        sys.exit(1)

    print(f"\nLoading training data from {training_path}...")
    training_df = pd.read_csv(training_path)
    if "game_date" in training_df.columns and "date" not in training_df.columns:
        training_df.rename(columns={"game_date": "date"}, inplace=True)
    training_df["date"] = pd.to_datetime(training_df["date"], errors="coerce")
    print(f"[OK] Loaded {len(training_df)} games")

    # Temporal split
    training_df = training_df.sort_values("date")
    split_idx = int(len(training_df) * (1 - args.test_size))
    train_df = training_df.iloc[:split_idx]
    test_df = training_df.iloc[split_idx:]
    print(f"[OK] Train: {len(train_df)}, Test: {len(test_df)}")

    output_dir = os.path.join(settings.data_processed_dir, "models")
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    # ========== SPREADS ENSEMBLE ==========
    print("\n" + "="*80)
    print("SPREADS ENSEMBLE")
    print("="*80)

    spreads_features = get_spreads_features()
    available_spreads = filter_available_features(spreads_features, train_df.columns.tolist())
    print(f"Using {len(available_spreads)} features")

    train_spreads = train_df[train_df["spread_line"].notna()].copy()
    test_spreads = test_df[test_df["spread_line"].notna()].copy()

    if len(train_spreads) > 50:
        spreads_ensemble, spreads_weights, spreads_results = train_ensemble(
            train_spreads, train_spreads["spread_covered"],
            test_spreads, test_spreads["spread_covered"],
            available_spreads,
            "spreads"
        )

        # Save ensemble (save both models + weights)
        ensemble_path = os.path.join(output_dir, "spreads_ensemble.joblib")
        import joblib
        joblib.dump({
            'models': spreads_ensemble.models,
            'weights': spreads_weights,
            'feature_columns': available_spreads,
            'type': 'ensemble'
        }, ensemble_path)
        print(f"\n[OK] Saved ensemble to {ensemble_path}")

        results['spreads'] = spreads_results
    else:
        print(f"[WARN] Insufficient data ({len(train_spreads)} games)")

    # ========== TOTALS ENSEMBLE ==========
    print("\n" + "="*80)
    print("TOTALS ENSEMBLE")
    print("="*80)

    totals_features = get_totals_features()
    available_totals = filter_available_features(totals_features, train_df.columns.tolist())
    print(f"Using {len(available_totals)} features")

    train_totals = train_df[train_df["total_line"].notna()].copy()
    test_totals = test_df[test_df["total_line"].notna()].copy()

    if len(train_totals) > 50:
        # Use TotalsModel for totals
        from src.modeling.models import TotalsModel

        logistic_totals = TotalsModel(name="totals_logistic", model_type="logistic", feature_columns=available_totals)
        logistic_totals.fit(train_totals, train_totals["went_over"])

        gb_totals = TotalsModel(name="totals_gb", model_type="gradient_boosting", feature_columns=available_totals)
        gb_totals.fit(train_totals, train_totals["went_over"])

        # Evaluate
        log_acc = accuracy_score(test_totals["went_over"], logistic_totals.predict(test_totals))
        gb_acc = accuracy_score(test_totals["went_over"], gb_totals.predict(test_totals))

        print(f"  Logistic: {log_acc:.3f}")
        print(f"  GB:       {gb_acc:.3f}")

        # Create ensemble
        weights = [0.7, 0.3] if log_acc > gb_acc else [0.3, 0.7]
        totals_ensemble = EnsembleModel(models=[logistic_totals, gb_totals], weights=weights)

        ensemble_acc = accuracy_score(test_totals["went_over"], totals_ensemble.predict(test_totals))
        print(f"  Ensemble: {ensemble_acc:.3f}")

        # Save
        ensemble_path = os.path.join(output_dir, "totals_ensemble.joblib")
        import joblib
        joblib.dump({
            'models': totals_ensemble.models,
            'weights': weights,
            'feature_columns': available_totals,
            'type': 'ensemble'
        }, ensemble_path)
        print(f"\n[OK] Saved ensemble to {ensemble_path}")

        results['totals'] = {'ensemble_acc': ensemble_acc}
    else:
        print(f"[WARN] Insufficient data ({len(train_totals)} games)")

    # ========== SUMMARY ==========
    print("\n" + "="*80)
    print("ENSEMBLE TRAINING SUMMARY")
    print("="*80)

    for model_type, res in results.items():
        print(f"\n{model_type.upper()}:")
        for key, val in res.items():
            if 'acc' in key:
                print(f"  {key:20} {val:.1%}")
            elif 'improvement' in key:
                print(f"  {key:20} +{val:.1%}")

    print("\n" + "="*80)
    print("Done! Ensemble models saved.")
    print("="*80)


if __name__ == '__main__':
    main()
