#!/usr/bin/env python3
"""
Quick validation backtest using fixed train/test split.

This is a FAST validation to verify formula changes work correctly.
For production validation, use the full walk-forward backtest.

Usage:
    python scripts/quick_backtest.py
"""
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.models import SpreadsModel, TotalsModel
from src.modeling.features import FeatureEngineer


def main():
    print("=" * 60)
    print("QUICK VALIDATION BACKTEST")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load training data
    data_path = PROJECT_ROOT / "data" / "processed" / "training_data.csv"
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    print(f"\nTotal games: {len(df)}")
    print(f"Games with spread_line: {df['spread_line'].notna().sum()}")
    print(f"Games with total_line: {df['total_line'].notna().sum()}")

    # Filter to games with betting lines
    df_valid = df[df["spread_line"].notna() & df["total_line"].notna()].copy()
    print(f"Games with both lines: {len(df_valid)}")

    if len(df_valid) < 500:
        print("[ERROR] Not enough data for validation")
        return 1

    # 70/30 train/test split
    split_idx = int(len(df_valid) * 0.7)
    train_df = df_valid.iloc[:split_idx].copy()
    test_df = df_valid.iloc[split_idx:].copy()

    print(f"\nTrain: {len(train_df)} games ({train_df['date'].min().date()} to {train_df['date'].max().date()})")
    print(f"Test: {len(test_df)} games ({test_df['date'].min().date()} to {test_df['date'].max().date()})")

    # Feature engineering
    fe = FeatureEngineer(lookback=10)

    def build_features_batch(games_df, historical_df):
        """Build features for a batch of games."""
        features_list = []
        for idx, game in games_df.iterrows():
            hist = historical_df[historical_df["date"] < game["date"]]
            if len(hist) < 30:
                continue
            feats = fe.build_game_features(game, hist)
            if feats:
                feats["spread_covered"] = game["spread_covered"]
                feats["total_over"] = game["total_over"]
                feats["spread_line"] = game["spread_line"]
                feats["total_line"] = game["total_line"]
                features_list.append(feats)
        return pd.DataFrame(features_list)

    print("\n[1] Building training features...")
    train_features = build_features_batch(train_df, train_df)
    print(f"    Built {len(train_features)} training samples")

    print("\n[2] Building test features...")
    test_features = build_features_batch(test_df, df_valid)
    print(f"    Built {len(test_features)} test samples")

    if len(train_features) < 100 or len(test_features) < 50:
        print("[ERROR] Not enough features generated")
        return 1

    # Train and evaluate SPREAD model
    print("\n[3] Training Spread Model...")
    spread_model = SpreadsModel(model_type="logistic", use_calibration=True)
    y_train_spread = train_features["spread_covered"].astype(int)
    spread_model.fit(train_features, y_train_spread)

    spread_preds = spread_model.predict(test_features)
    spread_proba = spread_model.predict_proba(test_features)[:, 1]
    y_test_spread = test_features["spread_covered"].astype(int)

    spread_accuracy = (spread_preds == y_test_spread).mean()
    spread_profit = sum([100/110 if p == a else -1 for p, a in zip(spread_preds, y_test_spread)])
    spread_roi = spread_profit / len(test_features)

    # High confidence
    high_conf_mask = (spread_proba >= 0.55) | (spread_proba <= 0.45)
    if high_conf_mask.sum() > 0:
        hc_spread_acc = (spread_preds[high_conf_mask] == y_test_spread.values[high_conf_mask]).mean()
        hc_spread_profit = sum([100/110 if p == a else -1 for p, a in
                                zip(spread_preds[high_conf_mask], y_test_spread.values[high_conf_mask])])
        hc_spread_roi = hc_spread_profit / high_conf_mask.sum()
    else:
        hc_spread_acc = hc_spread_roi = 0

    print(f"\n{'='*50}")
    print("SPREAD RESULTS")
    print(f"{'='*50}")
    print(f"Overall Accuracy: {spread_accuracy:.1%}")
    print(f"Overall ROI: {spread_roi:+.1%}")
    print(f"High Conf (>55%) Accuracy: {hc_spread_acc:.1%}")
    print(f"High Conf ROI: {hc_spread_roi:+.1%}")

    # Train and evaluate TOTAL model
    print("\n[4] Training Total Model...")
    total_model = TotalsModel(model_type="logistic", use_calibration=True)
    y_train_total = train_features["total_over"].astype(int)
    total_model.fit(train_features, y_train_total)

    total_preds = total_model.predict(test_features)
    total_proba = total_model.predict_proba(test_features)[:, 1]
    y_test_total = test_features["total_over"].astype(int)

    total_accuracy = (total_preds == y_test_total).mean()
    total_profit = sum([100/110 if p == a else -1 for p, a in zip(total_preds, y_test_total)])
    total_roi = total_profit / len(test_features)

    # High confidence
    high_conf_mask = (total_proba >= 0.55) | (total_proba <= 0.45)
    if high_conf_mask.sum() > 0:
        hc_total_acc = (total_preds[high_conf_mask] == y_test_total.values[high_conf_mask]).mean()
        hc_total_profit = sum([100/110 if p == a else -1 for p, a in
                              zip(total_preds[high_conf_mask], y_test_total.values[high_conf_mask])])
        hc_total_roi = hc_total_profit / high_conf_mask.sum()
    else:
        hc_total_acc = hc_total_roi = 0

    print(f"\n{'='*50}")
    print("TOTAL RESULTS")
    print(f"{'='*50}")
    print(f"Overall Accuracy: {total_accuracy:.1%}")
    print(f"Overall ROI: {total_roi:+.1%}")
    print(f"High Conf (>55%) Accuracy: {hc_total_acc:.1%}")
    print(f"High Conf ROI: {hc_total_roi:+.1%}")

    # Summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")

    if spread_accuracy >= 0.52 and total_accuracy >= 0.52:
        print("PASS: Both models above 52% accuracy threshold")
    else:
        print("WARNING: One or more models below 52% accuracy")

    if hc_spread_roi > 0 or hc_total_roi > 0:
        print("PASS: Positive ROI on high-confidence bets")
    else:
        print("WARNING: Negative ROI on high-confidence bets")

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
