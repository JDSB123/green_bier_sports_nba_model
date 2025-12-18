#!/usr/bin/env python3
"""
Fast backtest for all 6 core NBA betting markets.

Optimized: Pre-computes features once, then does walk-forward selection.

Usage:
    python scripts/backtest_fast.py
    python scripts/backtest_fast.py --markets fg_moneyline
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.models import (
    SpreadsModel,
    TotalsModel,
    MoneylineModel,
    FirstHalfSpreadsModel,
    FirstHalfTotalsModel,
    FirstHalfMoneylineModel,
)
from src.modeling.features import FeatureEngineer

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# Market configurations
MARKETS = {
    "fg_moneyline": {
        "name": "Full Game Moneyline",
        "model_class": MoneylineModel,
        "label_col": "home_win",
    },
    "1h_moneyline": {
        "name": "First Half Moneyline",
        "model_class": FirstHalfMoneylineModel,
        "label_col": "home_1h_win",
    },
}


def load_and_prepare_data() -> pd.DataFrame:
    """Load training data and create labels."""
    df = pd.read_csv(PROCESSED_DIR / "training_data.csv")
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date").reset_index(drop=True)
    
    # Create labels
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    
    # 1H labels
    if "home_1h" in df.columns:
        df["home_1h_win"] = (df["home_1h"] > df["away_1h"]).astype(int)
    elif "home_q1" in df.columns and "home_q2" in df.columns:
        df["home_1h"] = df["home_q1"].fillna(0) + df["home_q2"].fillna(0)
        df["away_1h"] = df["away_q1"].fillna(0) + df["away_q2"].fillna(0)
        df["home_1h_win"] = (df["home_1h"] > df["away_1h"]).astype(int)
    
    return df


def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-compute features for all games."""
    print("\n[INFO] Pre-computing features for all games...")
    
    fe = FeatureEngineer(lookback=10)
    all_features = []
    
    for idx, game in df.iterrows():
        if idx % 500 == 0:
            print(f"  Building features: {idx}/{len(df)}...")
        
        # Historical data for this game
        historical = df[df["date"] < game["date"]].copy()
        
        if len(historical) < 30:
            continue
        
        try:
            features = fe.build_game_features(game, historical)
            if features:
                # Add identifiers and labels
                features["_idx"] = idx
                features["_date"] = game["date"]
                features["_home_team"] = game["home_team"]
                features["_away_team"] = game["away_team"]
                features["home_win"] = game["home_win"]
                if "home_1h_win" in game:
                    features["home_1h_win"] = game["home_1h_win"]
                all_features.append(features)
        except Exception as e:
            continue
    
    features_df = pd.DataFrame(all_features)
    print(f"[OK] Built features for {len(features_df)} games")
    
    return features_df


def run_backtest(
    features_df: pd.DataFrame,
    market_key: str,
    min_training: int = 200,
    retrain_interval: int = 50,
) -> pd.DataFrame:
    """
    Run walk-forward backtest.
    
    Retrains model every `retrain_interval` games for speed.
    """
    config = MARKETS[market_key]
    label_col = config["label_col"]
    ModelClass = config["model_class"]
    
    print(f"\n{'='*60}")
    print(f"BACKTEST: {config['name']}")
    print(f"{'='*60}")
    
    if label_col not in features_df.columns:
        print(f"[WARN] Label column {label_col} not found. Skipping.")
        return pd.DataFrame()
    
    # Filter to valid labels
    valid_df = features_df[features_df[label_col].notna()].copy()
    valid_df = valid_df.sort_values("_date").reset_index(drop=True)
    
    if len(valid_df) < min_training + 50:
        print(f"[WARN] Not enough data: {len(valid_df)} games. Skipping.")
        return pd.DataFrame()
    
    print(f"Total games with features: {len(valid_df)}")
    
    results = []
    model = None
    last_train_idx = 0
    
    for i in range(min_training, len(valid_df)):
        if i % 200 == 0:
            print(f"  Processing {i}/{len(valid_df)}...")
        
        # Retrain periodically
        if model is None or (i - last_train_idx) >= retrain_interval:
            train_df = valid_df.iloc[:i].copy()
            
            try:
                model = ModelClass(model_type="logistic", use_calibration=True)
                y_train = train_df[label_col].astype(int)
                model.fit(train_df, y_train)
                last_train_idx = i
            except Exception as e:
                print(f"  [ERROR] Training failed at {i}: {e}")
                continue
        
        # Predict
        test_row = valid_df.iloc[[i]]
        
        try:
            proba = model.predict_proba(test_row)[0, 1]
            pred = 1 if proba >= 0.5 else 0
            actual = int(test_row[label_col].iloc[0])
            
            # Calculate profit (assuming -110 odds)
            if pred == actual:
                profit = 100 / 110
            else:
                profit = -1.0
            
            results.append({
                "date": test_row["_date"].iloc[0],
                "home_team": test_row["_home_team"].iloc[0],
                "away_team": test_row["_away_team"].iloc[0],
                "market": market_key,
                "predicted": pred,
                "actual": actual,
                "confidence": proba if pred == 1 else 1 - proba,
                "home_prob": proba,
                "profit": profit,
                "correct": 1 if pred == actual else 0,
            })
        except Exception as e:
            continue
    
    results_df = pd.DataFrame(results)
    print(f"[OK] Completed {len(results_df)} predictions")
    
    return results_df


def analyze_results(results_df: pd.DataFrame, market_key: str):
    """Analyze and print results."""
    if len(results_df) == 0:
        print("[WARN] No results to analyze")
        return {}
    
    accuracy = results_df["correct"].mean()
    roi = results_df["profit"].sum() / len(results_df)
    total_profit = results_df["profit"].sum()
    
    print(f"\n{'-'*40}")
    print(f"RESULTS: {MARKETS[market_key]['name']}")
    print(f"{'-'*40}")
    print(f"Total bets: {len(results_df)}")
    print(f"Accuracy: {accuracy:.1%}")
    print(f"ROI: {roi:+.1%}")
    print(f"Total profit: {total_profit:+.2f} units")
    
    # Confidence breakdown
    print(f"\nBy Confidence:")
    for low, high in [(0.5, 0.55), (0.55, 0.6), (0.6, 0.65), (0.65, 0.7), (0.7, 1.0)]:
        mask = (results_df["confidence"] >= low) & (results_df["confidence"] < high)
        if mask.sum() > 0:
            acc = results_df[mask]["correct"].mean()
            r = results_df[mask]["profit"].sum() / mask.sum()
            print(f"  {low:.0%}-{high:.0%}: {mask.sum():4d} bets, {acc:.1%} acc, {r:+.1%} ROI")
    
    # Calibration
    print(f"\nCalibration:")
    for low, high in [(0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8)]:
        mask = (results_df["home_prob"] >= low) & (results_df["home_prob"] < high)
        if mask.sum() > 0:
            pred = results_df[mask]["home_prob"].mean()
            actual = results_df[mask]["actual"].mean()
            print(f"  {low:.0%}-{high:.0%}: Pred={pred:.1%}, Actual={actual:.1%}, N={mask.sum()}")
    
    return {
        "market": market_key,
        "total_bets": len(results_df),
        "accuracy": accuracy,
        "roi": roi,
        "total_profit": total_profit,
    }


def main():
    parser = argparse.ArgumentParser(description="Fast backtest for all markets")
    parser.add_argument(
        "--markets",
        type=str,
        default="all",
        help="Comma-separated markets to backtest or 'all'",
    )
    parser.add_argument(
        "--min-training",
        type=int,
        default=200,
        help="Minimum games before first prediction",
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("FAST BACKTEST")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Determine markets
    if args.markets == "all":
        markets_to_test = list(MARKETS.keys())
    else:
        markets_to_test = [m.strip() for m in args.markets.split(",")]
    
    print(f"Markets: {', '.join(markets_to_test)}")
    
    # Load data
    df = load_and_prepare_data()
    print(f"[OK] Loaded {len(df)} games")
    
    # Build features once
    features_df = build_all_features(df)
    
    # Run backtests
    all_results = []
    all_summaries = []
    
    for market_key in markets_to_test:
        if market_key not in MARKETS:
            print(f"[WARN] Unknown market: {market_key}")
            continue
        
        results_df = run_backtest(features_df, market_key, args.min_training)
        
        if len(results_df) > 0:
            all_results.append(results_df)
            summary = analyze_results(results_df, market_key)
            all_summaries.append(summary)
    
    # Save results
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        output_path = PROCESSED_DIR / "backtest_fast_results.csv"
        combined.to_csv(output_path, index=False)
        print(f"\n[OK] Results saved to {output_path}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for s in all_summaries:
        print(f"{s['market']}: {s['accuracy']:.1%} acc, {s['roi']:+.1%} ROI, {s['total_bets']} bets")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
