#!/usr/bin/env python3
"""
Quick backtest on 2020+ data for full game and first half spreads/totals.
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

def main():
    print("=" * 60)
    print("BACKTEST: 2020+ Season Data")
    print("=" * 60)
    
    # Load data
    data_path = PROJECT_ROOT / "data/processed/training_data_enhanced.csv"
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    # Filter to 2023+
    df = df[df["date"] >= "2023-01-01"].copy()
    print(f"Games since 2023: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Check for required columns
    print(f"spread_line non-null: {df['spread_line'].notna().sum()}")
    print(f"total_line non-null: {df['total_line'].notna().sum()}")
    
    # Create labels
    df["home_margin"] = df["home_score"] - df["away_score"]
    df["actual_total"] = df["home_score"] + df["away_score"]
    
    # Full game labels
    if "spread_covered" not in df.columns:
        df["spread_covered"] = (df["home_margin"] > -df["spread_line"]).astype(int)
    if "went_over" not in df.columns:
        df["went_over"] = (df["actual_total"] > df["total_line"]).astype(int)
    
    # First half labels
    if "home_q1" in df.columns and "home_q2" in df.columns:
        df["home_1h"] = df["home_q1"].fillna(0) + df["home_q2"].fillna(0)
        df["away_1h"] = df["away_q1"].fillna(0) + df["away_q2"].fillna(0)
        df["actual_1h_margin"] = df["home_1h"] - df["away_1h"]
        df["actual_1h_total"] = df["home_1h"] + df["away_1h"]
        
        if "fh_spread_line" not in df.columns:
            df["fh_spread_line"] = df["spread_line"] / 2
        if "fh_total_line" not in df.columns:
            df["fh_total_line"] = df["total_line"] / 2
            
        df["fh_spread_covered"] = (df["actual_1h_margin"] > -df["fh_spread_line"]).astype(int)
        df["fh_went_over"] = (df["actual_1h_total"] > df["fh_total_line"]).astype(int)
    
    # Markets to test
    markets = {
        "FG Spread": ("spread_covered", "spread_line", SpreadsModel),
        "FG Total": ("went_over", "total_line", TotalsModel),
        "1H Spread": ("fh_spread_covered", "fh_spread_line", FirstHalfSpreadsModel),
        "1H Total": ("fh_went_over", "fh_total_line", FirstHalfTotalsModel),
    }
    
    fe = FeatureEngineer(lookback=10)
    min_training = 200
    
    all_results = {}
    
    for market_name, (label_col, line_col, ModelClass) in markets.items():
        print(f"\n{'='*60}")
        print(f"MARKET: {market_name}")
        print(f"{'='*60}")
        
        if label_col not in df.columns:
            print(f"  [SKIP] Missing label column: {label_col}")
            continue
            
        valid_df = df[df[label_col].notna() & df[line_col].notna()].copy()
        print(f"  Valid games: {len(valid_df)}")
        
        if len(valid_df) < min_training + 100:
            print(f"  [SKIP] Not enough data")
            continue
        
        # Walk-forward backtest (sample every 10th game for speed)
        results = []
        total = len(valid_df)
        
        for i in range(min_training, total, 20):  # Step by 20 for speed
            if i % 500 == 0:
                print(f"  Processing {i}/{total}...")
            
            train_df = valid_df.iloc[:i].copy()
            test_game = valid_df.iloc[i]
            
            try:
                # Build features
                train_features = []
                for idx in range(max(0, len(train_df) - 500), len(train_df)):  # Last 500 games
                    game = train_df.iloc[idx]
                    historical = train_df[train_df["date"] < game["date"]]
                    if len(historical) < 30:
                        continue
                    
                    features = fe.build_game_features(game, historical)
                    if features:
                        features[label_col] = game[label_col]
                        if line_col in game.index:
                            features[line_col] = game[line_col]
                        train_features.append(features)
                
                if len(train_features) < 50:
                    continue
                
                train_features_df = pd.DataFrame(train_features)
                
                # Train model
                model = ModelClass(model_type="logistic", use_calibration=True)
                y_train = train_features_df[label_col].astype(int)
                model.fit(train_features_df, y_train)
                
                # Predict test game
                historical = train_df.copy()
                test_features = fe.build_game_features(test_game, historical)
                if not test_features:
                    continue
                
                test_features_df = pd.DataFrame([test_features])
                proba = model.predict_proba(test_features_df)[0, 1]
                pred = 1 if proba >= 0.5 else 0
                actual = int(test_game[label_col])
                
                # Profit calculation (-110 odds)
                profit = 100/110 if pred == actual else -1.0
                
                results.append({
                    "date": test_game["date"],
                    "predicted": pred,
                    "actual": actual,
                    "confidence": proba if pred == 1 else 1 - proba,
                    "line": test_game[line_col],
                    "correct": 1 if pred == actual else 0,
                    "profit": profit,
                })
                
            except Exception as e:
                continue
        
        if results:
            results_df = pd.DataFrame(results)
            accuracy = results_df["correct"].mean()
            roi = results_df["profit"].sum() / len(results_df)
            
            # High confidence
            high_conf = results_df[results_df["confidence"] >= 0.55]
            high_conf_acc = high_conf["correct"].mean() if len(high_conf) > 0 else 0
            high_conf_roi = high_conf["profit"].sum() / len(high_conf) if len(high_conf) > 0 else 0
            
            all_results[market_name] = {
                "bets": len(results_df),
                "accuracy": accuracy,
                "roi": roi,
                "high_conf_bets": len(high_conf),
                "high_conf_acc": high_conf_acc,
                "high_conf_roi": high_conf_roi,
            }
            
            print(f"\n  RESULTS:")
            print(f"    Bets: {len(results_df)}")
            print(f"    Accuracy: {accuracy:.1%}")
            print(f"    ROI: {roi:+.1%}")
            print(f"    High Conf (55%+): {len(high_conf)} bets, {high_conf_acc:.1%} acc, {high_conf_roi:+.1%} ROI")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Market':<15} {'Bets':>8} {'Accuracy':>10} {'ROI':>10} {'HC Acc':>10} {'HC ROI':>10}")
    print("-" * 65)
    
    for market, stats in all_results.items():
        print(f"{market:<15} {stats['bets']:>8} {stats['accuracy']:>9.1%} {stats['roi']:>+9.1%} {stats['high_conf_acc']:>9.1%} {stats['high_conf_roi']:>+9.1%}")
    
    # Production ready assessment
    print("\n" + "=" * 60)
    print("PRODUCTION READINESS ASSESSMENT")
    print("=" * 60)
    
    for market, stats in all_results.items():
        if stats["accuracy"] >= 0.54 and stats["roi"] > 0:
            status = "✅ PRODUCTION READY"
        elif stats["accuracy"] >= 0.52:
            status = "⚠️ NEEDS TUNING"
        else:
            status = "❌ NOT READY"
        print(f"{market}: {status}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()


