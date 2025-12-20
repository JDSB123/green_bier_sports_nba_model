#!/usr/bin/env python3
"""
Fast O(n) backtest for NBA moneyline markets.

This script pre-computes features ONCE, then does walk-forward validation
on the pre-computed feature matrix. This is O(n) instead of O(n²).

Usage:
    python scripts/backtest_fast.py
"""
import sys
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Suppress warnings during feature computation
logging.getLogger().setLevel(logging.ERROR)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"


def load_data():
    """Load and prepare training data."""
    df = pd.read_csv(PROCESSED_DIR / "training_data.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Create outcome labels
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)

    # 1H outcomes
    if "home_q1" in df.columns and "home_q2" in df.columns:
        df["home_1h_score"] = df["home_q1"].fillna(0) + df["home_q2"].fillna(0)
        df["away_1h_score"] = df["away_q1"].fillna(0) + df["away_q2"].fillna(0)
        df["home_1h_win"] = (df["home_1h_score"] > df["away_1h_score"]).astype(int)

    # Q1 outcomes
    if "home_q1" in df.columns:
        df["home_q1_win"] = (df["home_q1"].fillna(0) > df["away_q1"].fillna(0)).astype(int)

    print(f"Loaded {len(df)} games from {df['date'].min().date()} to {df['date'].max().date()}")
    return df


def compute_team_stats(df, team, before_date, lookback=10):
    """Compute rolling stats for a team before a given date."""
    # Get team's games before this date
    home = df[(df["home_team"] == team) & (df["date"] < before_date)].copy()
    away = df[(df["away_team"] == team) & (df["date"] < before_date)].copy()

    home["team_score"] = home["home_score"]
    home["opp_score"] = home["away_score"]
    home["is_home"] = 1

    away["team_score"] = away["away_score"]
    away["opp_score"] = away["home_score"]
    away["is_home"] = 0

    all_games = pd.concat([
        home[["date", "team_score", "opp_score", "is_home"]],
        away[["date", "team_score", "opp_score", "is_home"]]
    ]).sort_values("date", ascending=False)

    if len(all_games) < 3:
        return None

    recent = all_games.head(lookback)

    return {
        "ppg": recent["team_score"].mean(),
        "papg": recent["opp_score"].mean(),
        "margin": (recent["team_score"] - recent["opp_score"]).mean(),
        "win_pct": (recent["team_score"] > recent["opp_score"]).mean(),
        "pace": recent["team_score"].mean() + recent["opp_score"].mean(),
        "games": len(all_games),
    }


def precompute_features(df):
    """
    Pre-compute features for ALL games in ONE pass.
    This is O(n) instead of O(n²).
    """
    print("Pre-computing features for all games...")

    features_list = []
    skipped = 0

    for i, game in df.iterrows():
        # Get stats for both teams using only data BEFORE this game
        home_stats = compute_team_stats(df, game["home_team"], game["date"])
        away_stats = compute_team_stats(df, game["away_team"], game["date"])

        if home_stats is None or away_stats is None:
            skipped += 1
            features_list.append(None)
            continue

        features = {
            "game_idx": i,
            "date": game["date"],
            "home_team": game["home_team"],
            "away_team": game["away_team"],

            # Home team features
            "home_ppg": home_stats["ppg"],
            "home_papg": home_stats["papg"],
            "home_margin": home_stats["margin"],
            "home_win_pct": home_stats["win_pct"],
            "home_pace": home_stats["pace"],

            # Away team features
            "away_ppg": away_stats["ppg"],
            "away_papg": away_stats["papg"],
            "away_margin": away_stats["margin"],
            "away_win_pct": away_stats["win_pct"],
            "away_pace": away_stats["pace"],

            # Differentials
            "ppg_diff": home_stats["ppg"] - away_stats["ppg"],
            "margin_diff": home_stats["margin"] - away_stats["margin"],
            "win_pct_diff": home_stats["win_pct"] - away_stats["win_pct"],

            # Labels
            "home_win": game["home_win"],
        }

        # Add 1H and Q1 labels if available
        if "home_1h_win" in game:
            features["home_1h_win"] = game["home_1h_win"]
        if "home_q1_win" in game:
            features["home_q1_win"] = game["home_q1_win"]

        features_list.append(features)

    # Filter out None entries and create DataFrame
    valid_features = [f for f in features_list if f is not None]
    features_df = pd.DataFrame(valid_features)

    print(f"  Computed features for {len(features_df)} games (skipped {skipped} with insufficient history)")

    return features_df


def walk_forward_backtest(features_df, label_col, min_training=80):
    """
    Walk-forward backtest on pre-computed features.
    This is now O(n) - just iterating through the feature matrix.
    """
    feature_cols = [
        "home_ppg", "home_papg", "home_margin", "home_win_pct", "home_pace",
        "away_ppg", "away_papg", "away_margin", "away_win_pct", "away_pace",
        "ppg_diff", "margin_diff", "win_pct_diff"
    ]

    results = []

    for i in range(min_training, len(features_df)):
        train = features_df.iloc[:i]
        test = features_df.iloc[i]

        X_train = train[feature_cols].values
        y_train = train[label_col].values

        X_test = test[feature_cols].values.reshape(1, -1)

        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train and predict
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_scaled, y_train)

        proba = model.predict_proba(X_test_scaled)[0, 1]
        pred = 1 if proba >= 0.5 else 0
        actual = int(test[label_col])

        # Profit at -110 odds
        profit = 100/110 if pred == actual else -1.0

        results.append({
            "date": test["date"],
            "home_team": test["home_team"],
            "away_team": test["away_team"],
            "predicted": pred,
            "actual": actual,
            "confidence": proba if pred == 1 else 1 - proba,
            "correct": 1 if pred == actual else 0,
            "profit": profit,
        })

    return pd.DataFrame(results)


def analyze_results(results_df, market_name):
    """Analyze and print results."""
    if len(results_df) == 0:
        print(f"\n{market_name}: No results")
        return None

    total = len(results_df)
    correct = results_df["correct"].sum()
    accuracy = correct / total
    roi = results_df["profit"].sum() / total
    total_profit = results_df["profit"].sum()

    # High confidence
    high_conf = results_df[results_df["confidence"] >= 0.6]
    high_conf_acc = high_conf["correct"].mean() if len(high_conf) > 0 else 0
    high_conf_roi = high_conf["profit"].sum() / len(high_conf) if len(high_conf) > 0 else 0

    print(f"\n{'='*50}")
    print(f"{market_name}")
    print(f"{'='*50}")
    print(f"  Total Bets:      {total}")
    print(f"  Accuracy:        {accuracy:.1%} ({correct}/{total})")
    print(f"  ROI:             {roi:+.1%}")
    print(f"  Total Profit:    {total_profit:+.2f} units")
    print(f"  High Conf (60%+):")
    print(f"    Bets:          {len(high_conf)}")
    print(f"    Accuracy:      {high_conf_acc:.1%}")
    print(f"    ROI:           {high_conf_roi:+.1%}")

    return {
        "market": market_name,
        "bets": total,
        "accuracy": accuracy,
        "roi": roi,
        "profit": total_profit,
        "high_conf_bets": len(high_conf),
        "high_conf_acc": high_conf_acc,
        "high_conf_roi": high_conf_roi,
    }


def main():
    print("=" * 60)
    print("FAST MONEYLINE BACKTEST (O(n) Algorithm)")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    df = load_data()

    # Pre-compute all features ONCE
    features_df = precompute_features(df)

    if len(features_df) < 100:
        print("[ERROR] Not enough games with valid features")
        return

    # Run backtests for each moneyline market
    markets = [
        ("home_win", "Full Game Moneyline"),
        ("home_1h_win", "First Half Moneyline"),
        ("home_q1_win", "First Quarter Moneyline"),
    ]

    all_results = []
    summaries = []

    for label_col, market_name in markets:
        if label_col not in features_df.columns:
            print(f"\n[SKIP] {market_name} - label column not found")
            continue

        print(f"\nRunning {market_name} backtest...")
        results = walk_forward_backtest(features_df, label_col, min_training=80)
        results["market"] = market_name
        all_results.append(results)

        summary = analyze_results(results, market_name)
        if summary:
            summaries.append(summary)

    # Save results
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        output_path = PROCESSED_DIR / "moneyline_backtest_results.csv"
        combined.to_csv(output_path, index=False)
        print(f"\n[OK] Results saved to {output_path}")

    # Summary
    print(f"\n{'='*60}")
    print("BACKTEST COMPLETE")
    print(f"{'='*60}")

    print("\nPRODUCTION READINESS:")
    for s in summaries:
        if s["accuracy"] >= 0.55 and s["roi"] > 0.05:
            status = "PRODUCTION READY"
        elif s["accuracy"] >= 0.52 and s["roi"] > 0:
            status = "NEEDS MONITORING"
        else:
            status = "NOT RECOMMENDED"
        print(f"  {s['market']}: {s['accuracy']:.1%} acc, {s['roi']:+.1%} ROI - {status}")

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
