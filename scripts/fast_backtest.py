#!/usr/bin/env python3
"""
Fast O(n) backtest for NBA moneyline markets.

This script pre-computes all features in a single pass, then does walk-forward
validation using the pre-computed features. Much faster than the O(n²) approach.

Usage:
    python scripts/fast_backtest.py
"""
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"


def load_data():
    """Load training data."""
    df = pd.read_csv(PROCESSED_DIR / "training_data.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Create labels
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)

    # 1H winner
    if "home_q1" in df.columns and "home_q2" in df.columns:
        df["home_1h_score"] = df["home_q1"].fillna(0) + df["home_q2"].fillna(0)
        df["away_1h_score"] = df["away_q1"].fillna(0) + df["away_q2"].fillna(0)
        df["home_1h_win"] = (df["home_1h_score"] > df["away_1h_score"]).astype(int)

    # Q1 winner
    if "home_q1" in df.columns:
        df["home_q1_win"] = (df["home_q1"].fillna(0) > df["away_q1"].fillna(0)).astype(int)

    return df


def compute_rolling_stats(df: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
    """
    Pre-compute rolling stats for ALL games in O(n) time.

    For each game, computes rolling stats based on games BEFORE that date (no leakage).
    """
    print(f"Pre-computing rolling stats for {len(df)} games...")

    # Build team game history
    team_games = {}

    features_list = []

    for idx, row in df.iterrows():
        game_date = row["date"]
        home = row["home_team"]
        away = row["away_team"]

        # Get stats for each team based on games BEFORE this date
        home_stats = get_team_stats(team_games.get(home, []), lookback)
        away_stats = get_team_stats(team_games.get(away, []), lookback)

        # Build feature dict
        features = {"game_idx": idx}

        if home_stats and away_stats:
            # Home team features
            for key, val in home_stats.items():
                features[f"home_{key}"] = val
            # Away team features
            for key, val in away_stats.items():
                features[f"away_{key}"] = val

            # Differentials
            features["ppg_diff"] = home_stats["ppg"] - away_stats["ppg"]
            features["margin_diff"] = home_stats["margin"] - away_stats["margin"]
            features["win_pct_diff"] = home_stats["win_pct"] - away_stats["win_pct"]

            # 1H differentials if available
            if "ppg_1h" in home_stats and "ppg_1h" in away_stats:
                features["ppg_1h_diff"] = home_stats["ppg_1h"] - away_stats["ppg_1h"]
                features["margin_1h_diff"] = home_stats["margin_1h"] - away_stats["margin_1h"]

            # Q1 differentials if available
            if "ppg_q1" in home_stats and "ppg_q1" in away_stats:
                features["ppg_q1_diff"] = home_stats["ppg_q1"] - away_stats["ppg_q1"]
                features["margin_q1_diff"] = home_stats["margin_q1"] - away_stats["margin_q1"]

            features["valid"] = True
        else:
            features["valid"] = False

        features_list.append(features)

        # Update team history AFTER processing (no leakage)
        # Store full game result and quarter scores
        game_result = {
            "date": game_date,
            "score": row["home_score"] if home == row["home_team"] else row["away_score"],
            "opp_score": row["away_score"] if home == row["home_team"] else row["home_score"],
            "is_home": True,
        }
        # Add quarter scores if available
        if "home_q1" in row.index:
            game_result["q1"] = row["home_q1"] if home == row["home_team"] else row["away_q1"]
            game_result["q1_opp"] = row["away_q1"] if home == row["home_team"] else row["home_q1"]
        if "home_q2" in row.index:
            game_result["q2"] = row["home_q2"] if home == row["home_team"] else row["away_q2"]
            game_result["q2_opp"] = row["away_q2"] if home == row["home_team"] else row["home_q2"]

        if home not in team_games:
            team_games[home] = []
        team_games[home].append(game_result)

        away_result = {
            "date": game_date,
            "score": row["away_score"],
            "opp_score": row["home_score"],
            "is_home": False,
        }
        if "home_q1" in row.index:
            away_result["q1"] = row["away_q1"]
            away_result["q1_opp"] = row["home_q1"]
        if "home_q2" in row.index:
            away_result["q2"] = row["away_q2"]
            away_result["q2_opp"] = row["home_q2"]

        if away not in team_games:
            team_games[away] = []
        team_games[away].append(away_result)

    features_df = pd.DataFrame(features_list)
    valid_count = features_df["valid"].sum()
    print(f"  Valid games with features: {valid_count}/{len(df)}")

    return features_df


def get_team_stats(games: list, lookback: int) -> dict:
    """Compute stats from team's recent games."""
    if len(games) < 3:
        return {}

    recent = games[-lookback:] if len(games) >= lookback else games

    scores = [g["score"] for g in recent]
    opp_scores = [g["opp_score"] for g in recent]
    margins = [g["score"] - g["opp_score"] for g in recent]
    wins = [1 if m > 0 else 0 for m in margins]

    stats = {
        "ppg": np.mean(scores),
        "papg": np.mean(opp_scores),
        "margin": np.mean(margins),
        "win_pct": np.mean(wins),
        "games": len(games),
    }

    # 1H stats if available
    q1_scores = [g.get("q1", np.nan) for g in recent if "q1" in g]
    q2_scores = [g.get("q2", np.nan) for g in recent if "q2" in g]
    q1_opp = [g.get("q1_opp", np.nan) for g in recent if "q1_opp" in g]
    q2_opp = [g.get("q2_opp", np.nan) for g in recent if "q2_opp" in g]

    if len(q1_scores) >= 3 and len(q2_scores) >= 3:
        h1_scores = [q1 + q2 for q1, q2 in zip(q1_scores, q2_scores) if not np.isnan(q1) and not np.isnan(q2)]
        h1_opp = [q1 + q2 for q1, q2 in zip(q1_opp, q2_opp) if not np.isnan(q1) and not np.isnan(q2)]

        if len(h1_scores) >= 3:
            stats["ppg_1h"] = np.mean(h1_scores)
            stats["papg_1h"] = np.mean(h1_opp)
            stats["margin_1h"] = np.mean([s - o for s, o in zip(h1_scores, h1_opp)])

    # Q1 stats
    if len(q1_scores) >= 3:
        valid_q1 = [(s, o) for s, o in zip(q1_scores, q1_opp) if not np.isnan(s) and not np.isnan(o)]
        if len(valid_q1) >= 3:
            stats["ppg_q1"] = np.mean([s for s, o in valid_q1])
            stats["papg_q1"] = np.mean([o for s, o in valid_q1])
            stats["margin_q1"] = np.mean([s - o for s, o in valid_q1])

    return stats


def run_backtest(df: pd.DataFrame, features_df: pd.DataFrame, market: str, min_training: int = 150):
    """Run walk-forward backtest using pre-computed features."""

    label_col = {
        "fg_moneyline": "home_win",
        "1h_moneyline": "home_1h_win",
        "q1_moneyline": "home_q1_win",
    }[market]

    # Feature columns based on market
    if market == "fg_moneyline":
        feature_cols = ["home_ppg", "home_papg", "home_margin", "home_win_pct",
                       "away_ppg", "away_papg", "away_margin", "away_win_pct",
                       "ppg_diff", "margin_diff", "win_pct_diff"]
    elif market == "1h_moneyline":
        feature_cols = ["home_ppg_1h", "home_papg_1h", "home_margin_1h",
                       "away_ppg_1h", "away_papg_1h", "away_margin_1h",
                       "ppg_1h_diff", "margin_1h_diff"]
    else:  # q1_moneyline
        feature_cols = ["home_ppg_q1", "home_papg_q1", "home_margin_q1",
                       "away_ppg_q1", "away_papg_q1", "away_margin_q1",
                       "ppg_q1_diff", "margin_q1_diff"]

    # Filter to valid games with all required features
    valid_mask = features_df["valid"] & features_df[feature_cols].notna().all(axis=1)
    valid_indices = features_df[valid_mask]["game_idx"].values

    if len(valid_indices) < min_training + 20:
        print(f"  [SKIP] Not enough valid games: {len(valid_indices)}")
        return pd.DataFrame()

    results = []

    for i in range(min_training, len(valid_indices)):
        train_idx = valid_indices[:i]
        test_idx = valid_indices[i]

        # Get training data
        train_X = features_df[features_df["game_idx"].isin(train_idx)][feature_cols].values
        train_y = df.loc[train_idx, label_col].values

        # Get test data
        test_X = features_df[features_df["game_idx"] == test_idx][feature_cols].values
        test_y = df.loc[test_idx, label_col]
        test_row = df.loc[test_idx]

        # Scale features
        scaler = StandardScaler()
        train_X_scaled = scaler.fit_transform(train_X)
        test_X_scaled = scaler.transform(test_X)

        # Train simple logistic regression
        model = LogisticRegression(max_iter=200, random_state=42)
        model.fit(train_X_scaled, train_y)

        # Predict
        proba = model.predict_proba(test_X_scaled)[0, 1]
        pred = 1 if proba >= 0.5 else 0
        actual = int(test_y)

        # Calculate profit (-110 odds)
        profit = 100/110 if pred == actual else -1.0

        results.append({
            "date": test_row["date"],
            "home_team": test_row["home_team"],
            "away_team": test_row["away_team"],
            "market": market,
            "predicted": pred,
            "actual": actual,
            "confidence": proba if pred == 1 else 1 - proba,
            "profit": profit,
            "correct": 1 if pred == actual else 0,
        })

    return pd.DataFrame(results)


def main():
    print("=" * 60)
    print("FAST MONEYLINE BACKTEST (O(n) Algorithm)")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    df = load_data()
    print(f"Loaded {len(df)} games: {df['date'].min().date()} to {df['date'].max().date()}")

    # Pre-compute all features in one pass
    features_df = compute_rolling_stats(df)

    # Run backtests
    markets = ["fg_moneyline", "1h_moneyline", "q1_moneyline"]
    all_results = []

    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)

    for market in markets:
        print(f"\n{market.upper()}:")
        results = run_backtest(df, features_df, market, min_training=150)

        if len(results) > 0:
            all_results.append(results)

            accuracy = results["correct"].mean()
            roi = results["profit"].sum() / len(results)
            total_profit = results["profit"].sum()

            # High confidence
            high_conf = results[results["confidence"] >= 0.6]
            high_conf_acc = high_conf["correct"].mean() if len(high_conf) > 0 else 0
            high_conf_roi = high_conf["profit"].sum() / len(high_conf) if len(high_conf) > 0 else 0

            print(f"  Total bets: {len(results)}")
            print(f"  Accuracy: {accuracy:.1%}")
            print(f"  ROI: {roi:+.1%}")
            print(f"  Total profit (units): {total_profit:+.1f}")
            print(f"  High conf (60%+): {len(high_conf)} bets, {high_conf_acc:.1%} acc, {high_conf_roi:+.1%} ROI")

    # Save results
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        output_path = PROCESSED_DIR / "moneyline_backtest_results.csv"
        combined.to_csv(output_path, index=False)
        print(f"\n[OK] Results saved to {output_path}")
        print(f"Total predictions: {len(combined)}")

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
