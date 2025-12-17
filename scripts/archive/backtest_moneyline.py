#!/usr/bin/env python3
"""
Backtest moneyline predictions (FG + 1H).

Walk-forward validation for moneyline models:
1. Train on historical data
2. Predict next game
3. Move window forward
4. Repeat

Analyzes performance by:
- Odds bucket (heavy favorite, moderate favorite, pick'em, dog)
- Confidence level
- Calibration

Usage:
    python scripts/backtest_moneyline.py
    python scripts/backtest_moneyline.py --min-training 100

Output:
    data/processed/backtest_moneyline_results.csv
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

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# Moneyline odds segments for analysis
ML_SEGMENTS = {
    "heavy_favorite": (-500, -200),
    "moderate_favorite": (-200, -120),
    "slight_favorite": (-120, -100),
    "pick_em": (-100, 100),
    "slight_dog": (100, 120),
    "moderate_dog": (120, 200),
    "heavy_dog": (200, 500),
}


def load_training_data() -> pd.DataFrame:
    """Load training data with game outcomes."""
    training_path = PROCESSED_DIR / "training_data.csv"
    
    if not training_path.exists():
        print(f"[ERROR] Training data not found: {training_path}")
        sys.exit(1)
    
    df = pd.read_csv(training_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    print(f"[OK] Loaded {len(df)} games")
    
    return df


def derive_ml_odds_from_spread(spread: float) -> tuple:
    """
    Derive approximate moneyline odds from spread.
    
    Empirical conversion based on typical NBA lines.
    """
    spread = abs(spread)
    
    if spread <= 1:
        fav_ml, dog_ml = -110, -110
    elif spread <= 2:
        fav_ml, dog_ml = -125, +105
    elif spread <= 3:
        fav_ml, dog_ml = -140, +120
    elif spread <= 4:
        fav_ml, dog_ml = -160, +135
    elif spread <= 5:
        fav_ml, dog_ml = -180, +150
    elif spread <= 6:
        fav_ml, dog_ml = -200, +170
    elif spread <= 7:
        fav_ml, dog_ml = -225, +185
    elif spread <= 8:
        fav_ml, dog_ml = -260, +210
    elif spread <= 9:
        fav_ml, dog_ml = -300, +240
    elif spread <= 10:
        fav_ml, dog_ml = -350, +275
    elif spread <= 12:
        fav_ml, dog_ml = -450, +350
    else:
        fav_ml, dog_ml = -600, +450
    
    return fav_ml, dog_ml


def calculate_ml_profit(bet_side: str, actual_winner: str, odds: int) -> float:
    """
    Calculate profit/loss for a moneyline bet.
    
    Args:
        bet_side: "home" or "away"
        actual_winner: "home" or "away"
        odds: American odds for the bet
    
    Returns:
        Profit/loss in units (1 unit risked)
    """
    if bet_side == actual_winner:
        # Won
        if odds > 0:
            return odds / 100
        else:
            return 100 / abs(odds)
    else:
        # Lost
        return -1.0


def run_walkforward_backtest(
    df: pd.DataFrame,
    min_training: int = 80,
    market: str = "fg",
) -> pd.DataFrame:
    """
    Run walk-forward backtest for moneyline.
    
    Args:
        df: Full dataset sorted by date
        min_training: Minimum games for training before first prediction
        market: "fg" for full game, "1h" for first half
    
    Returns:
        DataFrame with backtest results
    """
    print(f"\n{'='*60}")
    print(f"WALK-FORWARD BACKTEST: {market.upper()} MONEYLINE")
    print(f"{'='*60}")
    
    fe = FeatureEngineer(lookback=10)
    results = []
    
    # Determine label column
    if market == "1h":
        label_col = "home_1h_win"
        ModelClass = FirstHalfMoneylineModel
    else:
        label_col = "home_win"
        ModelClass = MoneylineModel
    
    # Create label if not present
    if label_col not in df.columns:
        if market == "1h":
            if "home_q1" in df.columns and "home_q2" in df.columns:
                df["home_1h_score"] = df["home_q1"].fillna(0) + df["home_q2"].fillna(0)
                df["away_1h_score"] = df["away_q1"].fillna(0) + df["away_q2"].fillna(0)
                df["home_1h_win"] = (df["home_1h_score"] > df["away_1h_score"]).astype(int)
            else:
                print("[WARN] No Q1/Q2 data available for 1H backtest")
                return pd.DataFrame()
        else:
            df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    
    total_games = len(df)
    
    for i in range(min_training, total_games):
        if i % 50 == 0:
            print(f"  Processing game {i}/{total_games}...")
        
        # Training data: all games before this one
        train_df = df.iloc[:i].copy()
        
        # Test game
        test_game = df.iloc[i]
        
        try:
            # Build features for training data
            train_features = []
            for idx, game in train_df.iterrows():
                historical = train_df[train_df["date"] < game["date"]]
                if len(historical) < 30:
                    continue
                
                features = fe.build_game_features(game, historical)
                if features:
                    features[label_col] = game[label_col] if label_col in game else None
                    train_features.append(features)
            
            if len(train_features) < 50:
                continue
            
            train_features_df = pd.DataFrame(train_features)
            train_features_df = train_features_df[train_features_df[label_col].notna()]
            
            if len(train_features_df) < 50:
                continue
            
            # Train model
            model = ModelClass(
                model_type="logistic",
                use_calibration=True,
            )
            
            y_train = train_features_df[label_col].astype(int)
            model.fit(train_features_df, y_train)
            
            # Build features for test game
            historical = train_df.copy()
            test_features = fe.build_game_features(test_game, historical)
            
            if not test_features:
                continue
            
            test_features_df = pd.DataFrame([test_features])
            
            # Predict
            proba = model.predict_proba(test_features_df)[0, 1]
            pred = 1 if proba >= 0.5 else 0
            
            # Get actual result
            actual = test_game[label_col] if label_col in test_game.index else None
            if pd.isna(actual):
                continue
            
            actual = int(actual)
            
            # Derive ML odds from spread if available
            spread = test_game.get("spread_line", 0)
            if pd.isna(spread):
                spread = 0
            
            fav_ml, dog_ml = derive_ml_odds_from_spread(spread)
            
            # Determine which side we're betting
            if pred == 1:
                bet_side = "home"
                odds = fav_ml if spread < 0 else dog_ml
            else:
                bet_side = "away"
                odds = dog_ml if spread < 0 else fav_ml
            
            # Calculate profit
            actual_winner = "home" if actual == 1 else "away"
            profit = calculate_ml_profit(bet_side, actual_winner, odds)
            
            # Determine odds segment
            segment = "unknown"
            for seg_name, (low, high) in ML_SEGMENTS.items():
                if low <= odds < high:
                    segment = seg_name
                    break
            
            results.append({
                "date": test_game["date"],
                "home_team": test_game["home_team"],
                "away_team": test_game["away_team"],
                "market": market,
                "predicted": pred,
                "actual": actual,
                "confidence": proba if pred == 1 else 1 - proba,
                "home_prob": proba,
                "bet_side": bet_side,
                "bet_odds": odds,
                "odds_segment": segment,
                "profit": profit,
                "correct": 1 if pred == actual else 0,
            })
            
        except Exception as e:
            continue
    
    results_df = pd.DataFrame(results)
    print(f"[OK] Completed {len(results_df)} predictions")
    
    return results_df


def analyze_results(results_df: pd.DataFrame, market: str = "fg"):
    """Analyze backtest results."""
    if len(results_df) == 0:
        print("[WARN] No results to analyze")
        return
    
    print(f"\n{'='*60}")
    print(f"RESULTS: {market.upper()} MONEYLINE")
    print(f"{'='*60}")
    
    # Overall performance
    accuracy = results_df["correct"].mean()
    roi = results_df["profit"].sum() / len(results_df)
    
    print(f"\nOverall Performance:")
    print(f"  Total Bets: {len(results_df)}")
    print(f"  Accuracy: {accuracy:.1%}")
    print(f"  ROI: {roi:+.1%}")
    print(f"  Total Profit: {results_df['profit'].sum():+.2f} units")
    
    # Performance by odds segment
    print(f"\nPerformance by Odds Segment:")
    for segment in ML_SEGMENTS.keys():
        seg_df = results_df[results_df["odds_segment"] == segment]
        if len(seg_df) > 0:
            seg_acc = seg_df["correct"].mean()
            seg_roi = seg_df["profit"].sum() / len(seg_df)
            print(f"  {segment:20s}: {len(seg_df):4d} bets, {seg_acc:.1%} acc, {seg_roi:+.1%} ROI")
    
    # Performance by confidence level
    print(f"\nPerformance by Confidence Level:")
    conf_bins = [(0.5, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 1.0)]
    for low, high in conf_bins:
        conf_df = results_df[(results_df["confidence"] >= low) & (results_df["confidence"] < high)]
        if len(conf_df) > 0:
            conf_acc = conf_df["correct"].mean()
            conf_roi = conf_df["profit"].sum() / len(conf_df)
            print(f"  {low:.0%}-{high:.0%}: {len(conf_df):4d} bets, {conf_acc:.1%} acc, {conf_roi:+.1%} ROI")
    
    # Calibration analysis
    print(f"\nCalibration Analysis:")
    prob_bins = [(0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8)]
    for low, high in prob_bins:
        prob_df = results_df[(results_df["home_prob"] >= low) & (results_df["home_prob"] < high)]
        if len(prob_df) > 0:
            predicted = prob_df["home_prob"].mean()
            actual = prob_df["actual"].mean()
            error = abs(predicted - actual)
            print(f"  {low:.0%}-{high:.0%}: Predicted={predicted:.1%}, Actual={actual:.1%}, Error={error:.1%}, N={len(prob_df)}")
    
    # Home vs Away betting
    print(f"\nHome vs Away Betting:")
    home_bets = results_df[results_df["bet_side"] == "home"]
    away_bets = results_df[results_df["bet_side"] == "away"]
    
    if len(home_bets) > 0:
        home_acc = home_bets["correct"].mean()
        home_roi = home_bets["profit"].sum() / len(home_bets)
        print(f"  Home bets: {len(home_bets):4d} bets, {home_acc:.1%} acc, {home_roi:+.1%} ROI")
    
    if len(away_bets) > 0:
        away_acc = away_bets["correct"].mean()
        away_roi = away_bets["profit"].sum() / len(away_bets)
        print(f"  Away bets: {len(away_bets):4d} bets, {away_acc:.1%} acc, {away_roi:+.1%} ROI")


def main():
    parser = argparse.ArgumentParser(description="Backtest moneyline predictions")
    parser.add_argument(
        "--min-training",
        type=int,
        default=80,
        help="Minimum games before first prediction",
    )
    parser.add_argument(
        "--market",
        choices=["fg", "1h", "both"],
        default="both",
        help="Market to backtest (fg, 1h, or both)",
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("MONEYLINE BACKTEST")
    print("=" * 60)
    print(f"Min training: {args.min_training} games")
    print(f"Market: {args.market}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    df = load_training_data()
    
    all_results = []
    
    # Backtest FG moneyline
    if args.market in ["fg", "both"]:
        fg_results = run_walkforward_backtest(df, args.min_training, "fg")
        if len(fg_results) > 0:
            analyze_results(fg_results, "fg")
            all_results.append(fg_results)
    
    # Backtest 1H moneyline
    if args.market in ["1h", "both"]:
        fh_results = run_walkforward_backtest(df, args.min_training, "1h")
        if len(fh_results) > 0:
            analyze_results(fh_results, "1h")
            all_results.append(fh_results)
    
    # Save results
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        output_path = PROCESSED_DIR / "backtest_moneyline_results.csv"
        combined.to_csv(output_path, index=False)
        print(f"\n[OK] Results saved to {output_path}")
    
    print(f"\n{'='*60}")
    print("BACKTEST COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
