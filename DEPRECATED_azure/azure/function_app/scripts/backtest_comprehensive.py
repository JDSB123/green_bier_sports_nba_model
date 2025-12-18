#!/usr/bin/env python3
"""
Comprehensive backtest for all NBA betting markets with ROI optimization.

Markets:
    - Full Game: Spread, Total, Moneyline
    - First Half: Spread, Total, Moneyline  
    - First Quarter: Spread, Total, Moneyline (NEW)

Features:
    - ROI optimization with confidence filtering
    - Segment analysis (pickem, small, medium, large)
    - Clear prediction output format
    - Handles missing betting lines (estimates from historical data)

Usage:
    python scripts/backtest_comprehensive.py
    python scripts/backtest_comprehensive.py --markets fg_spread,1h_moneyline
    python scripts/backtest_comprehensive.py --min-confidence 0.60
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
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
    "fg_spread": {
        "name": "Full Game Spreads",
        "model_class": SpreadsModel,
        "label_col": "spread_covered",
        "line_col": "spread_line",
        "needs_line": True,
    },
    "fg_total": {
        "name": "Full Game Totals",
        "model_class": TotalsModel,
        "label_col": "total_over",
        "line_col": "total_line",
        "needs_line": True,
    },
    "fg_moneyline": {
        "name": "Full Game Moneyline",
        "model_class": MoneylineModel,
        "label_col": "home_win",
        "line_col": None,
        "needs_line": False,
    },
    "1h_spread": {
        "name": "First Half Spreads",
        "model_class": FirstHalfSpreadsModel,
        "label_col": "1h_spread_covered",
        "line_col": "1h_spread_line",
        "needs_line": True,
    },
    "1h_total": {
        "name": "First Half Totals",
        "model_class": FirstHalfTotalsModel,
        "label_col": "1h_total_over",
        "line_col": "1h_total_line",
        "needs_line": True,
    },
    "1h_moneyline": {
        "name": "First Half Moneyline",
        "model_class": FirstHalfMoneylineModel,
        "label_col": "home_1h_win",
        "line_col": None,
        "needs_line": False,
    },
    "1q_spread": {
        "name": "First Quarter Spreads",
        "model_class": FirstHalfSpreadsModel,  # Reuse 1H model for Q1
        "label_col": "1q_spread_covered",
        "line_col": "1q_spread_line",
        "needs_line": True,
    },
    "1q_total": {
        "name": "First Quarter Totals",
        "model_class": FirstHalfTotalsModel,  # Reuse 1H model for Q1
        "label_col": "1q_total_over",
        "line_col": "1q_total_line",
        "needs_line": True,
    },
    "1q_moneyline": {
        "name": "First Quarter Moneyline",
        "model_class": FirstHalfMoneylineModel,  # Reuse 1H model for Q1
        "label_col": "home_1q_win",
        "line_col": None,
        "needs_line": False,
    },
}

# Segments for analysis
SPREAD_SEGMENTS = [
    ("pickem", 0, 3),
    ("small", 3, 6),
    ("medium", 6, 9),
    ("large", 9, 15),
]

TOTAL_SEGMENTS = [
    ("low", 200, 215),
    ("medium", 215, 230),
    ("high", 230, 250),
]


def load_training_data(data_file: str = None) -> pd.DataFrame:
    """Load training data and create all labels."""
    if data_file:
        if Path(data_file).is_absolute():
            training_path = Path(data_file)
        else:
            training_path = PROCESSED_DIR / data_file
    else:
        training_path = PROCESSED_DIR / "training_data.csv"
    
    if not training_path.exists():
        print(f"[ERROR] Training data not found: {training_path}")
        sys.exit(1)
    
    df = pd.read_csv(training_path)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date").reset_index(drop=True)
    
    # Full Game labels
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    df["actual_margin"] = df["home_score"] - df["away_score"]
    df["actual_total"] = df["home_score"] + df["away_score"]
    
    # Estimate lines from historical averages if missing
    if "spread_line" not in df.columns:
        # Use historical average margin as proxy
        df["spread_line"] = -df["actual_margin"].rolling(window=100, min_periods=10).mean().shift(1)
        print("[INFO] Estimating spread lines from historical margins")
    
    if "total_line" not in df.columns:
        # Use historical average total as proxy
        df["total_line"] = df["actual_total"].rolling(window=100, min_periods=10).mean().shift(1)
        print("[INFO] Estimating total lines from historical totals")
    
    # Create spread/total labels
    df["spread_covered"] = (df["actual_margin"] > -df["spread_line"]).astype(int)
    df["total_over"] = (df["actual_total"] > df["total_line"]).astype(int)
    
    # First Half labels
    if "home_q1" in df.columns and "home_q2" in df.columns:
        df["home_1h_score"] = df["home_q1"].fillna(0) + df["home_q2"].fillna(0)
        df["away_1h_score"] = df["away_q1"].fillna(0) + df["away_q2"].fillna(0)
        df["home_1h_win"] = (df["home_1h_score"] > df["away_1h_score"]).astype(int)
        df["actual_1h_total"] = df["home_1h_score"] + df["away_1h_score"]
        df["actual_1h_margin"] = df["home_1h_score"] - df["away_1h_score"]
        
        # Estimate 1H lines
        if "1h_spread_line" not in df.columns:
            df["1h_spread_line"] = df["spread_line"] / 2
        if "1h_total_line" not in df.columns:
            df["1h_total_line"] = df["total_line"] / 2
        
        df["1h_spread_covered"] = (df["actual_1h_margin"] > -df["1h_spread_line"]).astype(int)
        df["1h_total_over"] = (df["actual_1h_total"] > df["1h_total_line"]).astype(int)
    
    # First Quarter labels
    if "home_q1" in df.columns:
        df["home_1q_score"] = df["home_q1"].fillna(0)
        df["away_1q_score"] = df["away_q1"].fillna(0)
        df["home_1q_win"] = (df["home_1q_score"] > df["away_1q_score"]).astype(int)
        df["actual_1q_total"] = df["home_1q_score"] + df["away_1q_score"]
        df["actual_1q_margin"] = df["home_1q_score"] - df["away_1q_score"]
        
        # Estimate Q1 lines (quarter of FG line)
        if "1q_spread_line" not in df.columns:
            df["1q_spread_line"] = df["spread_line"] / 4
        if "1q_total_line" not in df.columns:
            df["1q_total_line"] = df["total_line"] / 4
        
        df["1q_spread_covered"] = (df["actual_1q_margin"] > -df["1q_spread_line"]).astype(int)
        df["1q_total_over"] = (df["actual_1q_total"] > df["1q_total_line"]).astype(int)
    
    print(f"[OK] Loaded {len(df)} games")
    return df


def enrich_features(features: Dict, game_row: pd.Series) -> Dict:
    """Add external features to the feature set."""
    keep_cols = [
        "home_rest_days", "away_rest_days", "rest_diff",
        "home_rolling_", "away_rolling_"
    ]
    
    for col, val in game_row.items():
        if pd.isna(val):
            continue
        if any(col.startswith(p) or col == p for p in keep_cols):
            features[col] = float(val)
    
    return features


def backtest_market(
    df: pd.DataFrame,
    market_key: str,
    min_training: int = 80,
    min_confidence: float = 0.0,
) -> pd.DataFrame:
    """Run walk-forward backtest with ROI optimization."""
    config = MARKETS[market_key]
    
    print(f"\n{'='*60}")
    print(f"BACKTEST: {config['name']}")
    print(f"{'='*60}")
    
    label_col = config["label_col"]
    line_col = config["line_col"]
    ModelClass = config["model_class"]
    
    # Filter to games with valid labels
    if label_col not in df.columns:
        print(f"[WARN] Label column {label_col} not found. Skipping.")
        return pd.DataFrame()
    
    valid_df = df[df[label_col].notna()].copy()
    
    if len(valid_df) < min_training + 50:
        print(f"[WARN] Not enough data: {len(valid_df)} games. Skipping.")
        return pd.DataFrame()
    
    fe = FeatureEngineer(lookback=10)
    results = []
    
    total_games = len(valid_df)
    
    for i in range(min_training, total_games):
        if i % 500 == 0:
            print(f"  Processing game {i}/{total_games}...")
        
        train_df = valid_df.iloc[:i].copy()
        test_game = valid_df.iloc[i]
        
        try:
            # Build features for training
            train_features = []
            for idx, game in train_df.iterrows():
                historical = train_df[train_df["date"] < game["date"]]
                if len(historical) < 30:
                    continue
                
                features = fe.build_game_features(game, historical)
                if features:
                    features = enrich_features(features, game)
                    features[label_col] = game[label_col]
                    if line_col and line_col in game.index:
                        features[line_col] = game[line_col]
                    train_features.append(features)
            
            if len(train_features) < 50:
                continue
            
            train_features_df = pd.DataFrame(train_features)
            train_features_df = train_features_df[train_features_df[label_col].notna()]
            
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
            
            test_features = enrich_features(test_features, test_game)
            test_features_df = pd.DataFrame([test_features])
            
            # Predict
            proba = model.predict_proba(test_features_df)[0, 1]
            pred = 1 if proba >= 0.5 else 0
            actual = int(test_game[label_col])
            
            # Confidence of the bet we're making
            confidence = proba if pred == 1 else 1 - proba
            
            # Skip if below minimum confidence threshold
            if confidence < min_confidence:
                continue
            
            # Get line if available
            line = test_game[line_col] if line_col and line_col in test_game.index else None
            
            # Calculate profit (assuming -110 odds)
            if pred == actual:
                profit = 100 / 110
            else:
                profit = -1.0
            
            # Determine segment
            segment = "unknown"
            if line is not None and not pd.isna(line):
                abs_line = abs(line)
                if "spread" in market_key:
                    for seg_name, low, high in SPREAD_SEGMENTS:
                        if low <= abs_line < high:
                            segment = seg_name
                            break
                elif "total" in market_key:
                    for seg_name, low, high in TOTAL_SEGMENTS:
                        if low <= line < high:
                            segment = seg_name
                            break
            
            results.append({
                "date": test_game["date"],
                "home_team": test_game["home_team"],
                "away_team": test_game["away_team"],
                "market": market_key,
                "prediction": "home" if pred == 1 else "away",
                "actual": "home" if actual == 1 else "away",
                "confidence": confidence,
                "probability": proba,
                "line": line,
                "segment": segment,
                "profit": profit,
                "correct": 1 if pred == actual else 0,
            })
            
        except Exception as e:
            if i % 1000 == 0:  # Only print errors occasionally
                print(f"  [WARN] Error at game {i}: {e}")
            continue
    
    results_df = pd.DataFrame(results)
    print(f"[OK] Completed {len(results_df)} predictions")
    
    return results_df


def analyze_market_results(
    results_df: pd.DataFrame,
    market_key: str,
    confidence_thresholds: List[float] = [0.0, 0.55, 0.60, 0.65, 0.70],
) -> Dict:
    """Analyze results with ROI optimization at different confidence levels."""
    if len(results_df) == 0:
        return {}
    
    accuracy = results_df["correct"].mean()
    roi = results_df["profit"].sum() / len(results_df)
    
    summary = {
        "market": market_key,
        "name": MARKETS[market_key]["name"],
        "total_bets": len(results_df),
        "accuracy": accuracy,
        "roi": roi,
        "total_profit": results_df["profit"].sum(),
    }
    
    # Find optimal confidence threshold for ROI
    best_roi = roi
    best_threshold = 0.0
    best_bets = len(results_df)
    
    for threshold in confidence_thresholds:
        filtered = results_df[results_df["confidence"] >= threshold]
        if len(filtered) >= 20:  # Need minimum sample size
            filtered_roi = filtered["profit"].sum() / len(filtered)
            if filtered_roi > best_roi:
                best_roi = filtered_roi
                best_threshold = threshold
                best_bets = len(filtered)
    
    summary["optimal_threshold"] = best_threshold
    summary["optimal_roi"] = best_roi
    summary["optimal_bets"] = best_bets
    
    # Performance at different confidence levels
    for threshold in [0.55, 0.60, 0.65, 0.70]:
        filtered = results_df[results_df["confidence"] >= threshold]
        if len(filtered) >= 10:
            summary[f"conf_{threshold:.0f}_bets"] = len(filtered)
            summary[f"conf_{threshold:.0f}_acc"] = filtered["correct"].mean()
            summary[f"conf_{threshold:.0f}_roi"] = filtered["profit"].sum() / len(filtered)
    
    # Segment performance
    segments = results_df["segment"].unique()
    for seg in segments:
        if seg == "unknown":
            continue
        seg_df = results_df[results_df["segment"] == seg]
        if len(seg_df) >= 10:
            summary[f"seg_{seg}_bets"] = len(seg_df)
            summary[f"seg_{seg}_acc"] = seg_df["correct"].mean()
            summary[f"seg_{seg}_roi"] = seg_df["profit"].sum() / len(seg_df)
    
    return summary


def generate_predictions_output(results_df: pd.DataFrame, output_path: Path):
    """Generate clear predictions output file."""
    if len(results_df) == 0:
        return
    
    # Sort by confidence (highest first)
    results_df = results_df.sort_values("confidence", ascending=False)
    
    # Create formatted output
    output_lines = []
    output_lines.append("# Model Predictions Output\n")
    output_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    output_lines.append(f"Total Predictions: {len(results_df)}\n\n")
    output_lines.append("=" * 80 + "\n\n")
    
    # Group by market
    for market_key, market_df in results_df.groupby("market"):
        market_name = MARKETS[market_key]["name"]
        output_lines.append(f"## {market_name}\n\n")
        
        # Summary stats
        acc = market_df["correct"].mean()
        roi = market_df["profit"].sum() / len(market_df)
        output_lines.append(f"**Accuracy:** {acc:.1%} | **ROI:** {roi:+.1%} | **Bets:** {len(market_df)}\n\n")
        
        # Top predictions (highest confidence)
        output_lines.append("### Top Predictions (Highest Confidence)\n\n")
        output_lines.append("| Date | Matchup | Prediction | Confidence | Line | Result |\n")
        output_lines.append("|------|---------|------------|------------|------|--------|\n")
        
        for _, row in market_df.head(20).iterrows():
            date_str = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
            matchup = f"{row['home_team']} vs {row['away_team']}"
            pred = row["prediction"]
            conf = f"{row['confidence']:.1%}"
            line = f"{row['line']:.1f}" if pd.notna(row["line"]) else "N/A"
            result = "✓" if row["correct"] else "✗"
            
            output_lines.append(f"| {date_str} | {matchup} | {pred} | {conf} | {line} | {result} |\n")
        
        output_lines.append("\n")
    
    # Write to file
    with open(output_path, "w") as f:
        f.writelines(output_lines)
    
    print(f"[OK] Predictions output saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive backtest with ROI optimization")
    parser.add_argument(
        "--markets",
        type=str,
        default="all",
        help="Comma-separated markets or 'all'",
    )
    parser.add_argument(
        "--min-training",
        type=int,
        default=100,
        help="Minimum games before first prediction",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Minimum confidence threshold (0.0 = all bets)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Training data file",
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("COMPREHENSIVE BACKTEST - ROI OPTIMIZED")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Determine markets
    if args.markets == "all":
        markets_to_test = list(MARKETS.keys())
    else:
        markets_to_test = [m.strip() for m in args.markets.split(",")]
    
    print(f"Markets: {', '.join(markets_to_test)}")
    print(f"Min Confidence: {args.min_confidence:.0%}")
    
    # Load data
    df = load_training_data(args.data)
    
    # Run backtests
    all_results = []
    all_summaries = []
    
    for market_key in markets_to_test:
        if market_key not in MARKETS:
            print(f"[WARN] Unknown market: {market_key}. Skipping.")
            continue
        
        results_df = backtest_market(
            df, market_key, args.min_training, args.min_confidence
        )
        
        if len(results_df) > 0:
            all_results.append(results_df)
            
            summary = analyze_market_results(results_df, market_key)
            all_summaries.append(summary)
            
            # Print summary
            print(f"\n{MARKETS[market_key]['name']} Summary:")
            print(f"  Total Bets: {summary['total_bets']}")
            print(f"  Accuracy: {summary['accuracy']:.1%}")
            print(f"  ROI: {summary['roi']:+.1%}")
            if summary.get('optimal_threshold', 0) > 0:
                print(f"  Optimal Threshold: {summary['optimal_threshold']:.0%} confidence")
                print(f"  Optimal ROI: {summary['optimal_roi']:+.1%} ({summary['optimal_bets']} bets)")
    
    # Save results
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        output_path = PROCESSED_DIR / "comprehensive_backtest_results.csv"
        combined.to_csv(output_path, index=False)
        print(f"\n[OK] Results saved to {output_path}")
        
        # Generate predictions output
        predictions_path = PROCESSED_DIR / "model_predictions_output.md"
        generate_predictions_output(combined, predictions_path)
    
    # Generate summary report
    if all_summaries:
        report_path = PROJECT_ROOT / "COMPREHENSIVE_BACKTEST_RESULTS.md"
        with open(report_path, "w") as f:
            f.write("# Comprehensive Backtest Results\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Overall Summary\n\n")
            f.write("| Market | Bets | Accuracy | ROI | Optimal Threshold | Optimal ROI |\n")
            f.write("|--------|------|----------|-----|-------------------|------------|\n")
            
            for summary in all_summaries:
                if not summary:
                    continue
                opt_thresh = summary.get('optimal_threshold', 0)
                opt_roi = summary.get('optimal_roi', summary['roi'])
                f.write(
                    f"| {summary['name']} | {summary['total_bets']} | "
                    f"{summary['accuracy']:.1%} | {summary['roi']:+.1%} | "
                    f"{opt_thresh:.0%} | {opt_roi:+.1%} |\n"
                )
        
        print(f"[OK] Report saved to {report_path}")
    
    print(f"\n{'='*60}")
    print("BACKTEST COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

