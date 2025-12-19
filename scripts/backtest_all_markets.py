#!/usr/bin/env python3
"""
Unified backtest for all NBA betting markets.

This is the consolidated backtester for all market types.

Markets:
    fg_spread    - Full Game Spreads (60.6% acc, +15.7% ROI)
    fg_total     - Full Game Totals (59.2% acc, +13.1% ROI)
    fg_moneyline - Full Game Moneyline
    1h_spread    - First Half Spreads (55.9% acc, +6.7% ROI)
    1h_total     - First Half Totals (58.2% acc, +11.1% ROI)
    1h_moneyline - First Half Moneyline

Method: Walk-forward validation (train on past, predict next game - NO LEAKAGE)

Usage:
    python scripts/backtest.py                              # All markets
    python scripts/backtest.py --markets fg_spread,fg_total # Specific markets
    python scripts/backtest.py --markets fg_spread          # Single market
    python scripts/backtest.py --min-training 100           # More training data

Output:
    data/processed/all_markets_backtest_results.csv
    ALL_MARKETS_BACKTEST_RESULTS.md (moved to docs/ during cleanup)
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
    "fg_spread": {
        "name": "Full Game Spreads",
        "model_class": SpreadsModel,
        "label_col": "spread_covered",
        "line_col": "spread_line",
    },
    "fg_total": {
        "name": "Full Game Totals",
        "model_class": TotalsModel,
        "label_col": "total_over",
        "line_col": "total_line",
    },
    "fg_moneyline": {
        "name": "Full Game Moneyline",
        "model_class": MoneylineModel,
        "label_col": "home_win",
        "line_col": None,
    },
    "1h_spread": {
        "name": "First Half Spreads",
        "model_class": FirstHalfSpreadsModel,
        "label_col": "1h_spread_covered",
        "line_col": "1h_spread_line",
    },
    "1h_total": {
        "name": "First Half Totals",
        "model_class": FirstHalfTotalsModel,
        "label_col": "1h_total_over",
        "line_col": "1h_total_line",
    },
    "1h_moneyline": {
        "name": "First Half Moneyline",
        "model_class": FirstHalfMoneylineModel,
        "label_col": "home_1h_win",
        "line_col": None,
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


def load_training_data() -> pd.DataFrame:
    """Load training data with game outcomes."""
    training_path = PROCESSED_DIR / "training_data.csv"
    
    if not training_path.exists():
        print(f"[ERROR] Training data not found: {training_path}")
        sys.exit(1)
    
    df = pd.read_csv(training_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    # Create derived labels
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    
    # Spread covered (home perspective)
    if "spread_line" in df.columns:
        df["actual_margin"] = df["home_score"] - df["away_score"]
        df["spread_covered"] = (df["actual_margin"] > -df["spread_line"]).astype(int)
    
    # Total over
    if "total_line" in df.columns:
        df["actual_total"] = df["home_score"] + df["away_score"]
        df["total_over"] = (df["actual_total"] > df["total_line"]).astype(int)
    
    # 1H labels
    if "home_q1" in df.columns and "home_q2" in df.columns:
        df["home_1h_score"] = df["home_q1"].fillna(0) + df["home_q2"].fillna(0)
        df["away_1h_score"] = df["away_q1"].fillna(0) + df["away_q2"].fillna(0)
        df["home_1h_win"] = (df["home_1h_score"] > df["away_1h_score"]).astype(int)
        df["actual_1h_total"] = df["home_1h_score"] + df["away_1h_score"]
        df["actual_1h_margin"] = df["home_1h_score"] - df["away_1h_score"]
        
        # 1H spread covered (approximate - use half of FG line)
        if "spread_line" in df.columns:
            df["1h_spread_line"] = df["spread_line"] / 2
            df["1h_spread_covered"] = (df["actual_1h_margin"] > -df["1h_spread_line"]).astype(int)
        
        # 1H total over (approximate - use half of FG line)
        if "total_line" in df.columns:
            df["1h_total_line"] = df["total_line"] / 2
            df["1h_total_over"] = (df["actual_1h_total"] > df["1h_total_line"]).astype(int)
    
    print(f"[OK] Loaded {len(df)} games")
    
    return df


def backtest_market(
    df: pd.DataFrame,
    market_key: str,
    min_training: int = 80,
) -> pd.DataFrame:
    """
    Run walk-forward backtest for a single market.
    """
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
        if i % 100 == 0:
            print(f"  Processing game {i}/{total_games}...")
        
        # Training data
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
            
            test_features_df = pd.DataFrame([test_features])
            
            # Predict
            proba = model.predict_proba(test_features_df)[0, 1]
            pred = 1 if proba >= 0.5 else 0
            actual = int(test_game[label_col])
            
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
                "predicted": pred,
                "actual": actual,
                "confidence": proba if pred == 1 else 1 - proba,
                "line": line,
                "segment": segment,
                "profit": profit,
                "correct": 1 if pred == actual else 0,
            })
            
        except Exception as e:
            continue
    
    results_df = pd.DataFrame(results)
    print(f"[OK] Completed {len(results_df)} predictions")
    
    return results_df


def analyze_market_results(results_df: pd.DataFrame, market_key: str) -> Dict:
    """Analyze results for a single market."""
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
    
    # High confidence performance
    high_conf = results_df[results_df["confidence"] >= 0.6]
    if len(high_conf) > 0:
        summary["high_conf_bets"] = len(high_conf)
        summary["high_conf_accuracy"] = high_conf["correct"].mean()
        summary["high_conf_roi"] = high_conf["profit"].sum() / len(high_conf)
    
    # Segment performance
    segments = results_df["segment"].unique()
    for seg in segments:
        seg_df = results_df[results_df["segment"] == seg]
        if len(seg_df) > 10:
            summary[f"seg_{seg}_bets"] = len(seg_df)
            summary[f"seg_{seg}_acc"] = seg_df["correct"].mean()
            summary[f"seg_{seg}_roi"] = seg_df["profit"].sum() / len(seg_df)
    
    return summary


def generate_summary_report(all_summaries: List[Dict]) -> str:
    """Generate markdown summary report."""
    report = []
    report.append("# All Markets Backtest Results\n")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append("---\n")
    
    report.append("## Overall Summary\n")
    report.append("| Market | Bets | Accuracy | ROI | High Conf Acc | High Conf ROI |")
    report.append("|--------|------|----------|-----|---------------|---------------|")
    
    for summary in all_summaries:
        if not summary:
            continue
        
        high_conf_acc = summary.get("high_conf_accuracy", 0)
        high_conf_roi = summary.get("high_conf_roi", 0)
        
        report.append(
            f"| {summary['name']} | {summary['total_bets']} | "
            f"{summary['accuracy']:.1%} | {summary['roi']:+.1%} | "
            f"{high_conf_acc:.1%} | {high_conf_roi:+.1%} |"
        )
    
    report.append("\n---\n")
    
    # Segment analysis
    report.append("## Performance by Segment\n")
    
    for summary in all_summaries:
        if not summary:
            continue
        
        market = summary["name"]
        report.append(f"### {market}\n")
        
        seg_data = [(k, v) for k, v in summary.items() if k.startswith("seg_") and k.endswith("_bets")]
        
        if seg_data:
            report.append("| Segment | Bets | Accuracy | ROI |")
            report.append("|---------|------|----------|-----|")
            
            for key, bets in seg_data:
                seg_name = key.replace("seg_", "").replace("_bets", "")
                acc = summary.get(f"seg_{seg_name}_acc", 0)
                roi = summary.get(f"seg_{seg_name}_roi", 0)
                report.append(f"| {seg_name} | {bets} | {acc:.1%} | {roi:+.1%} |")
            
            report.append("")
    
    report.append("---\n")
    
    # Recommendations
    report.append("## Recommendations\n")
    
    for summary in all_summaries:
        if not summary:
            continue
        
        market = summary["name"]
        accuracy = summary["accuracy"]
        roi = summary["roi"]
        
        status = "VALIDATED" if accuracy >= 0.55 and roi > 0.05 else "NEEDS_REVIEW" if accuracy >= 0.52 else "NOT_RECOMMENDED"
        
        report.append(f"- **{market}**: {status}")
        report.append(f"  - Accuracy: {accuracy:.1%}")
        report.append(f"  - ROI: {roi:+.1%}")
        
        if summary.get("high_conf_accuracy", 0) > accuracy:
            report.append(f"  - Consider high-confidence filtering ({summary['high_conf_accuracy']:.1%} acc)")
        
        report.append("")
    
    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Backtest all 6 markets")
    parser.add_argument(
        "--markets",
        type=str,
        default="all",
        help="Comma-separated markets to backtest (e.g., fg_spread,fg_total) or 'all'",
    )
    parser.add_argument(
        "--min-training",
        type=int,
        default=80,
        help="Minimum games before first prediction",
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("ALL MARKETS BACKTEST")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Determine which markets to test
    if args.markets == "all":
        markets_to_test = list(MARKETS.keys())
    else:
        markets_to_test = [m.strip() for m in args.markets.split(",")]
    
    print(f"Markets: {', '.join(markets_to_test)}")
    
    # Load data
    df = load_training_data()
    
    # Run backtests
    all_results = []
    all_summaries = []
    
    for market_key in markets_to_test:
        if market_key not in MARKETS:
            print(f"[WARN] Unknown market: {market_key}. Skipping.")
            continue
        
        results_df = backtest_market(df, market_key, args.min_training)
        
        if len(results_df) > 0:
            all_results.append(results_df)
            
            summary = analyze_market_results(results_df, market_key)
            all_summaries.append(summary)
            
            # Print summary
            print(f"\n{MARKETS[market_key]['name']} Summary:")
            print(f"  Bets: {summary['total_bets']}")
            print(f"  Accuracy: {summary['accuracy']:.1%}")
            print(f"  ROI: {summary['roi']:+.1%}")
    
    # Save combined results
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        output_path = PROCESSED_DIR / "all_markets_backtest_results.csv"
        combined.to_csv(output_path, index=False)
        print(f"\n[OK] Results saved to {output_path}")
    
    # Generate and save report
    report = generate_summary_report(all_summaries)
    report_path = PROJECT_ROOT / "ALL_MARKETS_BACKTEST_RESULTS.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"[OK] Report saved to {report_path}")
    
    print(f"\n{'='*60}")
    print("BACKTEST COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
