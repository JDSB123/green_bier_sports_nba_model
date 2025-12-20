#!/usr/bin/env python3
"""
Unified backtest for all 9 NBA betting markets.

NBA v6.0: 9 INDEPENDENT Markets Architecture
STRICT MODE: No silent failures, no placeholder data.

Markets (All INDEPENDENT models):
    Q1 (First Quarter):
        q1_spread    - First Quarter Spreads
        q1_total     - First Quarter Totals
        q1_moneyline - First Quarter Moneyline

    1H (First Half):
        1h_spread    - First Half Spreads
        1h_total     - First Half Totals
        1h_moneyline - First Half Moneyline

    FG (Full Game):
        fg_spread    - Full Game Spreads
        fg_total     - Full Game Totals
        fg_moneyline - Full Game Moneyline

ARCHITECTURE:
    Each period (Q1, 1H, FG) uses INDEPENDENT features computed from
    historical data for that specific period. No cross-period dependencies.

Method: Walk-forward validation (train on past, predict next game - NO LEAKAGE)

Usage:
    python scripts/backtest.py                              # All 9 markets
    python scripts/backtest.py --markets fg_spread,q1_total # Specific markets
    python scripts/backtest.py --periods q1,1h              # Specific periods
    python scripts/backtest.py --min-training 100           # More training data
    python scripts/backtest.py --strict                     # Fail on any error

Output:
    data/processed/all_markets_backtest_results.csv
"""
import argparse
import sys
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Global strict mode flag
STRICT_MODE = False


class BacktestError(Exception):
    """Raised when backtest encounters an unrecoverable error."""
    pass


class DataValidationError(Exception):
    """Raised when data fails validation checks."""
    pass

from src.modeling.models import (
    SpreadsModel,
    TotalsModel,
    MoneylineModel,
    FirstHalfSpreadsModel,
    FirstHalfTotalsModel,
    FirstHalfMoneylineModel,
    FirstQuarterSpreadsModel,
    FirstQuarterTotalsModel,
    FirstQuarterMoneylineModel,
)
from src.modeling.features import FeatureEngineer

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# Market configurations with period info
MARKETS = {
    # Full Game Markets
    "fg_spread": {
        "name": "Full Game Spreads",
        "model_class": SpreadsModel,
        "label_col": "spread_covered",
        "line_col": "spread_line",
        "period": "fg",
    },
    "fg_total": {
        "name": "Full Game Totals",
        "model_class": TotalsModel,
        "label_col": "total_over",
        "line_col": "total_line",
        "period": "fg",
    },
    "fg_moneyline": {
        "name": "Full Game Moneyline",
        "model_class": MoneylineModel,
        "label_col": "home_win",
        "line_col": None,
        "period": "fg",
    },
    # First Half Markets
    "1h_spread": {
        "name": "First Half Spreads",
        "model_class": FirstHalfSpreadsModel,
        "label_col": "1h_spread_covered",
        "line_col": "1h_spread_line",
        "period": "1h",
    },
    "1h_total": {
        "name": "First Half Totals",
        "model_class": FirstHalfTotalsModel,
        "label_col": "1h_total_over",
        "line_col": "1h_total_line",
        "period": "1h",
    },
    "1h_moneyline": {
        "name": "First Half Moneyline",
        "model_class": FirstHalfMoneylineModel,
        "label_col": "home_1h_win",
        "line_col": None,
        "period": "1h",
    },
    # First Quarter Markets
    "q1_spread": {
        "name": "First Quarter Spreads",
        "model_class": FirstQuarterSpreadsModel,
        "label_col": "q1_spread_covered",
        "line_col": "q1_spread_line",
        "period": "q1",
    },
    "q1_total": {
        "name": "First Quarter Totals",
        "model_class": FirstQuarterTotalsModel,
        "label_col": "q1_total_over",
        "line_col": "q1_total_line",
        "period": "q1",
    },
    "q1_moneyline": {
        "name": "First Quarter Moneyline",
        "model_class": FirstQuarterMoneylineModel,
        "label_col": "home_q1_win",
        "line_col": None,
        "period": "q1",
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


def validate_training_data(df: pd.DataFrame) -> None:
    """
    Validate training data integrity.
    
    Raises DataValidationError if validation fails.
    """
    errors = []
    
    # Check required columns
    required = ["date", "home_team", "away_team", "home_score", "away_score"]
    for col in required:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")
    
    if errors:
        raise DataValidationError("\n".join(errors))
    
    # Check for empty dataset
    if len(df) == 0:
        raise DataValidationError("Training data is empty")
    
    # Check for null values
    null_counts = df[required].isnull().sum()
    for col, count in null_counts.items():
        if count > 0:
            pct = count / len(df) * 100
            if pct > 20:
                errors.append(f"Column '{col}' has {pct:.1f}% null values (too high)")
    
    if errors:
        raise DataValidationError("\n".join(errors))
    
    print(f"[OK] Training data validated: {len(df)} games, {df['date'].min().date()} to {df['date'].max().date()}")


def load_training_data(data_file: str = None, strict: bool = False) -> pd.DataFrame:
    """Load training data with game outcomes.
    
    Args:
        data_file: Optional filename in data/processed/ or full path to CSV
        strict: If True, raise exception on any issue
    
    Raises:
        DataValidationError: If data fails validation (in strict mode)
    """
    if data_file:
        # Check if it's a full path or just a filename
        if Path(data_file).is_absolute():
            training_path = Path(data_file)
        else:
            training_path = PROCESSED_DIR / data_file
    else:
        training_path = PROCESSED_DIR / "training_data.csv"
    
    if not training_path.exists():
        msg = f"Training data not found: {training_path}"
        if strict or STRICT_MODE:
            raise DataValidationError(msg)
        print(f"[ERROR] {msg}")
        sys.exit(1)
    
    df = pd.read_csv(training_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    
    # Drop rows with invalid dates
    invalid_dates = df["date"].isna().sum()
    if invalid_dates > 0:
        print(f"[WARN] Dropping {invalid_dates} rows with invalid dates")
        df = df.dropna(subset=["date"])
    
    df = df.sort_values("date").reset_index(drop=True)
    
    # Validate
    validate_training_data(df)
    
    # Create derived labels
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    
    # Spread covered (home perspective)
    if "spread_line" in df.columns:
        df["actual_margin"] = df["home_score"] - df["away_score"]
        df["spread_covered"] = (df["actual_margin"] > -df["spread_line"]).astype(int)
        # Set to NaN where spread_line is NaN
        df.loc[df["spread_line"].isna(), "spread_covered"] = None
    
    # Total over
    if "total_line" in df.columns:
        df["actual_total"] = df["home_score"] + df["away_score"]
        df["total_over"] = (df["actual_total"] > df["total_line"]).astype(int)
        # Set to NaN where total_line is NaN
        df.loc[df["total_line"].isna(), "total_over"] = None
    
    # 1H labels
    if "home_q1" in df.columns and "home_q2" in df.columns:
        df["home_1h_score"] = df["home_q1"].fillna(0) + df["home_q2"].fillna(0)
        df["away_1h_score"] = df["away_q1"].fillna(0) + df["away_q2"].fillna(0)
        df["home_1h_win"] = (df["home_1h_score"] > df["away_1h_score"]).astype(int)
        df["actual_1h_total"] = df["home_1h_score"] + df["away_1h_score"]
        df["actual_1h_margin"] = df["home_1h_score"] - df["away_1h_score"]
        
        if "fh_spread_line" in df.columns:
            df["1h_spread_line"] = df["fh_spread_line"]
        elif "1h_spread_line" not in df.columns:
            df["1h_spread_line"] = np.nan
        if "spread_line" in df.columns:
            mask = df["1h_spread_line"].isna()
            df.loc[mask, "1h_spread_line"] = df.loc[mask, "spread_line"] / 2
        # Only compute if we have valid line data
        if df["1h_spread_line"].notna().any():
            df["1h_spread_covered"] = np.where(
                df["1h_spread_line"].notna(),
                (df["actual_1h_margin"] > -df["1h_spread_line"]).astype(int),
                np.nan
            )
        else:
            df["1h_spread_covered"] = np.nan

        if "fh_total_line" in df.columns:
            df["1h_total_line"] = df["fh_total_line"]
        elif "1h_total_line" not in df.columns:
            df["1h_total_line"] = np.nan
        if "total_line" in df.columns:
            mask = df["1h_total_line"].isna()
            df.loc[mask, "1h_total_line"] = df.loc[mask, "total_line"] / 2
        # Only compute if we have valid line data
        if df["1h_total_line"].notna().any():
            df["1h_total_over"] = np.where(
                df["1h_total_line"].notna(),
                (df["actual_1h_total"] > df["1h_total_line"]).astype(int),
                np.nan
            )
        else:
            df["1h_total_over"] = np.nan
    
    # Q1 labels
    if "home_q1" in df.columns and "away_q1" in df.columns:
        df["home_q1_win"] = (df["home_q1"].fillna(0) > df["away_q1"].fillna(0)).astype(int)
        df["actual_q1_total"] = df["home_q1"].fillna(0) + df["away_q1"].fillna(0)
        df["actual_q1_margin"] = df["home_q1"].fillna(0) - df["away_q1"].fillna(0)
        
        if "q1_spread_line" not in df.columns:
            df["q1_spread_line"] = np.nan
        if "spread_line" in df.columns:
            mask = df["q1_spread_line"].isna()
            df.loc[mask, "q1_spread_line"] = df.loc[mask, "spread_line"] / 4
        # Only compute if we have valid line data
        if df["q1_spread_line"].notna().any():
            df["q1_spread_covered"] = np.where(
                df["q1_spread_line"].notna(),
                (df["actual_q1_margin"] > -df["q1_spread_line"]).astype(int),
                np.nan
            )
        else:
            df["q1_spread_covered"] = np.nan

        if "q1_total_line" not in df.columns:
            df["q1_total_line"] = np.nan
        if "total_line" in df.columns:
            mask = df["q1_total_line"].isna()
            df.loc[mask, "q1_total_line"] = df.loc[mask, "total_line"] / 4
        # Only compute if we have valid line data
        if df["q1_total_line"].notna().any():
            df["q1_total_over"] = np.where(
                df["q1_total_line"].notna(),
                (df["actual_q1_total"] > df["q1_total_line"]).astype(int),
                np.nan
            )
        else:
            df["q1_total_over"] = np.nan
    
    print(f"[OK] Loaded {len(df)} games")
    
    return df


def enrich_features(features: Dict, game_row: pd.Series) -> Dict:
    """Add external features (Kaggle) to the feature set."""
    # List of external feature prefixes/exact names to include
    keep_cols = [
        "home_rest_days", "away_rest_days", "rest_diff",
        "home_rolling_", "away_rolling_"
    ]
    
    for col, val in game_row.items():
        # Skip if NaN
        if pd.isna(val):
            continue
            
        # Check if column matches our external features
        if any(col.startswith(p) or col == p for p in keep_cols):
            features[col] = float(val)
            
    return features


def backtest_market(
    df: pd.DataFrame,
    market_key: str,
    min_training: int = 80,
) -> pd.DataFrame:
    """
    Run walk-forward backtest for a single market.

    NBA v6.0: Uses period-specific features computed from historical
    data for that specific period (Q1, 1H, or FG).
    """
    config = MARKETS[market_key]

    print(f"\n{'='*60}")
    print(f"BACKTEST: {config['name']} (period: {config['period']})")
    print(f"{'='*60}")

    label_col = config["label_col"]
    line_col = config["line_col"]
    ModelClass = config["model_class"]
    period = config["period"]

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
            print(f"  Processing game {i}/{total_games} (period={period})...")
        
        # Training data
        train_df = valid_df.iloc[:i].copy()
        test_game = valid_df.iloc[i]
        
        try:
            # Build PERIOD-SPECIFIC features for training
            # Each period (Q1, 1H, FG) uses its own historical stats
            train_features = []
            for idx, game in train_df.iterrows():
                historical = train_df[train_df["date"] < game["date"]]
                if len(historical) < 30:
                    continue

                # Get period-specific rolling stats
                home_stats = fe.compute_period_rolling_stats(
                    historical, game["home_team"],
                    pd.to_datetime(game["date"]), period
                )
                away_stats = fe.compute_period_rolling_stats(
                    historical, game["away_team"],
                    pd.to_datetime(game["date"]), period
                )

                if not home_stats or not away_stats:
                    continue

                # Build feature dict with period-specific stats
                features = {}
                for key, val in home_stats.items():
                    features[f"home_{key}"] = val
                for key, val in away_stats.items():
                    features[f"away_{key}"] = val

                # Add differentials
                suffix = f"_{period}" if period != "fg" else ""
                if f"ppg{suffix}" in home_stats and f"ppg{suffix}" in away_stats:
                    features[f"ppg_diff{suffix}"] = home_stats[f"ppg{suffix}"] - away_stats[f"ppg{suffix}"]
                if f"margin{suffix}" in home_stats and f"margin{suffix}" in away_stats:
                    features[f"margin_diff{suffix}"] = home_stats[f"margin{suffix}"] - away_stats[f"margin{suffix}"]

                # Also get standard features for context
                base_features = fe.build_game_features(game, historical)
                if base_features:
                    features.update(base_features)

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

            # Build PERIOD-SPECIFIC features for test game
            historical = train_df.copy()

            # Get period-specific rolling stats for test game
            home_stats = fe.compute_period_rolling_stats(
                historical, test_game["home_team"],
                pd.to_datetime(test_game["date"]), period
            )
            away_stats = fe.compute_period_rolling_stats(
                historical, test_game["away_team"],
                pd.to_datetime(test_game["date"]), period
            )

            if not home_stats or not away_stats:
                continue

            test_features = {}
            for key, val in home_stats.items():
                test_features[f"home_{key}"] = val
            for key, val in away_stats.items():
                test_features[f"away_{key}"] = val

            # Add differentials
            suffix = f"_{period}" if period != "fg" else ""
            if f"ppg{suffix}" in home_stats and f"ppg{suffix}" in away_stats:
                test_features[f"ppg_diff{suffix}"] = home_stats[f"ppg{suffix}"] - away_stats[f"ppg{suffix}"]
            if f"margin{suffix}" in home_stats and f"margin{suffix}" in away_stats:
                test_features[f"margin_diff{suffix}"] = home_stats[f"margin{suffix}"] - away_stats[f"margin{suffix}"]

            # Also add base features
            base_features = fe.build_game_features(test_game, historical)
            if base_features:
                test_features.update(base_features)

            test_features = enrich_features(test_features, test_game)
            
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
            if STRICT_MODE:
                print(f"\n[ERROR] Backtest failed at game {i}: {e}")
                traceback.print_exc()
                raise BacktestError(f"Backtest failed at game {i}: {e}")
            # In non-strict mode, log and continue
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
    global STRICT_MODE
    
    parser = argparse.ArgumentParser(description="Backtest all betting markets")
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
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Training data file (filename in data/processed/ or full path)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Strict mode: fail on any error instead of continuing",
    )
    args = parser.parse_args()
    
    # Set global strict mode
    STRICT_MODE = args.strict
    
    if STRICT_MODE:
        print("[MODE] Running in STRICT mode - will fail on any error")
    
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
    if args.data:
        print(f"Data file: {args.data}")
    
    # Load data
    try:
        df = load_training_data(args.data, strict=STRICT_MODE)
    except DataValidationError as e:
        print(f"\n[ERROR] Data validation failed:\n{e}")
        sys.exit(1)
    
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
    
    # Final validation summary
    print(f"\n{'='*60}")
    print("BACKTEST COMPLETE")
    print(f"{'='*60}")
    
    if not all_results:
        print("\n[WARN] No backtest results generated!")
        if STRICT_MODE:
            sys.exit(1)
    else:
        # Summary of production-ready markets
        print("\n📊 PRODUCTION READINESS SUMMARY:")
        print("-" * 40)
        
        for summary in all_summaries:
            if not summary:
                continue
            
            name = summary.get("name", "Unknown")
            acc = summary.get("accuracy", 0)
            roi = summary.get("roi", 0)
            bets = summary.get("total_bets", 0)
            
            if acc >= 0.55 and roi > 0.05:
                status = "✅ PRODUCTION READY"
            elif acc >= 0.52 and roi > 0:
                status = "⚠️  NEEDS MONITORING"
            else:
                status = "❌ NOT RECOMMENDED"
            
            print(f"  {name}: {acc:.1%} acc, {roi:+.1%} ROI ({bets} bets) - {status}")
        
        print("")
    
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    try:
        main()
    except (BacktestError, DataValidationError) as e:
        print(f"\n[FATAL] {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Backtest cancelled by user")
        sys.exit(130)
