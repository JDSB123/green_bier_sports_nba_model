#!/usr/bin/env python3
"""
Train and optimize moneyline models for FG and 1H markets independently.

This script trains moneyline prediction models using margin-derived probabilities
and tests different confidence/probability threshold configurations.

Usage:
    # Train both FG and 1H moneyline models
    python scripts/train_moneyline_models.py

    # Train only FG moneyline
    python scripts/train_moneyline_models.py --market fg

    # Train only 1H moneyline
    python scripts/train_moneyline_models.py --market 1h
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import settings


class MoneylineBacktest:
    """Backtest moneyline predictions with various thresholds."""

    def __init__(self, period: str = "fg", margin_std: float = 12.0):
        """
        Initialize moneyline backtest.

        Args:
            period: "fg" or "1h"
            margin_std: Standard deviation for margin distribution
        """
        self.period = period
        self.margin_std = margin_std

    def american_to_implied(self, odds: int) -> float:
        """Convert American odds to implied probability."""
        if odds < 0:
            return abs(odds) / (abs(odds) + 100)
        else:
            return 100 / (odds + 100)

    def calculate_profit(self, won: bool, odds: int) -> float:
        """Calculate profit for 1 unit bet."""
        if not won:
            return -1.0

        if odds < 0:
            return 100 / abs(odds)
        else:
            return odds / 100

    def margin_to_win_prob(self, predicted_margin: float) -> float:
        """
        Convert predicted margin to win probability using normal distribution.

        P(home wins) = P(margin > 0) = Φ(predicted_margin / σ)
        """
        return float(norm.cdf(predicted_margin / self.margin_std))

    def backtest(
        self,
        df: pd.DataFrame,
        min_confidence: float = 0.52,
        min_edge: float = 0.0,
    ) -> Dict:
        """
        Run backtest on moneyline predictions.

        Args:
            df: DataFrame with games and predictions
            min_confidence: Minimum win probability to bet
            min_edge: Minimum edge vs implied odds to bet

        Returns:
            Dictionary with performance metrics
        """
        results = []

        for _, game in df.iterrows():
            # Get predicted margin
            if self.period == "1h":
                predicted_margin = game.get("predicted_margin_1h", np.nan)
                home_ml = game.get("1h_ml_home", np.nan)
                away_ml = game.get("1h_ml_away", np.nan)

                # Actual outcome
                if pd.notna(game.get("home_1h")) and pd.notna(game.get("away_1h")):
                    actual_home_win = int(game["home_1h"] > game["away_1h"])
                elif pd.notna(game.get("home_q1")) and pd.notna(game.get("home_q2")):
                    home_1h = game["home_q1"] + game["home_q2"]
                    away_1h = game["away_q1"] + game["away_q2"]
                    actual_home_win = int(home_1h > away_1h)
                else:
                    continue
            else:
                predicted_margin = game.get("predicted_margin", np.nan)
                home_ml = game.get("fg_ml_home", np.nan)
                away_ml = game.get("fg_ml_away", np.nan)

                # Actual outcome
                if pd.notna(game.get("home_score")) and pd.notna(game.get("away_score")):
                    actual_home_win = int(game["home_score"] > game["away_score"])
                else:
                    continue

            # Skip if missing data
            if pd.isna(predicted_margin):
                continue
            if pd.isna(home_ml) or pd.isna(away_ml):
                continue

            # Convert margin to win probability
            home_win_prob = self.margin_to_win_prob(predicted_margin)
            away_win_prob = 1 - home_win_prob

            # Determine which side to bet
            if home_win_prob >= 0.5:
                bet_side = "home"
                bet_prob = home_win_prob
                bet_odds = int(home_ml)
                won = (actual_home_win == 1)
            else:
                bet_side = "away"
                bet_prob = away_win_prob
                bet_odds = int(away_ml)
                won = (actual_home_win == 0)

            # Apply confidence threshold
            if bet_prob < min_confidence:
                continue

            # Calculate edge
            implied_prob = self.american_to_implied(bet_odds)
            edge = bet_prob - implied_prob

            # Apply edge threshold
            if edge < min_edge:
                continue

            # Calculate profit
            profit = self.calculate_profit(won, bet_odds)

            results.append({
                "date": game.get("date", game.get("game_date")),
                "home_team": game.get("home", game.get("home_team")),
                "away_team": game.get("away", game.get("away_team")),
                "side": bet_side,
                "probability": bet_prob,
                "odds": bet_odds,
                "implied_prob": implied_prob,
                "edge": edge,
                "won": won,
                "profit": profit,
            })

        if not results:
            return {
                "n_bets": 0,
                "accuracy": 0.0,
                "roi": 0.0,
                "total_profit": 0.0,
                "avg_odds": 0.0,
                "avg_edge": 0.0,
            }

        df_results = pd.DataFrame(results)

        return {
            "n_bets": len(df_results),
            "accuracy": df_results["won"].mean(),
            "roi": df_results["profit"].mean(),
            "total_profit": df_results["profit"].sum(),
            "avg_odds": df_results["odds"].mean(),
            "avg_edge": df_results["edge"].mean(),
            "bets": results,
        }


def optimize_thresholds(
    df: pd.DataFrame,
    period: str = "fg",
    margin_std: float = 12.0,
    confidence_range: Tuple[float, float, float] = (0.50, 0.70, 0.02),
    edge_range: Tuple[float, float, float] = (0.0, 0.10, 0.01),
    min_bets: int = 30,
) -> List[Dict]:
    """
    Optimize confidence and edge thresholds for moneyline betting.

    Args:
        df: DataFrame with historical games
        period: "fg" or "1h"
        margin_std: Standard deviation for margin distribution
        confidence_range: (min, max, step) for confidence threshold
        edge_range: (min, max, step) for edge threshold
        min_bets: Minimum bets required for valid configuration

    Returns:
        List of configurations sorted by ROI
    """
    backtester = MoneylineBacktest(period=period, margin_std=margin_std)

    confidence_values = np.arange(*confidence_range)
    edge_values = np.arange(*edge_range)

    results = []
    total_configs = len(confidence_values) * len(edge_values)

    print(f"\nOptimizing {period.upper()} moneyline thresholds...")
    print(f"Testing {total_configs} configurations...")

    for i, min_conf in enumerate(confidence_values):
        for min_edge in edge_values:
            metrics = backtester.backtest(
                df,
                min_confidence=min_conf,
                min_edge=min_edge,
            )

            if metrics["n_bets"] >= min_bets:
                results.append({
                    "min_confidence": round(min_conf, 3),
                    "min_edge": round(min_edge, 3),
                    "n_bets": metrics["n_bets"],
                    "accuracy": round(metrics["accuracy"], 4),
                    "roi": round(metrics["roi"], 4),
                    "total_profit": round(metrics["total_profit"], 2),
                    "avg_edge": round(metrics["avg_edge"], 4),
                })

        if (i + 1) % 5 == 0:
            print(f"  Progress: {i+1}/{len(confidence_values)} confidence levels tested")

    # Sort by ROI descending
    results.sort(key=lambda x: x["roi"], reverse=True)

    return results


def train_moneyline_model(
    period: str = "fg",
    data_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    test_cutoff: Optional[str] = None,
) -> Dict:
    """
    Train moneyline model for specified period.

    Args:
        period: "fg" or "1h"
        data_file: Path to training data CSV
        output_dir: Directory to save results
        test_cutoff: Date to split train/test (YYYY-MM-DD)

    Returns:
        Dictionary with training results and optimal parameters
    """
    print(f"\n{'='*70}")
    print(f"Training {period.upper()} Moneyline Model")
    print(f"{'='*70}")

    # Load data
    if data_file and os.path.exists(data_file):
        data_path = data_file
    else:
        data_path = os.path.join(settings.data_processed_dir, "training_data.csv")

    if not os.path.exists(data_path):
        print(f"Training data not found at {data_path}")
        return {}

    df = pd.read_csv(data_path, low_memory=False)

    # Standardize date column
    if "game_date" in df.columns:
        df["date"] = pd.to_datetime(df["game_date"], format="mixed", errors="coerce")
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], format="mixed", errors="coerce")

    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    print(f"Loaded {len(df)} games from {df['date'].min().date()} to {df['date'].max().date()}")

    # Filter for required columns
    if period == "1h":
        required = ["predicted_margin_1h"]
        margin_std = 12.0 * 0.6  # Adjust for first half
    else:
        required = ["predicted_margin"]
        margin_std = 12.0

    # Check if we have predicted margins
    if period == "1h" and "predicted_margin_1h" not in df.columns:
        print("Warning: predicted_margin_1h not found in data. Moneyline model requires margin predictions.")
        return {}
    elif period == "fg" and "predicted_margin" not in df.columns:
        print("Warning: predicted_margin not found in data. Moneyline model requires margin predictions.")
        return {}

    # Split into train/test
    if test_cutoff:
        cutoff_date = pd.to_datetime(test_cutoff)
        train_df = df[df["date"] < cutoff_date].copy()
        test_df = df[df["date"] >= cutoff_date].copy()
        print(f"\nTrain: {len(train_df)} games (before {test_cutoff})")
        print(f"Test:  {len(test_df)} games (after {test_cutoff})")
    else:
        # Use 80/20 split
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        print(f"\nTrain: {len(train_df)} games")
        print(f"Test:  {len(test_df)} games")

    # Optimize on training data
    print("\n" + "-"*70)
    print("Optimizing thresholds on TRAINING data")
    print("-"*70)

    train_results = optimize_thresholds(
        train_df,
        period=period,
        margin_std=margin_std,
        confidence_range=(0.50, 0.75, 0.01),
        edge_range=(0.0, 0.15, 0.005),
        min_bets=50,
    )

    # Show top 10 configurations
    print(f"\nTop 10 configurations (by ROI):")
    print(f"{'Conf':>6} {'Edge':>6} {'Bets':>6} {'Acc':>7} {'ROI':>8} {'Profit':>8}")
    print("-"*50)
    for config in train_results[:10]:
        print(
            f"{config['min_confidence']:>6.3f} "
            f"{config['min_edge']:>6.3f} "
            f"{config['n_bets']:>6} "
            f"{config['accuracy']:>7.1%} "
            f"{config['roi']:>8.2%} "
            f"{config['total_profit']:>8.1f}"
        )

    # Test best configuration on test set
    if train_results:
        best_config = train_results[0]

        print("\n" + "-"*70)
        print("Testing BEST configuration on TEST data")
        print("-"*70)
        print(f"Best config: min_confidence={best_config['min_confidence']}, min_edge={best_config['min_edge']}")

        backtester = MoneylineBacktest(period=period, margin_std=margin_std)
        test_metrics = backtester.backtest(
            test_df,
            min_confidence=best_config["min_confidence"],
            min_edge=best_config["min_edge"],
        )

        print(f"\nTest Results:")
        print(f"  Bets:         {test_metrics['n_bets']}")
        print(f"  Accuracy:     {test_metrics['accuracy']:.1%}")
        print(f"  ROI:          {test_metrics['roi']:+.2%}")
        print(f"  Total Profit: {test_metrics['total_profit']:+.1f} units")
        print(f"  Avg Edge:     {test_metrics['avg_edge']:+.2%}")

        # Save results
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

            results_file = os.path.join(output_dir, f"{period}_moneyline_optimization_results.json")
            with open(results_file, "w") as f:
                json.dump({
                    "period": period,
                    "margin_std": margin_std,
                    "best_config": best_config,
                    "test_metrics": {
                        "n_bets": test_metrics["n_bets"],
                        "accuracy": test_metrics["accuracy"],
                        "roi": test_metrics["roi"],
                        "total_profit": test_metrics["total_profit"],
                        "avg_edge": test_metrics["avg_edge"],
                    },
                    "top_10_train_configs": train_results[:10],
                }, f, indent=2)

            print(f"\nResults saved to: {results_file}")

        return {
            "period": period,
            "best_config": best_config,
            "train_metrics": best_config,
            "test_metrics": test_metrics,
        }

    return {}


def main():
    parser = argparse.ArgumentParser(description="Train and optimize moneyline models")
    parser.add_argument(
        "--market",
        choices=["fg", "1h", "all"],
        default="all",
        help="Which market to train: fg (full game), 1h (first half), or all",
    )
    parser.add_argument(
        "--data-file",
        help="Path to training data CSV",
    )
    parser.add_argument(
        "--output-dir",
        default="data/backtest_results",
        help="Directory to save optimization results",
    )
    parser.add_argument(
        "--test-cutoff",
        help="Date for train/test split (YYYY-MM-DD)",
    )

    args = parser.parse_args()

    results = {}

    if args.market in ["fg", "all"]:
        fg_results = train_moneyline_model(
            period="fg",
            data_file=args.data_file,
            output_dir=args.output_dir,
            test_cutoff=args.test_cutoff,
        )
        if fg_results:
            results["fg"] = fg_results

    if args.market in ["1h", "all"]:
        h1_results = train_moneyline_model(
            period="1h",
            data_file=args.data_file,
            output_dir=args.output_dir,
            test_cutoff=args.test_cutoff,
        )
        if h1_results:
            results["1h"] = h1_results

    # Summary
    print("\n" + "="*70)
    print("MONEYLINE OPTIMIZATION SUMMARY")
    print("="*70)

    for period, result in results.items():
        print(f"\n{period.upper()} Moneyline:")
        print(f"  Best Config:")
        print(f"    Min Confidence: {result['best_config']['min_confidence']:.3f}")
        print(f"    Min Edge:       {result['best_config']['min_edge']:.3f}")
        print(f"  Test Performance:")
        print(f"    Bets:           {result['test_metrics']['n_bets']}")
        print(f"    Accuracy:       {result['test_metrics']['accuracy']:.1%}")
        print(f"    ROI:            {result['test_metrics']['roi']:+.2%}")
        print(f"    Total Profit:   {result['test_metrics']['total_profit']:+.1f} units")


if __name__ == "__main__":
    main()
