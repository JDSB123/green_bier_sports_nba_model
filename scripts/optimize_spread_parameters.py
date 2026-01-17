#!/usr/bin/env python3
"""
Optimize Spread Market Parameters (1H and FG)

This script runs backtests across different parameter configurations to find
optimal settings for spread betting (both first half and full game).

It tests:
- Confidence thresholds (0.55, 0.60, 0.62, 0.65, 0.68, 0.70)
- Juice values (-105, -110, -115)
- Edge thresholds (0.0, 1.0, 2.0, 3.0)

Results are saved to a unique output file for independent analysis.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict

# Import from backtest_extended
from scripts.backtest_extended import (
    get_default_market_configs,
    run_backtest,
    BetResult,
)

@dataclass
class ParameterConfig:
    """Configuration for a single backtest run."""
    market: str  # fg_spread or 1h_spread
    min_confidence: float
    juice: int
    min_edge: float

    def to_key(self) -> str:
        """Generate unique key for this configuration."""
        return f"{self.market}_conf{self.min_confidence:.2f}_juice{self.juice}_edge{self.min_edge:.1f}"


@dataclass
class BacktestMetrics:
    """Metrics from a single backtest run."""
    config: ParameterConfig
    n_bets: int
    wins: int
    accuracy: float
    roi: float
    profit: float

    # Breakdown by confidence tiers
    tier_55_60_bets: int = 0
    tier_55_60_acc: float = 0.0
    tier_55_60_roi: float = 0.0

    tier_60_65_bets: int = 0
    tier_60_65_acc: float = 0.0
    tier_60_65_roi: float = 0.0

    tier_65_70_bets: int = 0
    tier_65_70_acc: float = 0.0
    tier_65_70_roi: float = 0.0

    tier_70_plus_bets: int = 0
    tier_70_plus_acc: float = 0.0
    tier_70_plus_roi: float = 0.0


def calculate_tier_metrics(bets: List[BetResult], low: float, high: float) -> tuple:
    """Calculate metrics for a confidence tier."""
    tier_bets = [b for b in bets if low <= b.confidence < high]
    if not tier_bets:
        return 0, 0.0, 0.0

    wins = sum(1 for b in tier_bets if b.won)
    acc = wins / len(tier_bets)
    profit = sum(b.profit for b in tier_bets if b.profit is not None)
    roi = (profit / len(tier_bets) * 100) if len(tier_bets) > 0 else 0.0

    return len(tier_bets), acc, roi


def run_single_backtest(
    df: pd.DataFrame,
    models_dir: Path,
    config: ParameterConfig,
) -> BacktestMetrics:
    """Run a single backtest with given parameters."""

    # Configure market
    market_configs = get_default_market_configs()
    market_config = market_configs[config.market]
    market_config.min_confidence = config.min_confidence
    market_config.juice = config.juice
    market_config.min_edge = config.min_edge

    # Run backtest
    results = run_backtest(
        df=df,
        models_dir=models_dir,
        market_configs={config.market: market_config},
        pricing_enabled=True,
    )

    bets = results[config.market]

    if not bets:
        return BacktestMetrics(
            config=config,
            n_bets=0,
            wins=0,
            accuracy=0.0,
            roi=0.0,
            profit=0.0,
        )

    # Overall metrics
    n_bets = len(bets)
    wins = sum(1 for b in bets if b.won)
    accuracy = wins / n_bets
    profit = sum(b.profit for b in bets if b.profit is not None)
    roi = (profit / n_bets * 100) if n_bets > 0 else 0.0

    # Tier metrics
    tier_55_60 = calculate_tier_metrics(bets, 0.55, 0.60)
    tier_60_65 = calculate_tier_metrics(bets, 0.60, 0.65)
    tier_65_70 = calculate_tier_metrics(bets, 0.65, 0.70)
    tier_70_plus = calculate_tier_metrics(bets, 0.70, 1.01)

    return BacktestMetrics(
        config=config,
        n_bets=n_bets,
        wins=wins,
        accuracy=accuracy,
        roi=roi,
        profit=profit,
        tier_55_60_bets=tier_55_60[0],
        tier_55_60_acc=tier_55_60[1],
        tier_55_60_roi=tier_55_60[2],
        tier_60_65_bets=tier_60_65[0],
        tier_60_65_acc=tier_60_65[1],
        tier_60_65_roi=tier_60_65[2],
        tier_65_70_bets=tier_65_70[0],
        tier_65_70_acc=tier_65_70[1],
        tier_65_70_roi=tier_65_70[2],
        tier_70_plus_bets=tier_70_plus[0],
        tier_70_plus_acc=tier_70_plus[1],
        tier_70_plus_roi=tier_70_plus[2],
    )


def optimize_spread_parameters(
    df: pd.DataFrame,
    models_dir: Path,
    markets: List[str] = None,
    confidence_values: List[float] = None,
    juice_values: List[int] = None,
    edge_values: List[float] = None,
) -> Dict[str, List[BacktestMetrics]]:
    """
    Run parameter optimization for spread markets.

    Returns:
        Dictionary of market -> list of backtest metrics
    """
    if markets is None:
        markets = ["fg_spread", "1h_spread"]

    if confidence_values is None:
        confidence_values = [0.55, 0.60, 0.62, 0.65, 0.68, 0.70]

    if juice_values is None:
        juice_values = [-105, -110, -115]

    if edge_values is None:
        edge_values = [0.0, 1.0, 2.0, 3.0]

    results = {market: [] for market in markets}

    total_runs = len(markets) * len(confidence_values) * len(juice_values) * len(edge_values)
    current_run = 0

    print("=" * 80)
    print("SPREAD PARAMETER OPTIMIZATION")
    print("=" * 80)
    print(f"Markets: {', '.join(markets)}")
    print(f"Confidence values: {confidence_values}")
    print(f"Juice values: {juice_values}")
    print(f"Edge values: {edge_values}")
    print(f"Total runs: {total_runs}")
    print("=" * 80)
    print()

    for market in markets:
        print(f"\n{market.upper()} OPTIMIZATION")
        print("-" * 80)

        for conf, juice, edge in product(confidence_values, juice_values, edge_values):
            current_run += 1
            config = ParameterConfig(
                market=market,
                min_confidence=conf,
                juice=juice,
                min_edge=edge,
            )

            print(f"[{current_run}/{total_runs}] Testing {market}: conf={conf:.2f}, juice={juice}, edge={edge:.1f}... ", end="", flush=True)

            metrics = run_single_backtest(df, models_dir, config)
            results[market].append(metrics)

            if metrics.n_bets > 0:
                print(f"{metrics.n_bets:4d} bets, {metrics.accuracy:5.1%} acc, {metrics.roi:+6.2f}% ROI")
            else:
                print("No bets (filters too strict)")

    return results


def print_optimization_results(results: Dict[str, List[BacktestMetrics]]):
    """Print formatted optimization results."""
    print("\n" + "=" * 80)
    print("OPTIMIZATION RESULTS SUMMARY")
    print("=" * 80)

    for market, metrics_list in results.items():
        if not metrics_list:
            continue

        print(f"\n{market.upper()}")
        print("-" * 80)

        # Filter out configs with no bets
        valid_metrics = [m for m in metrics_list if m.n_bets > 0]

        if not valid_metrics:
            print("No valid results (all configurations produced 0 bets)")
            continue

        # Sort by ROI descending
        sorted_by_roi = sorted(valid_metrics, key=lambda m: m.roi, reverse=True)

        print("\nTOP 10 BY ROI:")
        print(f"{'Rank':<5} {'Conf':<6} {'Juice':<7} {'Edge':<6} {'Bets':<6} {'Acc':<8} {'ROI':<10} {'Profit':<10}")
        print("-" * 80)

        for i, m in enumerate(sorted_by_roi[:10], 1):
            print(
                f"{i:<5} {m.config.min_confidence:<6.2f} {m.config.juice:<7} {m.config.min_edge:<6.1f} "
                f"{m.n_bets:<6} {m.accuracy:<8.1%} {m.roi:<10.2f}% {m.profit:<10.2f}u"
            )

        # Find best by accuracy (with min 50 bets)
        high_volume = [m for m in valid_metrics if m.n_bets >= 50]
        if high_volume:
            sorted_by_acc = sorted(high_volume, key=lambda m: m.accuracy, reverse=True)
            best_acc = sorted_by_acc[0]

            print(f"\nBEST ACCURACY (min 50 bets):")
            print(f"  Config: conf={best_acc.config.min_confidence:.2f}, juice={best_acc.config.juice}, edge={best_acc.config.min_edge:.1f}")
            print(f"  Results: {best_acc.n_bets} bets, {best_acc.accuracy:.1%} acc, {best_acc.roi:+.2f}% ROI, {best_acc.profit:+.2f}u")

        # Find best balanced (ROI > 0, accuracy > 52%, min 30 bets)
        balanced = [m for m in valid_metrics if m.n_bets >= 30 and m.accuracy >= 0.52 and m.roi > 0]
        if balanced:
            sorted_balanced = sorted(balanced, key=lambda m: (m.roi * m.accuracy), reverse=True)
            best_balanced = sorted_balanced[0]

            print(f"\nBEST BALANCED (ROI > 0, Acc > 52%, min 30 bets):")
            print(f"  Config: conf={best_balanced.config.min_confidence:.2f}, juice={best_balanced.config.juice}, edge={best_balanced.config.min_edge:.1f}")
            print(f"  Results: {best_balanced.n_bets} bets, {best_balanced.accuracy:.1%} acc, {best_balanced.roi:+.2f}% ROI, {best_balanced.profit:+.2f}u")


def save_results(results: Dict[str, List[BacktestMetrics]], output_path: Path):
    """Save results to JSON file."""
    output = {
        "timestamp": datetime.now().isoformat(),
        "markets": {},
    }

    for market, metrics_list in results.items():
        output["markets"][market] = []
        for m in metrics_list:
            output["markets"][market].append({
                "config": {
                    "min_confidence": m.config.min_confidence,
                    "juice": m.config.juice,
                    "min_edge": m.config.min_edge,
                },
                "metrics": {
                    "n_bets": m.n_bets,
                    "wins": m.wins,
                    "accuracy": m.accuracy,
                    "roi": m.roi,
                    "profit": m.profit,
                },
                "tiers": {
                    "55-60": {
                        "n_bets": m.tier_55_60_bets,
                        "accuracy": m.tier_55_60_acc,
                        "roi": m.tier_55_60_roi,
                    },
                    "60-65": {
                        "n_bets": m.tier_60_65_bets,
                        "accuracy": m.tier_60_65_acc,
                        "roi": m.tier_60_65_roi,
                    },
                    "65-70": {
                        "n_bets": m.tier_65_70_bets,
                        "accuracy": m.tier_65_70_acc,
                        "roi": m.tier_65_70_roi,
                    },
                    "70+": {
                        "n_bets": m.tier_70_plus_bets,
                        "accuracy": m.tier_70_plus_acc,
                        "roi": m.tier_70_plus_roi,
                    },
                },
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Optimize spread market parameters independently"
    )

    parser.add_argument(
        "--data",
        default="data/processed/training_data.csv",
        help="Path to training data CSV",
    )
    parser.add_argument(
        "--models-dir",
        default="models/production",
        help="Path to production models directory",
    )
    parser.add_argument(
        "--markets",
        default="fg_spread,1h_spread",
        help="Comma-separated markets to optimize (default: fg_spread,1h_spread)",
    )
    parser.add_argument(
        "--output",
        default="data/backtest_results/spread_optimization_results.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--start-date",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: test fewer parameter combinations",
    )

    args = parser.parse_args()

    # Parse markets
    markets = [m.strip() for m in args.markets.split(",")]

    # Load data
    data_path = Path(args.data)
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path, low_memory=False)

    # Parse dates
    df["date"] = pd.to_datetime(df.get("game_date", df.get("date")), errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    print(f"Loaded {len(df)} games from {df['date'].min().date()} to {df['date'].max().date()}")

    # Apply date filters
    if args.start_date:
        df = df[df["date"] >= pd.to_datetime(args.start_date)]
    if args.end_date:
        df = df[df["date"] <= pd.to_datetime(args.end_date)]

    print(f"After date filter: {len(df)} games")

    # Define parameter grids
    if args.quick:
        confidence_values = [0.55, 0.62, 0.68]
        juice_values = [-110]
        edge_values = [0.0, 2.0]
    else:
        confidence_values = [0.55, 0.60, 0.62, 0.65, 0.68, 0.70]
        juice_values = [-105, -110, -115]
        edge_values = [0.0, 1.0, 2.0, 3.0]

    # Run optimization
    results = optimize_spread_parameters(
        df=df,
        models_dir=Path(args.models_dir),
        markets=markets,
        confidence_values=confidence_values,
        juice_values=juice_values,
        edge_values=edge_values,
    )

    # Print results
    print_optimization_results(results)

    # Save results
    save_results(results, Path(args.output))


if __name__ == "__main__":
    main()
