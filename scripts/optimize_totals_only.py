#!/usr/bin/env python3
"""
Optimize TOTALS markets (1H and FG) independently.

This script runs comprehensive backtesting with parameter sweeps to find
optimal confidence and edge thresholds for totals betting.

Usage:
    python scripts/optimize_totals_only.py
"""
import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import backtest_production


def run_totals_optimization():
    """Run comprehensive totals optimization."""

    print("=" * 80)
    print("TOTALS MARKET OPTIMIZATION (1H + FG)")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Configuration
    data_path = "data/processed/training_data.csv"
    models_dir = "models/production"
    spread_juice = -110
    total_juice = -110

    # Filter to totals only
    markets_filter = ["fg_total", "1h_total"]

    # Parameter grids for optimization
    confidence_grid = [round(x, 2) for x in [0.55, 0.57, 0.59, 0.61, 0.63, 0.65, 0.67, 0.69, 0.71, 0.73, 0.75, 0.77, 0.79]]
    edge_grid = [round(x, 1) for x in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]]

    print("CONFIGURATION:")
    print(f"  Data: {data_path}")
    print(f"  Models: {models_dir}")
    print(f"  Markets: {', '.join(markets_filter)}")
    print(f"  Odds: {spread_juice}/{total_juice}")
    print(f"  Confidence grid: {min(confidence_grid)}-{max(confidence_grid)} (step={confidence_grid[1]-confidence_grid[0]})")
    print(f"  Edge grid: {min(edge_grid)}-{max(edge_grid)} (step={edge_grid[1]-edge_grid[0]})")
    print(f"  Total combinations: {len(confidence_grid) * len(edge_grid)}")
    print()

    # Run backtest once to get all bet results
    print("Running backtest to collect all bet results...")
    results = backtest_production.run_backtest(
        data_path=data_path,
        models_dir=models_dir,
        start_date=None,
        end_date=None,
        spread_juice=spread_juice,
        total_juice=total_juice,
        pricing_enabled=True,
        min_train_games=100,
        max_games=None,
        markets_filter=markets_filter,
    )

    print()
    print("=" * 80)
    print("BASELINE PERFORMANCE (No Filters)")
    print("=" * 80)

    # Show baseline performance
    baseline_summary = {}
    for market_key, bets in results.items():
        if not bets:
            continue

        n_bets = len(bets)
        n_wins = sum(1 for b in bets if b.won)
        accuracy = n_wins / n_bets
        total_profit = sum(b.profit for b in bets if b.profit is not None)
        roi = (total_profit / n_bets * 100) if n_bets > 0 else 0

        baseline_summary[market_key] = {
            "n_bets": n_bets,
            "wins": n_wins,
            "accuracy": accuracy,
            "roi": roi,
            "profit": total_profit,
        }

        print(f"\n{market_key.upper()}")
        print(f"  Total bets: {n_bets}")
        print(f"  Accuracy: {accuracy:.2%}")
        print(f"  ROI: {roi:+.2f}%")
        print(f"  Profit: {total_profit:+.2f} units")

    # Now sweep parameter grids
    print()
    print("=" * 80)
    print("PARAMETER OPTIMIZATION")
    print("=" * 80)

    optimization_results = {}

    for market_key, bets in results.items():
        if not bets:
            continue

        print(f"\nOptimizing {market_key.upper()}...")
        print(f"  Testing {len(confidence_grid) * len(edge_grid)} parameter combinations...")

        market_results = []

        for confidence_threshold in confidence_grid:
            for edge_threshold in edge_grid:
                # Filter bets based on thresholds
                filtered_bets = []
                for bet in bets:
                    # Calculate edge
                    if "total" in market_key:
                        edge = abs(bet.predicted_value - bet.line)
                    else:
                        edge = abs(bet.predicted_value + bet.line)

                    # Apply filters
                    if bet.confidence >= confidence_threshold and edge >= edge_threshold:
                        filtered_bets.append(bet)

                # Calculate metrics
                if len(filtered_bets) >= 30:  # Minimum 30 bets for statistical significance
                    n_bets = len(filtered_bets)
                    n_wins = sum(1 for b in filtered_bets if b.won)
                    accuracy = n_wins / n_bets
                    total_profit = sum(b.profit for b in filtered_bets if b.profit is not None)
                    roi = (total_profit / n_bets * 100) if n_bets > 0 else 0

                    market_results.append({
                        "confidence_threshold": confidence_threshold,
                        "edge_threshold": edge_threshold,
                        "n_bets": n_bets,
                        "wins": n_wins,
                        "accuracy": accuracy,
                        "roi": roi,
                        "profit": total_profit,
                    })

        # Sort by ROI (primary) and accuracy (secondary)
        market_results.sort(key=lambda x: (x["roi"], x["accuracy"]), reverse=True)

        optimization_results[market_key] = market_results

        # Show top 10 results
        print(f"\n  Top 10 parameter combinations for {market_key.upper()}:")
        print(f"  {'Conf':>6} {'Edge':>6} {'Bets':>6} {'Wins':>6} {'Acc':>7} {'ROI':>8} {'Profit':>8}")
        print(f"  {'-'*60}")

        for i, result in enumerate(market_results[:10], 1):
            print(f"  {result['confidence_threshold']:>6.2f} {result['edge_threshold']:>6.1f} "
                  f"{result['n_bets']:>6d} {result['wins']:>6d} "
                  f"{result['accuracy']:>6.1%} {result['roi']:>+7.2f}% {result['profit']:>+7.2f}u")

    # Save results
    output_dir = PROJECT_ROOT / "data" / "backtest_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"totals_optimization_{timestamp}.json"

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "data_path": data_path,
            "models_dir": models_dir,
            "markets": markets_filter,
            "spread_juice": spread_juice,
            "total_juice": total_juice,
            "confidence_grid": confidence_grid,
            "edge_grid": edge_grid,
        },
        "baseline": baseline_summary,
        "optimization_results": optimization_results,
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for market_key in markets_filter:
        if market_key in optimization_results and optimization_results[market_key]:
            best = optimization_results[market_key][0]
            baseline = baseline_summary[market_key]

            print(f"\n{market_key.upper()}")
            print(f"  Baseline (no filters):")
            print(f"    Bets: {baseline['n_bets']}, Accuracy: {baseline['accuracy']:.2%}, ROI: {baseline['roi']:+.2f}%")
            print(f"  Optimal parameters:")
            print(f"    Confidence >= {best['confidence_threshold']:.2f}")
            print(f"    Edge >= {best['edge_threshold']:.1f}")
            print(f"    Bets: {best['n_bets']}, Accuracy: {best['accuracy']:.2%}, ROI: {best['roi']:+.2f}%")
            print(f"    Profit: {best['profit']:+.2f} units")

            improvement_roi = best['roi'] - baseline['roi']
            improvement_acc = best['accuracy'] - baseline['accuracy']
            print(f"  Improvement:")
            print(f"    ROI: {improvement_roi:+.2f} percentage points")
            print(f"    Accuracy: {improvement_acc:+.2%} percentage points")

    print(f"\nResults saved to: {output_file}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return output_data


if __name__ == "__main__":
    results = run_totals_optimization()
