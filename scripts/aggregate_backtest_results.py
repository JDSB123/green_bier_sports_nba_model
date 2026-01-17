#!/usr/bin/env python3
"""
Aggregate Backtest Results

Combines individual market backtest results into a unified report.
Used after running parallel market backtests via Docker Compose.

Usage:
    python scripts/aggregate_backtest_results.py
    python scripts/aggregate_backtest_results.py --results-dir data/backtest_results
    python scripts/aggregate_backtest_results.py --output data/backtest_results/unified_report.json
"""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_market_results(results_dir: Path) -> Dict[str, Any]:
    """Load all market result files from directory."""
    markets = {}

    # Expected result files
    result_files = [
        "fg_spread_results.json",
        "fg_total_results.json",
        "fg_moneyline_results.json",
        "1h_spread_results.json",
        "1h_total_results.json",
        "1h_moneyline_results.json",
        "extended_backtest_results.json",  # From single-run backtest
    ]

    for filename in result_files:
        filepath = results_dir / filename
        if filepath.exists():
            try:
                with open(filepath) as f:
                    data = json.load(f)
                    # Merge markets from file
                    if "markets" in data:
                        for market_key, market_data in data["markets"].items():
                            markets[market_key] = market_data
                            markets[market_key]["source_file"] = filename
                    print(f"  [OK] Loaded: {filename}")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  [WARN] Failed to parse {filename}: {e}")
        else:
            print(f"  [SKIP] Not found: {filename}")

    return markets


def calculate_unified_summary(markets: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate unified summary statistics across all markets."""

    total_bets = 0
    total_wins = 0
    total_profit = 0.0
    has_profit = True

    # Group by market type
    fg_markets = {}
    h1_markets = {}
    spread_markets = {}
    total_markets = {}
    ml_markets = {}

    for market_key, data in markets.items():
        n_bets = data.get("n_bets", 0)
        wins = data.get("wins", 0)
        profit = data.get("profit")

        total_bets += n_bets
        total_wins += wins
        if profit is not None:
            total_profit += profit
        else:
            has_profit = False

        # Categorize
        if market_key.startswith("fg_"):
            fg_markets[market_key] = data
        if market_key.startswith("1h_"):
            h1_markets[market_key] = data
        if "spread" in market_key:
            spread_markets[market_key] = data
        if "total" in market_key:
            total_markets[market_key] = data
        if "moneyline" in market_key:
            ml_markets[market_key] = data

    summary = {
        "generated_at": datetime.now().isoformat(),
        "total_bets": total_bets,
        "total_wins": total_wins,
        "overall_accuracy": total_wins / total_bets if total_bets > 0 else 0,
        "total_profit": total_profit if has_profit else None,
        "overall_roi": (total_profit / total_bets * 100) if has_profit and total_bets > 0 else None,
        "markets": markets,
        "by_category": {
            "full_game": {
                "markets": list(fg_markets.keys()),
                "total_bets": sum(m.get("n_bets", 0) for m in fg_markets.values()),
                "accuracy": (
                    sum(m.get("wins", 0) for m in fg_markets.values()) /
                    sum(m.get("n_bets", 0) for m in fg_markets.values())
                    if sum(m.get("n_bets", 0) for m in fg_markets.values()) > 0 else 0
                ),
            },
            "first_half": {
                "markets": list(h1_markets.keys()),
                "total_bets": sum(m.get("n_bets", 0) for m in h1_markets.values()),
                "accuracy": (
                    sum(m.get("wins", 0) for m in h1_markets.values()) /
                    sum(m.get("n_bets", 0) for m in h1_markets.values())
                    if sum(m.get("n_bets", 0) for m in h1_markets.values()) > 0 else 0
                ),
            },
            "spreads": {
                "markets": list(spread_markets.keys()),
                "total_bets": sum(m.get("n_bets", 0) for m in spread_markets.values()),
                "accuracy": (
                    sum(m.get("wins", 0) for m in spread_markets.values()) /
                    sum(m.get("n_bets", 0) for m in spread_markets.values())
                    if sum(m.get("n_bets", 0) for m in spread_markets.values()) > 0 else 0
                ),
            },
            "totals": {
                "markets": list(total_markets.keys()),
                "total_bets": sum(m.get("n_bets", 0) for m in total_markets.values()),
                "accuracy": (
                    sum(m.get("wins", 0) for m in total_markets.values()) /
                    sum(m.get("n_bets", 0) for m in total_markets.values())
                    if sum(m.get("n_bets", 0) for m in total_markets.values()) > 0 else 0
                ),
            },
            "moneylines": {
                "markets": list(ml_markets.keys()),
                "total_bets": sum(m.get("n_bets", 0) for m in ml_markets.values()),
                "accuracy": (
                    sum(m.get("wins", 0) for m in ml_markets.values()) /
                    sum(m.get("n_bets", 0) for m in ml_markets.values())
                    if sum(m.get("n_bets", 0) for m in ml_markets.values()) > 0 else 0
                ),
            },
        },
    }

    return summary


def print_unified_report(summary: Dict[str, Any]):
    """Print formatted unified report."""

    print()
    print("=" * 80)
    print("UNIFIED BACKTEST REPORT")
    print("=" * 80)
    print(f"Generated: {summary['generated_at']}")
    print()

    # Overall stats
    print("-" * 80)
    print("OVERALL PERFORMANCE")
    print("-" * 80)
    print(f"Total Bets: {summary['total_bets']:,}")
    print(f"Total Wins: {summary['total_wins']:,}")
    print(f"Accuracy: {summary['overall_accuracy']:.1%}")
    if summary["overall_roi"] is not None:
        print(f"ROI: {summary['overall_roi']:+.2f}%")
        print(f"Profit: {summary['total_profit']:+.1f} units")
    print()

    # Per-market breakdown
    print("-" * 80)
    print("BY MARKET")
    print("-" * 80)
    print(f"{'Market':<15} {'Bets':>8} {'Wins':>8} {'Accuracy':>10} {'ROI':>10} {'Profit':>10}")
    print("-" * 80)

    for market_key, data in sorted(summary["markets"].items()):
        n_bets = data.get("n_bets", 0)
        wins = data.get("wins", 0)
        accuracy = data.get("accuracy", 0)
        roi = data.get("roi")
        profit = data.get("profit")

        roi_str = f"{roi:+.2f}%" if roi is not None else "N/A"
        profit_str = f"{profit:+.1f}u" if profit is not None else "N/A"

        print(f"{market_key:<15} {n_bets:>8} {wins:>8} {accuracy:>9.1%} {roi_str:>10} {profit_str:>10}")

    print()

    # By category
    print("-" * 80)
    print("BY CATEGORY")
    print("-" * 80)

    for cat_name, cat_data in summary["by_category"].items():
        if cat_data["total_bets"] > 0:
            print(f"  {cat_name.upper()}: {cat_data['total_bets']:,} bets, {cat_data['accuracy']:.1%} accuracy")
            print(f"    Markets: {', '.join(cat_data['markets'])}")

    print()
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Aggregate parallel backtest results")
    parser.add_argument(
        "--results-dir",
        default="data/backtest_results",
        help="Directory containing individual market result files",
    )
    parser.add_argument(
        "--output", "-o",
        default="data/backtest_results/unified_backtest_report.json",
        help="Output path for unified report",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        sys.exit(1)

    print("Loading market results...")
    markets = load_market_results(results_dir)

    if not markets:
        print("ERROR: No market results found!")
        sys.exit(1)

    print(f"\nLoaded {len(markets)} markets")

    # Calculate unified summary
    summary = calculate_unified_summary(markets)

    # Print report
    print_unified_report(summary)

    # Save to JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
