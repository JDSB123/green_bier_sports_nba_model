#!/usr/bin/env python3
"""
Analyze Spread Optimization Results

This script analyzes the backtest results from spread parameter optimization
and generates a comprehensive report with recommendations.
"""
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ResultSummary:
    """Summary of a single backtest result."""
    market: str
    confidence: float
    juice: int
    n_bets: int
    accuracy: float
    roi: float
    profit: float

    # High confidence subset (60%+)
    hc_n_bets: int = 0
    hc_accuracy: float = 0.0
    hc_roi: float = 0.0

    @property
    def score(self) -> float:
        """Composite score: ROI * accuracy * log(bets)."""
        if self.n_bets < 10:
            return 0.0
        import math
        return self.roi * self.accuracy * math.log(self.n_bets + 1)


def parse_filename(filename: str) -> Tuple[str, float, int]:
    """Parse market, confidence, and juice from filename."""
    # Expected format: {market}_conf{XX}_j{XXX}.json
    # Examples: fg_spread_conf55_j110.json, 1h_spread_conf62_j105.json

    match = re.match(r'(fg_spread|1h_spread)_conf(\d+)_j(\d+)\.json', filename)
    if not match:
        return None, None, None

    market = match.group(1)
    conf = float(match.group(2)) / 100.0  # 55 -> 0.55
    juice = -int(match.group(3))  # 110 -> -110

    return market, conf, juice


def load_result(filepath: Path) -> ResultSummary:
    """Load a single backtest result file."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    filename = filepath.name
    market, conf, juice = parse_filename(filename)

    if market is None:
        raise ValueError(f"Could not parse filename: {filename}")

    # Extract market-specific data
    market_data = data.get("markets", {}).get(market)

    if not market_data:
        return ResultSummary(
            market=market,
            confidence=conf,
            juice=juice,
            n_bets=0,
            accuracy=0.0,
            roi=0.0,
            profit=0.0,
        )

    # Main metrics
    n_bets = market_data.get("n_bets", 0)
    accuracy = market_data.get("accuracy", 0.0)
    roi = market_data.get("roi", 0.0) if market_data.get("roi") is not None else 0.0
    profit = market_data.get("profit", 0.0) if market_data.get("profit") is not None else 0.0

    # High confidence metrics (60%+)
    hc_data = market_data.get("high_conf", {})
    hc_n_bets = hc_data.get("n_bets", 0)
    hc_accuracy = hc_data.get("accuracy", 0.0)
    hc_roi = hc_data.get("roi", 0.0) if hc_data.get("roi") is not None else 0.0

    return ResultSummary(
        market=market,
        confidence=conf,
        juice=juice,
        n_bets=n_bets,
        accuracy=accuracy,
        roi=roi,
        profit=profit,
        hc_n_bets=hc_n_bets,
        hc_accuracy=hc_accuracy,
        hc_roi=hc_roi,
    )


def load_all_results(results_dir: Path) -> List[ResultSummary]:
    """Load all optimization results from directory."""
    results = []

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return results

    for filepath in results_dir.glob("*.json"):
        try:
            result = load_result(filepath)
            results.append(result)
        except Exception as e:
            print(f"Error loading {filepath.name}: {e}")

    return results


def print_optimization_report(results: List[ResultSummary], output_file: Path = None):
    """Generate and print comprehensive optimization report."""

    # Separate by market
    fg_results = [r for r in results if r.market == "fg_spread"]
    h1_results = [r for r in results if r.market == "1h_spread"]

    # Generate report text
    report_lines = []

    report_lines.append("=" * 100)
    report_lines.append("SPREAD PARAMETER OPTIMIZATION - ANALYSIS REPORT")
    report_lines.append("=" * 100)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Total configurations tested: {len(results)}")
    report_lines.append("")

    # ========================================================================
    # FG SPREAD ANALYSIS
    # ========================================================================

    report_lines.append("")
    report_lines.append("=" * 100)
    report_lines.append("FULL GAME (FG) SPREAD OPTIMIZATION")
    report_lines.append("=" * 100)
    report_lines.append("")

    if fg_results:
        # Filter valid results (min 10 bets)
        valid_fg = [r for r in fg_results if r.n_bets >= 10]

        if valid_fg:
            # Sort by different criteria
            by_roi = sorted(valid_fg, key=lambda r: r.roi, reverse=True)
            by_accuracy = sorted(valid_fg, key=lambda r: r.accuracy, reverse=True)
            by_profit = sorted(valid_fg, key=lambda r: r.profit, reverse=True)
            by_score = sorted(valid_fg, key=lambda r: r.score, reverse=True)

            # Top 10 by ROI
            report_lines.append("TOP 10 CONFIGURATIONS BY ROI:")
            report_lines.append(f"{'Rank':<5} {'Conf':<8} {'Juice':<7} {'Bets':<7} {'Accuracy':<10} {'ROI':<12} {'Profit':<12} {'HC Bets':<8} {'HC ROI':<10}")
            report_lines.append("-" * 100)

            for i, r in enumerate(by_roi[:10], 1):
                report_lines.append(
                    f"{i:<5} {r.confidence:<8.2f} {r.juice:<7} {r.n_bets:<7} {r.accuracy:<10.1%} {r.roi:<12.2f}% {r.profit:<12.2f}u {r.hc_n_bets:<8} {r.hc_roi:<10.2f}%"
                )

            report_lines.append("")

            # Top 5 by accuracy
            report_lines.append("TOP 5 CONFIGURATIONS BY ACCURACY:")
            report_lines.append(f"{'Rank':<5} {'Conf':<8} {'Juice':<7} {'Bets':<7} {'Accuracy':<10} {'ROI':<12} {'Profit':<12}")
            report_lines.append("-" * 100)

            for i, r in enumerate(by_accuracy[:5], 1):
                report_lines.append(
                    f"{i:<5} {r.confidence:<8.2f} {r.juice:<7} {r.n_bets:<7} {r.accuracy:<10.1%} {r.roi:<12.2f}% {r.profit:<12.2f}u"
                )

            report_lines.append("")

            # Top 5 by profit
            report_lines.append("TOP 5 CONFIGURATIONS BY TOTAL PROFIT:")
            report_lines.append(f"{'Rank':<5} {'Conf':<8} {'Juice':<7} {'Bets':<7} {'Accuracy':<10} {'ROI':<12} {'Profit':<12}")
            report_lines.append("-" * 100)

            for i, r in enumerate(by_profit[:5], 1):
                report_lines.append(
                    f"{i:<5} {r.confidence:<8.2f} {r.juice:<7} {r.n_bets:<7} {r.accuracy:<10.1%} {r.roi:<12.2f}% {r.profit:<12.2f}u"
                )

            report_lines.append("")

            # Best balanced (composite score)
            report_lines.append("RECOMMENDED CONFIGURATION (Balanced Score):")
            best_fg = by_score[0]
            report_lines.append(f"  Confidence Threshold: {best_fg.confidence:.2f}")
            report_lines.append(f"  Juice: {best_fg.juice}")
            report_lines.append(f"  Total Bets: {best_fg.n_bets}")
            report_lines.append(f"  Accuracy: {best_fg.accuracy:.1%}")
            report_lines.append(f"  ROI: {best_fg.roi:+.2f}%")
            report_lines.append(f"  Profit: {best_fg.profit:+.2f} units")
            if best_fg.hc_n_bets > 0:
                report_lines.append(f"  High Confidence (60%+): {best_fg.hc_n_bets} bets, {best_fg.hc_roi:+.2f}% ROI")
            report_lines.append("")

            # Parameter sensitivity analysis
            report_lines.append("PARAMETER SENSITIVITY ANALYSIS:")
            report_lines.append("")

            # Group by confidence
            report_lines.append("  By Confidence Threshold (averaged across juice values):")
            conf_groups = {}
            for r in valid_fg:
                if r.confidence not in conf_groups:
                    conf_groups[r.confidence] = []
                conf_groups[r.confidence].append(r)

            report_lines.append(f"  {'Confidence':<12} {'Avg Bets':<10} {'Avg Acc':<10} {'Avg ROI':<10}")
            report_lines.append("  " + "-" * 50)
            for conf in sorted(conf_groups.keys()):
                group = conf_groups[conf]
                avg_bets = sum(r.n_bets for r in group) / len(group)
                avg_acc = sum(r.accuracy for r in group) / len(group)
                avg_roi = sum(r.roi for r in group) / len(group)
                report_lines.append(f"  {conf:<12.2f} {avg_bets:<10.0f} {avg_acc:<10.1%} {avg_roi:<10.2f}%")

            report_lines.append("")

            # Group by juice
            report_lines.append("  By Juice (averaged across confidence values):")
            juice_groups = {}
            for r in valid_fg:
                if r.juice not in juice_groups:
                    juice_groups[r.juice] = []
                juice_groups[r.juice].append(r)

            report_lines.append(f"  {'Juice':<12} {'Avg Bets':<10} {'Avg Acc':<10} {'Avg ROI':<10}")
            report_lines.append("  " + "-" * 50)
            for juice in sorted(juice_groups.keys()):
                group = juice_groups[juice]
                avg_bets = sum(r.n_bets for r in group) / len(group)
                avg_acc = sum(r.accuracy for r in group) / len(group)
                avg_roi = sum(r.roi for r in group) / len(group)
                report_lines.append(f"  {juice:<12} {avg_bets:<10.0f} {avg_acc:<10.1%} {avg_roi:<10.2f}%")

        else:
            report_lines.append("No valid results (all configurations produced < 10 bets)")

    else:
        report_lines.append("No FG spread results found")

    # ========================================================================
    # 1H SPREAD ANALYSIS
    # ========================================================================

    report_lines.append("")
    report_lines.append("")
    report_lines.append("=" * 100)
    report_lines.append("FIRST HALF (1H) SPREAD OPTIMIZATION")
    report_lines.append("=" * 100)
    report_lines.append("")

    if h1_results:
        # Filter valid results (min 10 bets)
        valid_1h = [r for r in h1_results if r.n_bets >= 10]

        if valid_1h:
            # Sort by different criteria
            by_roi = sorted(valid_1h, key=lambda r: r.roi, reverse=True)
            by_accuracy = sorted(valid_1h, key=lambda r: r.accuracy, reverse=True)
            by_profit = sorted(valid_1h, key=lambda r: r.profit, reverse=True)
            by_score = sorted(valid_1h, key=lambda r: r.score, reverse=True)

            # Top 10 by ROI
            report_lines.append("TOP 10 CONFIGURATIONS BY ROI:")
            report_lines.append(f"{'Rank':<5} {'Conf':<8} {'Juice':<7} {'Bets':<7} {'Accuracy':<10} {'ROI':<12} {'Profit':<12} {'HC Bets':<8} {'HC ROI':<10}")
            report_lines.append("-" * 100)

            for i, r in enumerate(by_roi[:10], 1):
                report_lines.append(
                    f"{i:<5} {r.confidence:<8.2f} {r.juice:<7} {r.n_bets:<7} {r.accuracy:<10.1%} {r.roi:<12.2f}% {r.profit:<12.2f}u {r.hc_n_bets:<8} {r.hc_roi:<10.2f}%"
                )

            report_lines.append("")

            # Top 5 by accuracy
            report_lines.append("TOP 5 CONFIGURATIONS BY ACCURACY:")
            report_lines.append(f"{'Rank':<5} {'Conf':<8} {'Juice':<7} {'Bets':<7} {'Accuracy':<10} {'ROI':<12} {'Profit':<12}")
            report_lines.append("-" * 100)

            for i, r in enumerate(by_accuracy[:5], 1):
                report_lines.append(
                    f"{i:<5} {r.confidence:<8.2f} {r.juice:<7} {r.n_bets:<7} {r.accuracy:<10.1%} {r.roi:<12.2f}% {r.profit:<12.2f}u"
                )

            report_lines.append("")

            # Top 5 by profit
            report_lines.append("TOP 5 CONFIGURATIONS BY TOTAL PROFIT:")
            report_lines.append(f"{'Rank':<5} {'Conf':<8} {'Juice':<7} {'Bets':<7} {'Accuracy':<10} {'ROI':<12} {'Profit':<12}")
            report_lines.append("-" * 100)

            for i, r in enumerate(by_profit[:5], 1):
                report_lines.append(
                    f"{i:<5} {r.confidence:<8.2f} {r.juice:<7} {r.n_bets:<7} {r.accuracy:<10.1%} {r.roi:<12.2f}% {r.profit:<12.2f}u"
                )

            report_lines.append("")

            # Best balanced (composite score)
            report_lines.append("RECOMMENDED CONFIGURATION (Balanced Score):")
            best_1h = by_score[0]
            report_lines.append(f"  Confidence Threshold: {best_1h.confidence:.2f}")
            report_lines.append(f"  Juice: {best_1h.juice}")
            report_lines.append(f"  Total Bets: {best_1h.n_bets}")
            report_lines.append(f"  Accuracy: {best_1h.accuracy:.1%}")
            report_lines.append(f"  ROI: {best_1h.roi:+.2f}%")
            report_lines.append(f"  Profit: {best_1h.profit:+.2f} units")
            if best_1h.hc_n_bets > 0:
                report_lines.append(f"  High Confidence (60%+): {best_1h.hc_n_bets} bets, {best_1h.hc_roi:+.2f}% ROI")
            report_lines.append("")

            # Parameter sensitivity analysis
            report_lines.append("PARAMETER SENSITIVITY ANALYSIS:")
            report_lines.append("")

            # Group by confidence
            report_lines.append("  By Confidence Threshold (averaged across juice values):")
            conf_groups = {}
            for r in valid_1h:
                if r.confidence not in conf_groups:
                    conf_groups[r.confidence] = []
                conf_groups[r.confidence].append(r)

            report_lines.append(f"  {'Confidence':<12} {'Avg Bets':<10} {'Avg Acc':<10} {'Avg ROI':<10}")
            report_lines.append("  " + "-" * 50)
            for conf in sorted(conf_groups.keys()):
                group = conf_groups[conf]
                avg_bets = sum(r.n_bets for r in group) / len(group)
                avg_acc = sum(r.accuracy for r in group) / len(group)
                avg_roi = sum(r.roi for r in group) / len(group)
                report_lines.append(f"  {conf:<12.2f} {avg_bets:<10.0f} {avg_acc:<10.1%} {avg_roi:<10.2f}%")

            report_lines.append("")

            # Group by juice
            report_lines.append("  By Juice (averaged across confidence values):")
            juice_groups = {}
            for r in valid_1h:
                if r.juice not in juice_groups:
                    juice_groups[r.juice] = []
                juice_groups[r.juice].append(r)

            report_lines.append(f"  {'Juice':<12} {'Avg Bets':<10} {'Avg Acc':<10} {'Avg ROI':<10}")
            report_lines.append("  " + "-" * 50)
            for juice in sorted(juice_groups.keys()):
                group = juice_groups[juice]
                avg_bets = sum(r.n_bets for r in group) / len(group)
                avg_acc = sum(r.accuracy for r in group) / len(group)
                avg_roi = sum(r.roi for r in group) / len(group)
                report_lines.append(f"  {juice:<12} {avg_bets:<10.0f} {avg_acc:<10.1%} {avg_roi:<10.2f}%")

        else:
            report_lines.append("No valid results (all configurations produced < 10 bets)")

    else:
        report_lines.append("No 1H spread results found")

    # ========================================================================
    # FINAL RECOMMENDATIONS
    # ========================================================================

    report_lines.append("")
    report_lines.append("")
    report_lines.append("=" * 100)
    report_lines.append("FINAL RECOMMENDATIONS")
    report_lines.append("=" * 100)
    report_lines.append("")

    if valid_fg:
        best_fg = sorted([r for r in fg_results if r.n_bets >= 10], key=lambda r: r.score, reverse=True)[0]
        report_lines.append("FG SPREAD:")
        report_lines.append(f"  - Use confidence threshold: {best_fg.confidence:.2f}")
        report_lines.append(f"  - Expected performance: {best_fg.accuracy:.1%} accuracy, {best_fg.roi:+.2f}% ROI")
        report_lines.append(f"  - Volume: ~{best_fg.n_bets} bets per season")
        report_lines.append("")

    if valid_1h:
        best_1h = sorted([r for r in h1_results if r.n_bets >= 10], key=lambda r: r.score, reverse=True)[0]
        report_lines.append("1H SPREAD:")
        report_lines.append(f"  - Use confidence threshold: {best_1h.confidence:.2f}")
        report_lines.append(f"  - Expected performance: {best_1h.accuracy:.1%} accuracy, {best_1h.roi:+.2f}% ROI")
        report_lines.append(f"  - Volume: ~{best_1h.n_bets} bets per season")
        report_lines.append("")

    report_lines.append("=" * 100)

    # Print to console
    report_text = "\n".join(report_lines)
    print(report_text)

    # Save to file if specified
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(report_text)
        print(f"\nReport saved to: {output_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze spread optimization results")
    parser.add_argument(
        "--results-dir",
        default="data/backtest_results/spread_optimization",
        help="Directory containing optimization results",
    )
    parser.add_argument(
        "--output",
        default="data/backtest_results/spread_optimization_report.txt",
        help="Output report file path",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_file = Path(args.output)

    # Load all results
    results = load_all_results(results_dir)

    if not results:
        print(f"No results found in {results_dir}")
        print("Please run the optimization script first:")
        print("  run_spread_optimization.bat")
        return

    # Generate and print report
    print_optimization_report(results, output_file)


if __name__ == "__main__":
    main()
