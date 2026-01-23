#!/usr/bin/env python3
"""
Run Spread Parameter Optimization

This script programmatically runs historical_backtest_production.py with different
parameter combinations to find optimal settings for spread markets.
"""
import os
import sys
import subprocess
from pathlib import Path
from typing import List, Tuple
from itertools import product

# Project root
PROJECT_ROOT = Path(__file__).parent.parent


def run_backtest(
    market: str,
    confidence: float,
    juice: int,
    output_dir: Path,
    data_path: str = "data/processed/training_data.csv",
    models_dir: str = "models/production",
) -> bool:
    """
    Run a single backtest configuration.

    Returns:
        True if successful, False otherwise
    """
    # Build output filename
    conf_str = f"{int(confidence * 100)}"
    juice_str = f"{abs(juice)}"
    output_file = output_dir / f"{market}_conf{conf_str}_j{juice_str}.json"
    log_file = output_dir / f"{market}_conf{conf_str}_j{juice_str}.log"

    # Build command
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "historical_backtest_production.py"),
        f"--data={data_path}",
        f"--models-dir={models_dir}",
        f"--markets={market}",
        f"--spread-juice={juice}",
        f"--total-juice=-110",  # Not used for spread-only backtest, but required
        f"--output-json={output_file}",
    ]

    print(
        f"Running: {market}, conf={confidence:.2f}, juice={juice}... ", end="", flush=True)

    try:
        # Run backtest
        with open(log_file, 'w') as log:
            result = subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=300,  # 5 minute timeout
            )

        if result.returncode == 0:
            print("Success")
            return True
        else:
            print(f"Failed (exit code {result.returncode})")
            return False

    except subprocess.TimeoutExpired:
        print("Timeout")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run spread parameter optimization")
    parser.add_argument(
        "--markets",
        default="fg_spread,1h_spread",
        help="Comma-separated markets to optimize",
    )
    parser.add_argument(
        "--confidence",
        default="0.55,0.60,0.62,0.65,0.68,0.70",
        help="Comma-separated confidence thresholds to test",
    )
    parser.add_argument(
        "--juice",
        default="-105,-110",
        help="Comma-separated juice values to test (negative numbers)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/backtest_results/spread_optimization",
        help="Output directory for results",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: test fewer configurations",
    )

    args = parser.parse_args()

    # Parse arguments
    markets = [m.strip() for m in args.markets.split(",")]

    if args.quick:
        confidence_values = [0.55, 0.62, 0.68]
        juice_values = [-110]
    else:
        confidence_values = [float(c) for c in args.confidence.split(",")]
        juice_values = [int(j) for j in args.juice.split(",")]

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate total runs
    total_runs = len(markets) * len(confidence_values) * len(juice_values)

    print("=" * 80)
    print("SPREAD PARAMETER OPTIMIZATION")
    print("=" * 80)
    print(f"Markets: {', '.join(markets)}")
    print(f"Confidence values: {confidence_values}")
    print(f"Juice values: {juice_values}")
    print(f"Total runs: {total_runs}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    print()

    # Run all configurations
    successes = 0
    failures = 0
    current = 0

    for market in markets:
        print(f"\n{market.upper()} OPTIMIZATION")
        print("-" * 80)

        for conf, juice in product(confidence_values, juice_values):
            current += 1
            print(f"[{current}/{total_runs}] ", end="")

            success = run_backtest(
                market=market,
                confidence=conf,
                juice=juice,
                output_dir=output_dir,
            )

            if success:
                successes += 1
            else:
                failures += 1

    print()
    print("=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"Successful runs: {successes}/{total_runs}")
    print(f"Failed runs: {failures}/{total_runs}")
    print(f"Results saved to: {output_dir}")
    print()
    print("Run the analysis script to view results:")
    print("  python scripts/analyze_spread_optimization.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
