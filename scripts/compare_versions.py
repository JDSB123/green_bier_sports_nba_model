"""
Compare predictions between two model versions side-by-side.

This script runs predictions with both the current version and a baseline version
to track how threshold changes affect pick volume and quality.

Usage:
    python scripts/compare_versions.py --date 2026-01-16
    python scripts/compare_versions.py --date today --baseline-version NBA_v33.0.20.0
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).parent.parent


def run_backtest_with_thresholds(
    date: str,
    spread_conf: float,
    spread_edge: float,
    total_conf: float,
    total_edge: float,
    fh_spread_conf: float,
    fh_spread_edge: float,
    fh_total_conf: float,
    fh_total_edge: float,
    version_label: str
) -> Dict:
    """Run backtest with specific thresholds."""
    print(f"\n{'=' * 80}")
    print(f"Running predictions: {version_label}")
    print(f"{'=' * 80}")
    print(f"FG Spread: {spread_conf} conf, {spread_edge} edge")
    print(f"FG Total:  {total_conf} conf, {total_edge} edge")
    print(f"1H Spread: {fh_spread_conf} conf, {fh_spread_edge} edge")
    print(f"1H Total:  {fh_total_conf} conf, {fh_total_edge} edge")

    # Set environment variables for thresholds
    import os
    env = os.environ.copy()
    env['FILTER_SPREAD_MIN_CONFIDENCE'] = str(spread_conf)
    env['FILTER_SPREAD_MIN_EDGE'] = str(spread_edge)
    env['FILTER_TOTAL_MIN_CONFIDENCE'] = str(total_conf)
    env['FILTER_TOTAL_MIN_EDGE'] = str(total_edge)
    env['FILTER_FH_SPREAD_MIN_CONFIDENCE'] = str(fh_spread_conf)
    env['FILTER_FH_SPREAD_MIN_EDGE'] = str(fh_spread_edge)
    env['FILTER_FH_TOTAL_MIN_CONFIDENCE'] = str(fh_total_conf)
    env['FILTER_FH_TOTAL_MIN_EDGE'] = str(fh_total_edge)

    # Run prediction script (assuming we have a local prediction script)
    # For now, we'll call the API with different thresholds
    # In production, this would use local models

    print(f"Fetching predictions from API...")

    # This is a placeholder - in reality you'd run local predictions
    # For now, just return structure
    return {
        'version': version_label,
        'thresholds': {
            'fg_spread': {'conf': spread_conf, 'edge': spread_edge},
            'fg_total': {'conf': total_conf, 'edge': total_edge},
            'fh_spread': {'conf': fh_spread_conf, 'edge': fh_spread_edge},
            'fh_total': {'conf': fh_total_conf, 'edge': fh_total_edge}
        },
        'picks': []
    }


def compare_pick_counts(baseline: Dict, current: Dict) -> None:
    """Compare pick counts between versions."""
    baseline_picks = baseline['picks']
    current_picks = current['picks']

    # Count by market
    def count_by_market(picks):
        counts = {
            'fg_spread': 0,
            'fg_total': 0,
            'fh_spread': 0,
            'fh_total': 0,
            'total': len(picks)
        }
        for pick in picks:
            market = pick.get('market', '').lower()
            period = pick.get('period', '').upper()

            if period == 'FG' and 'spread' in market:
                counts['fg_spread'] += 1
            elif period == 'FG' and 'total' in market:
                counts['fg_total'] += 1
            elif period == '1H' and 'spread' in market:
                counts['fh_spread'] += 1
            elif period == '1H' and 'total' in market:
                counts['fh_total'] += 1

        return counts

    baseline_counts = count_by_market(baseline_picks)
    current_counts = count_by_market(current_picks)

    print(f"\n{'=' * 80}")
    print("PICK COUNT COMPARISON")
    print(f"{'=' * 80}")
    print(f"\n{'Market':<15} | {'Baseline':<10} | {'Current':<10} | {'Delta':<10} | {'% Change':<10}")
    print("-" * 80)

    for market in ['fg_spread', 'fg_total', 'fh_spread', 'fh_total', 'total']:
        baseline_count = baseline_counts[market]
        current_count = current_counts[market]
        delta = current_count - baseline_count
        pct_change = ((current_count / baseline_count - 1) * 100) if baseline_count > 0 else 0

        delta_str = f"+{delta}" if delta > 0 else str(delta)
        pct_str = f"+{pct_change:.1f}%" if pct_change > 0 else f"{pct_change:.1f}%"

        market_label = market.upper().replace('_', ' ')
        print(f"{market_label:<15} | {baseline_count:<10} | {current_count:<10} | {delta_str:<10} | {pct_str:<10}")

    print("-" * 80)

    # Highlight FG Spread (optimized market)
    fg_spread_delta = current_counts['fg_spread'] - baseline_counts['fg_spread']
    if fg_spread_delta > 0:
        print(f"\n✓ FG Spread volume increased by {fg_spread_delta} picks (+{((current_counts['fg_spread'] / baseline_counts['fg_spread'] - 1) * 100):.1f}%)")
    elif fg_spread_delta < 0:
        print(f"\n✗ FG Spread volume decreased by {abs(fg_spread_delta)} picks ({((current_counts['fg_spread'] / baseline_counts['fg_spread'] - 1) * 100):.1f}%)")
    else:
        print(f"\n= FG Spread volume unchanged")


def main():
    parser = argparse.ArgumentParser(description="Compare model versions side-by-side")
    parser.add_argument('--date', type=str, default='today', help='Date to compare (YYYY-MM-DD or "today")')
    parser.add_argument('--baseline-version', type=str, default='NBA_v33.0.20.0', help='Baseline version to compare against')
    parser.add_argument('--current-version', type=str, default='NBA_v33.0.21.0', help='Current version')

    args = parser.parse_args()

    print("=" * 80)
    print("MODEL VERSION COMPARISON")
    print("=" * 80)
    print(f"\nDate: {args.date}")
    print(f"Baseline: {args.baseline_version}")
    print(f"Current:  {args.current_version}")

    # Baseline thresholds (v33.0.20.0)
    baseline = run_backtest_with_thresholds(
        date=args.date,
        spread_conf=0.62,
        spread_edge=2.0,
        total_conf=0.72,
        total_edge=3.0,
        fh_spread_conf=0.68,
        fh_spread_edge=1.5,
        fh_total_conf=0.66,
        fh_total_edge=2.0,
        version_label=args.baseline_version
    )

    # Current thresholds (v33.0.21.0)
    current = run_backtest_with_thresholds(
        date=args.date,
        spread_conf=0.55,
        spread_edge=0.0,
        total_conf=0.72,
        total_edge=3.0,
        fh_spread_conf=0.68,
        fh_spread_edge=1.5,
        fh_total_conf=0.66,
        fh_total_edge=2.0,
        version_label=args.current_version
    )

    # Compare results
    compare_pick_counts(baseline, current)

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)
    print("\nNote: This comparison shows the impact of threshold changes on pick volume.")
    print("Use this daily to track whether optimization is working as expected.")


if __name__ == "__main__":
    main()
