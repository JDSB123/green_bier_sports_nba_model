"""
Compare picks between baseline (v20) and current (v21) thresholds using production API.

Since the API is running v21, this fetches current picks and simulates what v20 would have produced
by filtering the same predictions with v20 thresholds.

Usage:
    python scripts/compare_thresholds_api.py --date 2026-01-16
    python scripts/compare_thresholds_api.py --date today
"""

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
API_URL = "https://nba-gbsv-api.livelycoast-b48c3cb0.eastus.azurecontainerapps.io"


def fetch_predictions(date: str) -> dict:
    """Fetch raw predictions from API."""
    url = f"{API_URL}/slate/{date}"
    print(f"Fetching predictions from {url}...")

    try:
        result = subprocess.run(
            ["curl", "-s", url],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            print(f"Failed to fetch: {result.stderr}")
            return {}

        return json.loads(result.stdout)

    except Exception as e:
        print(f"Error: {e}")
        return {}


def filter_with_thresholds(predictions: list, thresholds: dict) -> list:
    """Filter predictions based on thresholds."""
    filtered = []

    for pred in predictions:
        pred_dict = pred.get('predictions', {})

        # Process each period and market
        for period in ['full_game', 'first_half']:
            period_key = period
            period_label = 'FG' if period == 'full_game' else '1H'

            if period_key not in pred_dict:
                continue

            for market in ['spread', 'total']:
                market_data = pred_dict[period_key].get(market, {})

                if not market_data or not market_data.get('passes_filter'):
                    continue

                confidence = market_data.get('confidence', 0)
                edge = abs(market_data.get('edge', 0))

                # Get threshold for this market/period
                threshold_key = f"{period_label.lower()}_{market}"
                threshold = thresholds.get(threshold_key, {'conf': 0, 'edge': 0})

                # Check if passes threshold
                if confidence >= threshold['conf'] and edge >= threshold['edge']:
                    filtered.append({
                        'matchup': pred.get('matchup', 'Unknown'),
                        'period': period_label,
                        'market': market,
                        'confidence': confidence,
                        'edge': edge,
                        'pick': market_data.get('bet_side', 'N/A')
                    })

    return filtered


def compare_results(baseline_picks: list, current_picks: list, date: str):
    """Display comparison between baseline and current picks."""
    # Count by market
    def count_by_market(picks):
        counts = {'fg_spread': 0, 'fg_total': 0, 'fh_spread': 0, 'fh_total': 0}
        for pick in picks:
            period = pick['period'].lower()
            market = pick['market']
            key = f"{period}_{market}"
            counts[key] = counts.get(key, 0) + 1
        return counts

    baseline_counts = count_by_market(baseline_picks)
    current_counts = count_by_market(current_picks)

    print("\n" + "=" * 80)
    print(f"THRESHOLD COMPARISON - {date}")
    print("=" * 80)

    print("\nBASELINE (v33.0.20.0):")
    print("  FG Spread: 0.62 conf, 2.0 edge")
    print("  FG Total:  0.72 conf, 3.0 edge")
    print("  1H Spread: 0.68 conf, 1.5 edge")
    print("  1H Total:  0.66 conf, 2.0 edge")

    print("\nCURRENT (v33.0.21.0 - OPTIMIZED):")
    print("  FG Spread: 0.55 conf, 0.0 edge  <-- OPTIMIZED")
    print("  FG Total:  0.72 conf, 3.0 edge")
    print("  1H Spread: 0.68 conf, 1.5 edge")
    print("  1H Total:  0.66 conf, 2.0 edge")

    print("\n" + "=" * 80)
    print("PICK COUNT COMPARISON")
    print("=" * 80)

    print(f"\n{'Market':<15} | {'Baseline':<10} | {'Current':<10} | {'Delta':<10} | {'% Change':<10}")
    print("-" * 80)

    markets = [
        ('fg_spread', 'FG Spread'),
        ('fg_total', 'FG Total'),
        ('fh_spread', '1H Spread'),
        ('fh_total', '1H Total')
    ]

    total_baseline = sum(baseline_counts.values())
    total_current = sum(current_counts.values())

    for key, label in markets:
        b_count = baseline_counts.get(key, 0)
        c_count = current_counts.get(key, 0)
        delta = c_count - b_count
        pct = ((c_count / b_count - 1) * 100) if b_count > 0 else (100 if c_count > 0 else 0)

        delta_str = f"+{delta}" if delta >= 0 else str(delta)
        pct_str = f"+{pct:.1f}%" if pct >= 0 else f"{pct:.1f}%"

        marker = " <-- OPTIMIZED" if key == 'fg_spread' else ""
        print(f"{label:<15} | {b_count:<10} | {c_count:<10} | {delta_str:<10} | {pct_str:<10}{marker}")

    print("-" * 80)
    delta_total = total_current - total_baseline
    pct_total = ((total_current / total_baseline - 1) * 100) if total_baseline > 0 else 0
    delta_str = f"+{delta_total}" if delta_total >= 0 else str(delta_total)
    pct_str = f"+{pct_total:.1f}%" if pct_total >= 0 else f"{pct_total:.1f}%"
    print(f"{'TOTAL':<15} | {total_baseline:<10} | {total_current:<10} | {delta_str:<10} | {pct_str:<10}")

    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    fg_spread_delta = current_counts.get('fg_spread', 0) - baseline_counts.get('fg_spread', 0)
    fg_spread_baseline = baseline_counts.get('fg_spread', 0)

    if fg_spread_delta > 0 and fg_spread_baseline > 0:
        increase_pct = (fg_spread_delta / fg_spread_baseline) * 100
        print(f"\nOK FG Spread volume increased by {fg_spread_delta} picks (+{increase_pct:.0f}%)")
        print(f"  Baseline: {fg_spread_baseline} picks")
        print(f"  Current:  {current_counts.get('fg_spread', 0)} picks")
        print("  Expected: ~7-10 picks/day during regular season")

        if current_counts.get('fg_spread', 0) >= 7:
            print("  Status: ON TRACK")
        elif current_counts.get('fg_spread', 0) >= 5:
            print("  Status: ACCEPTABLE (monitor)")
        else:
            print("  Status: BELOW TARGET (investigate)")
    elif fg_spread_delta == 0:
        print(f"\n= FG Spread volume unchanged ({current_counts.get('fg_spread', 0)} picks)")
        print("  Status: Optimization not having expected effect")
    else:
        print(f"\nNOT OK FG Spread volume decreased by {abs(fg_spread_delta)} picks")
        print("  Status: UNEXPECTED - investigate threshold configuration")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Compare baseline vs current thresholds")
    parser.add_argument('--date', type=str, default='today', help='Date (YYYY-MM-DD or "today")')
    args = parser.parse_args()

    date_str = args.date if args.date != 'today' else datetime.now().strftime('%Y-%m-%d')

    print("=" * 80)
    print("FETCHING PREDICTIONS")
    print("=" * 80)

    # Fetch predictions from API
    data = fetch_predictions(date_str)

    if not data or 'predictions' not in data:
        print("No predictions found")
        return

    predictions = data['predictions']
    print(f"Fetched {len(predictions)} game predictions")

    # Define thresholds for each version
    baseline_thresholds = {
        'fg_spread': {'conf': 0.62, 'edge': 2.0},
        'fg_total': {'conf': 0.72, 'edge': 3.0},
        'fh_spread': {'conf': 0.68, 'edge': 1.5},
        'fh_total': {'conf': 0.66, 'edge': 2.0}
    }

    current_thresholds = {
        'fg_spread': {'conf': 0.55, 'edge': 0.0},  # OPTIMIZED
        'fg_total': {'conf': 0.72, 'edge': 3.0},
        'fh_spread': {'conf': 0.68, 'edge': 1.5},
        'fh_total': {'conf': 0.66, 'edge': 2.0}
    }

    # Filter with both threshold sets
    baseline_picks = filter_with_thresholds(predictions, baseline_thresholds)
    current_picks = filter_with_thresholds(predictions, current_thresholds)

    # Compare results
    compare_results(baseline_picks, current_picks, date_str)

    print("\nDone! Use this daily to track optimization impact.")


if __name__ == "__main__":
    main()
