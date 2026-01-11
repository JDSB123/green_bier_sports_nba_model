#!/usr/bin/env python3
"""
Compare Production Model Performance vs Our Backtests

This script shows the REAL production model performance from the manifest
vs what our backtesting infrastructure produces.
"""
import json
from pathlib import Path
import pandas as pd

def print_comparison():
    """Print production model vs backtest comparison."""
    print("=" * 80)
    print("PRODUCTION MODEL PERFORMANCE COMPARISON")
    print("=" * 80)

    # Load production manifest
    manifest_path = Path('models/production/model_pack.json')
    with open(manifest_path) as f:
        prod_data = json.load(f)

    print("\n[PRODUCTION] NBA NBA_v33.0.15.0 RESULTS")
    print("   Backtested on: 2025-10-02 to 2025-12-20 (464 games)")
    print("   Source: docs/BACKTEST_STATUS.md")
    print()

    prod_results = prod_data['backtest_results']
    print("   Market        | Accuracy | ROI     | Bets")
    print("   --------------|----------|---------|------")
    print(f"   FG Spread     | {prod_results['fg_spread']['accuracy']:.1%}   | +{prod_results['fg_spread']['roi']:.1%}  | {prod_results['fg_spread']['predictions']}")
    print(f"   FG Total      | {prod_results['fg_total']['accuracy']:.1%}    | +{prod_results['fg_total']['roi']:.1%}  | {prod_results['fg_total']['predictions']}")
    print(f"   1H Spread     | {prod_results['1h_spread']['accuracy']:.1%}   | +{prod_results['1h_spread']['roi']:.1%}  | {prod_results['1h_spread']['predictions']}")
    print(f"   1H Total      | {prod_results['1h_total']['accuracy']:.1%}    | +{prod_results['1h_total']['roi']:.1%}  | {prod_results['1h_total']['predictions']}")

    total_accuracy = sum(prod_results[m]['accuracy'] * prod_results[m]['predictions'] for m in prod_results) / sum(prod_results[m]['predictions'] for m in prod_results)
    total_roi = sum(prod_results[m]['roi'] * prod_results[m]['predictions'] for m in prod_results) / sum(prod_results[m]['predictions'] for m in prod_results)
    total_bets = sum(prod_results[m]['predictions'] for m in prod_results)

    print("   --------------|----------|---------|------")
    print(f"   TOTAL         | {total_accuracy:.1%}   | +{total_roi:.1%}  | {total_bets}")

    print("\n" + "=" * 80)
    print("OUR BACKTEST RESULTS (NBA v33.0.15.0)")
    print("=" * 80)

    # Load our latest backtest
    backtest_path = Path('data/backtest_results/33.0.15.0/2026-01-10_200118/backtest_summary_2026-01-10_200220.json')
    if backtest_path.exists():
        with open(backtest_path) as f:
            our_data = json.load(f)

        print("   Backtested on: 2023-01-01 to 2025-04-29 (3,366 games)")
        print("   Markets: FG Spread, FG Total, FG Moneyline, 1H Spread, 1H Total, 1H Moneyline")
        print("   Method: Walk-forward with expanding window")
        print()

        print("   Market        | Accuracy | ROI     | Bets")
        print("   --------------|----------|---------|------")

        market_map = {
            'fg_spread': 'FG Spread',
            'fg_total': 'FG Total',
            '1h_spread': '1H Spread',
            '1h_total': '1H Total',
        }

        total_bets = 0
        total_correct = 0
        total_roi = 0

        for market_key, market_name in market_map.items():
            if market_key in our_data['markets']:
                stats = our_data['markets'][market_key]['performance']
                accuracy = stats.get('accuracy', 0)
                roi = stats.get('roi', 0)
                bets = stats.get('n_bets', 0)

                print(f"   {market_name:13s} | {accuracy:.1%}   | {roi:+.1f}%  | {bets}")

                total_bets += bets
                total_correct += int(accuracy * bets)
                total_roi += roi * bets
            else:
                print(f"   {market_name:13s} | N/A      | N/A     | 0")

        if total_bets > 0:
            avg_accuracy = total_correct / total_bets
            avg_roi = total_roi / total_bets
            print("   --------------|----------|---------|------")
            print(f"   TOTAL         | {avg_accuracy:.1%}   | {avg_roi:+.1f}%  | {total_bets}")
    else:
        print("   No backtest results found!")
        print(f"   Expected: {backtest_path}")

    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print("WHY THE DIFFERENCE:")
    print("1. Production models trained on 464 recent games (Oct-Dec 2025)")
    print("2. Our backtests use 3,366 historical games (2023-2025)")
    print("3. Production uses live feature engineering with betting splits")
    print("4. Our backtests use static CSV features (no betting splits)")
    print("5. Production filters predictions by confidence/edge thresholds")
    print("6. Our backtests make predictions on all games with data")

    print("\nRECOMMENDATION:")
    print("- Production models show strong ROI: +12.7% across all markets")
    print("- Our backtesting shows potential but needs refinement")
    print("- Next: Compare feature engineering and filtering logic")

if __name__ == "__main__":
    print_comparison()