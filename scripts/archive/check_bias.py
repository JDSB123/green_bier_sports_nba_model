#!/usr/bin/env python3
"""Quick script to check for home/away bias in model picks."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings


def main():
    parser = argparse.ArgumentParser(description="Check for home/away bias in model picks.")
    parser.add_argument(
        "--analysis-file",
        type=Path,
        default=Path(settings.data_processed_dir) / "slate_analysis_20251214.json",
        help="Path to analysis JSON file (default: data/processed/slate_analysis_YYYYMMDD.json)",
    )
    args = parser.parse_args()

    analysis_file = args.analysis_file
    if not analysis_file.exists():
        print(f"❌ Analysis file not found: {analysis_file}")
        print(f"   Please provide a valid path with --analysis-file")
        sys.exit(1)

    with open(analysis_file, 'r') as f:
        data = json.load(f)

    home_spread_picks = 0
    away_spread_picks = 0
    over_picks = 0
    under_picks = 0

    print("=" * 60)
    print("TODAY'S SPREAD PICKS BREAKDOWN")
    print("=" * 60)

    for game in data:
        ce = game.get('comprehensive_edge', {})
        fg = ce.get('full_game', {})
        
        home_team = game['home_team']
        away_team = game['away_team']
        
        # Spread
        spread = fg.get('spread', {})
        if spread.get('pick'):
            pick_team = spread['pick']
            market_line = spread.get('market_line', 0)
            model_margin = spread.get('model_margin', 0)
            edge = spread.get('edge', 0)
            
            if pick_team == home_team:
                home_spread_picks += 1
                side = "HOME"
            else:
                away_spread_picks += 1
                side = "AWAY"
            
            print(f"\n{side}: {pick_team}")
            print(f"  Game: {away_team} @ {home_team}")
            print(f"  Market Line: {market_line:+.1f}")
            print(f"  Model Margin: {model_margin:+.1f}")
            print(f"  Edge: {edge:+.1f} pts")
        
        # Total
        total = fg.get('total', {})
        if total.get('pick'):
            if total['pick'] == 'OVER':
                over_picks += 1
            else:
                under_picks += 1

    print("\n")
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total_spread = home_spread_picks + away_spread_picks
    print(f"SPREAD PICKS: {home_spread_picks} HOME vs {away_spread_picks} AWAY")
    if total_spread > 0:
        print(f"Home Percentage: {home_spread_picks/total_spread*100:.1f}%")
        print(f"Away Percentage: {away_spread_picks/total_spread*100:.1f}%")

    print(f"\nTOTAL PICKS: {over_picks} OVER vs {under_picks} UNDER")

    # Check for systematic bias
    print("\n")
    print("=" * 60)
    print("BIAS ANALYSIS")
    print("=" * 60)

    # Look at model margins vs market lines
    print("\nModel Margin vs Market Line Analysis:")
    for game in data:
        ce = game.get('comprehensive_edge', {})
        fg = ce.get('full_game', {})
        spread = fg.get('spread', {})
        
        if spread:
            home_team = game['home_team']
            market_line = spread.get('market_line', 0)  # Negative = home favored
            model_margin = spread.get('model_margin', 0)  # Positive = home favored
            
            # Model thinks home wins by model_margin
            # Market thinks home wins by -market_line (since market_line is home spread)
            market_expected_margin = -market_line  # If line is -9.5, market expects home by 9.5
            
            diff = model_margin - market_expected_margin
            
            print(f"{home_team}:")
            print(f"  Market expects home by: {market_expected_margin:+.1f}")
            print(f"  Model expects home by: {model_margin:+.1f}")
            print(f"  Model vs Market diff: {diff:+.1f} (positive = model more bullish on home)")


if __name__ == "__main__":
    main()
