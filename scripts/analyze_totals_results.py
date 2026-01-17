#!/usr/bin/env python3
"""
Analyze totals optimization results and provide recommendations.

This script reads the optimization results JSON files and provides:
- Detailed analysis of optimal parameters
- Comparison across different metrics
- Recommendations for production deployment
- Visualization-ready data summaries

Usage:
    python scripts/analyze_totals_results.py
    python scripts/analyze_totals_results.py --fg-results data/backtest_results/fg_total_optimization.json --1h-results data/backtest_results/1h_total_optimization.json
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def load_optimization_results(file_path: str) -> Dict:
    """Load optimization results from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def analyze_market(market_name: str, results: List[Dict]) -> Dict:
    """Analyze optimization results for a single market."""

    if not results:
        return {"error": "No results available"}

    # Top result (best ROI)
    best = results[0]

    # Find best by accuracy
    best_accuracy = max(results, key=lambda x: x['accuracy'])

    # Find best volume (most bets with positive ROI)
    positive_roi = [r for r in results if r.get('roi', 0) > 0]
    best_volume = max(positive_roi, key=lambda x: x['n_bets']) if positive_roi else None

    # Find sweet spot (good ROI with reasonable volume)
    sweet_spot_candidates = [
        r for r in results
        if r.get('roi', 0) > 2.0 and r['n_bets'] >= 50
    ]
    sweet_spot = sweet_spot_candidates[0] if sweet_spot_candidates else None

    analysis = {
        "market": market_name,
        "best_roi": {
            "confidence": best['confidence_threshold'],
            "edge": best['edge_threshold'],
            "n_bets": best['n_bets'],
            "accuracy": best['accuracy'],
            "roi": best['roi'],
            "profit": best.get('total_profit', best.get('profit', 0)),
        },
        "best_accuracy": {
            "confidence": best_accuracy['confidence_threshold'],
            "edge": best_accuracy['edge_threshold'],
            "n_bets": best_accuracy['n_bets'],
            "accuracy": best_accuracy['accuracy'],
            "roi": best_accuracy.get('roi', 0),
            "profit": best_accuracy.get('total_profit', best_accuracy.get('profit', 0)),
        },
    }

    if best_volume:
        analysis["best_volume"] = {
            "confidence": best_volume['confidence_threshold'],
            "edge": best_volume['edge_threshold'],
            "n_bets": best_volume['n_bets'],
            "accuracy": best_volume['accuracy'],
            "roi": best_volume.get('roi', 0),
            "profit": best_volume.get('total_profit', best_volume.get('profit', 0)),
        }

    if sweet_spot:
        analysis["sweet_spot"] = {
            "confidence": sweet_spot['confidence_threshold'],
            "edge": sweet_spot['edge_threshold'],
            "n_bets": sweet_spot['n_bets'],
            "accuracy": sweet_spot['accuracy'],
            "roi": sweet_spot.get('roi', 0),
            "profit": sweet_spot.get('total_profit', sweet_spot.get('profit', 0)),
        }

    # Top 10 combinations
    analysis["top_10"] = [
        {
            "confidence": r['confidence_threshold'],
            "edge": r['edge_threshold'],
            "n_bets": r['n_bets'],
            "accuracy": r['accuracy'],
            "roi": r.get('roi', 0),
            "profit": r.get('total_profit', r.get('profit', 0)),
        }
        for r in results[:10]
    ]

    return analysis


def print_analysis(analysis: Dict):
    """Print formatted analysis."""
    market = analysis['market']

    print("\n" + "=" * 80)
    print(f"{market.upper()} ANALYSIS")
    print("=" * 80)

    print("\n1. BEST ROI COMBINATION:")
    best = analysis['best_roi']
    print(f"   Confidence >= {best['confidence']:.2f}")
    print(f"   Edge >= {best['edge']:.1f} points")
    print(f"   Metrics: {best['n_bets']} bets, {best['accuracy']:.2%} accuracy, "
          f"{best['roi']:.2f}% ROI, {best['profit']:+.2f} units profit")

    print("\n2. BEST ACCURACY COMBINATION:")
    best_acc = analysis['best_accuracy']
    print(f"   Confidence >= {best_acc['confidence']:.2f}")
    print(f"   Edge >= {best_acc['edge']:.1f} points")
    print(f"   Metrics: {best_acc['n_bets']} bets, {best_acc['accuracy']:.2%} accuracy, "
          f"{best_acc['roi']:.2f}% ROI, {best_acc['profit']:+.2f} units profit")

    if 'best_volume' in analysis:
        print("\n3. BEST VOLUME (Most bets with positive ROI):")
        best_vol = analysis['best_volume']
        print(f"   Confidence >= {best_vol['confidence']:.2f}")
        print(f"   Edge >= {best_vol['edge']:.1f} points")
        print(f"   Metrics: {best_vol['n_bets']} bets, {best_vol['accuracy']:.2%} accuracy, "
              f"{best_vol['roi']:.2f}% ROI, {best_vol['profit']:+.2f} units profit")

    if 'sweet_spot' in analysis:
        print("\n4. SWEET SPOT (ROI > 2% with 50+ bets):")
        sweet = analysis['sweet_spot']
        print(f"   Confidence >= {sweet['confidence']:.2f}")
        print(f"   Edge >= {sweet['edge']:.1f} points")
        print(f"   Metrics: {sweet['n_bets']} bets, {sweet['accuracy']:.2%} accuracy, "
              f"{sweet['roi']:.2f}% ROI, {sweet['profit']:+.2f} units profit")

    print("\n5. TOP 10 COMBINATIONS:")
    print(f"   {'Rank':<6} {'Conf':>6} {'Edge':>6} {'Bets':>6} {'Acc':>7} {'ROI':>8} {'Profit':>9}")
    print(f"   {'-'*60}")
    for i, combo in enumerate(analysis['top_10'], 1):
        print(f"   {i:<6} {combo['confidence']:>6.2f} {combo['edge']:>6.1f} "
              f"{combo['n_bets']:>6d} {combo['accuracy']:>6.1%} "
              f"{combo['roi']:>+7.2f}% {combo['profit']:>+8.2f}u")


def compare_markets(fg_analysis: Dict, h1_analysis: Dict):
    """Compare FG and 1H totals optimization results."""
    print("\n" + "=" * 80)
    print("MARKET COMPARISON")
    print("=" * 80)

    print("\n" + "-" * 80)
    print("RECOMMENDED PARAMETERS (Best ROI)")
    print("-" * 80)

    fg_best = fg_analysis['best_roi']
    h1_best = h1_analysis['best_roi']

    print(f"\nFG TOTALS:")
    print(f"  Confidence >= {fg_best['confidence']:.2f}")
    print(f"  Edge >= {fg_best['edge']:.1f} points")
    print(f"  Expected: {fg_best['n_bets']} bets, {fg_best['accuracy']:.2%} accuracy, "
          f"{fg_best['roi']:.2f}% ROI")

    print(f"\n1H TOTALS:")
    print(f"  Confidence >= {h1_best['confidence']:.2f}")
    print(f"  Edge >= {h1_best['edge']:.1f} points")
    print(f"  Expected: {h1_best['n_bets']} bets, {h1_best['accuracy']:.2%} accuracy, "
          f"{h1_best['roi']:.2f}% ROI")

    print("\n" + "-" * 80)
    print("DEPLOYMENT RECOMMENDATIONS")
    print("-" * 80)

    # Check if sweet spot exists for both
    if 'sweet_spot' in fg_analysis and 'sweet_spot' in h1_analysis:
        print("\n✓ RECOMMENDED: Use 'Sweet Spot' parameters (balance of ROI and volume)")

        fg_sweet = fg_analysis['sweet_spot']
        h1_sweet = h1_analysis['sweet_spot']

        print(f"\nFG TOTALS (Sweet Spot):")
        print(f"  Confidence >= {fg_sweet['confidence']:.2f}")
        print(f"  Edge >= {fg_sweet['edge']:.1f} points")

        print(f"\n1H TOTALS (Sweet Spot):")
        print(f"  Confidence >= {h1_sweet['confidence']:.2f}")
        print(f"  Edge >= {h1_sweet['edge']:.1f} points")
    else:
        print("\n⚠ CAUTION: No 'Sweet Spot' found for one or both markets")
        print("  Consider using Best ROI parameters but monitor bet volume")

    print("\n" + "-" * 80)
    print("IMPLEMENTATION CHECKLIST")
    print("-" * 80)
    print("""
1. Update src/config.py with new thresholds:
   - FILTER_TOTAL_MIN_CONFIDENCE
   - FILTER_TOTAL_MIN_EDGE
   - FILTER_1H_TOTAL_MIN_CONFIDENCE
   - FILTER_1H_TOTAL_MIN_EDGE

2. Set environment variables (production):
   - FILTER_TOTAL_MIN_CONFIDENCE=X.XX
   - FILTER_TOTAL_MIN_EDGE=X.X
   - FILTER_1H_TOTAL_MIN_CONFIDENCE=X.XX
   - FILTER_1H_TOTAL_MIN_EDGE=X.X

3. Test on recent data (validation):
   - Run backtest on most recent 30-50 games
   - Verify metrics match expectations
   - Monitor for any anomalies

4. Deploy to production:
   - Update configuration
   - Restart prediction service
   - Monitor initial bets closely

5. Ongoing monitoring:
   - Track actual vs expected performance
   - Re-optimize monthly or quarterly
   - Adjust parameters as needed
    """)


def generate_summary_report(fg_results_path: str, h1_results_path: str, output_path: str):
    """Generate comprehensive summary report."""

    print("=" * 80)
    print("TOTALS MARKET OPTIMIZATION ANALYSIS")
    print("=" * 80)
    print(f"\nLoading results...")
    print(f"  FG Totals: {fg_results_path}")
    print(f"  1H Totals: {h1_results_path}")

    # Load results
    fg_data = load_optimization_results(fg_results_path)
    h1_data = load_optimization_results(h1_results_path)

    # Extract results
    if 'results' in fg_data:  # From optimize_confidence_thresholds.py format
        fg_results = fg_data['results'].get('fg_total', [])
        h1_results = h1_data['results'].get('1h_total', [])
    elif 'optimization_results' in fg_data:  # From optimize_totals_only.py format
        fg_results = fg_data['optimization_results'].get('fg_total', [])
        h1_results = h1_data['optimization_results'].get('1h_total', [])
    else:
        print("ERROR: Unknown results format")
        return

    # Analyze each market
    fg_analysis = analyze_market("FG Totals", fg_results)
    h1_analysis = analyze_market("1H Totals", h1_results)

    # Print analyses
    print_analysis(fg_analysis)
    print_analysis(h1_analysis)

    # Compare markets
    compare_markets(fg_analysis, h1_analysis)

    # Save summary
    summary = {
        "fg_totals": fg_analysis,
        "1h_totals": h1_analysis,
        "comparison": {
            "fg_best_params": fg_analysis['best_roi'],
            "1h_best_params": h1_analysis['best_roi'],
        }
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"Summary saved to: {output_file}")
    print(f"{'=' * 80}\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Analyze totals optimization results")
    parser.add_argument(
        "--fg-results",
        default="data/backtest_results/fg_total_optimization.json",
        help="Path to FG totals optimization results JSON",
    )
    parser.add_argument(
        "--1h-results",
        default="data/backtest_results/1h_total_optimization.json",
        help="Path to 1H totals optimization results JSON",
    )
    parser.add_argument(
        "--output",
        default="data/backtest_results/totals_optimization_summary.json",
        help="Path to save summary report",
    )

    args = parser.parse_args()

    # Check if files exist
    if not Path(args.fg_results).exists():
        print(f"ERROR: FG results file not found: {args.fg_results}")
        print("\nPlease run the optimization first:")
        print("  python scripts/optimize_totals_only.py")
        print("  OR")
        print("  run_totals_optimization.bat")
        return

    if not Path(args.__1h_results).exists():
        print(f"ERROR: 1H results file not found: {args.__1h_results}")
        print("\nPlease run the optimization first:")
        print("  python scripts/optimize_totals_only.py")
        print("  OR")
        print("  run_totals_optimization.bat")
        return

    generate_summary_report(args.fg_results, args.__1h_results, args.output)


if __name__ == "__main__":
    main()
