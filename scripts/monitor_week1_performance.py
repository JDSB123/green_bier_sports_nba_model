"""
Week 1 Performance Monitoring Script

Tracks daily performance of NBA_v33.0.21.0 optimization deployment.
Monitors FG Spread performance specifically to validate optimization results.

Usage:
    python scripts/monitor_week1_performance.py --date 2026-01-16
    python scripts/monitor_week1_performance.py --week-summary
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent


def load_daily_picks(date_str: str) -> Optional[Dict]:
    """Load picks for a specific date from API output."""
    picks_file = PROJECT_ROOT / f"data/picks/picks_{date_str}.json"

    if not picks_file.exists():
        print(f"No picks file found for {date_str}")
        return None

    with open(picks_file, 'r') as f:
        return json.load(f)


def analyze_daily_picks(picks_data: Dict, date_str: str) -> Dict:
    """Analyze picks for a single day."""
    if not picks_data or 'picks' not in picks_data:
        return {
            'date': date_str,
            'total_picks': 0,
            'fg_spread_picks': 0,
            'fg_total_picks': 0,
            'fh_spread_picks': 0,
            'fh_total_picks': 0,
            'elite_picks': 0,
            'strong_picks': 0,
        }

    picks = picks_data['picks']

    # Count by market
    fg_spread = [p for p in picks if p.get('market') == 'spread' and p.get('period') == 'FG']
    fg_total = [p for p in picks if p.get('market') == 'total' and p.get('period') == 'FG']
    fh_spread = [p for p in picks if p.get('market') == 'spread' and p.get('period') == '1H']
    fh_total = [p for p in picks if p.get('market') == 'total' and p.get('period') == '1H']

    # Count by fire rating
    elite = [p for p in picks if p.get('fire_rating', 0) >= 4]
    strong = [p for p in picks if p.get('fire_rating', 0) == 3]

    # FG Spread specific metrics
    fg_spread_avg_conf = sum(p.get('confidence', 0) for p in fg_spread) / len(fg_spread) if fg_spread else 0
    fg_spread_avg_edge = sum(p.get('edge', 0) for p in fg_spread) / len(fg_spread) if fg_spread else 0

    return {
        'date': date_str,
        'total_picks': len(picks),
        'fg_spread_picks': len(fg_spread),
        'fg_total_picks': len(fg_total),
        'fh_spread_picks': len(fh_spread),
        'fh_total_picks': len(fh_total),
        'elite_picks': len(elite),
        'strong_picks': len(strong),
        'fg_spread_avg_confidence': round(fg_spread_avg_conf, 3),
        'fg_spread_avg_edge': round(fg_spread_avg_edge, 2),
    }


def print_daily_report(stats: Dict):
    """Print formatted daily report."""
    print("\n" + "=" * 80)
    print(f"DAILY REPORT: {stats['date']}")
    print("=" * 80)
    print(f"\nTotal Picks: {stats['total_picks']}")
    print(f"\nMarket Breakdown:")
    print(f"  FG Spread: {stats['fg_spread_picks']} picks")
    print(f"  FG Total:  {stats['fg_total_picks']} picks")
    print(f"  1H Spread: {stats['fh_spread_picks']} picks")
    print(f"  1H Total:  {stats['fh_total_picks']} picks")
    print(f"\nQuality Distribution:")
    print(f"  ELITE (4+ fire):  {stats['elite_picks']} picks")
    print(f"  STRONG (3 fire):  {stats['strong_picks']} picks")

    if stats['fg_spread_picks'] > 0:
        print(f"\nFG Spread Metrics (OPTIMIZED MARKET):")
        print(f"  Avg Confidence: {stats['fg_spread_avg_confidence']:.1%}")
        print(f"  Avg Edge:       {stats['fg_spread_avg_edge']:.2f} pts")
        print(f"\n  Target: 50-70 picks/week (7-10/day)")
        print(f"  Current: {stats['fg_spread_picks']} picks today")

        if stats['fg_spread_picks'] >= 7:
            print("  Status: ON TRACK ✅")
        elif stats['fg_spread_picks'] >= 5:
            print("  Status: ACCEPTABLE ⚠️")
        else:
            print("  Status: BELOW TARGET ⚠️")


def print_week_summary(daily_stats: List[Dict]):
    """Print weekly summary report."""
    if not daily_stats:
        print("No data available for week summary")
        return

    total_picks = sum(s['total_picks'] for s in daily_stats)
    fg_spread_picks = sum(s['fg_spread_picks'] for s in daily_stats)
    fg_total_picks = sum(s['fg_total_picks'] for s in daily_stats)
    fh_spread_picks = sum(s['fh_spread_picks'] for s in daily_stats)
    fh_total_picks = sum(s['fh_total_picks'] for s in daily_stats)
    elite_picks = sum(s['elite_picks'] for s in daily_stats)
    strong_picks = sum(s['strong_picks'] for s in daily_stats)

    days_tracked = len(daily_stats)

    print("\n" + "=" * 80)
    print(f"WEEK 1 SUMMARY: {days_tracked} days tracked")
    print("=" * 80)
    print(f"\nTotal Picks: {total_picks} ({total_picks/days_tracked:.1f}/day)")
    print(f"\nMarket Breakdown:")
    print(f"  FG Spread: {fg_spread_picks} picks ({fg_spread_picks/days_tracked:.1f}/day)")
    print(f"  FG Total:  {fg_total_picks} picks ({fg_total_picks/days_tracked:.1f}/day)")
    print(f"  1H Spread: {fh_spread_picks} picks ({fh_spread_picks/days_tracked:.1f}/day)")
    print(f"  1H Total:  {fh_total_picks} picks ({fh_total_picks/days_tracked:.1f}/day)")
    print(f"\nQuality Distribution:")
    print(f"  ELITE (4+ fire):  {elite_picks} picks ({elite_picks/total_picks*100:.1f}%)")
    print(f"  STRONG (3 fire):  {strong_picks} picks ({strong_picks/total_picks*100:.1f}%)")

    print(f"\n" + "=" * 80)
    print("FG SPREAD VALIDATION (OPTIMIZED MARKET)")
    print("=" * 80)
    print(f"\nVolume Target: 50-70 picks/week")
    print(f"Actual Volume: {fg_spread_picks} picks")

    if days_tracked >= 7:
        print(f"Projected Week: {fg_spread_picks} picks")
    else:
        projected = fg_spread_picks / days_tracked * 7
        print(f"Projected Week: {projected:.0f} picks (based on {days_tracked} days)")

    if fg_spread_picks >= 50:
        print("Volume Status: ON TARGET ✅")
        validation_status = "PASS"
    elif fg_spread_picks >= 35:
        print("Volume Status: ACCEPTABLE ⚠️")
        validation_status = "MONITOR"
    else:
        print("Volume Status: BELOW TARGET ❌")
        validation_status = "INVESTIGATE"

    print(f"\n" + "=" * 80)
    print("DECISION RECOMMENDATION")
    print("=" * 80)

    if validation_status == "PASS":
        print("\n✅ Week 1 validation PASSING")
        print("\nRECOMMENDED NEXT STEP:")
        print("  Deploy Option B: FG Totals Optimization")
        print("  Command: python scripts/deploy_option_b.py")
        print("\n  Expected Impact:")
        print("    - FG Total: 0.72 → 0.55 conf, 3.0 → 0.0 edge")
        print("    - Expected: +12.12% ROI, 58.73% accuracy")
        print("    - Volume: ~2,721 bets/season")
    elif validation_status == "MONITOR":
        print("\n⚠️ Week 1 validation ACCEPTABLE but not optimal")
        print("\nRECOMMENDED NEXT STEP:")
        print("  Continue monitoring for 2-3 more days")
        print("  If volume stabilizes above 50/week, deploy Option B")
    else:
        print("\n❌ Week 1 validation BELOW TARGET")
        print("\nRECOMMENDED NEXT STEP:")
        print("  Investigate why FG Spread volume is low")
        print("  Options:")
        print("    1. Check if games are being filtered correctly")
        print("    2. Verify thresholds are active in production")
        print("    3. Consider adding slight edge filter (0.5-1.0)")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Monitor Week 1 performance")
    parser.add_argument('--date', type=str, help='Date to analyze (YYYY-MM-DD)')
    parser.add_argument('--week-summary', action='store_true', help='Show weekly summary')
    parser.add_argument('--start-date', type=str, default='2026-01-16', help='Week start date')

    args = parser.parse_args()

    if args.week_summary:
        # Analyze full week
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        daily_stats = []

        for i in range(7):
            date = start_date + timedelta(days=i)
            date_str = date.strftime('%Y-%m-%d')
            picks_data = load_daily_picks(date_str)

            if picks_data:
                stats = analyze_daily_picks(picks_data, date_str)
                daily_stats.append(stats)

        print_week_summary(daily_stats)

    elif args.date:
        # Analyze single day
        picks_data = load_daily_picks(args.date)

        if picks_data:
            stats = analyze_daily_picks(picks_data, args.date)
            print_daily_report(stats)
        else:
            print(f"No picks data available for {args.date}")
            print(f"\nTo save today's picks, run:")
            print(f"  curl https://nba-gbsv-api.livelycoast-b48c3cb0.eastus.azurecontainerapps.io/slate/today > data/picks/picks_{args.date}.json")

    else:
        # Default: analyze today
        today = datetime.now().strftime('%Y-%m-%d')
        picks_data = load_daily_picks(today)

        if picks_data:
            stats = analyze_daily_picks(picks_data, today)
            print_daily_report(stats)
        else:
            print(f"No picks data available for today ({today})")
            print(f"\nTo save today's picks, run:")
            print(f"  curl https://nba-gbsv-api.livelycoast-b48c3cb0.eastus.azurecontainerapps.io/slate/today > data/picks/picks_{today}.json")


if __name__ == "__main__":
    main()
