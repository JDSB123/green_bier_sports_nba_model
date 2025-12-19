#!/usr/bin/env python3
"""
Update pick tracker with results from previous day.

This script should be run the following morning at 8am CST to update
the status and results of picks from the previous day.

Usage:
    python scripts/update_pick_tracker.py [--date YYYY-MM-DD]
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.betting_card_export import update_tracker_results

CST = ZoneInfo("America/Chicago")


def main():
    parser = argparse.ArgumentParser(
        description="Update pick tracker with results from previous day"
    )
    parser.add_argument(
        "--date",
        help="Date to update (YYYY-MM-DD). Defaults to yesterday.",
    )
    args = parser.parse_args()
    
    target_date = None
    if args.date:
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    
    print("=" * 80)
    print("📊 PICK TRACKER UPDATE")
    print("=" * 80)
    print(f"Current time: {datetime.now(CST).strftime('%A, %B %d, %Y at %I:%M %p CST')}")
    if target_date:
        print(f"Target date: {target_date.strftime('%A, %B %d, %Y')}")
    else:
        now_cst = datetime.now(CST)
        target_date = (now_cst - timedelta(days=1)).date()
        print(f"Target date: {target_date.strftime('%A, %B %d, %Y')} (yesterday)")
    print("")
    
    update_tracker_results(target_date)
    
    print("\n" + "=" * 80)
    print("✅ Tracker update complete!")
    print("=" * 80)
    print("\n⚠️  NOTE: Please manually update the following columns in the tracker:")
    print("   - Result: Final score or outcome")
    print("   - Win/Loss: W or L")
    print("   - Notes: Any additional notes")
    print(f"\n   File: C:\\Users\\JB\\Green Bier Capital\\Early Stage Sport Ventures - Documents\\Daily Picks\\tracker_bombay711_daily_picks.xlsx")


if __name__ == "__main__":
    main()

