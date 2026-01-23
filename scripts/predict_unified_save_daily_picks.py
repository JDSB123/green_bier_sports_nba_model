"""
Save Daily Picks for Monitoring

Downloads picks from production API and saves locally for performance tracking.

Usage:
    python scripts/predict_unified_save_daily_picks.py                    # Save today's picks
    python scripts/predict_unified_save_daily_picks.py --date 2026-01-17  # Save specific date
"""

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PICKS_DIR = PROJECT_ROOT / "data" / "picks"
API_URL = "https://nba-gbsv-api.livelycoast-b48c3cb0.eastus.azurecontainerapps.io"


def save_picks_for_date(date_str: str) -> bool:
    """Save picks for a specific date."""
    # Create picks directory if it doesn't exist
    PICKS_DIR.mkdir(parents=True, exist_ok=True)

    output_file = PICKS_DIR / f"picks_{date_str}.json"

    # Check if file already exists
    if output_file.exists():
        print(f"Picks for {date_str} already saved at {output_file}")
        overwrite = input("Overwrite? (y/n): ").strip().lower()
        if overwrite != 'y':
            print("Skipped")
            return False

    # Download picks from API
    url = f"{API_URL}/slate/{date_str}"
    print(f"Downloading picks from {url}...")

    try:
        result = subprocess.run(
            ["curl", "-s", url],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            print(f"Failed to download picks: {result.stderr}")
            return False

        # Parse JSON to validate
        picks_data = json.loads(result.stdout)

        # Save to file
        with open(output_file, 'w') as f:
            json.dump(picks_data, f, indent=2)

        print(
            f"Saved {len(picks_data.get('picks', []))} picks to {output_file}")

        # Show summary
        if 'picks' in picks_data:
            picks = picks_data['picks']
            fg_spread = sum(1 for p in picks if p.get('market')
                            == 'spread' and p.get('period') == 'FG')
            fg_total = sum(1 for p in picks if p.get('market')
                           == 'total' and p.get('period') == 'FG')
            fh_spread = sum(1 for p in picks if p.get('market')
                            == 'spread' and p.get('period') == '1H')
            fh_total = sum(1 for p in picks if p.get('market')
                           == 'total' and p.get('period') == '1H')

            print(f"\nPick Summary for {date_str}:")
            print(f"  Total:     {len(picks)}")
            print(f"  FG Spread: {fg_spread} (OPTIMIZED)")
            print(f"  FG Total:  {fg_total}")
            print(f"  1H Spread: {fh_spread}")
            print(f"  1H Total:  {fh_total}")

        return True

    except subprocess.TimeoutExpired:
        print("Request timed out")
        return False
    except json.JSONDecodeError as e:
        print(f"Invalid JSON response: {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Save daily picks for monitoring")
    parser.add_argument('--date', type=str,
                        help='Date to save (YYYY-MM-DD), defaults to today')

    args = parser.parse_args()

    if args.date:
        date_str = args.date
    else:
        date_str = datetime.now().strftime('%Y-%m-%d')

    print("=" * 80)
    print(f"SAVING PICKS FOR {date_str}")
    print("=" * 80)

    success = save_picks_for_date(date_str)

    if success:
        print("\n" + "=" * 80)
        print("NEXT STEPS")
        print("=" * 80)
        print("\nTo analyze today's picks:")
        print(
            f"  python scripts/monitor_week1_performance.py --date {date_str}")
        print("\nTo see weekly summary:")
        print("  python scripts/monitor_week1_performance.py --week-summary")
    else:
        print("\nFailed to save picks")


if __name__ == "__main__":
    main()
