"""
Collect public betting percentages and save to data directory.

This script captures pre-game betting splits for building historical training data.
Run daily before games to collect RLM/sharp money signals.

Usage:
    # Collect today's splits (default: Action Network)
    # Appends to history by default (use --no-append-history to skip)
    python scripts/collect_betting_splits.py

    # Collect and save to historical CSV + JSON snapshot
    python scripts/collect_betting_splits.py --save

    # Use specific source
    python scripts/collect_betting_splits.py --source action_network

Data Sources:
    - action_network: Best data (public API with premium indicators when available)
    - auto: Tries sources in order of quality
    - mock: Fake data for testing only

Output:
    - data/processed/betting_splits.json: Today's splits snapshot
    - data/splits/historical_splits.csv: Cumulative training data
"""
import asyncio
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.ingestion.betting_splits import (
    fetch_public_betting_splits,
    fetch_splits_sbro,
    scrape_splits_covers,
    _create_mock_splits_for_games,
    splits_to_features,
)
from src.ingestion import the_odds

DATA_DIR = PROJECT_ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"
CST = ZoneInfo("America/Chicago")


async def main():
    """Collect betting splits and save to file."""
    parser = argparse.ArgumentParser(
        description="Collect public betting splits for training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick check of today's splits
    python scripts/collect_betting_splits.py

    # Collect and save to training history (default behavior)
    python scripts/collect_betting_splits.py --save

    # Use Action Network specifically
    python scripts/collect_betting_splits.py --source action_network --save
        """,
    )
    parser.add_argument(
        "--source",
        choices=["auto", "action_network", "sbro", "covers", "mock"],
        default="auto",
        help="Data source for betting splits (default: auto)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save splits to data/processed/betting_splits.json",
    )
    parser.add_argument(
        "--append-history",
        dest="append_history",
        action="store_true",
        default=True,
        help="Append to historical CSV for training (data/splits/historical_splits.csv) [default]",
    )
    parser.add_argument(
        "--no-append-history",
        dest="append_history",
        action="store_false",
        help="Skip appending to historical CSV",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("BETTING SPLITS COLLECTION")
    print("=" * 80)
    print(f"Source: {args.source}")
    print(f"Time: {datetime.now(CST).strftime('%Y-%m-%d %I:%M %p CST')}")

    # Fetch upcoming games from The Odds API
    print("\nFetching upcoming games from The Odds API...")
    try:
        games = await the_odds.fetch_odds()
        print(f"  [OK] Found {len(games)} upcoming games")
    except Exception as e:
        print(f"  [ERROR] Failed to fetch games: {e}")
        return

    if not games:
        print("\n[WARN] No upcoming games found")
        return

    # Fetch betting splits
    print(f"\nFetching betting splits from {args.source}...")
    try:
        splits_dict = await fetch_public_betting_splits(games, source=args.source)
        print(f"  [OK] Loaded splits for {len(splits_dict)} games")
    except Exception as e:
        print(f"  [ERROR] Failed to fetch splits: {e}")
        import traceback
        traceback.print_exc()
        return

    # Display splits
    print("\n" + "=" * 80)
    print("BETTING SPLITS SUMMARY")
    print("=" * 80)

    for game_key, splits in splits_dict.items():
        print(f"\n{splits.away_team} @ {splits.home_team}")
        print(f"  Spread: {splits.spread_line:+.1f}")
        print(f"  Public: {splits.spread_home_ticket_pct:.1f}% home / "
              f"{splits.spread_away_ticket_pct:.1f}% away")
        print(f"  Money:  {splits.spread_home_money_pct:.1f}% home / "
              f"{splits.spread_away_money_pct:.1f}% away")

        if splits.spread_rlm:
            print(f"  [!] RLM DETECTED - Sharp side: {splits.sharp_spread_side}")

        ticket_money_diff = splits.spread_home_ticket_pct - splits.spread_home_money_pct
        if abs(ticket_money_diff) > 10:
            print(f"  [$] Ticket/Money divergence: {ticket_money_diff:+.1f}% "
                  f"(sharps on {'away' if ticket_money_diff > 0 else 'home'})")

        print(f"  Total: {splits.total_line:.1f}")
        print(f"  Public: {splits.over_ticket_pct:.1f}% over / "
              f"{splits.under_ticket_pct:.1f}% under")

        if splits.total_rlm:
            print(f"  [!] RLM DETECTED - Sharp side: {splits.sharp_total_side}")

        print(f"  Source: {splits.source}")

    # Save to file if requested
    if args.save:
        output_dir = DATA_DIR / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "betting_splits.json"

        # Convert to serializable format
        splits_data = {}
        for game_key, splits in splits_dict.items():
            splits_data[game_key] = {
                "event_id": splits.event_id,
                "home_team": splits.home_team,
                "away_team": splits.away_team,
                "game_time": splits.game_time.isoformat() if splits.game_time else None,
                "features": splits_to_features(splits),
                "source": splits.source,
                "updated_at": splits.updated_at.isoformat() if splits.updated_at else None,
            }

        with open(output_file, "w") as f:
            json.dump(splits_data, f, indent=2)

        print(f"\n[OK] Saved betting splits to {output_file}")

    # Append to historical CSV for training data
    if args.append_history:
        SPLITS_DIR.mkdir(parents=True, exist_ok=True)
        history_file = SPLITS_DIR / "historical_splits.csv"

        # Build rows for CSV
        rows = []
        collection_time = datetime.now(CST)
        for game_key, splits in splits_dict.items():
            features = splits_to_features(splits)
            row = {
                "collection_date": collection_time.strftime("%Y-%m-%d"),
                "collection_time": collection_time.isoformat(),
                "event_id": splits.event_id,
                "home_team": splits.home_team,
                "away_team": splits.away_team,
                "game_time": (
                    splits.game_time.isoformat() if splits.game_time else None
                ),
                "source": splits.source,
                **features,  # Flatten all features into columns
            }
            rows.append(row)

        new_df = pd.DataFrame(rows)

        # Append or create
        if history_file.exists():
            existing_df = pd.read_csv(history_file)
            original_count = len(existing_df)
            # Deduplicate by event_id + collection_date (keep latest)
            combined = pd.concat([existing_df, new_df], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=["event_id", "collection_date"], keep="last"
            )
            combined.to_csv(history_file, index=False)
            # Net new = final count - original count (accounts for replacements)
            net_new = len(combined) - original_count
            replaced = len(new_df) - net_new if net_new < len(new_df) else 0
            print(f"\n[OK] Added {len(new_df)} rows to {history_file}")
            if replaced > 0:
                print(f"     ({replaced} replaced existing, {net_new} net new)")
            print(f"     Total rows: {len(combined)}")
        else:
            new_df.to_csv(history_file, index=False)
            print(f"\n[OK] Created {history_file} with {len(new_df)} rows")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
