"""
Collect public betting percentages and save to data directory.

Usage:
    python scripts/collect_betting_splits.py [--source sbro|covers|mock]
"""
import asyncio
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

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
CST = ZoneInfo("America/Chicago")


async def main():
    """Collect betting splits and save to file."""
    parser = argparse.ArgumentParser(description="Collect public betting splits")
    parser.add_argument(
        "--source",
        choices=["auto", "sbro", "covers", "mock"],
        default="auto",
        help="Data source for betting splits",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save splits to data/processed/betting_splits.json",
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

    print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
