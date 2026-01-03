#!/usr/bin/env python3
"""
Inspect and verify Action Network data accuracy by comparing with public website.
"""

import asyncio
import sys
import os
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.ingestion.betting_splits import fetch_splits_action_network
from src.config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def inspect_action_network_data():
    """Inspect Action Network data and show what's being extracted."""
    print("Inspecting Action Network betting splits data...")
    print("=" * 60)

    # Check credentials
    has_username = bool(settings.action_network_username)
    has_password = bool(settings.action_network_password)

    print(f"Credentials configured: {'YES' if has_username and has_password else 'NO'}")
    print()

    try:
        # Fetch the raw data
        splits = await fetch_splits_action_network()

        print(f"Successfully fetched {len(splits)} games")
        print()

        if not splits:
            print("No games found!")
            return

        # Show detailed breakdown of first game
        game = splits[0]
        print("SAMPLE GAME DATA BREAKDOWN:")
        print("-" * 40)
        print(f"Event ID: {game.event_id}")
        print(f"Teams: {game.away_team} @ {game.home_team}")
        print(f"Game Time: {game.game_time}")
        print(f"Source: {game.source}")
        print()

        print("SPREAD BETTING DATA:")
        print(f"  Line: {game.spread_line:+.1f} (home team)")
        print(f"  Public Tickets: {game.spread_home_ticket_pct:.1f}% on home, {game.spread_away_ticket_pct:.1f}% on away")
        print(f"  Money Split: {game.spread_home_money_pct:.1f}% on home, {game.spread_away_money_pct:.1f}% on away")
        print(f"  Line Movement: Open={game.spread_open:+.1f}, Current={game.spread_current:+.1f}")
        print()

        print("TOTAL BETTING DATA:")
        print(f"  Line: {game.total_line:.1f} points")
        print(f"  Public Tickets: {game.over_ticket_pct:.1f}% over, {game.under_ticket_pct:.1f}% under")
        print(f"  Money Split: {game.over_money_pct:.1f}% over, {game.under_money_pct:.1f}% under")
        print(f"  Line Movement: Open={game.total_open:.1f}, Current={game.total_current:.1f}")
        print()

        # Show additional fields if available
        if hasattr(game, 'is_rlm_spread'):
            print(f"Reverse Line Movement (Spread): {getattr(game, 'is_rlm_spread', 'N/A')}")
        if hasattr(game, 'is_rlm_total'):
            print(f"Reverse Line Movement (Total): {getattr(game, 'is_rlm_total', 'N/A')}")
        print()

        print("VERIFICATION INSTRUCTIONS:")
        print("-" * 40)
        print("To verify accuracy:")
        print("1. Visit: https://www.actionnetwork.com/nba/public-betting")
        print("2. Find the same game:", game.away_team, "@", game.home_team)
        print("3. Compare the percentages shown on Action Network with the data above")
        print("4. Check if ticket splits and money splits match")
        print()

        # Show summary for all games
        print(f"SUMMARY OF ALL {len(splits)} GAMES:")
        print("-" * 40)
        for i, g in enumerate(splits[:5]):  # Show first 5 games
            has_spread_data = g.spread_home_ticket_pct != 50.0 or g.spread_away_ticket_pct != 50.0
            has_total_data = g.over_ticket_pct != 50.0 or g.under_ticket_pct != 50.0

            status = ""
            if has_spread_data and has_total_data:
                status = "FULL DATA"
            elif has_spread_data or has_total_data:
                status = "PARTIAL DATA"
            else:
                status = "NO DATA"

            print("2d")

        if len(splits) > 5:
            print(f"  ... and {len(splits) - 5} more games")

        print()
        print("DATA QUALITY SUMMARY:")
        print("-" * 40)
        full_data = sum(1 for g in splits if (g.spread_home_ticket_pct != 50.0 or g.spread_away_ticket_pct != 50.0) and (g.over_ticket_pct != 50.0 or g.under_ticket_pct != 50.0))
        partial_data = sum(1 for g in splits if ((g.spread_home_ticket_pct != 50.0 or g.spread_away_ticket_pct != 50.0) or (g.over_ticket_pct != 50.0 or g.under_ticket_pct != 50.0)) and not ((g.spread_home_ticket_pct != 50.0 or g.spread_away_ticket_pct != 50.0) and (g.over_ticket_pct != 50.0 or g.under_ticket_pct != 50.0)))
        no_data = len(splits) - full_data - partial_data

        print(f"  Full data (spread + total): {full_data} games")
        print(f"  Partial data: {partial_data} games")
        print(f"  No betting data: {no_data} games")

    except Exception as e:
        print(f"Inspection failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(inspect_action_network_data())