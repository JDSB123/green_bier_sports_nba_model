#!/usr/bin/env python3
"""
Debug Action Network API - show raw response data.
"""

import asyncio
import sys
import os
import json
import httpx

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.config import settings


async def debug_action_network_raw():
    """Fetch raw Action Network data to see what's available."""
    print("Fetching raw Action Network data...")
    print("=" * 50)

    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            # Public scoreboard API
            scoreboard_url = "https://api.actionnetwork.com/web/v1/scoreboard/nba"

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/json",
                "Origin": "https://www.actionnetwork.com",
                "Referer": "https://www.actionnetwork.com/nba/public-betting",
            }

            response = await client.get(scoreboard_url, headers=headers)

            if response.status_code != 200:
                print(f"API request failed: {response.status_code}")
                return

            data = response.json()
            games = data.get("games", [])

            print(f"Found {len(games)} games in response")
            print()

            if games:
                # Show structure of first game
                game = games[0]
                print("RAW GAME STRUCTURE (first game):")
                print("-" * 40)
                print(json.dumps(game, indent=2, default=str)[:2000])  # First 2000 chars

                # Show odds structure specifically
                if "odds" in game and game["odds"]:
                    print("\n\nODDS STRUCTURE:")
                    print("-" * 20)
                    odds = game["odds"][0] if game["odds"] else {}
                    print(json.dumps(odds, indent=2, default=str))

                # Show team structure
                if "teams" in game and game["teams"]:
                    print("\n\nTEAM STRUCTURE:")
                    print("-" * 20)
                    teams = game["teams"]
                    print(json.dumps(teams[:2], indent=2, default=str))  # First 2 teams

    except Exception as e:
        print(f"Debug failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(debug_action_network_raw())