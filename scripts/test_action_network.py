#!/usr/bin/env python3
"""
Test Action Network integration with premium authentication.
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.ingestion.betting_splits import fetch_splits_action_network
from src.config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_action_network():
    """Test Action Network betting splits fetching."""
    print("Testing Action Network integration...")
    print()

    # Check credentials
    has_username = bool(settings.action_network_username)
    has_password = bool(settings.action_network_password)

    print("Credential Status:")
    print(f"  Username configured: {'YES' if has_username else 'NO'}")
    print(f"  Password configured: {'YES' if has_password else 'NO'}")
    print()

    if has_username and has_password:
        print("Premium credentials detected - will attempt authenticated access")
        print("This should use Action PRO/Labs features if your subscription is active")
    else:
        print("No premium credentials - will use public API")
        print("To enable premium access, set ACTION_NETWORK_USERNAME and ACTION_NETWORK_PASSWORD")
    print()

    try:
        # Test fetching today's games
        print("Fetching betting splits...")
        splits = await fetch_splits_action_network()

        print(f"Successfully fetched {len(splits)} games")
        print()

        if splits:
            print("Sample game data:")
            game = splits[0]
            print(f"  {game.away_team} @ {game.home_team}")
            if hasattr(game, 'spread_home_pct') and game.spread_home_pct:
                print(".1f")
                print(".1f")
                print(".1f")
            print()

        print("Action Network integration test completed successfully!")

    except Exception as e:
        print(f"Test failed: {e}")
        return False

    return True


if __name__ == "__main__":
    success = asyncio.run(test_action_network())
    sys.exit(0 if success else 1)