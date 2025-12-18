"""
Run all available standardization steps over existing raw data.

Assumes ingestion has already written raw JSON under data/raw/*.
"""
from __future__ import annotations
import os
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.collect_the_odds import collect_the_odds
from scripts.collect_api_basketball import process_api_basketball
from src.processing.betsapi import process_betsapi_events


def main() -> int:
    print("Standardizing The Odds...")
    odds_path = collect_the_odds()
    print(f"  -> {odds_path}")

    print("Standardizing BetsAPI events...")
    betsapi_path = process_betsapi_events()
    print(f"  -> {betsapi_path}")

    print("Standardizing API-Basketball games...")
    api_basketball_path = process_api_basketball()
    print(f"  -> {api_basketball_path}")

    print("\nAll standardizations complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
