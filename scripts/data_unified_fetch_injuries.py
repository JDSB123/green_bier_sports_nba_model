#!/usr/bin/env python3
"""
Fetch NBA injury data from ESPN and API-Basketball.

This script fetches current injury reports, enriches them with player stats,
and saves to data/processed/injuries.csv for use in model predictions.

Usage:
    python scripts/data_unified_fetch_injuries.py
"""
from src.config import settings
from src.ingestion.injuries import (
    fetch_all_injuries,
    save_injuries,
    enrich_injuries_with_stats,
)
import os
import sys
import asyncio
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


async def main():
    print("="*80)
    print("NBA INJURY DATA FETCHER")
    print("="*80)

    # Fetch injuries from all sources
    print("\nFetching injury data from available sources...")
    injuries = await fetch_all_injuries()

    if not injuries:
        print("[WARN] No injury data fetched")
        return

    print(f"[OK] Fetched {len(injuries)} injury reports")

    # Display summary
    status_counts = {}
    team_counts = {}
    for inj in injuries:
        status_counts[inj.status] = status_counts.get(inj.status, 0) + 1
        team_counts[inj.team] = team_counts.get(inj.team, 0) + 1

    print("\nInjury Status Breakdown:")
    for status, count in sorted(status_counts.items(), key=lambda x: -x[1]):
        print(f"  {status:15} {count:3} players")

    print(f"\nTeams with injuries: {len(team_counts)}")
    print("Top 5 teams by injury count:")
    for team, count in sorted(team_counts.items(), key=lambda x: -x[1])[:5]:
        print(f"  {team:25} {count:2} players")

    # Enrich with player stats (if available)
    print("\nEnriching with player statistics...")
    injuries = await enrich_injuries_with_stats(injuries)

    players_with_stats = sum(1 for inj in injuries if inj.ppg > 0)
    print(f"[OK] Found stats for {players_with_stats}/{len(injuries)} players")

    # Save to CSV
    print(f"\nSaving to {settings.data_processed_dir}/injuries.csv...")
    out_path = await save_injuries(injuries)
    print(f"[OK] Saved to: {out_path}")

    # Show high-impact injuries (out players scoring >15 PPG)
    print("\nHigh-Impact Injuries (OUT status, >15 PPG):")
    high_impact = [inj for inj in injuries if inj.status ==
                   'out' and inj.ppg > 15]
    if high_impact:
        for inj in sorted(high_impact, key=lambda x: -x.ppg)[:10]:
            print(
                f"  {inj.player_name:25} ({inj.team:20}) {inj.ppg:5.1f} PPG - {inj.injury_type or 'N/A'}")
    else:
        print("  None found")

    print("\n" + "="*80)
    print("Done!")
    print("="*80)


if __name__ == '__main__':
    asyncio.run(main())
