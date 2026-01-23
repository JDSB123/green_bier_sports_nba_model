"""
Data Ingestion Pipeline - COMPREHENSIVE.

Ingests data from ALL configured sources with intelligent caching:
- The Odds API (FG odds, 1H odds, events, scores, betting splits)
- API-Basketball (teams, games, statistics, standings)
- Action Network (betting splits - requires login)
- ESPN (schedule, injuries - FREE)

Caching reduces redundant API calls by ~60%.

Usage:
    python scripts/data_unified_ingest_all.py              # Full ingestion with caching
    python scripts/data_unified_ingest_all.py --refresh    # Force refresh (bypass cache)
    python scripts/data_unified_ingest_all.py --slate      # Optimized for today's slate only
    python scripts/data_unified_ingest_all.py --legacy     # Old behavior (no caching)
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Any, Awaitable, Callable

from src.config import settings
from src.ingestion import injuries, the_odds
from src.ingestion.api_basketball import APIBasketballClient
from src.utils.api_cache import api_cache

FetchCallable = Callable[..., Awaitable[Any]]
SaveCallable = Callable[[Any], Awaitable[str]]


async def _run_source(
    name: str,
    fetch_fn: FetchCallable,
    save_fn: SaveCallable,
    *fetch_args: Any,
    **fetch_kwargs: Any,
) -> str:
    """Run a single data source ingestion."""
    print(f"Starting ingestion: {name}")
    payload = await fetch_fn(*fetch_args, **fetch_kwargs)
    path = await save_fn(payload)
    print(f"{name} saved to {path}")
    return path


async def _ingest_api_basketball(essential_only: bool = False) -> list[str]:
    """Ingest API-Basketball endpoints.

    Args:
        essential_only: If True, only ingest TIER 1 (teams, games, statistics, box scores).
                       If False, ingest all tiers including reference data.
    """
    print("\n" + "=" * 60)
    print("API-BASKETBALL INGESTION")
    print("=" * 60)

    client = APIBasketballClient()

    if essential_only:
        print("Mode: ESSENTIAL (Tier 1 only - fastest)")
        results = await client.ingest_essential()
    else:
        print("Mode: FULL (All tiers)")
        results = await client.ingest_all()

    paths = [r.path for r in results if r.path]
    print(f"\nAPI-Basketball: {len(results)} endpoints ingested")
    return paths


async def main_legacy(essential_only: bool = False) -> None:
    """Run legacy ingestion pipeline (no caching).

    Args:
        essential_only: If True, use essential-only mode for API-Basketball.
    """
    print("=" * 60)
    print("NBA DATA INGESTION PIPELINE (LEGACY MODE)")
    print("=" * 60)
    print(f"Season: {settings.current_season}")
    print(f"Mode: {'ESSENTIAL' if essential_only else 'FULL'}")
    print()

    # Collect all tasks
    tasks: list[Awaitable[Any]] = [
        _run_source(
            "the_odds",
            the_odds.fetch_odds,
            the_odds.save_odds,
        ),
        _ingest_api_basketball(essential_only=essential_only),
        _run_source(
            "injuries",
            injuries.fetch_all_injuries,
            injuries.save_injuries,
        ),
    ]

    # Run all ingestion tasks
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Report results
    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)

    errors = [r for r in results if isinstance(r, Exception)]
    if errors:
        print(f"\n{len(errors)} errors occurred:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("\nAll sources ingested successfully!")


async def main_comprehensive(force_refresh: bool = False, slate_only: bool = False) -> None:
    """Run COMPREHENSIVE ingestion with caching.

    Uses ALL available endpoints with intelligent caching.

    Args:
        force_refresh: If True, bypass cache for all calls
        slate_only: If True, only fetch data needed for today's slate
    """
    from src.ingestion.comprehensive import ComprehensiveIngestion

    print("=" * 60)
    print("NBA DATA INGESTION - COMPREHENSIVE (WITH CACHING)")
    print("=" * 60)
    print(f"Season: {settings.current_season}")
    print(f"Mode: {'SLATE ONLY' if slate_only else 'FULL'}")
    print(f"Force Refresh: {force_refresh}")
    print()

    # Show cache stats before
    stats_before = api_cache.get_stats()
    if stats_before.get("mode") == "STRICT":
        print(
            f"Cache Status: STRICT MODE (Memory only: {stats_before.get('memory_count', 0)} entries)")
    else:
        print(
            f"Cache Status: {stats_before.get('file_count', 0)} cached entries ({stats_before.get('total_size_mb', 0)} MB)")
    print()

    ingestion = ComprehensiveIngestion(force_refresh=force_refresh)

    if slate_only:
        await ingestion.ingest_for_slate()
    else:
        await ingestion.ingest_all()

    # Report results
    status = ingestion.get_status()

    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)
    print(
        f"Endpoints: {status['successful']} successful, {status['failed']} failed")
    print(f"Total Records: {status['total_records']}")
    print(
        f"API Calls: {status['api_calls']} (saved ~{status['cache_hits']} via cache)")
    print()

    if status['errors']:
        print(f"Errors ({len(status['errors'])}):")
        for error in status['errors']:
            print(f"  - {error}")
    else:
        print("No errors!")

    # Show what data sources were used
    print()
    print("Data Sources Summary:")
    by_source = {}
    for result in ingestion.status.results:
        if result.source not in by_source:
            by_source[result.source] = {"endpoints": 0, "records": 0}
        by_source[result.source]["endpoints"] += 1
        by_source[result.source]["records"] += result.record_count

    for source, info in by_source.items():
        print(
            f"  {source}: {info['endpoints']} endpoints, {info['records']} records")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NBA Data Ingestion Pipeline")
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Use legacy ingestion (no caching, fewer endpoints)",
    )
    parser.add_argument(
        "--essential",
        action="store_true",
        help="Essential mode (legacy only): Tier 1 endpoints only",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force refresh: Bypass all caches",
    )
    parser.add_argument(
        "--slate",
        action="store_true",
        help="Slate mode: Only fetch data needed for today's picks",
    )
    args = parser.parse_args()

    if args.legacy:
        asyncio.run(main_legacy(essential_only=args.essential))
    else:
        asyncio.run(main_comprehensive(
            force_refresh=args.refresh,
            slate_only=args.slate,
        ))
