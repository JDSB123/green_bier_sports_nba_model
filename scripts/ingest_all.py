"""
Data Ingestion Pipeline.

Ingests data from all configured sources:
- API-Basketball (tiered endpoints - essential vs full)
- The Odds API (live betting odds)
- Injury data

Usage:
    python scripts/ingest_all.py              # Full ingestion (all endpoints)
    python scripts/ingest_all.py --essential  # Essential endpoints only (faster)
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Any, Awaitable, Callable

from src.config import settings
from src.ingestion import injuries, the_odds
from src.ingestion.api_basketball import APIBasketballClient

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


async def main(essential_only: bool = False) -> None:
    """Run full ingestion pipeline.

    Args:
        essential_only: If True, use essential-only mode for API-Basketball.
    """
    print("=" * 60)
    print("NBA DATA INGESTION PIPELINE")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NBA Data Ingestion Pipeline")
    parser.add_argument(
        "--essential",
        action="store_true",
        help="Essential mode: Only ingest Tier 1 endpoints (faster, fewer API calls)",
    )
    args = parser.parse_args()

    asyncio.run(main(essential_only=args.essential))
