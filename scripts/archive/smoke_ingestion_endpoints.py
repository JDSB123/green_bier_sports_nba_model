"""
Smoke test for ingestion endpoints: The Odds and BetsAPI.

Usage:
    python scripts/smoke_ingestion_endpoints.py
"""
from __future__ import annotations
import asyncio
from typing import Any

from src.config import settings
from src.ingestion import the_odds, betsapi


def summarize(payload: Any) -> str:
    if isinstance(payload, dict):
        keys = list(payload.keys())
        return f"dict keys={keys[:5]}"
    if isinstance(payload, list):
        return f"list len={len(payload)}"
    return str(type(payload))


async def main() -> int:
    if not settings.the_odds_api_key:
        print("THE_ODDS_API_KEY missing; set it in .env")
        return 1
    if not settings.betsapi_key:
        print("BETSAPI_KEY missing; set it in .env")
        return 1

    try:
        odds = await the_odds.fetch_odds()
        print(f"[OK] the_odds.fetch_odds: {summarize(odds)}")
    except Exception as e:
        print(f"[ERR] the_odds.fetch_odds: {e.__class__.__name__}: {e}")

    try:
        events = await betsapi.fetch_events()
        print(f"[OK] betsapi.fetch_events: {summarize(events)}")
    except Exception as e:
        print(f"[ERR] betsapi.fetch_events: {e.__class__.__name__}: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
