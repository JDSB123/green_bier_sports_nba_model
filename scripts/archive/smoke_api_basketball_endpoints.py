"""
Quick smoke test for API-Basketball endpoints (v1 host).

Usage:
    python scripts/smoke_api_basketball_endpoints.py [--season 2025-2026] [--team 139] [--game 391053]
"""
from __future__ import annotations
import argparse
import asyncio
import os
from typing import Any, Awaitable, Callable

from src.config import settings
from src.ingestion import api_basketball


def summarize(payload: Any) -> str:
    if isinstance(payload, dict):
        keys = list(payload.keys())
        return f"dict keys={keys[:5]}"
    if isinstance(payload, list):
        return f"list len={len(payload)}"
    return str(type(payload))


async def probe(name: str, fn: Callable[[], Awaitable[Any]]) -> None:
    try:
        data = await fn()
        print(f"[OK] {name}: {summarize(data)}")
    except Exception as exc:
        print(f"[ERR] {name}: {exc.__class__.__name__}: {exc}")


async def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test API-Basketball endpoints")
    parser.add_argument("--season", default="2025-2026", help="Season string (e.g., 2025-2026)")
    parser.add_argument("--team", type=int, default=139, help="Team id to probe")
    parser.add_argument("--game", type=int, default=391053, help="Game id to probe for stats/odds")
    args = parser.parse_args()

    if not settings.api_basketball_key:
        print("API_BASKETBALL_KEY missing; set it in .env to run this smoke test.")
        return 1

    league = api_basketball.NBA_LEAGUE_ID
    season = args.season
    team_id = args.team
    game_id = args.game

    await probe("teams search Denver", lambda: api_basketball.fetch_teams(search="Denver"))
    await probe(
        "games season+league",
        lambda: api_basketball.fetch_games(season=season, league=league),
    )
    await probe(
        "statistics team",
        lambda: api_basketball.fetch_statistics(league=league, season=season, team=team_id),
    )
    await probe(
        "players by team+season",
        lambda: api_basketball.fetch_players(team=team_id, season=season),
    )
    await probe(
        "standings",
        lambda: api_basketball.fetch_standings(league=league, season=season),
    )
    await probe(
        "standings stages",
        lambda: api_basketball.fetch_standings_stages(league=league, season=season),
    )
    await probe(
        "standings groups",
        lambda: api_basketball.fetch_standings_groups(league=league, season=season),
    )
    await probe(
        "odds by league/season",
        lambda: api_basketball.fetch_odds(league=league, season=season),
    )
    await probe("bookmakers", api_basketball.fetch_bookmakers)
    await probe("bets", api_basketball.fetch_bets)
    await probe(
        "game stats teams",
        lambda: api_basketball.fetch_game_stats_teams(id=game_id),
    )
    await probe(
        "game stats players",
        lambda: api_basketball.fetch_game_stats_players(id=game_id),
    )
    await probe(
        "h2h 132-134",
        lambda: api_basketball.fetch_h2h(h2h="132-134", league=league, season=season),
    )

    print("\nAPI-Basketball smoke test complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
