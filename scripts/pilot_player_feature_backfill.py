#!/usr/bin/env python3
"""
Pilot coverage check for player/box-score endpoints (ESPN + API-Basketball).

This does NOT create a leakage-safe training set. It only verifies that
historical endpoints can be accessed for recent games and records success rates.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from src.config import settings
from src.ingestion.api_basketball import fetch_game_stats_players, fetch_games
from src.ingestion.espn import fetch_espn_box_score, fetch_espn_schedule
from src.ingestion.standardize import normalize_team_to_espn


@dataclass
class CoverageResult:
    game_id: int | str | None
    game_date: str
    home_team: str
    away_team: str
    espn_game_id: str | None
    espn_ok: bool
    api_b_ok: bool
    api_b_players: int | None
    error_espn: str | None
    error_api_b: str | None


async def _build_espn_schedule_map(dates: list[str]) -> Dict[Tuple[str, str, str], str]:
    schedule_map: Dict[Tuple[str, str, str], str] = {}
    for date_str in dates:
        try:
            games = await fetch_espn_schedule(dates=[date_str])
        except Exception:
            continue
        for game in games:
            away = game.away_team
            home = game.home_team
            key = (date_str, away, home)
            schedule_map[key] = game.game_id
    return schedule_map


async def _build_api_b_schedule_map(dates: list[str]) -> Dict[Tuple[str, str, str], int]:
    schedule_map: Dict[Tuple[str, str, str], int] = {}
    for date_str in dates:
        try:
            resp = await fetch_games(
                date=date_str,
                league=12,
                season=settings.current_season,
            )
            games = resp.get("response", []) if isinstance(resp, dict) else []
        except Exception:
            continue
        for game in games:
            teams = game.get("teams", {})
            away = str(teams.get("away", {}).get("name", "")).lower()
            home = str(teams.get("home", {}).get("name", "")).lower()
            game_id = game.get("id")
            if away and home and game_id:
                schedule_map[(date_str, away, home)] = int(game_id)
    return schedule_map


async def _process_game(row, schedule_map, api_b_map, sem, use_espn, use_api_b) -> CoverageResult:
    async with sem:
        game_date = pd.to_datetime(row["game_date"], errors="coerce")
        date_key = game_date.strftime("%Y%m%d") if pd.notna(game_date) else ""
        date_key_api = game_date.strftime("%Y-%m-%d") if pd.notna(game_date) else ""
        away_norm, _ = normalize_team_to_espn(str(row["away_team"]), source="pilot")
        home_norm, _ = normalize_team_to_espn(str(row["home_team"]), source="pilot")

        espn_game_id = schedule_map.get((date_key, away_norm, home_norm))
        api_b_game_id = api_b_map.get(
            (date_key_api, str(row["away_team"]).lower(), str(row["home_team"]).lower())
        )
        espn_ok = False
        api_b_ok = False
        api_b_players = None
        error_espn = None
        error_api_b = None

        if use_espn and espn_game_id:
            try:
                data = await fetch_espn_box_score(espn_game_id)
                espn_ok = bool(data.get("teams"))
            except Exception as exc:
                error_espn = str(exc)

        if use_api_b and api_b_game_id:
            try:
                resp = await fetch_game_stats_players(game=api_b_game_id)
                players = resp.get("response", []) if isinstance(resp, dict) else []
                api_b_players = len(players)
                api_b_ok = api_b_players > 0
            except Exception as exc:
                error_api_b = str(exc)

        return CoverageResult(
            game_id=api_b_game_id,
            game_date=date_key,
            home_team=str(row["home_team"]),
            away_team=str(row["away_team"]),
            espn_game_id=espn_game_id,
            espn_ok=espn_ok,
            api_b_ok=api_b_ok,
            api_b_players=api_b_players,
            error_espn=error_espn,
            error_api_b=error_api_b,
        )


async def main_async(args) -> int:
    training_path = Path(args.training_file)
    if not training_path.exists():
        raise SystemExit(f"Training file not found: {training_path}")

    df = pd.read_csv(training_path, low_memory=False)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce", format="mixed")
    df["game_date"] = df["game_date"].dt.tz_localize(None)

    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=int(args.days))
    df = df[df["game_date"] >= cutoff].copy()
    if df.empty:
        print("[WARN] No games in selected window.")
        return 0

    df = df.sort_values("game_date")
    if args.max_games:
        df = df.head(int(args.max_games))

    dates = sorted({d.strftime("%Y%m%d") for d in df["game_date"].dropna()})
    schedule_map = await _build_espn_schedule_map(dates) if args.use_espn else {}
    api_b_dates = sorted({d.strftime("%Y-%m-%d") for d in df["game_date"].dropna()})
    api_b_map = await _build_api_b_schedule_map(api_b_dates) if args.use_api_b else {}

    sem = asyncio.Semaphore(int(args.concurrency))
    tasks = [
        _process_game(row, schedule_map, api_b_map, sem, args.use_espn, args.use_api_b)
        for _, row in df.iterrows()
    ]
    results = []
    for coro in asyncio.as_completed(tasks):
        results.append(await coro)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame([r.__dict__ for r in results])
    out_df.to_csv(out_path, index=False)

    total = len(out_df)
    espn_cov = (out_df["espn_ok"].mean() * 100) if args.use_espn else None
    api_cov = (out_df["api_b_ok"].mean() * 100) if args.use_api_b else None

    print(f"[OK] Pilot coverage written to {out_path}")
    print(f"  Games checked: {total}")
    if espn_cov is not None:
        print(f"  ESPN box score coverage: {espn_cov:.1f}%")
    if api_cov is not None:
        print(f"  API-B player stats coverage: {api_cov:.1f}%")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pilot coverage check for ESPN/API-Basketball historical endpoints",
    )
    parser.add_argument(
        "--training-file",
        type=str,
        default=str(Path(settings.data_processed_dir) / "training_data.csv"),
        help="Path to training data CSV",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=60,
        help="Number of days back to check (default: 60)",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Optional cap on number of games to check",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Concurrency level for API calls (default: 3)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(
            Path("archive")
            / "analysis"
            / f"pilot_player_features_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
        ),
        help="Output CSV path",
    )
    parser.add_argument(
        "--no-espn",
        dest="use_espn",
        action="store_false",
        help="Skip ESPN box score checks",
    )
    parser.add_argument(
        "--no-api-b",
        dest="use_api_b",
        action="store_false",
        help="Skip API-Basketball player stats checks",
    )
    parser.set_defaults(use_espn=True, use_api_b=True)

    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
