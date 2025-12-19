"""
Standalone ingestion script for BallDontLie historical NBA data.

Fetches historical games with scores, team stats, and player data.
Usage: python scripts/collect_balldontlie.py [--seasons 2023 2024] [--all-pages]
"""
from __future__ import annotations
import argparse
import asyncio
import datetime as dt
import json
import os
import glob
from typing import Any, Dict, List

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings

BALLDONTLIE_BASE = "https://api.balldontlie.io/v1"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
)
async def fetch_games_page(
    page: int = 1,
    per_page: int = 100,
    seasons: List[int] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> Dict[str, Any]:
    """Fetch a page of games from BallDontLie API."""
    params: Dict[str, Any] = {"page": page, "per_page": per_page}
    if seasons:
        for season in seasons:
            params.setdefault("seasons[]", []).append(season)
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date

    headers = {}
    api_key = os.getenv("BALLDONTLIE_API_KEY", "")
    if api_key:
        headers["Authorization"] = api_key

    async with httpx.AsyncClient(timeout=30, headers=headers) as client:
        resp = await client.get(f"{BALLDONTLIE_BASE}/games", params=params)
        resp.raise_for_status()
        return resp.json()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
)
async def fetch_team_stats(
    season: int,
) -> Dict[str, Any]:
    """Fetch team season averages."""
    headers = {}
    api_key = os.getenv("BALLDONTLIE_API_KEY", "")
    if api_key:
        headers["Authorization"] = api_key

    async with httpx.AsyncClient(timeout=30, headers=headers) as client:
        resp = await client.get(
            f"{BALLDONTLIE_BASE}/season_averages",
            params={"season": season}
        )
        resp.raise_for_status()
        return resp.json()


async def fetch_all_games(
    seasons: List[int] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    max_pages: int = 50,
) -> List[Dict[str, Any]]:
    """Fetch all games with pagination."""
    all_games: List[Dict[str, Any]] = []
    page = 1

    while page <= max_pages:
        print(f"Fetching page {page}...")
        response = await fetch_games_page(
            page=page,
            per_page=100,
            seasons=seasons,
            start_date=start_date,
            end_date=end_date,
        )

        games = response.get("data", [])
        if not games:
            break

        all_games.extend(games)

        meta = response.get("meta", {})
        next_page = meta.get("next_page")
        if not next_page:
            break

        page = next_page
        await asyncio.sleep(0.5)  # Rate limiting

    return all_games


async def save_balldontlie_data(
    data: List[Dict[str, Any]],
    data_type: str = "games",
) -> str:
    """Save BallDontLie data to JSON file."""
    out_dir = os.path.join(settings.data_raw_dir, "balldontlie")
    os.makedirs(out_dir, exist_ok=True)
    ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(out_dir, f"{data_type}_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return out_path


def normalize_balldontlie_games(games: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize BallDontLie games to standard format."""
    rows: List[Dict[str, Any]] = []

    for game in games:
        home_team = game.get("home_team", {})
        visitor_team = game.get("visitor_team", {})

        home_score = game.get("home_team_score")
        away_score = game.get("visitor_team_score")

        # Only include completed games with scores
        if home_score is not None and away_score is not None and home_score > 0:
            rows.append({
                "source": "balldontlie",
                "game_id": game.get("id"),
                "date": game.get("date"),
                "season": game.get("season"),
                "home_team": home_team.get("full_name", ""),
                "home_team_abbr": home_team.get("abbreviation", ""),
                "away_team": visitor_team.get("full_name", ""),
                "away_team_abbr": visitor_team.get("abbreviation", ""),
                "home_score": home_score,
                "away_score": away_score,
                "total_score": home_score + away_score,
                "home_margin": home_score - away_score,
                "status": game.get("status", ""),
            })

    return rows


def process_balldontlie(
    output_csv: str = "data/processed/historical_games.csv",
) -> str:
    """Process raw BallDontLie JSON files into normalized CSV."""
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    raw_pattern = os.path.join(settings.data_raw_dir, "balldontlie", "games_*.json")
    raw_files = glob.glob(raw_pattern)

    all_rows: List[Dict[str, Any]] = []
    for file_path in raw_files:
        with open(file_path, "r", encoding="utf-8") as f:
            games = json.load(f)
        if isinstance(games, list):
            rows = normalize_balldontlie_games(games)
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows, columns=[
        "source",
        "game_id",
        "date",
        "season",
        "home_team",
        "home_team_abbr",
        "away_team",
        "away_team_abbr",
        "home_score",
        "away_score",
        "total_score",
        "home_margin",
        "status",
    ])

    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.drop_duplicates(subset=["game_id"], keep="last")
        df = df.sort_values("date", na_position="last")

    df.to_csv(output_csv, index=False)
    return output_csv


async def main():
    parser = argparse.ArgumentParser(description="Collect historical NBA data from BallDontLie")
    parser.add_argument("--seasons", type=int, nargs="+", help="Seasons to fetch (e.g., 2023 2024)")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--max-pages", type=int, default=50, help="Maximum pages to fetch")
    parser.add_argument("--process-only", action="store_true", help="Only process existing raw files")
    args = parser.parse_args()

    if not args.process_only:
        # Default to current season if not specified
        seasons = args.seasons or [2024]
        print(f"Fetching games for seasons: {seasons}")

        games = await fetch_all_games(
            seasons=seasons,
            start_date=args.start_date,
            end_date=args.end_date,
            max_pages=args.max_pages,
        )

        path = await save_balldontlie_data(games, "games")
        print(f"Raw data saved to {path}")
        print(f"Fetched {len(games)} games")

    # Process to CSV
    output = process_balldontlie()
    df = pd.read_csv(output)
    print(f"Processed {len(df)} historical games to {output}")


if __name__ == "__main__":
    asyncio.run(main())
