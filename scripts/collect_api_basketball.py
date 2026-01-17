"""
Standalone ingestion script for API-Basketball game outcomes.

Fetches NBA game results including final scores for training labels.
Usage: python scripts/collect_api_basketball.py [--season 2024] [--date 2024-12-01]
"""
from __future__ import annotations
import argparse
import asyncio
import datetime as dt
import json
import os
import sys
import glob
from typing import Any, Dict, List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.config import settings
from src.ingestion.api_basketball import APIBasketballClient


NBA_LEAGUE_ID = 12  # NBA league ID in API-Basketball


async def fetch_nba_games(
    season: int | None = None,
    date: str | None = None,
) -> Dict[str, Any]:
    """Fetch NBA games by season or date."""
    client = APIBasketballClient()
    if date:
        result = await client.fetch_games_by_date(date)
        return result.data
    result = await client.fetch_games()
    return result.data


def _latest_file(pattern: str) -> str | None:
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def _safe_load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_game_outcomes(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Normalize API-Basketball response to extract game outcomes.

    Returns rows with:
    - game_id, date, home_team, away_team
    - home_score, away_score, total_score
    - spread_result (home margin), status
    """
    rows: List[Dict[str, Any]] = []

    response = payload.get("response", [])
    for game in response:
        game_id = game.get("id")
        date_info = game.get("date", "")
        status = game.get("status", {})

        teams = game.get("teams", {})
        home_team = teams.get("home", {}).get("name", "")
        away_team = teams.get("away", {}).get("name", "")

        scores = game.get("scores", {})
        home_score = scores.get("home", {}).get("total")
        away_score = scores.get("away", {}).get("total")

        # Attempt to extract halftime scores if present in the payload.
        home_ht = None
        away_ht = None
        # Common providers include a 'halftime' object
        ht = scores.get("halftime") or scores.get("ht") or scores.get("1h")
        if isinstance(ht, dict):
            home_ht = ht.get("home") or ht.get("home_score")
            away_ht = ht.get("away") or ht.get("away_score")
        # Fallback: some payloads provide period-level details
        if home_ht is None or away_ht is None:
            periods = game.get("periods") or []
            if isinstance(periods, list) and len(periods) >= 2:
                try:
                    p1 = periods[0]
                    p2 = periods[1]
                    h1 = p1.get("home") or p1.get("home_score")
                    a1 = p1.get("away") or p1.get("away_score")
                    h2 = p2.get("home") or p2.get("home_score")
                    a2 = p2.get("away") or p2.get("away_score")
                    if all(v is not None for v in (h1, a1, h2, a2)):
                        home_ht = int(h1) + int(h2)
                        away_ht = int(a1) + int(a2)
                except Exception:
                    pass

        # Only include finished games with valid scores
        game_status = status.get("long", "")
        if home_score is not None and away_score is not None:
            total_score = home_score + away_score
            home_margin = home_score - away_score  # For spread calculations

            rows.append({
                "source": "api_basketball",
                "game_id": game_id,
                "date": date_info,
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
                "home_halftime_score": home_ht,
                "away_halftime_score": away_ht,
                "total_score": total_score,
                "home_margin": home_margin,
                "status": game_status,
            })

    return rows


def process_api_basketball(
    output_csv: str = "data/processed/game_outcomes.csv",
) -> str:
    """Process raw API-Basketball JSON files into normalized CSV."""
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Find all API-Basketball raw files
    raw_pattern = os.path.join(settings.data_raw_dir, "api_basketball", "games_*.json")
    raw_files = glob.glob(raw_pattern)

    all_rows: List[Dict[str, Any]] = []
    for file_path in raw_files:
        payload = _safe_load_json(file_path)
        rows = normalize_game_outcomes(payload)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows, columns=[
        "source",
        "game_id",
        "date",
        "home_team",
        "away_team",
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
    parser = argparse.ArgumentParser(description="Collect NBA game outcomes from API-Basketball")
    parser.add_argument("--season", type=int, help="NBA season year (e.g., 2024 for 2024-25)")
    parser.add_argument("--date", type=str, help="Specific date (YYYY-MM-DD)")
    parser.add_argument("--process-only", action="store_true", help="Only process existing raw files")
    args = parser.parse_args()

    target_date = args.date
    if not target_date and not args.season and not args.process_only:
        # Default to fetching yesterday's games if no args provided
        yesterday = dt.date.today() - dt.timedelta(days=1)
        target_date = yesterday.strftime("%Y-%m-%d")

    if not args.process_only:
        # Fetch new data
        print(f"Fetching NBA games (season={args.season}, date={target_date})...")
        data = await fetch_nba_games(season=args.season, date=target_date)

        # Save raw JSON
        client = APIBasketballClient()
        path = client._save("games", data)
        print(f"Raw data saved to {path}")

        games_count = len(data.get("response", []))
        print(f"Fetched {games_count} games")

    # Process to CSV
    output = process_api_basketball()
    df = pd.read_csv(output)
    print(f"Processed {len(df)} game outcomes to {output}")


if __name__ == "__main__":
    asyncio.run(main())
