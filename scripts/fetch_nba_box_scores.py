#!/usr/bin/env python3
"""
Fetch NBA box scores from NBA.com API using nba_api.

This fills the gap in wyattowalsh/basketball dataset (which ends June 2023).
Fetches 2023-24 and 2024-25 season box scores.

NO API KEY REQUIRED - uses official NBA.com public API.
"""
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# NBA API imports
from nba_api.stats.endpoints import leaguegamefinder, boxscoretraditionalv2
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from nba_api.stats.static import teams

OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "nba_api"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_all_nba_teams():
    """Get list of all NBA team IDs."""
    nba_teams = teams.get_teams()
    return {t["id"]: t["full_name"] for t in nba_teams}


def fetch_season_games(season: str) -> pd.DataFrame:
    """Fetch all games for a season."""
    print(f"\n  Fetching games for {season}...", flush=True)
    
    # LeagueGameFinder gets all games
    gamefinder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        league_id_nullable="00",  # NBA
        season_type_nullable="Regular Season",
    )
    
    games = gamefinder.get_data_frames()[0]
    print(f"    Found {len(games)} game records", flush=True)
    
    return games


def fetch_box_score(game_id: str) -> dict:
    """Fetch box score for a single game."""
    try:
        box = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
        team_stats = box.get_data_frames()[1]  # Team stats
        return team_stats.to_dict("records") if len(team_stats) > 0 else None
    except Exception as e:
        print(f"      Error fetching {game_id}: {e}", flush=True)
        return None


def fetch_season_box_scores(season: str, delay: float = 0.6) -> pd.DataFrame:
    """Fetch all box scores for a season."""
    print(f"\n[FETCHING] Season {season} box scores...", flush=True)
    
    # Get games
    games = fetch_season_games(season)
    
    # Get unique game IDs
    game_ids = games["GAME_ID"].unique()
    print(f"  Unique games: {len(game_ids)}", flush=True)
    
    # Fetch box scores
    all_box_scores = []
    for i, game_id in enumerate(game_ids):
        if i % 50 == 0:
            print(f"    Progress: {i}/{len(game_ids)}", flush=True)
        
        box = fetch_box_score(game_id)
        if box:
            all_box_scores.extend(box)
        
        time.sleep(delay)  # Be nice to NBA.com API
    
    df = pd.DataFrame(all_box_scores)
    print(f"  Fetched {len(df)} box score records", flush=True)
    
    return df


def main():
    print("="*80, flush=True)
    print(" FETCHING NBA BOX SCORES FROM NBA.COM API", flush=True)
    print(" Using nba_api (no API key required)", flush=True)
    print("="*80, flush=True)
    
    seasons = ["2023-24", "2024-25"]
    
    for season in seasons:
        # Check if already fetched
        output_file = OUTPUT_DIR / f"box_scores_{season.replace('-', '_')}.csv"
        
        if output_file.exists():
            df = pd.read_csv(output_file)
            print(f"\n[SKIP] {season} already fetched: {len(df)} records", flush=True)
            continue
        
        # Fetch
        df = fetch_season_box_scores(season)
        
        # Save
        df.to_csv(output_file, index=False)
        print(f"  Saved to: {output_file}", flush=True)
    
    # Summary
    print("\n" + "="*80, flush=True)
    print(" SUMMARY", flush=True)
    print("="*80, flush=True)
    
    for season in seasons:
        output_file = OUTPUT_DIR / f"box_scores_{season.replace('-', '_')}.csv"
        if output_file.exists():
            df = pd.read_csv(output_file)
            print(f"\n  {season}:", flush=True)
            print(f"    Records: {len(df)}", flush=True)
            print(f"    Columns: {df.columns.tolist()[:10]}...", flush=True)


if __name__ == "__main__":
    main()
