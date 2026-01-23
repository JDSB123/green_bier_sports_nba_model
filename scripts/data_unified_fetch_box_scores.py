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
import random
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


def fetch_box_score(game_id: str, max_retries: int = 3) -> dict:
    """Fetch box score for a single game with retry logic."""
    for attempt in range(max_retries):
        try:
            box = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id, timeout=60)
            team_stats = box.get_data_frames()[1]  # Team stats
            return team_stats.to_dict("records") if len(team_stats) > 0 else None
        except Exception as e:
            if attempt < max_retries - 1:
                wait = (2 ** attempt) + random.random()  # Exponential backoff
                time.sleep(wait)
            else:
                print(f"      [FAIL] {game_id}: {str(e)[:50]}", flush=True)
                return None


def fetch_season_box_scores(season: str, delay: float = 1.5) -> pd.DataFrame:
    """Fetch all box scores for a season with checkpointing."""
    print(f"\n[FETCHING] Season {season} box scores...", flush=True)
    
    # Check for existing partial data
    checkpoint_file = OUTPUT_DIR / f"box_scores_{season.replace('-', '_')}_partial.csv"
    fetched_ids = set()
    all_box_scores = []
    
    if checkpoint_file.exists():
        existing = pd.read_csv(checkpoint_file)
        fetched_ids = set(existing["GAME_ID"].unique())
        all_box_scores = existing.to_dict("records")
        print(f"  Resuming: {len(fetched_ids)} games already fetched", flush=True)
    
    # Get games
    games = fetch_season_games(season)
    
    # Get unique game IDs, skip already fetched
    game_ids = [gid for gid in games["GAME_ID"].unique() if gid not in fetched_ids]
    print(f"  Games to fetch: {len(game_ids)}", flush=True)
    
    # Fetch box scores
    errors = 0
    for i, game_id in enumerate(game_ids):
        if i % 25 == 0:
            print(f"    Progress: {i}/{len(game_ids)} (errors: {errors})", flush=True)
        
        # Random jitter to avoid patterns
        time.sleep(delay + random.random())
        
        box = fetch_box_score(game_id)
        if box:
            all_box_scores.extend(box)
            errors = 0  # Reset error count on success
        else:
            errors += 1
        
        # Save checkpoint every 100 games
        if (i + 1) % 100 == 0:
            df = pd.DataFrame(all_box_scores)
            df.to_csv(checkpoint_file, index=False)
            print(f"    [CHECKPOINT] Saved {len(df)} records", flush=True)
        
        # If too many consecutive errors, pause
        if errors >= 10:
            print(f"    [PAUSE] Too many errors, waiting 60s...", flush=True)
            time.sleep(60)
            errors = 0
    
    df = pd.DataFrame(all_box_scores)
    print(f"  Fetched {len(df)} box score records", flush=True)
    
    return df


def main():
    print("="*80, flush=True)
    print(" FETCHING NBA BOX SCORES FROM NBA.COM API", flush=True)
    print(" Using nba_api (no API key required)", flush=True)
    print(" With retry logic, checkpointing, and rate limit handling", flush=True)
    print("="*80, flush=True)
    
    seasons = ["2023-24", "2024-25"]
    
    for season in seasons:
        # Check if already fetched
        output_file = OUTPUT_DIR / f"box_scores_{season.replace('-', '_')}.csv"
        checkpoint_file = OUTPUT_DIR / f"box_scores_{season.replace('-', '_')}_partial.csv"
        
        if output_file.exists():
            df = pd.read_csv(output_file)
            print(f"\n[SKIP] {season} already fetched: {len(df)} records", flush=True)
            continue
        
        # Fetch
        df = fetch_season_box_scores(season)
        
        # Save final
        df.to_csv(output_file, index=False)
        print(f"  Saved to: {output_file}", flush=True)
        
        # Remove checkpoint
        if checkpoint_file.exists():
            checkpoint_file.unlink()
    
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
        else:
            checkpoint_file = OUTPUT_DIR / f"box_scores_{season.replace('-', '_')}_partial.csv"
            if checkpoint_file.exists():
                df = pd.read_csv(checkpoint_file)
                print(f"\n  {season} [PARTIAL]:", flush=True)
                print(f"    Records: {len(df)}", flush=True)


if __name__ == "__main__":
    main()
