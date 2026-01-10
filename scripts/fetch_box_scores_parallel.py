#!/usr/bin/env python3
"""
Parallel box score fetcher - fetches a specific range of games for a season.
Usage: python fetch_box_scores_parallel.py <season> <start_idx> <end_idx> <worker_id>
Example: python fetch_box_scores_parallel.py 2024-25 0 400 1
"""
from __future__ import annotations

import sys
import time
import random
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nba_api.stats.endpoints import leaguegamefinder, boxscoretraditionalv2
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "nba_api"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_box_score(game_id: str, max_retries: int = 3) -> dict:
    """Fetch box score with retry logic."""
    for attempt in range(max_retries):
        try:
            box = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id, timeout=60)
            team_stats = box.get_data_frames()[1]
            return team_stats.to_dict("records") if len(team_stats) > 0 else None
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep((2 ** attempt) + random.random())
            else:
                return None


def main():
    if len(sys.argv) != 5:
        print("Usage: python fetch_box_scores_parallel.py <season> <start_idx> <end_idx> <worker_id>")
        sys.exit(1)
    
    season = sys.argv[1]
    start_idx = int(sys.argv[2])
    end_idx = int(sys.argv[3])
    worker_id = sys.argv[4]
    
    print(f"[WORKER {worker_id}] Fetching {season} games {start_idx}-{end_idx}", flush=True)
    
    # Get all game IDs
    gamefinder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        league_id_nullable="00",
        season_type_nullable="Regular Season",
    )
    games = gamefinder.get_data_frames()[0]
    game_ids = list(games["GAME_ID"].unique())
    
    # Get our slice
    game_ids = game_ids[start_idx:end_idx]
    print(f"[WORKER {worker_id}] Processing {len(game_ids)} games", flush=True)
    
    output_file = OUTPUT_DIR / f"box_scores_{season.replace('-', '_')}_worker{worker_id}.csv"
    
    all_box = []
    for i, gid in enumerate(game_ids):
        if i % 25 == 0:
            print(f"[WORKER {worker_id}] Progress: {i}/{len(game_ids)}", flush=True)
        
        time.sleep(1.5 + random.random())  # Rate limit
        box = fetch_box_score(gid)
        if box:
            all_box.extend(box)
        
        # Checkpoint every 50
        if (i + 1) % 50 == 0:
            df = pd.DataFrame(all_box)
            df.to_csv(output_file, index=False)
    
    # Final save
    df = pd.DataFrame(all_box)
    df.to_csv(output_file, index=False)
    print(f"[WORKER {worker_id}] DONE - Saved {len(df)} records to {output_file}", flush=True)


if __name__ == "__main__":
    main()
