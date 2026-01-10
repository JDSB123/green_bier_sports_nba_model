#!/usr/bin/env python3
"""
Fetch 2025-26 box scores using BoxScoreTraditionalV3.
V2 is deprecated for 2025-26 season.
"""
import sys
import time
import random
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nba_api.stats.endpoints import leaguegamefinder, boxscoretraditionalv3

OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "nba_api"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_box_score_v3(game_id: str, max_retries: int = 3) -> list:
    """Fetch box score using V3 API."""
    for attempt in range(max_retries):
        try:
            box = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id)
            data = box.get_dict()
            
            # V3 structure is different
            box_data = data.get("boxScoreTraditional", {})
            home = box_data.get("homeTeam", {})
            away = box_data.get("awayTeam", {})
            
            records = []
            for team in [home, away]:
                stats = team.get("statistics", {})
                if stats:
                    record = {
                        "GAME_ID": game_id,
                        "TEAM_ID": team.get("teamId"),
                        "TEAM_NAME": team.get("teamName"),
                        "TEAM_ABBREVIATION": team.get("teamTricode"),
                        "TEAM_CITY": team.get("teamCity"),
                        "FGM": stats.get("fieldGoalsMade"),
                        "FGA": stats.get("fieldGoalsAttempted"),
                        "FG_PCT": stats.get("fieldGoalsPercentage"),
                        "FG3M": stats.get("threePointersMade"),
                        "FG3A": stats.get("threePointersAttempted"),
                        "FG3_PCT": stats.get("threePointersPercentage"),
                        "FTM": stats.get("freeThrowsMade"),
                        "FTA": stats.get("freeThrowsAttempted"),
                        "FT_PCT": stats.get("freeThrowsPercentage"),
                        "OREB": stats.get("reboundsOffensive"),
                        "DREB": stats.get("reboundsDefensive"),
                        "REB": stats.get("reboundsTotal"),
                        "AST": stats.get("assists"),
                        "STL": stats.get("steals"),
                        "BLK": stats.get("blocks"),
                        "TO": stats.get("turnovers"),
                        "PF": stats.get("foulsPersonal"),
                        "PTS": stats.get("points"),
                        "PLUS_MINUS": stats.get("plusMinusPoints"),
                    }
                    records.append(record)
            
            return records if len(records) == 2 else None
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep((2 ** attempt) + random.random())
            else:
                print(f"  [FAIL] {game_id}: {str(e)[:50]}", flush=True)
                return None


def main():
    print("="*80, flush=True)
    print(" FETCHING 2025-26 BOX SCORES (using V3 API)", flush=True)
    print("="*80, flush=True)
    
    output_file = OUTPUT_DIR / "box_scores_2025_26.csv"
    
    # Get all game IDs
    print("\nFetching game list...", flush=True)
    gamefinder = leaguegamefinder.LeagueGameFinder(
        season_nullable="2025-26",
        league_id_nullable="00",
        season_type_nullable="Regular Season",
    )
    games = gamefinder.get_data_frames()[0]
    game_ids = list(games["GAME_ID"].unique())
    print(f"  Found {len(game_ids)} unique games", flush=True)
    
    # Fetch box scores
    all_records = []
    for i, gid in enumerate(game_ids):
        if i % 25 == 0:
            print(f"  Progress: {i}/{len(game_ids)}", flush=True)
        
        time.sleep(1.5 + random.random())
        records = fetch_box_score_v3(gid)
        if records:
            all_records.extend(records)
        
        # Checkpoint every 100
        if (i + 1) % 100 == 0:
            df = pd.DataFrame(all_records)
            df.to_csv(output_file.with_suffix(".partial.csv"), index=False)
    
    # Save final
    df = pd.DataFrame(all_records)
    df.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}", flush=True)
    print(f"Total: {len(df)} records, {df['GAME_ID'].nunique()} games", flush=True)


if __name__ == "__main__":
    main()
