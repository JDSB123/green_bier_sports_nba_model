#!/usr/bin/env python3
"""
Fetch quarter-by-quarter scores for 2025-26 season from NBA API.

Uses BoxScoreSummaryV3 which has period1Score, period2Score, etc.
"""
from __future__ import annotations

import sys
import time
import random
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nba_api.stats.endpoints import leaguegamefinder, boxscoresummaryv3
from src.data.standardization import standardize_team_name

OUTPUT_FILE = PROJECT_ROOT / "data" / "raw" / "nba_api" / "quarter_scores_2025_26.csv"


def fetch_season_games(season: str = "2025-26") -> pd.DataFrame:
    """Fetch all games for 2025-26 season."""
    print(f"[1/2] Fetching game list for {season}...")
    
    gamefinder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        league_id_nullable="00",
        season_type_nullable="Regular Season",
    )
    
    games = gamefinder.get_data_frames()[0]
    game_ids = games["GAME_ID"].unique()
    print(f"      Found {len(game_ids)} unique games")
    return game_ids


def fetch_quarter_scores(game_id: str, max_retries: int = 3) -> dict:
    """Fetch quarter scores for a single game."""
    for attempt in range(max_retries):
        try:
            box = boxscoresummaryv3.BoxScoreSummaryV3(game_id=game_id, timeout=60)
            dfs = box.get_data_frames()
            
            # DataFrame 4 has team line scores
            if len(dfs) > 4:
                line_score = dfs[4]
                if len(line_score) == 2:  # Home and away
                    game_info = dfs[0].iloc[0] if len(dfs[0]) > 0 else {}
                    
                    home_row = line_score[line_score["teamId"] == game_info.get("homeTeamId")].iloc[0] if "homeTeamId" in game_info else line_score.iloc[1]
                    away_row = line_score[line_score["teamId"] == game_info.get("awayTeamId")].iloc[0] if "awayTeamId" in game_info else line_score.iloc[0]
                    
                    return {
                        "game_id": game_id,
                        "game_date": game_info.get("gameTimeUTC", ""),
                        "home_team": standardize_team_name(home_row.get("teamName", "")),
                        "away_team": standardize_team_name(away_row.get("teamName", "")),
                        "home_q1": home_row.get("period1Score"),
                        "home_q2": home_row.get("period2Score"),
                        "home_q3": home_row.get("period3Score"),
                        "home_q4": home_row.get("period4Score"),
                        "home_score": home_row.get("score"),
                        "away_q1": away_row.get("period1Score"),
                        "away_q2": away_row.get("period2Score"),
                        "away_q3": away_row.get("period3Score"),
                        "away_q4": away_row.get("period4Score"),
                        "away_score": away_row.get("score"),
                    }
            return None
        except Exception as e:
            if attempt < max_retries - 1:
                wait = (2 ** attempt) + random.random()
                time.sleep(wait)
            else:
                return None


def main():
    print("="*60)
    print("FETCHING 2025-26 QUARTER SCORES")
    print("="*60)
    
    # Check for existing partial
    existing = set()
    all_scores = []
    if OUTPUT_FILE.exists():
        df_existing = pd.read_csv(OUTPUT_FILE)
        existing = set(df_existing["game_id"].astype(str))
        all_scores = df_existing.to_dict("records")
        print(f"      Resuming: {len(existing)} games already fetched")
    
    game_ids = fetch_season_games("2025-26")
    to_fetch = [gid for gid in game_ids if str(gid) not in existing]
    
    print(f"\n[2/2] Fetching quarter scores for {len(to_fetch)} games...")
    
    errors = 0
    for i, game_id in enumerate(to_fetch):
        if i % 25 == 0:
            print(f"      Progress: {i}/{len(to_fetch)} (errors: {errors})")
            # Save checkpoint
            if all_scores:
                pd.DataFrame(all_scores).to_csv(OUTPUT_FILE, index=False)
        
        result = fetch_quarter_scores(game_id)
        if result:
            all_scores.append(result)
        else:
            errors += 1
        
        # Rate limit
        time.sleep(0.8 + random.random() * 0.4)
    
    # Save final
    df = pd.DataFrame(all_scores)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n      Saved {len(df)} games to {OUTPUT_FILE}")
    
    # Compute 1H totals
    df["home_1h"] = df["home_q1"].fillna(0) + df["home_q2"].fillna(0)
    df["away_1h"] = df["away_q1"].fillna(0) + df["away_q2"].fillna(0)
    df["1h_total_actual"] = df["home_1h"] + df["away_1h"]
    df["fg_total_actual"] = df["home_score"] + df["away_score"]
    
    print(f"\n      1H totals computed: {df['1h_total_actual'].notna().sum()}/{len(df)}")
    print(f"      Sample:")
    print(df[["game_date", "home_team", "away_team", "home_1h", "away_1h", "1h_total_actual"]].head(5))


if __name__ == "__main__":
    main()
