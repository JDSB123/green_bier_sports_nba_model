#!/usr/bin/env python3
"""
Fetch player box scores and compute inactive players for recent seasons.

The wyattowalsh/basketball dataset on Kaggle is stale (last updated June 2023).
This script fetches player-level box scores from NBA API for 2023-24, 2024-25, 2025-26
and computes inactive players (on roster but DNP).

Output:
    data/raw/nba_api/player_box_scores_YYYY_YY.csv
    data/raw/nba_api/inactive_players_YYYY_YY.csv
"""
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nba_api.stats.endpoints import (
    leaguegamefinder,
    boxscoretraditionalv3,
    commonteamroster,
)
from nba_api.stats.static import teams

OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "nba_api"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Rate limiting
REQUEST_DELAY = 0.6  # seconds between requests


def get_all_teams() -> List[Dict]:
    """Get all NBA teams."""
    return teams.get_teams()


def get_team_roster(team_id: str, season: str) -> pd.DataFrame:
    """Get roster for a team in a season."""
    try:
        time.sleep(REQUEST_DELAY)
        roster = commonteamroster.CommonTeamRoster(team_id=team_id, season=season)
        df = roster.get_data_frames()[0]
        return df
    except Exception as e:
        print(f"    Error getting roster for {team_id}: {e}")
        return pd.DataFrame()


def get_season_games(season: str) -> pd.DataFrame:
    """Get all games for a season."""
    print(f"  Fetching games for {season}...")
    time.sleep(REQUEST_DELAY)
    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        league_id_nullable="00",  # NBA
        season_type_nullable="Regular Season"
    )
    games = finder.get_data_frames()[0]
    # Get unique game IDs
    game_ids = games["GAME_ID"].unique()
    print(f"    Found {len(game_ids)} games")
    return games


def get_player_box_score(game_id: str) -> pd.DataFrame:
    """Get player box score for a game using V3 endpoint."""
    try:
        time.sleep(REQUEST_DELAY)
        box = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id)
        dfs = box.get_data_frames()
        if len(dfs) > 0 and len(dfs[0]) > 0:
            return dfs[0]  # Player stats
        return pd.DataFrame()
    except Exception as e:
        # V3 may not work for all games, try different approach
        return pd.DataFrame()


def fetch_season_player_data(season: str, max_games: int = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch player box scores and compute inactive players for a season.
    
    Returns:
        (player_box_scores_df, inactive_players_df)
    """
    print(f"\n{'='*60}")
    print(f"FETCHING PLAYER DATA FOR {season}")
    print(f"{'='*60}")
    
    # Get all games
    games_df = get_season_games(season)
    if games_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    game_ids = games_df["GAME_ID"].unique()
    if max_games:
        game_ids = game_ids[:max_games]
    
    # Get all team rosters for the season
    print(f"  Fetching team rosters...")
    all_teams = get_all_teams()
    rosters = {}
    for team in all_teams:
        roster_df = get_team_roster(str(team["id"]), season)
        if not roster_df.empty:
            rosters[team["id"]] = set(roster_df["PLAYER_ID"].tolist())
    print(f"    Got rosters for {len(rosters)} teams")
    
    # Fetch player box scores
    print(f"  Fetching player box scores for {len(game_ids)} games...")
    all_player_stats = []
    all_inactive = []
    
    for i, game_id in enumerate(game_ids):
        if i % 50 == 0:
            print(f"    Processing game {i+1}/{len(game_ids)}...")
        
        player_df = get_player_box_score(game_id)
        if not player_df.empty:
            player_df["GAME_ID"] = game_id
            all_player_stats.append(player_df)
            
            # Compute inactive: players on roster but not in box score
            game_info = games_df[games_df["GAME_ID"] == game_id].iloc[0]
            team_id = game_info["TEAM_ID"]
            
            if team_id in rosters:
                players_in_game = set(player_df[player_df["teamId"] == team_id]["personId"].tolist()) if "personId" in player_df.columns else set()
                inactive_players = rosters[team_id] - players_in_game
                
                for player_id in inactive_players:
                    all_inactive.append({
                        "game_id": game_id,
                        "player_id": player_id,
                        "team_id": team_id,
                        "season": season
                    })
    
    # Combine results
    player_box_df = pd.concat(all_player_stats, ignore_index=True) if all_player_stats else pd.DataFrame()
    inactive_df = pd.DataFrame(all_inactive) if all_inactive else pd.DataFrame()
    
    print(f"  Results: {len(player_box_df)} player stats, {len(inactive_df)} inactive records")
    
    return player_box_df, inactive_df


def main():
    """Fetch player data for recent seasons."""
    seasons = ["2023-24", "2024-25", "2025-26"]
    
    for season in seasons:
        season_tag = season.replace("-", "_")
        
        # Check if we already have this data
        player_file = OUTPUT_DIR / f"player_box_scores_{season_tag}.csv"
        inactive_file = OUTPUT_DIR / f"inactive_players_{season_tag}.csv"
        
        if player_file.exists() and inactive_file.exists():
            print(f"\n{season}: Data already exists, skipping...")
            continue
        
        player_df, inactive_df = fetch_season_player_data(season, max_games=100)  # Limit for testing
        
        if not player_df.empty:
            player_df.to_csv(player_file, index=False)
            print(f"  Saved: {player_file}")
        
        if not inactive_df.empty:
            inactive_df.to_csv(inactive_file, index=False)
            print(f"  Saved: {inactive_file}")
    
    print("\n" + "="*60)
    print("DONE")
    print("="*60)


if __name__ == "__main__":
    main()
