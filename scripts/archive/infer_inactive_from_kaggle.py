#!/usr/bin/env python3
"""
Infer inactive players from Kaggle box scores to fill injury data gap.

Logic:
1. For each game, identify "active roster" = players who played in RECENT games for that team
2. If a player was on active roster but NOT in box score for a game = INACTIVE
3. Compute injury impact score based on player value (PPG, experience, draft status)

This fills the gap where nba_database inactive_players.csv only covers through June 2023.

Data source: eoinamoore/historical-nba-data-and-player-box-scores (Kaggle)
- PlayerStatistics.csv: Box scores for every player, every game (1947-present)
- Players.csv: Player biographical info (draft info, height, weight)
- Games.csv: All games with dates, teams, scores
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Set, Tuple, Optional

import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
KAGGLE_DIR = DATA_DIR / "external" / "kaggle_nba"
OUTPUT_DIR = DATA_DIR / "processed"

# Team name mappings (city -> abbreviation)
TEAM_CITY_TO_ABBR = {
    'Atlanta': 'ATL', 'Boston': 'BOS', 'Brooklyn': 'BKN', 'Charlotte': 'CHA',
    'Chicago': 'CHI', 'Cleveland': 'CLE', 'Dallas': 'DAL', 'Denver': 'DEN',
    'Detroit': 'DET', 'Golden State': 'GSW', 'Houston': 'HOU', 'Indiana': 'IND',
    'Los Angeles Clippers': 'LAC', 'Los Angeles Lakers': 'LAL', 'LA Clippers': 'LAC',
    'LA Lakers': 'LAL', 'Memphis': 'MEM', 'Miami': 'MIA', 'Milwaukee': 'MIL',
    'Minnesota': 'MIN', 'New Orleans': 'NOP', 'New York': 'NYK', 'Oklahoma City': 'OKC',
    'Orlando': 'ORL', 'Philadelphia': 'PHI', 'Phoenix': 'PHX', 'Portland': 'POR',
    'Sacramento': 'SAC', 'San Antonio': 'SAS', 'Toronto': 'TOR', 'Utah': 'UTA',
    'Washington': 'WAS',
}

# Map team ID to abbreviation (from NBA API)
TEAM_ID_TO_ABBR = {
    1610612737: 'ATL', 1610612738: 'BOS', 1610612751: 'BKN', 1610612766: 'CHA',
    1610612741: 'CHI', 1610612739: 'CLE', 1610612742: 'DAL', 1610612743: 'DEN',
    1610612765: 'DET', 1610612744: 'GSW', 1610612745: 'HOU', 1610612754: 'IND',
    1610612746: 'LAC', 1610612747: 'LAL', 1610612763: 'MEM', 1610612748: 'MIA',
    1610612749: 'MIL', 1610612750: 'MIN', 1610612740: 'NOP', 1610612752: 'NYK',
    1610612760: 'OKC', 1610612753: 'ORL', 1610612755: 'PHI', 1610612756: 'PHX',
    1610612757: 'POR', 1610612758: 'SAC', 1610612759: 'SAS', 1610612761: 'TOR',
    1610612762: 'UTA', 1610612764: 'WAS',
}


def city_to_abbr(city: str, name: str = '') -> str:
    """Convert team city/name to abbreviation."""
    if pd.isna(city):
        return ''
    
    city = str(city).strip()
    name = str(name).strip() if name else ''
    
    # Direct city match
    if city in TEAM_CITY_TO_ABBR:
        return TEAM_CITY_TO_ABBR[city]
    
    # LA special case
    if city == 'Los Angeles' or city == 'LA':
        if 'Clipper' in name:
            return 'LAC'
        elif 'Laker' in name:
            return 'LAL'
    
    # Try city + name combo
    combo = f"{city} {name}"
    for key, abbr in TEAM_CITY_TO_ABBR.items():
        if key in combo:
            return abbr
    
    return city[:3].upper()


def load_kaggle_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load Kaggle datasets."""
    print("Loading Kaggle data...")
    
    # Load with proper dtypes
    player_stats = pd.read_csv(
        KAGGLE_DIR / "PlayerStatistics.csv",
        dtype={'gameId': str, 'personId': str}
    )
    
    players = pd.read_csv(
        KAGGLE_DIR / "Players.csv",
        dtype={'personId': str}
    )
    
    games = pd.read_csv(
        KAGGLE_DIR / "Games.csv",
        dtype={'gameId': str, 'hometeamId': str, 'awayteamId': str}
    )
    
    print(f"  PlayerStatistics: {len(player_stats):,} records")
    print(f"  Players: {len(players):,} players")
    print(f"  Games: {len(games):,} games")
    
    # Parse dates
    player_stats['game_date'] = pd.to_datetime(player_stats['gameDateTimeEst'], errors='coerce')
    games['game_date'] = pd.to_datetime(games['gameDateTimeEst'], errors='coerce')
    
    # Add team abbreviations to player stats
    player_stats['team_abbr'] = player_stats.apply(
        lambda r: city_to_abbr(r.get('playerteamCity', ''), r.get('playerteamName', '')), 
        axis=1
    )
    
    # Add team abbreviations to games
    games['home_team_abbr'] = games.apply(
        lambda r: city_to_abbr(r.get('hometeamCity', ''), r.get('hometeamName', '')), 
        axis=1
    )
    games['away_team_abbr'] = games.apply(
        lambda r: city_to_abbr(r.get('awayteamCity', ''), r.get('awayteamName', '')), 
        axis=1
    )
    
    return player_stats, players, games


def build_player_value_index(player_stats: pd.DataFrame, players: pd.DataFrame) -> Dict[str, float]:
    """Build player value index based on recent PPG and career stats."""
    print("\nBuilding player value index...")
    
    # Filter to 2023+ for recent performance
    recent = player_stats[player_stats['game_date'] >= '2023-01-01'].copy()
    print(f"  Recent games (2023+): {len(recent):,} player-game records")
    
    # Calculate per-game stats
    player_ppg = recent.groupby('personId').agg({
        'points': 'mean',
        'numMinutes': 'mean'
    }).reset_index()
    player_ppg.columns = ['player_id', 'ppg', 'mpg']
    
    # Merge with player draft info
    player_ppg = player_ppg.merge(
        players[['personId', 'draftRound', 'draftNumber']].rename(columns={'personId': 'player_id'}),
        on='player_id',
        how='left'
    )
    
    # Compute value score (0-10 scale)
    # PPG component (max 5 points): 20+ PPG = 5 points
    player_ppg['ppg_score'] = np.clip(player_ppg['ppg'] / 4.0, 0, 5)
    
    # MPG component (max 2 points): 30+ MPG = 2 points (starters more valuable)
    player_ppg['mpg_score'] = np.clip(player_ppg['mpg'] / 15.0, 0, 2)
    
    # Draft component (max 3 points)
    # 1st round = 2 points, Top 5 = 2.5, #1 overall = 3
    player_ppg['draft_score'] = 0.0
    player_ppg.loc[player_ppg['draftRound'] == 1, 'draft_score'] = 2.0
    player_ppg.loc[(player_ppg['draftNumber'] <= 5) & (player_ppg['draftNumber'] > 0), 'draft_score'] = 2.5
    player_ppg.loc[player_ppg['draftNumber'] == 1, 'draft_score'] = 3.0
    
    # Total value score
    player_ppg['value_score'] = player_ppg['ppg_score'] + player_ppg['mpg_score'] + player_ppg['draft_score']
    player_ppg['value_score'] = np.clip(player_ppg['value_score'], 0, 10)
    
    value_index = dict(zip(player_ppg['player_id'].astype(str), player_ppg['value_score']))
    
    print(f"  Built value index for {len(value_index):,} players")
    print(f"  Value score range: {min(value_index.values()):.2f} - {max(value_index.values()):.2f}")
    
    # Show top players
    top_5 = sorted(value_index.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"  Top 5 by value: {top_5}")
    
    return value_index


def get_active_roster(player_stats: pd.DataFrame, team_abbr: str, game_date: datetime, 
                      lookback_days: int = 14) -> Set[str]:
    """Get 'active roster' = players who played in recent games for team."""
    cutoff = game_date - timedelta(days=lookback_days)
    
    mask = (
        (player_stats['team_abbr'] == team_abbr) &
        (player_stats['game_date'] >= cutoff) &
        (player_stats['game_date'] < game_date)
    )
    recent = player_stats[mask]
    
    return set(recent['personId'].astype(str).unique())


def process_games_for_inactive(player_stats: pd.DataFrame, games: pd.DataFrame,
                               value_index: Dict[str, float],
                               start_date: str = '2023-07-01') -> pd.DataFrame:
    """
    Process games to infer inactive players.
    
    Args:
        start_date: Start from this date (after nba_database coverage ends)
    
    Returns:
        DataFrame with inactive player records
    """
    print(f"\nProcessing games from {start_date}...")
    
    # Filter games
    games_filtered = games[games['game_date'] >= start_date].copy()
    print(f"  Games to process: {len(games_filtered):,}")
    
    inactive_records = []
    games_processed = 0
    
    for idx, game in games_filtered.iterrows():
        games_processed += 1
        if games_processed % 500 == 0:
            print(f"  Processed {games_processed}/{len(games_filtered)} games...")
        
        game_id = str(game['gameId'])
        game_date = game['game_date']
        home_team = game['home_team_abbr']
        away_team = game['away_team_abbr']
        
        if pd.isna(game_date) or not home_team or not away_team:
            continue
        
        # Get who actually played in this game
        game_mask = player_stats['gameId'] == game_id
        game_players = player_stats[game_mask]
        
        home_played = set(game_players[game_players['team_abbr'] == home_team]['personId'].astype(str))
        away_played = set(game_players[game_players['team_abbr'] == away_team]['personId'].astype(str))
        
        # Get active rosters (who should have been available)
        home_roster = get_active_roster(player_stats, home_team, game_date)
        away_roster = get_active_roster(player_stats, away_team, game_date)
        
        # Inactive = on recent roster but didn't play this game
        home_inactive = home_roster - home_played
        away_inactive = away_roster - away_played
        
        # Record inactive players
        for player_id in home_inactive:
            inactive_records.append({
                'game_id': game_id,
                'game_date': game_date.strftime('%Y-%m-%d'),
                'player_id': player_id,
                'team_abbreviation': home_team,
                'value_score': value_index.get(player_id, 0.0)
            })
        
        for player_id in away_inactive:
            inactive_records.append({
                'game_id': game_id,
                'game_date': game_date.strftime('%Y-%m-%d'),
                'player_id': player_id,
                'team_abbreviation': away_team,
                'value_score': value_index.get(player_id, 0.0)
            })
    
    result = pd.DataFrame(inactive_records)
    print(f"\n  Total inactive records: {len(result):,}")
    
    return result


def compute_game_injury_impact(inactive_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate inactive player data to per-game injury impact.
    
    Returns DataFrame with:
    - game_id, game_date
    - home_team, home_injury_impact
    - away_team, away_injury_impact
    """
    print("\nComputing per-game injury impact...")
    
    if len(inactive_df) == 0:
        return pd.DataFrame()
    
    # Group by game and team
    game_impact = inactive_df.groupby(['game_id', 'game_date', 'team_abbreviation']).agg({
        'value_score': 'sum',
        'player_id': 'count'
    }).reset_index()
    game_impact.columns = ['game_id', 'game_date', 'team', 'injury_impact', 'inactive_count']
    
    print(f"  Game-team records: {len(game_impact):,}")
    
    return game_impact


def main():
    print("="*60)
    print("INFER INACTIVE PLAYERS FROM KAGGLE BOX SCORES")
    print("="*60)
    print("Purpose: Fill injury data gap for 2023H2, 2024, 2025, 2026")
    print("="*60)
    
    # Check if Kaggle data exists
    if not (KAGGLE_DIR / "PlayerStatistics.csv").exists():
        print(f"\n✗ Kaggle data not found at {KAGGLE_DIR}")
        print("  Run: kaggle datasets download -d eoinamoore/historical-nba-data-and-player-box-scores")
        return 1
    
    # Load data
    player_stats, players, games = load_kaggle_data()
    
    # Build value index
    value_index = build_player_value_index(player_stats, players)
    
    # Process games (starting after nba_database coverage ends)
    inactive_df = process_games_for_inactive(
        player_stats, games, value_index,
        start_date='2023-07-01'  # nba_database ends ~June 2023
    )
    
    # Save inactive records
    output_file = OUTPUT_DIR / "inactive_players_kaggle_supplement.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    inactive_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved inactive records: {output_file}")
    
    # Compute and save game-level impact
    game_impact = compute_game_injury_impact(inactive_df)
    impact_file = OUTPUT_DIR / "injury_impact_by_game.csv"
    game_impact.to_csv(impact_file, index=False)
    print(f"✓ Saved game impact: {impact_file}")
    
    # Show summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total inactive records: {len(inactive_df):,}")
    if len(inactive_df) > 0:
        print(f"Unique games: {inactive_df['game_id'].nunique():,}")
        print(f"Unique players: {inactive_df['player_id'].nunique():,}")
        print(f"Date range: {inactive_df['game_date'].min()} to {inactive_df['game_date'].max()}")
        print(f"Avg value score: {inactive_df['value_score'].mean():.2f}")
        
        # By year
        inactive_df['year'] = pd.to_datetime(inactive_df['game_date']).dt.year
        by_year = inactive_df.groupby('year').agg({'game_id': 'nunique', 'player_id': 'count'})
        print(f"\nBy year:")
        for year, row in by_year.iterrows():
            print(f"  {year}: {row['game_id']} games, {row['player_id']} inactive records")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
