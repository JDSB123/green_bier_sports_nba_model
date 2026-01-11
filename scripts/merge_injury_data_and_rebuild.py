#!/usr/bin/env python3
"""
Merge injury data from multiple sources and rebuild training data.

Sources:
1. nba_database inactive_players.csv (through June 2023)
2. Kaggle-inferred inactive players (July 2023 - present)

This creates a unified injury impact feature for training.
"""

import sys
from pathlib import Path
from typing import Dict

import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"


def load_nba_database_injury_data() -> pd.DataFrame:
    """Load injury data from nba_database inactive_players.csv."""
    # Try multiple possible locations
    possible_paths = [
        EXTERNAL_DIR / "nba_database" / "csv" / "inactive_players.csv",
        EXTERNAL_DIR / "nba_database" / "inactive_players.csv",
        DATA_DIR / "raw" / "inactive_players.csv",
    ]
    
    for path in possible_paths:
        if path.exists():
            print(f"  Loading nba_database: {path}")
            df = pd.read_csv(path)
            print(f"    {len(df):,} records")
            return df
    
    print("  ⚠️ nba_database inactive_players.csv not found")
    return pd.DataFrame()


def load_kaggle_injury_data() -> pd.DataFrame:
    """Load injury data inferred from Kaggle box scores."""
    path = PROCESSED_DIR / "inactive_players_kaggle_supplement.csv"
    if path.exists():
        print(f"  Loading Kaggle supplement: {path}")
        df = pd.read_csv(path)
        print(f"    {len(df):,} records")
        return df
    
    print("  ⚠️ Kaggle supplement not found")
    return pd.DataFrame()


def load_player_info() -> pd.DataFrame:
    """Load player info for value scoring."""
    # Try Kaggle players first
    kaggle_players = EXTERNAL_DIR / "kaggle_nba" / "Players.csv"
    if kaggle_players.exists():
        print(f"  Loading Kaggle players: {kaggle_players}")
        return pd.read_csv(kaggle_players, dtype={'personId': str})
    
    # Fallback to nba_database
    nba_players = EXTERNAL_DIR / "nba_database" / "csv" / "common_player_info.csv"
    if nba_players.exists():
        print(f"  Loading nba_database players: {nba_players}")
        return pd.read_csv(nba_players)
    
    return pd.DataFrame()


def compute_player_value_scores(inactive_df: pd.DataFrame, players_df: pd.DataFrame) -> Dict[str, float]:
    """Compute value scores for players missing scores."""
    if players_df.empty:
        return {}
    
    # Build value index from player info
    value_index = {}
    
    # Check for different column naming conventions
    id_col = next((c for c in players_df.columns if 'person' in c.lower() and 'id' in c.lower()), 
                  next((c for c in players_df.columns if 'player' in c.lower() and 'id' in c.lower()), None))
    
    if not id_col:
        return {}
    
    for _, row in players_df.iterrows():
        player_id = str(row[id_col])
        
        # Draft bonus
        draft_score = 0.0
        if 'draftRound' in row and row['draftRound'] == 1:
            draft_score = 2.0
        if 'draftNumber' in row and pd.notna(row['draftNumber']):
            if row['draftNumber'] <= 5:
                draft_score = 2.5
            if row['draftNumber'] == 1:
                draft_score = 3.0
        
        # Experience bonus (if available)
        exp_score = 0.0
        if 'season_exp' in row and pd.notna(row['season_exp']):
            exp_score = min(row['season_exp'] * 0.3, 2.0)
        
        value_index[player_id] = draft_score + exp_score
    
    return value_index


def merge_injury_sources() -> pd.DataFrame:
    """Merge injury data from all sources."""
    print("\n=== Loading Injury Data Sources ===")
    
    # Load sources
    nba_db_df = load_nba_database_injury_data()
    kaggle_df = load_kaggle_injury_data()
    
    dfs = []
    
    # Process nba_database data
    if not nba_db_df.empty:
        # Standardize columns
        nba_db_df = nba_db_df.rename(columns={
            'GAME_ID': 'game_id',
            'PLAYER_ID': 'player_id',
            'TEAM_ABBREVIATION': 'team_abbreviation'
        })
        
        # Add game_date if missing (join with games data)
        if 'game_date' not in nba_db_df.columns:
            nba_db_df['game_date'] = None
        
        # Add value_score if missing
        if 'value_score' not in nba_db_df.columns:
            nba_db_df['value_score'] = 3.0  # Default moderate value
        
        nba_db_df['source'] = 'nba_database'
        dfs.append(nba_db_df[['game_id', 'player_id', 'team_abbreviation', 'value_score', 'source']])
    
    # Process Kaggle data
    if not kaggle_df.empty:
        kaggle_df['source'] = 'kaggle'
        dfs.append(kaggle_df[['game_id', 'player_id', 'team_abbreviation', 'value_score', 'source']])
    
    if not dfs:
        print("⚠️ No injury data sources found!")
        return pd.DataFrame()
    
    # Combine
    merged = pd.concat(dfs, ignore_index=True)
    
    # Remove duplicates (prefer Kaggle as it has better value scores)
    merged = merged.drop_duplicates(subset=['game_id', 'player_id'], keep='last')
    
    print(f"\n=== Merged Injury Data ===")
    print(f"  Total records: {len(merged):,}")
    print(f"  Unique games: {merged['game_id'].nunique():,}")
    print(f"  By source:")
    for source, count in merged['source'].value_counts().items():
        print(f"    {source}: {count:,}")
    
    return merged


def compute_game_injury_impact(inactive_df: pd.DataFrame) -> pd.DataFrame:
    """Compute injury impact per game-team combination."""
    print("\n=== Computing Game Injury Impact ===")
    
    if inactive_df.empty:
        return pd.DataFrame()
    
    # Aggregate by game and team
    impact = inactive_df.groupby(['game_id', 'team_abbreviation']).agg({
        'value_score': 'sum',
        'player_id': 'count'
    }).reset_index()
    impact.columns = ['game_id', 'team', 'injury_impact', 'inactive_count']
    
    print(f"  {len(impact):,} game-team records")
    print(f"  Avg injury impact: {impact['injury_impact'].mean():.2f}")
    print(f"  Avg inactive count: {impact['inactive_count'].mean():.1f}")
    
    return impact


def update_training_data(training_file: Path, game_impact: pd.DataFrame) -> pd.DataFrame:
    """Update training data with injury impact features."""
    print(f"\n=== Updating Training Data ===")
    print(f"  Loading: {training_file}")
    
    df = pd.read_csv(training_file, low_memory=False)
    print(f"  {len(df):,} games")
    
    if game_impact.empty:
        print("  ⚠️ No game impact data, keeping existing injury features")
        return df
    
    # Load Kaggle games to build lookup keys
    kaggle_games_file = DATA_DIR / "external" / "kaggle_nba" / "Games.csv"
    if not kaggle_games_file.exists():
        print(f"  ⚠️ Kaggle games file not found: {kaggle_games_file}")
        return df
    
    kaggle_games = pd.read_csv(kaggle_games_file, dtype={'gameId': str}, low_memory=False)
    kaggle_games['date'] = pd.to_datetime(kaggle_games['gameDateTimeEst'], format='mixed', utc=True).dt.strftime('%Y-%m-%d')
    
    # Team name mapping
    TEAM_MAP = {
        'Atlanta': 'atlanta hawks', 'Boston': 'boston celtics', 'Brooklyn': 'brooklyn nets',
        'Charlotte': 'charlotte hornets', 'Chicago': 'chicago bulls', 'Cleveland': 'cleveland cavaliers',
        'Dallas': 'dallas mavericks', 'Denver': 'denver nuggets', 'Detroit': 'detroit pistons',
        'Golden State': 'golden state warriors', 'Houston': 'houston rockets', 'Indiana': 'indiana pacers',
        'Memphis': 'memphis grizzlies', 'Miami': 'miami heat', 'Milwaukee': 'milwaukee bucks',
        'Minnesota': 'minnesota timberwolves', 'New Orleans': 'new orleans pelicans', 'New York': 'new york knicks',
        'Oklahoma City': 'oklahoma city thunder', 'Orlando': 'orlando magic', 'Philadelphia': 'philadelphia 76ers',
        'Phoenix': 'phoenix suns', 'Portland': 'portland trail blazers', 'Sacramento': 'sacramento kings',
        'San Antonio': 'san antonio spurs', 'Toronto': 'toronto raptors', 'Utah': 'utah jazz', 
        'Washington': 'washington wizards', 'Los Angeles': 'los angeles lakers',
    }
    
    def get_team_name(city, name):
        city = str(city) if pd.notna(city) else ''
        name = str(name) if pd.notna(name) else ''
        if 'Los Angeles' in city or city == 'LA':
            if 'Clipper' in name:
                return 'los angeles clippers'
            return 'los angeles lakers'
        return TEAM_MAP.get(city, f"{city.lower()} {name.lower()}")
    
    # Build lookup from gameId to training key format (date_home_away)
    kaggle_games['home_name'] = kaggle_games.apply(lambda r: get_team_name(r['hometeamCity'], r.get('hometeamName', '')), axis=1)
    kaggle_games['away_name'] = kaggle_games.apply(lambda r: get_team_name(r['awayteamCity'], r.get('awayteamName', '')), axis=1)
    kaggle_games['training_key'] = kaggle_games['date'] + '_' + kaggle_games['home_name'] + '_' + kaggle_games['away_name']
    
    # Create gameId -> training_key mapping
    game_id_to_key = dict(zip(kaggle_games['gameId'].astype(str), kaggle_games['training_key']))
    game_id_to_home = dict(zip(kaggle_games['gameId'].astype(str), kaggle_games['home_name']))
    game_id_to_away = dict(zip(kaggle_games['gameId'].astype(str), kaggle_games['away_name']))
    
    # Build impact lookup keyed by training_key
    # game_impact has: game_id, game_date, team (abbr), injury_impact, inactive_count
    
    # Need to map team abbr to team name
    ABBR_TO_NAME = {
        'ATL': 'atlanta hawks', 'BOS': 'boston celtics', 'BKN': 'brooklyn nets',
        'CHA': 'charlotte hornets', 'CHI': 'chicago bulls', 'CLE': 'cleveland cavaliers',
        'DAL': 'dallas mavericks', 'DEN': 'denver nuggets', 'DET': 'detroit pistons',
        'GSW': 'golden state warriors', 'HOU': 'houston rockets', 'IND': 'indiana pacers',
        'LAC': 'los angeles clippers', 'LAL': 'los angeles lakers', 'MEM': 'memphis grizzlies',
        'MIA': 'miami heat', 'MIL': 'milwaukee bucks', 'MIN': 'minnesota timberwolves',
        'NOP': 'new orleans pelicans', 'NYK': 'new york knicks', 'OKC': 'oklahoma city thunder',
        'ORL': 'orlando magic', 'PHI': 'philadelphia 76ers', 'PHX': 'phoenix suns',
        'POR': 'portland trail blazers', 'SAC': 'sacramento kings', 'SAS': 'san antonio spurs',
        'TOR': 'toronto raptors', 'UTA': 'utah jazz', 'WAS': 'washington wizards',
    }
    
    # Build impact lookup: (training_key, 'home'|'away') -> impact
    impact_lookup = {}
    for _, row in game_impact.iterrows():
        game_id = str(row['game_id'])
        team_abbr = row['team']
        training_key = game_id_to_key.get(game_id, '')
        home_name = game_id_to_home.get(game_id, '')
        away_name = game_id_to_away.get(game_id, '')
        team_name = ABBR_TO_NAME.get(team_abbr, team_abbr.lower())
        
        if not training_key:
            continue
        
        # Determine if this is home or away
        if team_name == home_name:
            impact_lookup[(training_key, 'home')] = row['injury_impact']
        elif team_name == away_name:
            impact_lookup[(training_key, 'away')] = row['injury_impact']
    
    print(f"  Built impact lookup with {len(impact_lookup)} entries")
    
    # Update training data
    home_impacts = []
    away_impacts = []
    matched = 0
    
    for idx, row in df.iterrows():
        training_key = row.get('game_id', '')
        
        home_impact = impact_lookup.get((training_key, 'home'), 0.0)
        away_impact = impact_lookup.get((training_key, 'away'), 0.0)
        
        if (training_key, 'home') in impact_lookup or (training_key, 'away') in impact_lookup:
            matched += 1
        
        home_impacts.append(home_impact)
        away_impacts.append(away_impact)
    
    # Update columns
    df['home_injury_impact'] = home_impacts
    df['away_injury_impact'] = away_impacts
    
    # Add differential
    df['injury_impact_diff'] = df['home_injury_impact'] - df['away_injury_impact']
    
    print(f"  Matched {matched}/{len(df)} games ({100*matched/len(df):.1f}%)")
    print(f"  Home injury impact: mean={df['home_injury_impact'].mean():.2f}, max={df['home_injury_impact'].max():.2f}")
    print(f"  Away injury impact: mean={df['away_injury_impact'].mean():.2f}, max={df['away_injury_impact'].max():.2f}")
    
    return df


def main():
    print("="*60)
    print("MERGE INJURY DATA AND REBUILD TRAINING")
    print("="*60)
    
    # Merge injury sources
    merged_inactive = merge_injury_sources()
    
    # Save merged inactive players
    merged_file = PROCESSED_DIR / "inactive_players_merged.csv"
    merged_inactive.to_csv(merged_file, index=False)
    print(f"\n✓ Saved merged inactive players: {merged_file}")
    
    # Compute game-level impact
    game_impact = compute_game_injury_impact(merged_inactive)
    
    # Save game impact
    impact_file = PROCESSED_DIR / "injury_impact_merged.csv"
    game_impact.to_csv(impact_file, index=False)
    print(f"✓ Saved game impact: {impact_file}")
    
    # Update training data
    training_file = PROCESSED_DIR / "training_data_complete_2023.csv"
    if training_file.exists():
        updated_df = update_training_data(training_file, game_impact)
        
        # Save updated training data
        output_file = PROCESSED_DIR / "training_data_complete_2023_with_injuries.csv"
        updated_df.to_csv(output_file, index=False)
        print(f"✓ Saved updated training data: {output_file}")
        
        # Show coverage
        print("\n=== FINAL COVERAGE ===")
        total = len(updated_df)
        with_injury = ((updated_df['home_injury_impact'] > 0) | (updated_df['away_injury_impact'] > 0)).sum()
        print(f"Games with injury data: {with_injury}/{total} ({100*with_injury/total:.1f}%)")
        
        # By year - parse from game_id (format: YYYY-MM-DD_team_team)
        updated_df['year'] = updated_df['game_id'].apply(lambda x: int(str(x)[:4]) if pd.notna(x) else 0)
        by_year = updated_df.groupby('year').apply(
            lambda g: ((g['home_injury_impact'] > 0) | (g['away_injury_impact'] > 0)).sum()
        )
        total_by_year = updated_df.groupby('year').size()
        
        print("\nBy year:")
        for year in sorted(by_year.index):
            if year == 0:
                continue
            pct = 100 * by_year[year] / total_by_year[year] if total_by_year[year] > 0 else 0
            print(f"  {year}: {by_year[year]}/{total_by_year[year]} ({pct:.1f}%)")
    else:
        print(f"\n⚠️ Training file not found: {training_file}")
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
