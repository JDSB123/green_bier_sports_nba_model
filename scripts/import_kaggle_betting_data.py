#!/usr/bin/env python3
"""
Import Kaggle NBA betting data for enhanced backtesting.

Reads: data/raw/kaggle/nba_2008-2025.csv (23K+ games with betting lines)
Outputs: data/processed/training_data_kaggle.csv

This data provides 17 seasons of historical betting data with:
- Full game scores + quarter scores
- Spreads, totals, moneylines
- Second half lines
- Regular season vs playoffs

Usage:
    python scripts/import_kaggle_betting_data.py
    python scripts/import_kaggle_betting_data.py --merge  # Merge with existing training_data.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings

# Kaggle data uses lowercase abbreviations - map to canonical team IDs
KAGGLE_TEAM_MAP = {
    # Standard abbreviations
    'atl': 'Atlanta Hawks',
    'bos': 'Boston Celtics',
    'bkn': 'Brooklyn Nets',
    'cha': 'Charlotte Hornets',
    'chi': 'Chicago Bulls',
    'cle': 'Cleveland Cavaliers',
    'dal': 'Dallas Mavericks',
    'den': 'Denver Nuggets',
    'det': 'Detroit Pistons',
    'gsw': 'Golden State Warriors',
    'hou': 'Houston Rockets',
    'ind': 'Indiana Pacers',
    'lac': 'Los Angeles Clippers',
    'lal': 'Los Angeles Lakers',
    'mem': 'Memphis Grizzlies',
    'mia': 'Miami Heat',
    'mil': 'Milwaukee Bucks',
    'min': 'Minnesota Timberwolves',
    'nop': 'New Orleans Pelicans',
    'nyk': 'New York Knicks',
    'okc': 'Oklahoma City Thunder',
    'orl': 'Orlando Magic',
    'phi': 'Philadelphia 76ers',
    'phx': 'Phoenix Suns',
    'por': 'Portland Trail Blazers',
    'sac': 'Sacramento Kings',
    'sas': 'San Antonio Spurs',
    'sa': 'San Antonio Spurs',  # Alternate
    'tor': 'Toronto Raptors',
    'uta': 'Utah Jazz',
    'utah': 'Utah Jazz',  # Alternate
    'was': 'Washington Wizards',
    # Historical teams / alternate names
    'nj': 'Brooklyn Nets',  # New Jersey Nets -> Brooklyn Nets
    'njn': 'Brooklyn Nets',
    'sea': 'Oklahoma City Thunder',  # Seattle SuperSonics -> OKC
    'van': 'Memphis Grizzlies',  # Vancouver Grizzlies -> Memphis
    'cha_old': 'Charlotte Hornets',  # Charlotte Bobcats
    'noh': 'New Orleans Pelicans',  # New Orleans Hornets
    'nok': 'New Orleans Pelicans',  # New Orleans/Oklahoma City Hornets
    'gs': 'Golden State Warriors',
    'no': 'New Orleans Pelicans',
    'ny': 'New York Knicks',
    'la': 'Los Angeles Lakers',
}


def map_team_name(abbrev: str) -> str:
    """Map Kaggle abbreviation to canonical team name."""
    abbrev_lower = abbrev.lower().strip()
    return KAGGLE_TEAM_MAP.get(abbrev_lower, abbrev)


def load_kaggle_data(path: Path) -> pd.DataFrame:
    """Load and validate Kaggle NBA betting data."""
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} games from Kaggle dataset")
    print(f"Seasons: {df['season'].min()} - {df['season'].max()}")
    return df


def transform_kaggle_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform Kaggle data to match training_data.csv format.
    
    Input columns:
        season, date, regular, playoffs, away, home, score_away, score_home,
        q1_away, q2_away, q3_away, q4_away, ot_away, q1_home, q2_home, q3_home, q4_home, ot_home,
        whos_favored, spread, total, moneyline_away, moneyline_home, h2_spread, h2_total
    
    Output columns (matching existing format):
        game_id, date, home_team, away_team, home_score, away_score,
        home_q1, home_q2, home_q3, home_q4, away_q1, away_q2, away_q3, away_q4,
        home_margin, total_score, home_1h, away_1h, home_1h_margin, total_1h,
        spread_line, total_line, moneyline_home, moneyline_away,
        spread_covered, went_over, home_win
    """
    out = pd.DataFrame()
    
    # Generate game IDs
    out['game_id'] = df.apply(
        lambda r: f"kaggle_{r['season']}_{r['date']}_{r['away']}_{r['home']}", axis=1
    )
    
    # Date
    out['date'] = pd.to_datetime(df['date'])
    
    # Team names (mapped to canonical)
    out['home_team'] = df['home'].apply(map_team_name)
    out['away_team'] = df['away'].apply(map_team_name)
    
    # Scores
    out['home_score'] = df['score_home'].astype(int)
    out['away_score'] = df['score_away'].astype(int)
    
    # Quarter scores
    out['home_q1'] = df['q1_home'].fillna(0).astype(int)
    out['home_q2'] = df['q2_home'].fillna(0).astype(int)
    out['home_q3'] = df['q3_home'].fillna(0).astype(int)
    out['home_q4'] = df['q4_home'].fillna(0).astype(int)
    out['away_q1'] = df['q1_away'].fillna(0).astype(int)
    out['away_q2'] = df['q2_away'].fillna(0).astype(int)
    out['away_q3'] = df['q3_away'].fillna(0).astype(int)
    out['away_q4'] = df['q4_away'].fillna(0).astype(int)
    
    # Calculated fields
    out['home_margin'] = out['home_score'] - out['away_score']
    out['total_score'] = out['home_score'] + out['away_score']
    
    # First half scores
    out['home_1h'] = out['home_q1'] + out['home_q2']
    out['away_1h'] = out['away_q1'] + out['away_q2']
    out['home_1h_margin'] = out['home_1h'] - out['away_1h']
    out['total_1h'] = out['home_1h'] + out['away_1h']
    
    # Betting lines (Kaggle spread is positive for home favorite)
    # Convert: if home is favored, spread should be negative
    out['spread_line'] = df.apply(
        lambda r: -r['spread'] if r['whos_favored'] == 'home' else r['spread'],
        axis=1
    )
    out['total_line'] = df['total']
    out['moneyline_home'] = df['moneyline_home']
    out['moneyline_away'] = df['moneyline_away']
    
    # First half lines (approximation: ~half of full game)
    out['fh_spread_line'] = out['spread_line'] / 2.0
    out['fh_total_line'] = df['h2_total']  # h2_total is actually 1H total in this dataset
    
    # Betting outcomes
    # Spread covered: home team beats the spread
    # If spread_line is -5.5, home needs to win by 6+ to cover
    out['spread_covered'] = (out['home_margin'] + out['spread_line'] > 0).astype(int)
    
    # Total went over
    out['went_over'] = (out['total_score'] > out['total_line']).astype(int)
    
    # Moneyline (home win)
    out['home_win'] = (out['home_margin'] > 0).astype(int)
    
    # First half outcomes
    out['fh_spread_covered'] = (out['home_1h_margin'] + out['fh_spread_line'] > 0).astype(int)
    out['fh_went_over'] = (out['total_1h'] > out['fh_total_line']).astype(int)
    out['fh_home_win'] = (out['home_1h_margin'] > 0).astype(int)
    
    # Metadata
    out['is_playoffs'] = df['playoffs'].astype(int)
    out['season'] = df['season']
    out['source'] = 'kaggle'
    
    # Sort by date
    out = out.sort_values('date').reset_index(drop=True)
    
    return out


def main():
    parser = argparse.ArgumentParser(description="Import Kaggle NBA betting data")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "kaggle" / "nba_2008-2025.csv",
        help="Path to Kaggle CSV file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(settings.data_processed_dir) / "training_data_kaggle.csv",
        help="Output path for processed data",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge with existing training_data.csv (deduplicates by date+teams)",
    )
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"[ERROR] Kaggle data not found at {args.input}")
        print("   Run: kaggle datasets download -d cviaxmiwnptr/nba-betting-data-october-2007-to-june-2024 -p data/raw/kaggle --unzip")
        return 1
    
    # Load and transform
    print("=" * 60)
    print("IMPORTING KAGGLE NBA BETTING DATA")
    print("=" * 60)
    
    raw_df = load_kaggle_data(args.input)
    df = transform_kaggle_data(raw_df)
    
    print(f"\nTransformed {len(df):,} games")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Teams: {df['home_team'].nunique()} unique")
    
    # Stats
    print("\n--- Betting Outcome Stats ---")
    print(f"Spread Cover Rate: {df['spread_covered'].mean():.1%}")
    print(f"Over Rate: {df['went_over'].mean():.1%}")
    print(f"Home Win Rate: {df['home_win'].mean():.1%}")
    
    # Merge with existing if requested
    if args.merge:
        existing_path = Path(settings.data_processed_dir) / "training_data.csv"
        if existing_path.exists():
            existing_df = pd.read_csv(existing_path)
            existing_df['date'] = pd.to_datetime(existing_df['date'])
            existing_df['source'] = 'api'
            
            # Add missing columns to existing data
            for col in df.columns:
                if col not in existing_df.columns:
                    existing_df[col] = None
            
            # Combine and deduplicate
            combined = pd.concat([df, existing_df], ignore_index=True)
            
            # Deduplicate by date + home_team + away_team (prefer API data)
            combined = combined.sort_values(['date', 'source'], ascending=[True, False])
            combined = combined.drop_duplicates(
                subset=['date', 'home_team', 'away_team'],
                keep='last'  # Keep API data when available
            )
            combined = combined.sort_values('date').reset_index(drop=True)
            
            print(f"\n--- Merged Dataset ---")
            print(f"Kaggle: {len(df):,} games")
            print(f"Existing: {len(existing_df):,} games")
            print(f"Combined (deduplicated): {len(combined):,} games")
            
            df = combined
    
    # Save
    df.to_csv(args.output, index=False)
    print(f"\n[OK] Saved to {args.output}")
    
    # Also save as enhanced training data for backtest
    enhanced_path = Path(settings.data_processed_dir) / "training_data_enhanced.csv"
    df.to_csv(enhanced_path, index=False)
    print(f"[OK] Also saved to {enhanced_path}")
    
    print("\n--- Next Steps ---")
    print("1. Run backtest with enhanced data:")
    print("   python scripts/backtest.py --data training_data_kaggle.csv")
    print("2. Or update training_data.csv:")
    print("   cp data/processed/training_data_kaggle.csv data/processed/training_data.csv")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
