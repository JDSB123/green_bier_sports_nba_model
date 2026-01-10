#!/usr/bin/env python3
"""
Ingest the comprehensive NBA database from Kaggle (wyattowalsh/basketball).

This dataset contains:
- 60,000+ games back to 1946
- Quarter-by-quarter line scores (essential for 1H backtesting)
- Complete box scores per game
- Play-by-play data
- Player profiles and statistics
- Team information and history

Data source: https://www.kaggle.com/datasets/wyattowalsh/basketball
Updated daily via automated pipeline from stats.nba.com

Usage:
    pip install kagglehub
    python scripts/ingest_nba_database.py
    python scripts/ingest_nba_database.py --extract-linescores  # Extract Q1-Q4 scores
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "data" / "external" / "nba_database"


def download_dataset() -> Path:
    """Download the NBA database using kagglehub."""
    try:
        import kagglehub
    except ImportError:
        logger.error("kagglehub not installed. Run: pip install kagglehub")
        raise

    logger.info("Downloading wyattowalsh/basketball dataset...")
    path = kagglehub.dataset_download("wyattowalsh/basketball")
    logger.info(f"Downloaded to: {path}")
    return Path(path)


def list_tables(db_path: Path) -> list[str]:
    """List all tables in the SQLite database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    return tables


def get_table_info(db_path: Path, table_name: str) -> pd.DataFrame:
    """Get column info for a table."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"PRAGMA table_info({table_name})", conn)
    conn.close()
    return df


def extract_table(db_path: Path, table_name: str, limit: int | None = None) -> pd.DataFrame:
    """Extract a table from the SQLite database."""
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM {table_name}"
    if limit:
        query += f" LIMIT {limit}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def extract_line_scores(db_path: Path) -> pd.DataFrame:
    """
    Extract line scores (quarter-by-quarter) for all games.
    
    This is CRITICAL for 1H backtesting - provides actual Q1, Q2, Q3, Q4 scores.
    """
    conn = sqlite3.connect(db_path)
    
    # Try common table names for line scores
    possible_tables = ["line_score", "linescores", "game_line_score", "line_scores"]
    
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    all_tables = [row[0].lower() for row in cursor.fetchall()]
    
    # Find line score table
    line_score_table = None
    for table in possible_tables:
        if table in all_tables:
            line_score_table = table
            break
    
    # Also check for tables containing 'line' or 'score'
    if not line_score_table:
        for table in all_tables:
            if 'line' in table and 'score' in table:
                line_score_table = table
                break
    
    if not line_score_table:
        # List all tables to help debug
        logger.warning(f"No line_score table found. Available tables: {all_tables[:20]}...")
        conn.close()
        return pd.DataFrame()
    
    logger.info(f"Found line score table: {line_score_table}")
    df = pd.read_sql_query(f"SELECT * FROM {line_score_table}", conn)
    conn.close()
    
    logger.info(f"Extracted {len(df)} line score records")
    logger.info(f"Columns: {list(df.columns)}")
    
    return df


def extract_games(db_path: Path) -> pd.DataFrame:
    """Extract game data with dates and scores."""
    conn = sqlite3.connect(db_path)
    
    # Try common table names
    possible_tables = ["game", "games", "game_info", "game_summary"]
    
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    all_tables = [row[0].lower() for row in cursor.fetchall()]
    
    game_table = None
    for table in possible_tables:
        if table in all_tables:
            game_table = table
            break
    
    if not game_table:
        logger.warning(f"No game table found. Available tables: {all_tables[:20]}...")
        conn.close()
        return pd.DataFrame()
    
    logger.info(f"Found game table: {game_table}")
    df = pd.read_sql_query(f"SELECT * FROM {game_table}", conn)
    conn.close()
    
    logger.info(f"Extracted {len(df)} game records")
    return df


def analyze_database(db_path: Path) -> dict:
    """Analyze the database structure and contents."""
    logger.info(f"\n{'='*60}")
    logger.info("ANALYZING NBA DATABASE")
    logger.info(f"{'='*60}")
    
    tables = list_tables(db_path)
    logger.info(f"\nFound {len(tables)} tables:")
    
    analysis = {"tables": {}}
    
    for table in sorted(tables):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        conn.close()
        
        analysis["tables"][table] = count
        logger.info(f"  {table}: {count:,} rows")
    
    return analysis


def save_key_tables(db_path: Path, output_dir: Path):
    """Save key tables as CSV for easier access."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tables = list_tables(db_path)
    
    # Priority tables for NBA backtesting
    priority_tables = [
        "game", "games",
        "line_score", "linescores", "line_scores",
        "team", "teams",
        "player", "players",
        "game_summary",
        "other_stats",
    ]
    
    saved = []
    for table in tables:
        table_lower = table.lower()
        # Save priority tables or small reference tables
        if any(p in table_lower for p in priority_tables):
            df = extract_table(db_path, table)
            if len(df) > 0:
                output_path = output_dir / f"{table}.csv"
                df.to_csv(output_path, index=False)
                saved.append((table, len(df), output_path))
                logger.info(f"  Saved {table}: {len(df):,} rows -> {output_path.name}")
    
    return saved


def main():
    parser = argparse.ArgumentParser(description="Ingest NBA database from Kaggle")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Path to existing nba.sqlite file (skips download)"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze database structure"
    )
    parser.add_argument(
        "--extract-linescores",
        action="store_true",
        help="Extract line scores (Q1-Q4) for 1H backtesting"
    )
    parser.add_argument(
        "--extract-games",
        action="store_true",
        help="Extract game data"
    )
    parser.add_argument(
        "--save-tables",
        action="store_true",
        help="Save key tables as CSV"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("NBA DATABASE INGESTION (wyattowalsh/basketball)")
    print("=" * 60)
    
    # Download or use existing
    if args.db_path:
        db_path = args.db_path
        if not db_path.exists():
            logger.error(f"Database not found: {db_path}")
            return 1
    else:
        dataset_path = download_dataset()
        # Find the SQLite file
        sqlite_files = list(dataset_path.glob("*.sqlite")) + list(dataset_path.glob("**/*.sqlite"))
        if not sqlite_files:
            # Also check for .db files
            sqlite_files = list(dataset_path.glob("*.db")) + list(dataset_path.glob("**/*.db"))
        
        if not sqlite_files:
            logger.error(f"No SQLite database found in {dataset_path}")
            logger.info(f"Contents: {list(dataset_path.iterdir())}")
            return 1
        
        db_path = sqlite_files[0]
        logger.info(f"Using database: {db_path}")
    
    # Analyze
    if args.analyze or not any([args.extract_linescores, args.extract_games, args.save_tables]):
        analysis = analyze_database(db_path)
    
    # Extract line scores
    if args.extract_linescores:
        logger.info("\n[EXTRACTING LINE SCORES]")
        df = extract_line_scores(db_path)
        if len(df) > 0:
            output_path = OUTPUT_DIR / "line_scores.csv"
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f"Saved line scores to {output_path}")
    
    # Extract games
    if args.extract_games:
        logger.info("\n[EXTRACTING GAMES]")
        df = extract_games(db_path)
        if len(df) > 0:
            output_path = OUTPUT_DIR / "games.csv"
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f"Saved games to {output_path}")
    
    # Save key tables
    if args.save_tables:
        logger.info("\n[SAVING KEY TABLES]")
        save_key_tables(db_path, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"\nDatabase location: {db_path}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
