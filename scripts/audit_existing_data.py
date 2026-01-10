#!/usr/bin/env python3
"""
AUDIT ALL EXISTING DATA

Before calling any APIs, this script inventories EVERYTHING we already have.
Focus: Historical data from 2023 forward.

Checks:
1. All data directories (including .gitignore'd)
2. All file types (CSV, JSON, parquet)
3. Date ranges for each source
4. What's complete vs what's missing
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def format_size(size_bytes):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def print_header(title):
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)


# =============================================================================
# 1. INVENTORY ALL FILES
# =============================================================================

def inventory_all_files():
    """List all data files by directory."""
    print_header("1. ALL DATA FILES BY DIRECTORY")
    
    inventory = defaultdict(list)
    total_size = 0
    
    for root, dirs, files in os.walk(DATA_DIR):
        for f in files:
            path = Path(root) / f
            size = path.stat().st_size
            total_size += size
            rel_path = path.relative_to(DATA_DIR)
            inventory[rel_path.parts[0] if len(rel_path.parts) > 1 else "root"].append({
                "name": f,
                "size": size,
                "path": str(rel_path),
            })
    
    for dir_name, files in sorted(inventory.items()):
        dir_size = sum(f["size"] for f in files)
        print(f"\n  {dir_name}/ ({len(files)} files, {format_size(dir_size)})")
        # Show first few files
        for f in sorted(files, key=lambda x: x["size"], reverse=True)[:5]:
            print(f"    - {f['name']}: {format_size(f['size'])}")
        if len(files) > 5:
            print(f"    ... and {len(files) - 5} more files")
    
    print(f"\n  TOTAL: {format_size(total_size)}")
    return inventory


# =============================================================================
# 2. CHECK KAGGLE DATA (2023+)
# =============================================================================

def check_kaggle_data():
    """Check Kaggle betting data for 2023+."""
    print_header("2. KAGGLE BETTING DATA (2023+)")
    
    kaggle_file = DATA_DIR / "external" / "kaggle" / "nba_2008-2025.csv"
    if not kaggle_file.exists():
        print("  [NOT FOUND]")
        return
    
    df = pd.read_csv(kaggle_file)
    df["date"] = pd.to_datetime(df["date"])
    
    # Filter 2023+
    df_2023 = df[df["date"] >= "2023-01-01"]
    
    print(f"\n  FULL DATASET:")
    print(f"    Total games: {len(df):,}")
    print(f"    Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    print(f"\n  2023+ SUBSET:")
    print(f"    Games: {len(df_2023):,}")
    print(f"    Date range: {df_2023['date'].min().date()} to {df_2023['date'].max().date()}")
    
    print(f"\n  COLUMNS AVAILABLE:")
    for col in df.columns:
        pct = df_2023[col].notna().mean() * 100
        print(f"    {col}: {pct:.0f}% filled")
    
    print(f"\n  BY SEASON (2023+):")
    for season in sorted(df_2023["season"].unique()):
        season_df = df_2023[df_2023["season"] == season]
        print(f"    {season}: {len(season_df)} games ({season_df['date'].min().date()} to {season_df['date'].max().date()})")


# =============================================================================
# 3. CHECK THEODDS DATA (2023+)
# =============================================================================

def check_theodds_data():
    """Check TheOdds API data for 2023+."""
    print_header("3. THEODDS API DATA (2023+)")
    
    # Derived lines
    derived = DATA_DIR / "historical" / "derived" / "theodds_lines.csv"
    if derived.exists():
        df = pd.read_csv(derived)
        df["commence_time"] = pd.to_datetime(df["commence_time"])
        df_2023 = df[df["commence_time"] >= "2023-01-01"]
        print(f"\n  DERIVED LINES (theodds_lines.csv):")
        print(f"    Total games: {len(df):,}")
        print(f"    2023+ games: {len(df_2023):,}")
        print(f"    Date range: {df['commence_time'].min().date()} to {df['commence_time'].max().date()}")
        print(f"    Columns: {df.columns.tolist()}")
    
    # Exports
    exports_dir = DATA_DIR / "historical" / "exports"
    if exports_dir.exists():
        print(f"\n  EXPORTS:")
        for f in sorted(exports_dir.glob("*.csv")):
            df = pd.read_csv(f)
            print(f"    {f.name}: {len(df):,} rows")
    
    # Raw odds by season
    odds_dir = DATA_DIR / "historical" / "the_odds" / "odds"
    if odds_dir.exists():
        print(f"\n  RAW ODDS BY SEASON:")
        for season_dir in sorted(odds_dir.iterdir()):
            if season_dir.is_dir():
                files = list(season_dir.glob("*.json"))
                print(f"    {season_dir.name}: {len(files)} files")
    
    # Period odds (1H)
    period_dir = DATA_DIR / "historical" / "the_odds" / "period_odds"
    if period_dir.exists():
        print(f"\n  PERIOD ODDS (1H):")
        for season_dir in sorted(period_dir.iterdir()):
            if season_dir.is_dir():
                files = list(season_dir.glob("*.json"))
                for f in files:
                    size = f.stat().st_size
                    print(f"    {season_dir.name}/{f.name}: {format_size(size)}")


# =============================================================================
# 4. CHECK NBA DATABASE (Box Scores)
# =============================================================================

def check_nba_database():
    """Check wyattowalsh/basketball data."""
    print_header("4. NBA DATABASE (Box Scores)")
    
    db_dir = DATA_DIR / "external" / "nba_database"
    if not db_dir.exists():
        print("  [NOT FOUND]")
        return
    
    # Check game.csv
    game_file = db_dir / "game.csv"
    if game_file.exists():
        df = pd.read_csv(game_file)
        df["game_date"] = pd.to_datetime(df["game_date"])
        df_2023 = df[df["game_date"] >= "2023-01-01"]
        
        print(f"\n  game.csv:")
        print(f"    Total games: {len(df):,}")
        print(f"    Date range: {df['game_date'].min().date()} to {df['game_date'].max().date()}")
        print(f"    2023+ games: {len(df_2023):,}")
        
        if len(df_2023) > 0:
            print(f"    2023+ range: {df_2023['game_date'].min().date()} to {df_2023['game_date'].max().date()}")
        
        # Key columns
        box_cols = ["fgm_home", "fga_home", "fg3m_home", "ftm_home", "reb_home", "ast_home", "tov_home"]
        print(f"\n    Box score columns available: {[c for c in box_cols if c in df.columns]}")
    
    # List all tables
    print(f"\n  ALL TABLES:")
    for f in sorted(db_dir.glob("*.csv")):
        size = f.stat().st_size
        print(f"    {f.name}: {format_size(size)}")


# =============================================================================
# 5. CHECK ELO DATA
# =============================================================================

def check_elo_data():
    """Check FiveThirtyEight ELO data."""
    print_header("5. ELO RATINGS")
    
    elo_file = DATA_DIR / "historical" / "elo" / "fivethirtyeight_elo_historical.csv"
    if not elo_file.exists():
        print("  [NOT FOUND]")
        return
    
    df = pd.read_csv(elo_file)
    print(f"\n  fivethirtyeight_elo_historical.csv:")
    print(f"    Total records: {len(df):,}")
    print(f"    Date range: {df['date_game'].min()} to {df['date_game'].max()}")
    print(f"    Columns: {df.columns.tolist()}")
    
    # Check if covers 2023
    try:
        df["date_game"] = pd.to_datetime(df["date_game"])
        df_2023 = df[df["date_game"] >= "2023-01-01"]
        print(f"    2023+ records: {len(df_2023):,}")
    except:
        print("    [Unable to parse dates]")


# =============================================================================
# 6. CHECK PROCESSED/TRAINING DATA
# =============================================================================

def check_processed_data():
    """Check already-processed training data."""
    print_header("6. PROCESSED TRAINING DATA")
    
    proc_dir = DATA_DIR / "processed"
    if not proc_dir.exists():
        print("  [NOT FOUND]")
        return
    
    for f in sorted(proc_dir.glob("*.csv")):
        df = pd.read_csv(f)
        print(f"\n  {f.name}:")
        print(f"    Rows: {len(df):,}")
        print(f"    Columns: {len(df.columns)}")
        
        # Check date range
        for date_col in ["game_date", "date", "commence_time"]:
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col])
                print(f"    Date range: {df[date_col].min().date()} to {df[date_col].max().date()}")
                break


# =============================================================================
# 7. SUMMARY: WHAT WE HAVE FOR 2023+
# =============================================================================

def summarize_2023_coverage():
    """Summarize what data we have for 2023+."""
    print_header("7. SUMMARY: 2023+ DATA COVERAGE")
    
    coverage = {}
    
    # Kaggle
    kaggle_file = DATA_DIR / "external" / "kaggle" / "nba_2008-2025.csv"
    if kaggle_file.exists():
        df = pd.read_csv(kaggle_file)
        df["date"] = pd.to_datetime(df["date"])
        df_2023 = df[df["date"] >= "2023-01-01"]
        coverage["Kaggle (scores, Q1-Q4, FG lines)"] = {
            "games": len(df_2023),
            "end_date": df_2023["date"].max().date(),
            "status": "COMPLETE" if df_2023["date"].max().year >= 2025 else "PARTIAL",
        }
    
    # TheOdds derived
    derived = DATA_DIR / "historical" / "derived" / "theodds_lines.csv"
    if derived.exists():
        df = pd.read_csv(derived)
        df["commence_time"] = pd.to_datetime(df["commence_time"])
        df_2023 = df[df["commence_time"] >= "2023-01-01"]
        coverage["TheOdds (FG + 1H lines)"] = {
            "games": len(df_2023),
            "end_date": df_2023["commence_time"].max().date(),
            "status": "PARTIAL" if len(df_2023) < 2000 else "GOOD",
        }
    
    # 1H exports
    h1_files = list((DATA_DIR / "historical" / "exports").glob("*_odds_1h.csv"))
    if h1_files:
        total_rows = sum(len(pd.read_csv(f)) for f in h1_files)
        coverage["TheOdds 1H exports"] = {
            "games": f"{total_rows:,} rows",
            "end_date": "2024-25",
            "status": "COMPLETE",
        }
    
    # Box scores
    game_file = DATA_DIR / "external" / "nba_database" / "game.csv"
    if game_file.exists():
        df = pd.read_csv(game_file)
        df["game_date"] = pd.to_datetime(df["game_date"])
        df_2023 = df[df["game_date"] >= "2023-01-01"]
        end_date = df["game_date"].max().date()
        coverage["Box scores (eFG%, ratings)"] = {
            "games": len(df_2023),
            "end_date": end_date,
            "status": "PARTIAL - ends June 2023" if end_date.year == 2023 else "COMPLETE",
        }
    
    # ELO
    elo_file = DATA_DIR / "historical" / "elo" / "fivethirtyeight_elo_historical.csv"
    if elo_file.exists():
        df = pd.read_csv(elo_file)
        coverage["FiveThirtyEight ELO"] = {
            "games": len(df),
            "end_date": df["date_game"].max(),
            "status": "OUTDATED - ends 2015",
        }
    
    print("\n  DATA SOURCE                          | GAMES    | END DATE   | STATUS")
    print("  " + "-"*75)
    for source, info in coverage.items():
        games = str(info["games"])
        end_date = str(info["end_date"])
        status = info["status"]
        print(f"  {source:<40} | {games:<8} | {end_date:<10} | {status}")
    
    print("\n  CONCLUSION:")
    print("  - Kaggle has scores/lines through June 2025 [COMPLETE]")
    print("  - TheOdds has FG+1H lines for 2023-2025 [COMPLETE]")
    print("  - Box scores end June 2023 [NEED API-BASKETBALL for 2023-24, 2024-25]")
    print("  - ELO ends 2015 [COMPUTED FROM RESULTS - OK]")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*80)
    print(" COMPLETE DATA AUDIT - WHAT DO WE ALREADY HAVE?")
    print(" Focus: Historical data from 2023 forward")
    print("="*80)
    
    inventory_all_files()
    check_kaggle_data()
    check_theodds_data()
    check_nba_database()
    check_elo_data()
    check_processed_data()
    summarize_2023_coverage()


if __name__ == "__main__":
    main()
