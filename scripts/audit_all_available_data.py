#!/usr/bin/env python3
"""
COMPREHENSIVE DATA AUDIT

Inventories ALL available historical data across all sources and identifies:
1. What data EXISTS
2. What data we are USING
3. What data we are MISSING/NOT USING
4. Recommendations for maximizing feature extraction

This is NOT about checking formatting - it's about ensuring we leverage EVERYTHING.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"


def format_size(size_bytes):
    """Format bytes to human readable."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def print_section(title):
    print(f"\n{'='*80}")
    print(f" {title}")
    print('='*80)


def get_dir_size(path):
    """Get total size of directory."""
    total = 0
    for p in path.rglob('*'):
        if p.is_file():
            total += p.stat().st_size
    return total


# =============================================================================
# 1. INVENTORY ALL DATA SOURCES
# =============================================================================

def inventory_kaggle():
    """Audit Kaggle data."""
    print_section("1. KAGGLE DATA (nba_2008-2025.csv)")
    
    kaggle_file = DATA_DIR / "external" / "kaggle" / "nba_2008-2025.csv"
    if not kaggle_file.exists():
        print("  [NOT FOUND]")
        return {}
    
    df = pd.read_csv(kaggle_file)
    print(f"  File size: {format_size(kaggle_file.stat().st_size)}")
    print(f"  Games: {len(df):,}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Columns: {len(df.columns)}")
    
    # Categorize columns
    cols = df.columns.tolist()
    score_cols = [c for c in cols if 'score' in c.lower() or c.startswith('q') or c.startswith('ot')]
    line_cols = [c for c in cols if 'spread' in c.lower() or 'total' in c.lower() or 'moneyline' in c.lower()]
    team_cols = [c for c in cols if 'home' in c.lower() or 'away' in c.lower() and 'score' not in c.lower()]
    
    print(f"\n  AVAILABLE DATA:")
    print(f"    Score columns ({len(score_cols)}): {score_cols}")
    print(f"    Line columns ({len(line_cols)}): {line_cols}")
    print(f"    Team columns ({len(team_cols)}): {team_cols}")
    
    # Check data completeness
    print(f"\n  DATA COMPLETENESS:")
    for col in cols:
        pct = df[col].notna().mean() * 100
        if pct < 100:
            print(f"    {col}: {pct:.1f}% filled")
    
    return {
        "games": len(df),
        "columns": cols,
        "date_range": (df['date'].min(), df['date'].max()),
    }


def inventory_theodds():
    """Audit The Odds API data."""
    print_section("2. THE ODDS API DATA")
    
    theodds_dir = DATA_DIR / "historical" / "the_odds"
    if not theodds_dir.exists():
        print("  [NOT FOUND]")
        return {}
    
    # FG odds
    odds_dir = theodds_dir / "odds"
    events_dir = theodds_dir / "events"
    period_dir = theodds_dir / "period_odds"
    
    result = {}
    
    # Count FG odds
    if odds_dir.exists():
        fg_files = list(odds_dir.rglob("*.json"))
        fg_size = get_dir_size(odds_dir)
        print(f"\n  FG ODDS (Full Game):")
        print(f"    Files: {len(fg_files)}")
        print(f"    Size: {format_size(fg_size)}")
        
        # Sample to see structure
        if fg_files:
            with open(fg_files[0]) as f:
                sample = json.load(f)
            events = sample if isinstance(sample, list) else sample.get("data", [])
            if events:
                e = events[0]
                bookmakers = e.get("bookmakers", [])
                markets = set()
                for bm in bookmakers:
                    for mkt in bm.get("markets", []):
                        markets.add(mkt.get("key", ""))
                print(f"    Markets available: {sorted(markets)}")
                print(f"    Bookmakers per game: ~{len(bookmakers)}")
        
        result["fg_files"] = len(fg_files)
    
    # Count 1H/Q1 odds
    if period_dir.exists():
        h1_files = list(period_dir.rglob("*1h*.json"))
        q1_files = list(period_dir.rglob("*q1*.json"))
        period_size = get_dir_size(period_dir)
        print(f"\n  PERIOD ODDS (1H/Q1):")
        print(f"    1H files: {len(h1_files)}")
        print(f"    Q1 files: {len(q1_files)}")
        print(f"    Size: {format_size(period_size)}")
        
        if h1_files:
            with open(h1_files[0]) as f:
                sample = json.load(f)
            events = sample.get("data", [])
            if events:
                e = events[0].get("data", events[0])
                bookmakers = e.get("bookmakers", [])
                markets = set()
                for bm in bookmakers:
                    for mkt in bm.get("markets", []):
                        markets.add(mkt.get("key", ""))
                print(f"    1H Markets: {sorted(markets)}")
        
        result["h1_files"] = len(h1_files)
        result["q1_files"] = len(q1_files)
    
    # Events data
    if events_dir.exists():
        event_files = list(events_dir.rglob("*.json"))
        print(f"\n  EVENTS (game metadata):")
        print(f"    Files: {len(event_files)}")
        result["event_files"] = len(event_files)
    
    return result


def inventory_elo():
    """Audit FiveThirtyEight ELO data."""
    print_section("3. FIVETHIRTYEIGHT ELO DATA")
    
    elo_dir = DATA_DIR / "raw" / "github"
    if not elo_dir.exists():
        print("  [NOT FOUND]")
        return {}
    
    elo_files = list(elo_dir.rglob("*elo*.csv"))
    for f in elo_files:
        print(f"\n  {f.name}:")
        print(f"    Size: {format_size(f.stat().st_size)}")
        df = pd.read_csv(f)
        print(f"    Records: {len(df):,}")
        print(f"    Columns: {df.columns.tolist()}")
        if 'date' in df.columns or 'date_game' in df.columns:
            date_col = 'date' if 'date' in df.columns else 'date_game'
            print(f"    Date range: {df[date_col].min()} to {df[date_col].max()}")
    
    return {"files": len(elo_files)}


def inventory_nba_database():
    """Audit wyattowalsh/basketball dataset."""
    print_section("4. WYATTOWALSH/BASKETBALL DATABASE")
    
    db_dir = DATA_DIR / "external" / "nba_database"
    if not db_dir.exists():
        print("  [NOT FOUND] - Consider running: python scripts/ingest_nba_database.py")
        return {}
    
    # Check for SQLite database
    db_files = list(db_dir.rglob("*.sqlite"))
    csv_files = list(db_dir.rglob("*.csv"))
    
    print(f"  SQLite databases: {len(db_files)}")
    print(f"  CSV extracts: {len(csv_files)}")
    print(f"  Total size: {format_size(get_dir_size(db_dir))}")
    
    # List available tables/CSVs
    if csv_files:
        print(f"\n  AVAILABLE DATA TABLES:")
        for csv_file in sorted(csv_files)[:20]:
            df = pd.read_csv(csv_file, nrows=5)
            print(f"    {csv_file.name}: {len(df.columns)} columns")
    
    return {"db_files": len(db_files), "csv_files": len(csv_files)}


def inventory_api_basketball():
    """Audit API-Basketball data."""
    print_section("5. API-BASKETBALL DATA")
    
    api_dir = DATA_DIR / "raw" / "api_basketball"
    if not api_dir.exists():
        print("  [NOT FOUND]")
        return {}
    
    subdirs = [d for d in api_dir.iterdir() if d.is_dir()]
    total_files = 0
    
    for subdir in subdirs:
        files = list(subdir.rglob("*.json"))
        total_files += len(files)
        if files:
            print(f"\n  {subdir.name}/:")
            print(f"    Files: {len(files)}")
            
            # Sample structure
            with open(files[0]) as f:
                sample = json.load(f)
            if isinstance(sample, dict):
                print(f"    Keys: {list(sample.keys())[:10]}")
    
    return {"files": total_files}


def inventory_processed():
    """Audit processed/training data."""
    print_section("6. PROCESSED/TRAINING DATA (Current)")
    
    proc_dir = DATA_DIR / "processed"
    if not proc_dir.exists():
        print("  [NOT FOUND]")
        return {}
    
    csv_files = list(proc_dir.glob("*.csv"))
    for f in sorted(csv_files):
        print(f"\n  {f.name}:")
        print(f"    Size: {format_size(f.stat().st_size)}")
        df = pd.read_csv(f, nrows=1)
        print(f"    Columns: {len(df.columns)}")
        
        # Read full to get row count
        df_full = pd.read_csv(f)
        print(f"    Rows: {len(df_full):,}")
    
    return {"files": len(csv_files)}


# =============================================================================
# 2. ANALYZE WHAT WE'RE USING VS NOT USING
# =============================================================================

def analyze_feature_utilization():
    """Analyze which features we're actually using."""
    print_section("7. FEATURE UTILIZATION ANALYSIS")
    
    training_file = DATA_DIR / "processed" / "training_data_complete_2023.csv"
    if not training_file.exists():
        print("  [Training data not found]")
        return
    
    df = pd.read_csv(training_file)
    
    # Categorize columns
    categories = {
        "identifiers": [],
        "betting_lines": [],
        "scores": [],
        "rolling_stats": [],
        "rest_features": [],
        "streak_features": [],
        "elo_features": [],
        "h2h_features": [],
        "temporal": [],
        "implied_prob": [],
        "labels": [],
        "other": [],
    }
    
    for col in df.columns:
        col_lower = col.lower()
        if col in ["match_key", "date_str", "home_team", "away_team", "game_date"]:
            categories["identifiers"].append(col)
        elif "spread_line" in col_lower or "total_line" in col_lower or "ml_" in col_lower:
            categories["betting_lines"].append(col)
        elif "score" in col_lower or "_q" in col_lower or "margin" in col_lower or "total_actual" in col_lower:
            categories["scores"].append(col)
        elif "rolling" in col_lower or "_ppg" in col_lower or "_papg" in col_lower or "_win_pct" in col_lower:
            categories["rolling_stats"].append(col)
        elif "rest" in col_lower or "b2b" in col_lower:
            categories["rest_features"].append(col)
        elif "streak" in col_lower:
            categories["streak_features"].append(col)
        elif "elo" in col_lower:
            categories["elo_features"].append(col)
        elif "h2h" in col_lower:
            categories["h2h_features"].append(col)
        elif col in ["season", "day_of_week", "month", "season_progress", "is_playoffs"]:
            categories["temporal"].append(col)
        elif "implied" in col_lower or "_prob" in col_lower:
            categories["implied_prob"].append(col)
        elif "covered" in col_lower or "over" in col_lower or "win" in col_lower:
            categories["labels"].append(col)
        else:
            categories["other"].append(col)
    
    print("\n  FEATURES BY CATEGORY:")
    for cat, cols in categories.items():
        if cols:
            print(f"\n    {cat.upper()} ({len(cols)}):")
            for c in cols[:10]:
                pct = df[c].notna().mean() * 100
                print(f"      - {c}: {pct:.0f}% filled")
            if len(cols) > 10:
                print(f"      ... and {len(cols) - 10} more")


def identify_missing_features():
    """Identify features we COULD compute but aren't."""
    print_section("8. MISSING FEATURES (Opportunities)")
    
    print("""
  FEATURES WE SHOULD ADD:
  
  FROM KAGGLE (Q1-Q4 Scores):
    - Q1 spread/total outcomes (for Q1 model)
    - Q2 spread/total outcomes
    - 3Q spread/total outcomes
    - Q4 spread/total outcomes
    - Scoring pace by quarter
    - Quarter-by-quarter trends
    
  FROM BOX SCORES (if available):
    - Team offensive/defensive ratings
    - Pace (possessions per game)
    - Four Factors: eFG%, TOV%, ORB%, FT rate
    - Assist-to-turnover ratio
    - Rebounding differential
    - 3-point attempt rate
    - Free throw rate
    
  FROM PLAY-BY-PLAY (if available):
    - Clutch performance metrics
    - Run differential (momentum)
    - Lead changes frequency
    - Largest leads
    
  FROM PLAYER DATA (if available):
    - Injuries/inactive impact
    - Star player performance
    - Key player minutes
    - Bench depth scoring
    
  LINE MOVEMENT:
    - Opening vs closing spread
    - Opening vs closing total
    - Line movement direction
    - Steam moves
    
  ADVANCED:
    - Days since last game vs opponent (revenge)
    - Conference/division game flag
    - Time zone travel
    - Altitude (Denver home games)
    - Game importance (playoff implications)
    - Team motivation factors
    """)


def summarize_data_gaps():
    """Summary of data gaps and recommendations."""
    print_section("9. DATA GAPS & RECOMMENDATIONS")
    
    print("""
  CURRENT GAPS:
  
  1. Q1 LINES (From TheOdds API):
     - Available from May 2023
     - We have period_odds but may not be extracting Q1
     
  2. LINE MOVEMENT:
     - TheOdds API has historical snapshots
     - Not currently extracting opening vs closing lines
     
  3. ADVANCED STATS:
     - API-Basketball provides team stats
     - Not currently incorporating into training data
     
  4. PLAYER-LEVEL:
     - Injury data available but not used
     - Key player metrics not computed
     
  RECOMMENDATIONS:
  
  1. EXTRACT Q1 LINES from period_odds (already have data)
  2. COMPUTE LINE MOVEMENT from historical odds snapshots
  3. ADD TEAM STATS from API-Basketball
  4. ADD PLAYER IMPACT from injuries/inactive data
  5. COMPUTE ADVANCED METRICS from box scores
    """)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 80)
    print(" COMPREHENSIVE DATA AUDIT - ALL AVAILABLE HISTORICAL DATA")
    print("=" * 80)
    
    results = {}
    
    results["kaggle"] = inventory_kaggle()
    results["theodds"] = inventory_theodds()
    results["elo"] = inventory_elo()
    results["nba_db"] = inventory_nba_database()
    results["api_basketball"] = inventory_api_basketball()
    results["processed"] = inventory_processed()
    
    analyze_feature_utilization()
    identify_missing_features()
    summarize_data_gaps()
    
    print_section("10. ACTION ITEMS")
    print("""
  TO MAXIMIZE HISTORICAL DATA:
  
  [1] Run: python scripts/ingest_nba_database.py
      -> Downloads full NBA database (1946-2023)
      -> Extracts box scores, play-by-play, player data
      
  [2] Run: python scripts/extract_team_advanced_stats.py (CREATE)
      -> Extract offensive/defensive ratings
      -> Compute four factors
      
  [3] Run: python scripts/extract_line_movement.py (CREATE)
      -> Compute opening vs closing lines
      -> Identify steam moves
      
  [4] Run: python scripts/build_complete_training_data.py --all-features
      -> Incorporate all extracted features
    """)


if __name__ == "__main__":
    main()
