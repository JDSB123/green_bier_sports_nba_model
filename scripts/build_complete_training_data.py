#!/usr/bin/env python3
"""
Build complete training data from API-Basketball games files.

Extracts:
- Full game scores
- Q1-Q4 scores for 1H analysis
- All finished games
"""
import json
import sys
from pathlib import Path
from glob import glob

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "api_basketball"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def extract_games_from_json(filepath: Path) -> list:
    """Extract games from a single JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    games = []
    for game in data.get("response", []):
        # Skip non-finished games
        status = game.get("status", {})
        if status.get("short") != "FT":
            continue
        
        teams = game.get("teams", {})
        scores = game.get("scores", {})
        
        home_scores = scores.get("home", {})
        away_scores = scores.get("away", {})
        
        # Extract all score data
        game_data = {
            "game_id": game.get("id"),
            "date": game.get("date"),
            "home_team": teams.get("home", {}).get("name"),
            "away_team": teams.get("away", {}).get("name"),
            "home_score": home_scores.get("total"),
            "away_score": away_scores.get("total"),
            "home_q1": home_scores.get("quarter_1"),
            "home_q2": home_scores.get("quarter_2"),
            "home_q3": home_scores.get("quarter_3"),
            "home_q4": home_scores.get("quarter_4"),
            "away_q1": away_scores.get("quarter_1"),
            "away_q2": away_scores.get("quarter_2"),
            "away_q3": away_scores.get("quarter_3"),
            "away_q4": away_scores.get("quarter_4"),
        }
        
        # Skip if missing critical data
        if not all([game_data["home_team"], game_data["away_team"], 
                   game_data["home_score"], game_data["away_score"]]):
            continue
        
        games.append(game_data)
    
    return games


def main():
    print("=" * 60)
    print("BUILDING COMPLETE TRAINING DATA")
    print("=" * 60)
    
    # Find all games files
    games_files = list(RAW_DIR.glob("games_*.json"))
    print(f"Found {len(games_files)} games files")
    
    # Also check fixtures cache files
    fixtures_files = list(RAW_DIR.glob("fixtures_cache_*.json"))
    print(f"Found {len(fixtures_files)} fixtures cache files")
    
    all_games = []
    seen_ids = set()
    
    # Process games files
    for filepath in games_files:
        games = extract_games_from_json(filepath)
        for game in games:
            if game["game_id"] not in seen_ids:
                seen_ids.add(game["game_id"])
                all_games.append(game)
    
    print(f"Extracted {len(all_games)} unique games from games files")
    
    # Process fixtures cache files (different structure)
    for filepath in fixtures_files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for game in data.get("response", []):
                if game.get("id") in seen_ids:
                    continue
                
                status = game.get("status", {})
                if status.get("short") != "FT":
                    continue
                
                teams = game.get("teams", {})
                scores = game.get("scores", {})
                
                home_scores = scores.get("home", {})
                away_scores = scores.get("away", {})
                
                game_data = {
                    "game_id": game.get("id"),
                    "date": game.get("date"),
                    "home_team": teams.get("home", {}).get("name"),
                    "away_team": teams.get("away", {}).get("name"),
                    "home_score": home_scores.get("total"),
                    "away_score": away_scores.get("total"),
                    "home_q1": home_scores.get("quarter_1"),
                    "home_q2": home_scores.get("quarter_2"),
                    "home_q3": home_scores.get("quarter_3"),
                    "home_q4": home_scores.get("quarter_4"),
                    "away_q1": away_scores.get("quarter_1"),
                    "away_q2": away_scores.get("quarter_2"),
                    "away_q3": away_scores.get("quarter_3"),
                    "away_q4": away_scores.get("quarter_4"),
                }
                
                if all([game_data["home_team"], game_data["away_team"],
                       game_data["home_score"], game_data["away_score"]]):
                    seen_ids.add(game_data["game_id"])
                    all_games.append(game_data)
        except Exception as e:
            continue
    
    print(f"Total unique games: {len(all_games)}")
    
    if not all_games:
        print("[ERROR] No games found!")
        sys.exit(1)
    
    # Create DataFrame
    df = pd.DataFrame(all_games)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    # Remove duplicates by game_id
    df = df.drop_duplicates(subset=["game_id"], keep="first")
    
    print(f"After dedup: {len(df)} games")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Calculate derived columns
    df["home_margin"] = df["home_score"] - df["away_score"]
    df["total_score"] = df["home_score"] + df["away_score"]
    
    # 1H scores
    df["home_1h"] = df["home_q1"].fillna(0) + df["home_q2"].fillna(0)
    df["away_1h"] = df["away_q1"].fillna(0) + df["away_q2"].fillna(0)
    df["home_1h_margin"] = df["home_1h"] - df["away_1h"]
    df["total_1h"] = df["home_1h"] + df["away_1h"]
    
    # Show stats
    print(f"\nQ1-Q4 data available: {df['home_q1'].notna().sum()} games")
    print(f"Home win rate: {(df['home_margin'] > 0).mean():.1%}")
    print(f"Home 1H lead rate: {(df['home_1h_margin'] > 0).mean():.1%}")
    
    # Save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / "training_data.csv"
    df.to_csv(output_path, index=False)
    print(f"\n[OK] Saved {len(df)} games to {output_path}")
    
    # Show column summary
    print(f"\nColumns: {df.columns.tolist()}")


if __name__ == "__main__":
    main()
