#!/usr/bin/env python3
"""Convert game_outcomes.csv to training_data.csv format for backtesting."""
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

def main():
    # Load game outcomes
    df = pd.read_csv(PROCESSED_DIR / "game_outcomes.csv")
    print(f"Loaded {len(df)} games")
    
    # Ensure required columns
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    # Filter to finished games with valid scores
    df = df[df["status"] == "Game Finished"]
    df = df[df["home_score"].notna() & df["away_score"].notna()]
    print(f"After filtering: {len(df)} games")
    
    # Save as training_data.csv
    output_path = PROCESSED_DIR / "training_data.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

if __name__ == "__main__":
    main()
