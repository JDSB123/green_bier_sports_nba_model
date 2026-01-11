#!/usr/bin/env python3
"""Check current data coverage and identify gaps."""
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

print("=== CURRENT DATA COVERAGE ===")
df = pd.read_csv(PROJECT_ROOT / "data/processed/training_data_complete_2023.csv", low_memory=False)
df["game_date"] = pd.to_datetime(df["game_date"], format="mixed")
print(f"Training data: {df['game_date'].min().date()} to {df['game_date'].max().date()}")
print(f"Total games: {len(df)}")

# Injury coverage
injury_covered = (df["has_injury_data"] == 1).sum()
print(f"\nInjury data coverage: {injury_covered}/{len(df)} ({injury_covered/len(df)*100:.1f}%)")

# By calendar year
print("\nGames by calendar year:")
for year, count in df.groupby(df["game_date"].dt.year).size().items():
    has_inj = (df[df["game_date"].dt.year == year]["has_injury_data"] == 1).sum()
    print(f"  {year}: {count} games, {has_inj} with injury data ({has_inj/count*100:.1f}%)")

# What we need
print("\n=== GAPS TO FILL ===")
print("Need inactive player data for:")
for year in [2023, 2024, 2025, 2026]:
    year_games = len(df[df["game_date"].dt.year == year])
    year_inj = (df[df["game_date"].dt.year == year]["has_injury_data"] == 1).sum()
    gap = year_games - year_inj
    if gap > 0:
        print(f"  {year}: {gap} games missing injury data")
