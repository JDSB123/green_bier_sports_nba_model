#!/usr/bin/env python3
"""Quick check of training data."""
import pandas as pd
import os

# Check main file
print("=" * 60)
print("MAIN FILE: training_data_complete_2023.csv")
print("=" * 60)
df = pd.read_csv('data/processed/training_data_complete_2023.csv', low_memory=False)
print(f"Total rows: {len(df)}")
print(f"Columns: {len(df.columns)}")

date_col = 'game_date' if 'game_date' in df.columns else 'date'
df['dt'] = pd.to_datetime(df[date_col], errors='coerce')
df = df.dropna(subset=['dt'])

print(f"Valid games: {len(df)}")
print(f"Date range: {df['dt'].min()} to {df['dt'].max()}")

# Season breakdown
df['season'] = df['dt'].apply(lambda x: f"{x.year-1}-{str(x.year)[2:]}" if x.month < 7 else f"{x.year}-{str(x.year+1)[2:]}")
print("\nGames by season:")
print(df.groupby('season').size().sort_index())

# Check key columns
print("\nKey columns available:")
key_cols = ['home_score', 'away_score', 'fg_spread_line', 'fg_total_line', 
            '1h_spread_line', '1h_total_line', 'fg_spread_covered', 'fg_total_over',
            '1h_spread_covered', '1h_total_over', 'home_elo', 'away_elo']
for col in key_cols:
    if col in df.columns:
        non_null = df[col].notna().sum()
        pct = non_null / len(df) * 100
        print(f"  {col}: {non_null} ({pct:.1f}%)")
    else:
        print(f"  {col}: NOT FOUND")

# Check 2025-26 file
print("\n" + "=" * 60)
print("2025-26 FILE: training_data_2025_26.csv")
print("=" * 60)
if os.path.exists('data/processed/training_data_2025_26.csv'):
    df2 = pd.read_csv('data/processed/training_data_2025_26.csv', low_memory=False)
    print(f"Total rows: {len(df2)}")
    date_col2 = 'game_date' if 'game_date' in df2.columns else 'date'
    df2['dt'] = pd.to_datetime(df2[date_col2], errors='coerce')
    df2 = df2.dropna(subset=['dt'])
    print(f"Valid games: {len(df2)}")
    print(f"Date range: {df2['dt'].min()} to {df2['dt'].max()}")
else:
    print("File not found!")

# Check Kaggle data
print("\n" + "=" * 60)
print("KAGGLE FILE: nba_2008-2025.csv")
print("=" * 60)
if os.path.exists('data/external/kaggle/nba_2008-2025.csv'):
    dfk = pd.read_csv('data/external/kaggle/nba_2008-2025.csv', low_memory=False)
    print(f"Total rows: {len(dfk)}")
    print(f"Columns: {list(dfk.columns)[:15]}...")
    date_col = [c for c in dfk.columns if 'date' in c.lower()]
    if date_col:
        dfk['dt'] = pd.to_datetime(dfk[date_col[0]], errors='coerce')
        dfk = dfk.dropna(subset=['dt'])
        print(f"Date range: {dfk['dt'].min()} to {dfk['dt'].max()}")
else:
    print("File not found!")
