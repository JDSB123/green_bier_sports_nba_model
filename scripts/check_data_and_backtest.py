#!/usr/bin/env python3
"""Quick check of data availability and run a sample backtest."""
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/processed")
training_file = DATA_DIR / "training_data.csv"

if not training_file.exists():
    print(f"[ERROR] Training data not found: {training_file}")
    exit(1)

df = pd.read_csv(training_file)
print(f"\nData Shape: {df.shape}")
print(f"Columns ({len(df.columns)}): {', '.join(df.columns[:30])}")
if len(df.columns) > 30:
    print(f"... and {len(df.columns) - 30} more")

print(f"\nDate range: {df['date'].min()} to {df['date'].max()}")

# Check for required columns
required_cols = {
    "spread_line": "spread_line" in df.columns,
    "total_line": "total_line" in df.columns,
    "home_score": "home_score" in df.columns,
    "away_score": "away_score" in df.columns,
    "home_q1": "home_q1" in df.columns,
    "home_q2": "home_q2" in df.columns,
}

print("\nRequired Columns:")
for col, exists in required_cols.items():
    status = "[OK]" if exists else "[MISSING]"
    print(f"  {status} {col}")

print(f"\nNon-null rows: {len(df.dropna(subset=['home_score', 'away_score']))}")

