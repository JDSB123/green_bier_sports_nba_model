#!/usr/bin/env python3
"""Check 1H lines coverage gaps."""
import pandas as pd
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"

# Load training data
df = pd.read_csv(DATA / "processed" / "training_data_complete_2023.csv", low_memory=False)
df["game_date"] = pd.to_datetime(df["game_date"], format="mixed")
df["year_month"] = df["game_date"].dt.to_period("M")

# Check 1H coverage by month
print("1H SPREAD COVERAGE BY MONTH:")
print("=" * 70)

coverage = df.groupby("year_month").agg(
    games=("match_key", "count"),
    has_1h_spread=("1h_spread_line", lambda x: x.notna().sum()),
).reset_index()
coverage["pct"] = coverage["has_1h_spread"] / coverage["games"] * 100

for _, row in coverage.iterrows():
    bar = "#" * int(row["pct"] / 5)
    print(f"{row['year_month']}: {row['has_1h_spread']:3.0f}/{row['games']:3.0f} ({row['pct']:5.1f}%) {bar}")

print()
print("SUMMARY:")
total = len(df)
has_1h = df["1h_spread_line"].notna().sum()
print(f"  Total games: {total}")
print(f"  With 1H lines: {has_1h} ({has_1h/total*100:.1f}%)")
print(f"  Missing 1H: {total - has_1h}")

# Check source of 1H data
print()
print("1H DATA SOURCES:")
print("=" * 70)

# Check 1H exports
exports_dir = DATA / "historical" / "exports"
for f in sorted(exports_dir.glob("*_odds_1h.csv")):
    exp_df = pd.read_csv(f)
    exp_df["commence_time"] = pd.to_datetime(exp_df["commence_time"])
    print(f"  {f.name}: {len(exp_df):,} rows, {exp_df['commence_time'].min().date()} to {exp_df['commence_time'].max().date()}")

# Check 2025-26 all_markets
all_markets = DATA / "historical" / "the_odds" / "2025-2026_all_markets.csv"
if all_markets.exists():
    am = pd.read_csv(all_markets)
    am["commence_time"] = pd.to_datetime(am["commence_time"])
    h1_cols = [c for c in am.columns if "h1" in c.lower() or "1h" in c.lower()]
    print(f"  2025-2026_all_markets.csv: {len(am)} events")
    print(f"    1H columns: {h1_cols}")
    if "h1_spread" in am.columns:
        print(f"    h1_spread coverage: {am['h1_spread'].notna().sum()}/{len(am)} ({am['h1_spread'].notna().mean()*100:.1f}%)")

# Check TheOdds API 1H availability start
print()
print("CONCLUSION:")
print("  TheOdds API 1H/Q1 markets became available in May 2023")
print("  Games before May 2023 (or without 1H odds published) will be missing 1H lines")
