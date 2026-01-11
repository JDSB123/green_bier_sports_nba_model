#!/usr/bin/env python3
"""Merge 4 worker outputs into single 2025-2026_all_markets.csv."""
import json
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data" / "historical" / "the_odds" / "2025-2026"

# Load and merge all worker JSON files
all_data = []
for w in range(1, 5):
    json_file = DATA_DIR / f"all_markets_w{w}.json"
    if json_file.exists():
        with open(json_file) as f:
            worker_data = json.load(f)
            all_data.extend(worker_data)
            print(f"Worker {w}: {len(worker_data)} events")

print(f"Total events: {len(all_data)}")

# Dedupe by event ID
seen = set()
unique_data = []
for e in all_data:
    if e["id"] not in seen:
        seen.add(e["id"])
        unique_data.append(e)
print(f"Unique events: {len(unique_data)}")

# Save merged JSON
merged_json = DATA_DIR / "2025-2026_all_markets.json"
with open(merged_json, "w") as f:
    json.dump(unique_data, f)
print(f"Saved: {merged_json.name}")

# Convert to flattened CSV with consensus lines
rows = []
for event in unique_data:
    row = {
        "id": event.get("id"),
        "commence_time": event.get("commence_time"),
        "home_team": event.get("home_team"),
        "away_team": event.get("away_team"),
    }
    
    # Aggregate odds by market type
    market_lines = {}  # market -> list of (point, price)
    
    for bm in event.get("bookmakers", []):
        for mkt in bm.get("markets", []):
            mkt_key = mkt.get("key")
            for outcome in mkt.get("outcomes", []):
                name = outcome.get("name", "")
                price = outcome.get("price")
                point = outcome.get("point")
                
                key = f"{mkt_key}_{name}"
                if key not in market_lines:
                    market_lines[key] = {"points": [], "prices": []}
                if point is not None:
                    market_lines[key]["points"].append(point)
                if price is not None:
                    market_lines[key]["prices"].append(price)
    
    # Compute consensus (median point, median price)
    for key, vals in market_lines.items():
        if vals["points"]:
            row[f"{key}_point"] = pd.Series(vals["points"]).median()
        if vals["prices"]:
            row[f"{key}_price"] = pd.Series(vals["prices"]).median()
    
    rows.append(row)

df = pd.DataFrame(rows)

# Extract consensus lines for key markets (per row based on home team)
def extract_home_spread(row, prefix="spreads_"):
    """Extract spread for the home team."""
    home = row["home_team"]
    col = f"{prefix}{home}_point"
    return row.get(col, None)

# FG Spread (home team)
df["fg_spread"] = df.apply(lambda r: extract_home_spread(r, "spreads_"), axis=1)

# 1H Spread (home team)
df["h1_spread"] = df.apply(lambda r: extract_home_spread(r, "spreads_h1_"), axis=1)

# Q1 Spread (home team)
df["q1_spread"] = df.apply(lambda r: extract_home_spread(r, "spreads_q1_"), axis=1)

# FG Total
df["fg_total"] = df.get("totals_Over_point", None)

# 1H Total
df["h1_total"] = df.get("totals_h1_Over_point", None)

# Q1 Total
df["q1_total"] = df.get("totals_q1_Over_point", None)

# Save CSV
csv_file = DATA_DIR / "2025-2026_all_markets.csv"
df.to_csv(csv_file, index=False)
print(f"Saved: {csv_file.name} ({len(df)} rows, {len(df.columns)} columns)")

# Report coverage
print("\nMarket Coverage:")
for col in ["fg_spread", "fg_total", "h1_spread", "h1_total", "q1_spread", "q1_total"]:
    if col in df.columns:
        pct = df[col].notna().mean() * 100
        print(f"  {col}: {df[col].notna().sum()}/{len(df)} ({pct:.1f}%)")
