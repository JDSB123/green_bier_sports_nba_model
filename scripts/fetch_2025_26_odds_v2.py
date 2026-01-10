#!/usr/bin/env python3
"""
Fetch 2025-26 NBA betting lines from TheOdds API - V2 with correct market names.
"""
import os
import sys
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

API_KEY = os.environ.get("THE_ODDS_API_KEY", "4a0b80471d1ebeeb74c358fa0fcc4a27")
BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "basketball_nba"

OUTPUT_DIR = PROJECT_ROOT / "data" / "historical" / "the_odds" / "2025-2026"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_historical_odds(date_str: str, markets: list[str]) -> dict:
    """Fetch historical odds for a specific date."""
    url = f"{BASE_URL}/historical/sports/{SPORT}/odds"
    
    params = {
        "apiKey": API_KEY,
        "regions": "us",
        "markets": ",".join(markets),
        "oddsFormat": "american",
        "date": f"{date_str}T12:00:00Z",
    }
    
    try:
        resp = httpx.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 422:
            return {"data": []}
        else:
            print(f"    Error {resp.status_code}: {resp.text[:200]}", flush=True)
            return {"data": []}
    except Exception as e:
        print(f"    Error: {e}", flush=True)
        return {"data": []}


def main():
    if len(sys.argv) != 4:
        print("Usage: python fetch_2025_26_odds_v2.py <market_group> <start_date> <end_date>")
        print("  market_group: fg, 1h")
        print("  start_date/end_date: YYYY-MM-DD")
        sys.exit(1)
    
    market_group = sys.argv[1]
    start_date = datetime.strptime(sys.argv[2], "%Y-%m-%d")
    end_date = datetime.strptime(sys.argv[3], "%Y-%m-%d")
    
    # Correct market names per TheOdds API docs
    market_map = {
        "fg": ["h2h", "spreads", "totals"],
        "1h": ["h2h_1st_half", "spreads_1st_half", "totals_1st_half"],
        "1q": ["h2h_1st_quarter", "spreads_1st_quarter", "totals_1st_quarter"],
        "alt": ["alternate_spreads", "alternate_totals"],
    }
    
    markets = market_map.get(market_group, ["h2h", "spreads", "totals"])
    prefix = f"{market_group}_{start_date.strftime('%m%d')}_{end_date.strftime('%m%d')}"
    
    print(f"Fetching {market_group} ({','.join(markets)}) from {start_date.date()} to {end_date.date()}", flush=True)
    
    all_data = []
    current = start_date
    
    while current <= end_date:
        date_str = current.strftime("%Y-%m-%d")
        result = fetch_historical_odds(date_str, markets)
        events = result.get("data", [])
        if events:
            all_data.extend(events)
            print(f"  {date_str}: {len(events)} games", flush=True)
        current += timedelta(days=1)
        time.sleep(0.5)
    
    # Save
    output_file = OUTPUT_DIR / f"{prefix}.json"
    with open(output_file, "w") as f:
        json.dump(all_data, f)
    
    print(f"\nDONE - Saved {len(all_data)} records to {output_file.name}", flush=True)
    
    # CSV
    if all_data:
        rows = []
        for event in all_data:
            row = {
                "id": event.get("id"),
                "commence_time": event.get("commence_time"),
                "home_team": event.get("home_team"),
                "away_team": event.get("away_team"),
            }
            for bm in event.get("bookmakers", []):
                book = bm.get("key", "unknown")
                for mkt in bm.get("markets", []):
                    mkt_key = mkt.get("key")
                    for outcome in mkt.get("outcomes", []):
                        name = outcome.get("name", "")
                        price = outcome.get("price")
                        point = outcome.get("point")
                        col_base = f"{book}_{mkt_key}_{name}"
                        row[f"{col_base}_price"] = price
                        if point is not None:
                            row[f"{col_base}_point"] = point
            rows.append(row)
        
        df = pd.DataFrame(rows)
        csv_file = OUTPUT_DIR / f"{prefix}.csv"
        df.to_csv(csv_file, index=False)
        print(f"  Also saved CSV: {csv_file.name} ({len(df)} rows)", flush=True)


if __name__ == "__main__":
    main()
