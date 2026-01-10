#!/usr/bin/env python3
"""
Fetch 2025-26 NBA betting lines from TheOdds API.
Fetches: h2h, spreads, totals for full game AND first half.

Usage: python fetch_2025_26_odds.py <market_group> <worker_id>
  market_group: fg (full game) or 1h (first half)
  worker_id: 1, 2, 3, or 4
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

# Set API key
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
            # No data for this date
            return {"data": []}
        else:
            print(f"    Error {resp.status_code}: {resp.text[:100]}", flush=True)
            return {"data": []}
    except Exception as e:
        print(f"    Error: {e}", flush=True)
        return {"data": []}


def fetch_events(date_str: str) -> list:
    """Fetch events for a date."""
    url = f"{BASE_URL}/historical/sports/{SPORT}/events"
    
    params = {
        "apiKey": API_KEY,
        "date": f"{date_str}T12:00:00Z",
    }
    
    try:
        resp = httpx.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("data", [])
        return []
    except:
        return []


def get_date_range(worker_id: int) -> tuple[datetime, datetime]:
    """Get date range for this worker (split Oct 2025 - Jan 2026)."""
    # Full range: 2025-10-01 to 2026-01-10
    start = datetime(2025, 10, 1)
    end = datetime(2026, 1, 10)
    
    total_days = (end - start).days
    chunk = total_days // 4
    
    if worker_id == 1:
        return start, start + timedelta(days=chunk)
    elif worker_id == 2:
        return start + timedelta(days=chunk), start + timedelta(days=chunk*2)
    elif worker_id == 3:
        return start + timedelta(days=chunk*2), start + timedelta(days=chunk*3)
    else:
        return start + timedelta(days=chunk*3), end


def main():
    if len(sys.argv) != 3:
        print("Usage: python fetch_2025_26_odds.py <market_group> <worker_id>")
        print("  market_group: fg, 1h, h2h, spreads, totals, events")
        print("  worker_id: 1, 2, 3, 4")
        sys.exit(1)
    
    market_group = sys.argv[1]
    worker_id = int(sys.argv[2])
    
    # Define markets for each group
    market_map = {
        "fg": ["h2h", "spreads", "totals"],
        "1h": ["h2h_h1", "spreads_h1", "totals_h1"],
        "h2h": ["h2h"],
        "spreads": ["spreads"],
        "totals": ["totals"],
        "events": [],  # Special case
    }
    
    markets = market_map.get(market_group, ["h2h", "spreads", "totals"])
    
    start_date, end_date = get_date_range(worker_id)
    
    print(f"[WORKER {worker_id}] Fetching {market_group} from {start_date.date()} to {end_date.date()}", flush=True)
    
    all_data = []
    current = start_date
    
    while current <= end_date:
        date_str = current.strftime("%Y-%m-%d")
        
        if market_group == "events":
            data = fetch_events(date_str)
            if data:
                all_data.extend(data)
                print(f"  {date_str}: {len(data)} events", flush=True)
        else:
            result = fetch_historical_odds(date_str, markets)
            events = result.get("data", [])
            if events:
                all_data.extend(events)
                print(f"  {date_str}: {len(events)} games", flush=True)
        
        current += timedelta(days=1)
        time.sleep(0.5)  # Rate limit
    
    # Save results
    output_file = OUTPUT_DIR / f"{market_group}_w{worker_id}.json"
    with open(output_file, "w") as f:
        json.dump(all_data, f)
    
    print(f"\n[WORKER {worker_id}] DONE - Saved {len(all_data)} records to {output_file.name}", flush=True)
    
    # Also save as CSV if we have data
    if all_data and market_group != "events":
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
        csv_file = OUTPUT_DIR / f"{market_group}_w{worker_id}.csv"
        df.to_csv(csv_file, index=False)
        print(f"  Also saved CSV: {csv_file.name} ({len(df)} rows)", flush=True)


if __name__ == "__main__":
    main()
