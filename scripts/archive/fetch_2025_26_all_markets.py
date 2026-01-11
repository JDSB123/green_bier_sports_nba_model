#!/usr/bin/env python3
"""
Fetch 2025-26 NBA ALL MARKETS using event-level endpoint.
Premier subscription includes: FG, 1H, Q1, alternates, player props.

Uses /events/{event_id}/odds endpoint which supports all market types.

Usage: python fetch_2025_26_all_markets.py <worker_id> [date_start] [date_end]
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

# All premium markets
ALL_MARKETS = [
    "h2h", "spreads", "totals",                    # Full Game
    "h2h_h1", "spreads_h1", "totals_h1",           # First Half
    "h2h_q1", "spreads_q1", "totals_q1",           # First Quarter
    "alternate_spreads", "alternate_totals",       # Alternates
]


def get_historical_events(date_str: str) -> list:
    """Get events for a specific date from historical endpoint."""
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
    except Exception as e:
        print(f"    Error getting events: {e}", flush=True)
        return []


def get_event_odds(event_id: str, date_str: str, markets: list) -> dict:
    """Get odds for a specific event from historical endpoint."""
    url = f"{BASE_URL}/historical/sports/{SPORT}/events/{event_id}/odds"
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
            # Response is nested under 'data' key for historical endpoints
            result = resp.json()
            if "data" in result:
                return result["data"]  # Extract the nested data
            return result
        else:
            return {}
    except Exception as e:
        return {}


def get_date_range(worker_id: int) -> tuple:
    """Get date range for this worker."""
    # Full 2025-26 season range: Oct 2025 - Jan 2026
    start = datetime(2025, 10, 22)  # Season starts ~Oct 22
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
    if len(sys.argv) < 2:
        print("Usage: python fetch_2025_26_all_markets.py <worker_id>")
        print("  worker_id: 1, 2, 3, or 4")
        sys.exit(1)
    
    worker_id = int(sys.argv[1])
    start_date, end_date = get_date_range(worker_id)
    
    print(f"[WORKER {worker_id}] Fetching ALL markets from {start_date.date()} to {end_date.date()}", flush=True)
    print(f"  Markets: {', '.join(ALL_MARKETS)}", flush=True)
    
    all_data = []
    current = start_date
    total_events = 0
    
    while current <= end_date:
        date_str = current.strftime("%Y-%m-%d")
        
        # Get events for this date
        events = get_historical_events(date_str)
        
        if events:
            for event in events:
                event_id = event.get("id")
                if not event_id:
                    continue
                
                # Get odds for this event
                odds = get_event_odds(event_id, date_str, ALL_MARKETS)
                
                if odds and odds.get("bookmakers"):
                    record = {
                        "id": event_id,
                        "commence_time": event.get("commence_time"),
                        "home_team": event.get("home_team"),
                        "away_team": event.get("away_team"),
                        "bookmakers": odds.get("bookmakers", []),
                    }
                    all_data.append(record)
                    total_events += 1
                
                time.sleep(0.3)  # Rate limit
            
            print(f"  {date_str}: {len(events)} events", flush=True)
        
        current += timedelta(days=1)
        time.sleep(0.2)
    
    # Save JSON
    output_file = OUTPUT_DIR / f"all_markets_w{worker_id}.json"
    with open(output_file, "w") as f:
        json.dump(all_data, f)
    
    print(f"\n[WORKER {worker_id}] DONE - Saved {total_events} events to {output_file.name}", flush=True)
    
    # Convert to CSV with flattened structure
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
        csv_file = OUTPUT_DIR / f"all_markets_w{worker_id}.csv"
        df.to_csv(csv_file, index=False)
        print(f"  Also saved CSV: {csv_file.name} ({len(df)} rows)", flush=True)
        
        # Report markets found
        mkt_cols = set()
        for col in df.columns:
            for mkt in ALL_MARKETS:
                if mkt in col:
                    mkt_cols.add(mkt)
                    break
        print(f"  Markets in data: {sorted(mkt_cols)}", flush=True)


if __name__ == "__main__":
    main()
