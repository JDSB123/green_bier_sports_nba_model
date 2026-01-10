#!/usr/bin/env python3
"""
Fetch LIVE 1H odds for today's and upcoming NBA games.
TheOdds API supports 1H markets for live/upcoming games (not historical).
"""
import os
import sys
import json
from datetime import datetime
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


def fetch_live_odds(markets: list[str]) -> list:
    """Fetch current/upcoming odds."""
    url = f"{BASE_URL}/sports/{SPORT}/odds"
    
    params = {
        "apiKey": API_KEY,
        "regions": "us",
        "markets": ",".join(markets),
        "oddsFormat": "american",
    }
    
    resp = httpx.get(url, params=params, timeout=30)
    print(f"Status: {resp.status_code}", flush=True)
    
    if resp.status_code == 200:
        return resp.json()
    else:
        print(f"Error: {resp.text[:200]}", flush=True)
        return []


def main():
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"Fetching LIVE 1H odds for {today}", flush=True)
    print("=" * 60, flush=True)
    
    # Fetch FG markets
    print("\n[FULL GAME MARKETS]", flush=True)
    fg_markets = ["h2h", "spreads", "totals"]
    fg_data = fetch_live_odds(fg_markets)
    print(f"Found {len(fg_data)} events with FG odds", flush=True)
    
    # Fetch 1H markets
    print("\n[FIRST HALF MARKETS]", flush=True)
    h1_markets = ["h2h_1st_half", "spreads_1st_half", "totals_1st_half"]
    h1_data = fetch_live_odds(h1_markets)
    print(f"Found {len(h1_data)} events with 1H odds", flush=True)
    
    # Fetch Q1 markets
    print("\n[FIRST QUARTER MARKETS]", flush=True)
    q1_markets = ["h2h_1st_quarter", "spreads_1st_quarter", "totals_1st_quarter"]
    q1_data = fetch_live_odds(q1_markets)
    print(f"Found {len(q1_data)} events with Q1 odds", flush=True)
    
    # Print sample
    if fg_data:
        print("\n[SAMPLE GAMES]", flush=True)
        for event in fg_data[:5]:
            print(f"  {event.get('away_team')} @ {event.get('home_team')} - {event.get('commence_time')}", flush=True)
    
    # Save all
    all_data = {
        "fetch_time": datetime.now().isoformat(),
        "full_game": fg_data,
        "first_half": h1_data,
        "first_quarter": q1_data,
    }
    
    output_file = OUTPUT_DIR / f"live_odds_{today}.json"
    with open(output_file, "w") as f:
        json.dump(all_data, f, indent=2)
    
    print(f"\nSaved to {output_file.name}", flush=True)
    
    # Convert 1H to CSV if we have data
    if h1_data:
        rows = []
        for event in h1_data:
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
        csv_file = OUTPUT_DIR / f"live_1h_{today}.csv"
        df.to_csv(csv_file, index=False)
        print(f"1H CSV: {csv_file.name} ({len(df)} rows)", flush=True)
    
    print("\n" + "=" * 60, flush=True)
    print("SUMMARY:", flush=True)
    print(f"  FG events: {len(fg_data)}", flush=True)
    print(f"  1H events: {len(h1_data)}", flush=True)
    print(f"  Q1 events: {len(q1_data)}", flush=True)


if __name__ == "__main__":
    main()
