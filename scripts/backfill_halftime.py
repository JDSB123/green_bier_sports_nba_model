"""Backfill real halftime scores into processed training data.

This script reads `data/processed/training_data.csv`, queries the API-Basketball
`/games` endpoint for each game date, extracts explicit halftime scores when
available, and writes `data/processed/training_data_fh.csv` with the added
fields:

- `home_halftime_score`, `away_halftime_score`
- `fh_home_win` (0/1)
- `fh_total_line`, `fh_went_over`
- `fh_spread_line`, `fh_spread_covered`

The script is idempotent and caches fixture responses under
`data/raw/api_basketball/fixtures_cache_{date}.json` to avoid repeated API
calls.
"""
from __future__ import annotations
import os
import time
import json
from typing import Dict, Any

import httpx
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, ROOT)

from src.config import settings


HEADERS = {"x-apisports-key": settings.api_basketball_key}


def cache_path_for_date(date: str) -> str:
    ddir = os.path.join(settings.data_raw_dir, "api_basketball")
    os.makedirs(ddir, exist_ok=True)
    return os.path.join(ddir, f"fixtures_cache_{date}.json")


def fetch_fixtures_for_date(date: str, season: str = None) -> Dict[str, Any]:
    """Fetch fixtures (cached). Returns parsed JSON or empty dict on failure."""
    cpath = cache_path_for_date(date)
    if os.path.exists(cpath):
        try:
            with open(cpath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    url = f"{settings.api_basketball_base_url}/games"
    params = {"date": date, "league": 12, "timezone": "America/New_York"}
    if season:
        params["season"] = season
    try:
        resp = httpx.get(url, params=params, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        with open(cpath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        time.sleep(0.3)
        return data
    except Exception as e:
        print(f"  API error for {date}: {e}")
        return {}


def extract_halftime_from_fixture(item: Dict[str, Any]) -> tuple | None:
    """Given a fixture dict, return (home_ht, away_ht) if available else None."""
    # API-Basketball structure: item['scores'] with 'home'/'away' dicts containing 'quarter_1','quarter_2', etc.
    scores = item.get("scores") or {}
    home_scores = scores.get("home") or {}
    away_scores = scores.get("away") or {}
    
    # Try summing first two quarters
    try:
        h_q1 = home_scores.get("quarter_1")
        h_q2 = home_scores.get("quarter_2")
        a_q1 = away_scores.get("quarter_1")
        a_q2 = away_scores.get("quarter_2")
        if all(v is not None for v in (h_q1, h_q2, a_q1, a_q2)):
            return int(h_q1) + int(h_q2), int(a_q1) + int(a_q2)
    except Exception:
        pass
    
    # Fallback: look for explicit 'halftime' key
    for k in ("score", "scores"):
        sc = item.get(k) or {}
        ht = sc.get("halftime") or sc.get("half_time")
        if ht and isinstance(ht, dict):
            h = ht.get("home")
            a = ht.get("away")
            if h is not None and a is not None:
                try:
                    return int(h), int(a)
                except Exception:
                    pass
    return None


def normalize_team_name(name: str) -> str:
    """Normalize team name for matching."""
    name = name.lower().strip()
    # Common abbreviations and variants
    replacements = {
        "trail blazers": "trailblazers",
        "portland": "trailblazers",
        "la lakers": "lakers",
        "la clippers": "clippers",
        "golden state": "warriors",
        "san antonio": "spurs",
        "new york": "knicks",
        "oklahoma city": "thunder",
    }
    for k, v in replacements.items():
        if k in name:
            return v
    # Extract team name (last word usually)
    parts = name.split()
    return parts[-1] if parts else name

def find_fixture_for_game(fixtures: Dict[str, Any], home: str, away: str) -> dict | None:
    """Attempt to locate matching fixture by team names."""
    items = fixtures.get("response") or []
    if not isinstance(items, list):
        return None
    
    home_norm = normalize_team_name(home)
    away_norm = normalize_team_name(away)
    
    for it in items:
        teams = it.get("teams") or {}
        ht_data = teams.get("home") or {}
        at_data = teams.get("away") or {}
        ht = normalize_team_name(ht_data.get("name") or "")
        at = normalize_team_name(at_data.get("away") or "")
        
        if home_norm == ht and away_norm == at:
            return it
        # Partial match
        if (home_norm in ht or ht in home_norm) and (away_norm in at or at in away_norm):
            return it
    return None


def main():
    in_path = os.path.join(settings.data_processed_dir, "training_data.csv")
    out_path = os.path.join(settings.data_processed_dir, "training_data_fh.csv")
    if not os.path.exists(in_path):
        print("training_data.csv not found; aborting")
        return

    df = pd.read_csv(in_path)

    # Track which rows we fill
    filled = 0

    # We'll operate on a copy and add columns
    out = df.copy()
    if "home_halftime_score" not in out.columns:
        out["home_halftime_score"] = pd.NA
    if "away_halftime_score" not in out.columns:
        out["away_halftime_score"] = pd.NA

    unique_dates = sorted(out[~out["home_halftime_score"].notna()]["date"].unique())
    print(f"Fetching fixtures for {len(unique_dates)} unique dates...")
    
    for i, date in enumerate(unique_dates):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(unique_dates)} dates processed, {filled} rows filled")
        
        # Derive season from date for better API results
        try:
            dt_obj = pd.to_datetime(date)
            year = dt_obj.year
            season = f"{year}-{year+1}" if dt_obj.month >= 10 else f"{year-1}-{year}"
        except:
            season = None
        
        fixtures = fetch_fixtures_for_date(str(date), season=season)
        if not fixtures:
            continue
        
        for idx, row in out[out["home_halftime_score"].isna() & (out["date"] == date)].iterrows():
            home = str(row["home_team"]) if not pd.isna(row["home_team"]) else ""
            away = str(row["away_team"]) if not pd.isna(row["away_team"]) else ""
            fixture = find_fixture_for_game(fixtures, home, away)
            if not fixture:
                continue
            ht = extract_halftime_from_fixture(fixture)
            if ht:
                out.at[idx, "home_halftime_score"] = ht[0]
                out.at[idx, "away_halftime_score"] = ht[1]
                filled += 1

    # Derive FH targets where present
    if "home_halftime_score" in out.columns and "away_halftime_score" in out.columns:
        out["fh_home_win"] = (out["home_halftime_score"] > out["away_halftime_score"]).astype(float)
    if "total_line" in out.columns and "home_halftime_score" in out.columns:
        out["fh_total_line"] = out["total_line"] / 2.0
        out["fh_went_over"] = ((out["home_halftime_score"].fillna(0).astype(float) + out["away_halftime_score"].fillna(0).astype(float)) > out["fh_total_line"]).astype(float)
    if "spread_line" in out.columns and "home_halftime_score" in out.columns:
        out["fh_spread_line"] = out["spread_line"] / 2.0
        out["fh_spread_covered"] = (((out["home_halftime_score"].fillna(0).astype(float) - out["away_halftime_score"].fillna(0).astype(float)) - out["fh_spread_line"]) > 0).astype(float)

    out.to_csv(out_path, index=False)
    print(f"Wrote first-half augmented training data to {out_path}; filled {filled} rows")


if __name__ == "__main__":
    main()
