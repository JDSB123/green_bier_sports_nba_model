import os
import json
import glob
from typing import List, Dict, Any

import pandas as pd


def _latest_file(pattern: str) -> str | None:
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def _safe_load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_the_odds(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    # Expected structure per The Odds API (generic):
    # events -> bookmakers -> markets
    # Allow top-level list payloads
    if isinstance(payload, list):
        events = payload
    else:
        events = payload.get("data") or payload.get("events") or []
    for e in events:
        event_id = e.get("id") or e.get("event_id")
        home = (e.get("home_team") or e.get("teams", [None, None])[0])
        away = (e.get("away_team") or e.get("teams", [None, None])[1])
        start_time = e.get("commence_time") or e.get("start_time")
        bookmakers = e.get("bookmakers") or []
        for b in bookmakers:
            bookmaker = b.get("title") or b.get("key")
            ts = b.get("last_update")
            markets = b.get("markets") or b.get("markets", [])
            # Support typical market keys: h2h (moneyline), spreads, totals
            for m in markets:
                market_key = m.get("key") or m.get("market")
                outcomes = m.get("outcomes") or m.get("outcomes", [])
                for o in outcomes:
                    rows.append({
                        "source": "the_odds",
                        "event_id": event_id,
                        "home_team": home,
                        "away_team": away,
                        "start_time": start_time,
                        "bookmaker": bookmaker,
                        "market": market_key,
                        "participant": o.get("name"),
                        "price": o.get("price") or o.get("odds"),
                        "line": o.get("point") or o.get("line"),
                        "last_update": ts,
                    })
    return rows


def _normalize_betsapi(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    # BetsAPI shapes can vary; we best-effort extract similar columns
    events = payload.get("results") or payload.get("events") or []
    for e in events:
        event_id = e.get("id") or e.get("event_id")
        home = e.get("home") or e.get("home_team")
        away = e.get("away") or e.get("away_team")
        start_time = e.get("time") or e.get("start_time")
        odds_sets = e.get("odds") or e.get("bookmakers") or []
        for b in odds_sets:
            bookmaker = b.get("bookmaker") or b.get("name") or b.get("key")
            ts = b.get("updated_at") or b.get("last_update")
            for market_key in (
                "h2h",
                "spreads",
                "totals",
                "moneyline",
                "handicap",
                "over_under",
            ):
                market = b.get(market_key)
                if not market:
                    continue
                outcomes = (
                    market if isinstance(market, list)
                    else (market.get("outcomes") or [])
                )
                for o in outcomes:
                    rows.append({
                        "source": "betsapi",
                        "event_id": event_id,
                        "home_team": home,
                        "away_team": away,
                        "start_time": start_time,
                        "bookmaker": bookmaker,
                        "market": market_key,
                        "participant": o.get("name") or o.get("participant"),
                        "price": o.get("price") or o.get("odds"),
                        "line": o.get("point") or o.get("line"),
                        "last_update": ts,
                    })
    return rows


def collect_and_standardize(
    output_csv: str = "data/processed/odds.csv",
) -> str:
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Locate latest raw files
    the_odds_path = _latest_file("data/raw/the_odds/odds_*.json")
    betsapi_path = _latest_file("data/raw/betsapi/odds_*.json")

    rows: List[Dict[str, Any]] = []

    if the_odds_path and os.path.exists(the_odds_path):
        payload = _safe_load_json(the_odds_path)
        rows.extend(_normalize_the_odds(payload))

    if betsapi_path and os.path.exists(betsapi_path):
        payload = _safe_load_json(betsapi_path)
        rows.extend(_normalize_betsapi(payload))

    if not rows:
        # Still write an empty file with columns to keep pipelines happy
        df = pd.DataFrame(columns=[
            "source",
            "event_id",
            "home_team",
            "away_team",
            "start_time",
            "bookmaker",
            "market",
            "participant",
            "price",
            "line",
            "last_update",
        ])
        df.to_csv(output_csv, index=False)
        return output_csv

    df = pd.DataFrame(rows)

    # Basic cleaning
    # Parse timestamps when present
    for col in ("start_time", "last_update"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Sort for readability
    df = df.sort_values(
        ["start_time", "event_id", "bookmaker", "market"],
        na_position="last",
    )

    df.to_csv(output_csv, index=False)
    return output_csv


def main():
    path = collect_and_standardize()
    print(f"Wrote standardized odds to {path}")


if __name__ == "__main__":
    main()
