import glob
import json
import os
from typing import Any, Dict, List

import pandas as pd


def _latest_file(pattern: str) -> str | None:
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def _safe_load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_the_odds(payload: Dict[str, Any] | List[Dict[str, Any]]):
    rows: List[Dict[str, Any]] = []
    # events -> bookmakers -> markets
    events = (
        payload
        if isinstance(payload, list)
        else (payload.get("data") or payload.get("events") or [])
    )
    for e in events:
        event_id = e.get("id") or e.get("event_id")
        home = e.get("home_team") or e.get("teams", [None, None])[0]
        away = e.get("away_team") or e.get("teams", [None, None])[1]
        start_time = e.get("commence_time") or e.get("start_time")
        bookmakers = e.get("bookmakers") or []
        for b in bookmakers:
            bookmaker = b.get("title") or b.get("key")
            ts = b.get("last_update")
            markets = b.get("markets") or []
            for m in markets:
                market_key = m.get("key") or m.get("market")
                outcomes = m.get("outcomes") or []
                for o in outcomes:
                    rows.append(
                        {
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
                        }
                    )
    return rows


def collect_the_odds(
    output_csv: str = "data/processed/odds_the_odds.csv",
) -> str:
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    the_odds_path = _latest_file("data/raw/the_odds/odds_*.json")
    rows: List[Dict[str, Any]] = []
    if the_odds_path and os.path.exists(the_odds_path):
        payload = _safe_load_json(the_odds_path)
        rows.extend(_normalize_the_odds(payload))

    df = pd.DataFrame(
        rows,
        columns=[
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
        ],
    )
    if not df.empty:
        df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
        df["last_update"] = pd.to_datetime(df["last_update"], errors="coerce")
        df = df.sort_values(
            ["start_time", "event_id", "bookmaker", "market"],
            na_position="last",
        )
    df.to_csv(output_csv, index=False)
    return output_csv


def main():
    path = collect_the_odds()
    print(f"Wrote standardized The Odds to {path}")


if __name__ == "__main__":
    main()
