import os
import glob
from typing import List

import pandas as pd


def merge_odds(
    inputs: List[str] | None = None,
    output_csv: str = "data/processed/odds_merged.csv",
) -> str:
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    if inputs is None:
        inputs = glob.glob("data/processed/odds_*.csv")
    frames: List[pd.DataFrame] = []
    for path in inputs:
        try:
            df = pd.read_csv(path)
            frames.append(df)
        except Exception:
            pass
    if not frames:
        pd.DataFrame(columns=[
            "source", "event_id", "home_team", "away_team", "start_time",
            "bookmaker", "market", "participant", "price",
            "line", "last_update",
        ]).to_csv(output_csv, index=False)
        return output_csv
    df = pd.concat(frames, ignore_index=True)
    # Optional: drop dupes across sources/bookmakers
    df = df.drop_duplicates(
        subset=[
            "event_id", "bookmaker", "market", "participant", "line", "price",
        ],
        keep="last",
    )
    df.to_csv(output_csv, index=False)
    return output_csv


def main():
    path = merge_odds()
    print(f"Wrote merged odds to {path}")


if __name__ == "__main__":
    main()
