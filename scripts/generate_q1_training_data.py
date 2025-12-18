#!/usr/bin/env python3
"""
Generate first-quarter (Q1) training dataset with rolling features + real betting lines.

Output:
    - data/processed/q1_training_data.parquet
    - data/processed/q1_training_data.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.modeling.features import FeatureEngineer

LOOKBACK_GAMES = 10
MIN_HISTORY = 4
OUTPUT_PARQUET = Path(settings.data_processed_dir) / "q1_training_data.parquet"
OUTPUT_CSV = Path(settings.data_processed_dir) / "q1_training_data.csv"


def load_training_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Training data not found at {path}. Run build_fresh_training_data.py first."
        )

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    if "game_id" not in df.columns:
        df["game_id"] = (
            df["date"].dt.strftime("%Y%m%d")
            + "_"
            + df["away_team"].str.replace(" ", "_")
            + "_at_"
            + df["home_team"].str.replace(" ", "_")
        )
    return df


def build_team_long(df: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict] = []
    for _, row in df.iterrows():
        records.append(
            {
                "game_id": row["game_id"],
                "date": row["date"],
                "team": row["home_team"],
                "opponent": row["away_team"],
                "is_home": 1,
                "q1_points": row.get("home_q1"),
                "q1_allowed": row.get("away_q1"),
                "fg_points": row.get("home_score"),
                "fg_allowed": row.get("away_score"),
                "fg_margin": (row.get("home_score") or 0) - (row.get("away_score") or 0),
                "q1_margin": (row.get("home_q1") or 0) - (row.get("away_q1") or 0),
                "win": 1 if row.get("home_score", 0) > row.get("away_score", 0) else 0,
                "q1_win": 1 if row.get("home_q1", 0) > row.get("away_q1", 0) else 0,
            }
        )
        records.append(
            {
                "game_id": row["game_id"],
                "date": row["date"],
                "team": row["away_team"],
                "opponent": row["home_team"],
                "is_home": 0,
                "q1_points": row.get("away_q1"),
                "q1_allowed": row.get("home_q1"),
                "fg_points": row.get("away_score"),
                "fg_allowed": row.get("home_score"),
                "fg_margin": (row.get("away_score") or 0) - (row.get("home_score") or 0),
                "q1_margin": (row.get("away_q1") or 0) - (row.get("home_q1") or 0),
                "win": 1 if row.get("away_score", 0) > row.get("home_score", 0) else 0,
                "q1_win": 1 if row.get("away_q1", 0) > row.get("home_q1", 0) else 0,
            }
        )
    team_long = pd.DataFrame(records)
    team_long = team_long.sort_values(["team", "date"]).reset_index(drop=True)

    # Rest days + B2B flags
    team_long["rest_days"] = (
        team_long.groupby("team")["date"].diff().dt.days.subtract(1).clip(lower=0)
    )
    team_long["rest_days"] = team_long["rest_days"].fillna(3)
    team_long["is_b2b"] = (team_long["rest_days"] == 0).astype(int)

    def rolling(series: pd.Series) -> pd.Series:
        return (
            series.shift()
            .rolling(window=LOOKBACK_GAMES, min_periods=MIN_HISTORY)
            .mean()
        )

    grouped = team_long.groupby("team")
    team_long["ppg_q1_roll"] = grouped["q1_points"].transform(rolling)
    team_long["margin_q1_roll"] = grouped["q1_margin"].transform(rolling)
    team_long["ppg_fg_roll"] = grouped["fg_points"].transform(rolling)
    team_long["margin_fg_roll"] = grouped["fg_margin"].transform(rolling)
    team_long["win_pct_roll"] = grouped["win"].transform(rolling)
    team_long["q1_win_pct_roll"] = grouped["q1_win"].transform(rolling)

    return team_long


def pivot_features(team_long: pd.DataFrame) -> pd.DataFrame:
    home_cols = {
        "ppg_q1_roll": "home_ppg_q1",
        "margin_q1_roll": "home_margin_q1",
        "ppg_fg_roll": "home_ppg_fg",
        "margin_fg_roll": "home_margin_fg",
        "win_pct_roll": "home_win_pct",
        "rest_days": "home_rest_days",
        "is_b2b": "home_b2b",
        "q1_win_pct_roll": "home_q1_win_pct",
    }
    away_cols = {k: v.replace("home", "away") for k, v in home_cols.items()}

    home_feats = (
        team_long[team_long["is_home"] == 1][["game_id"] + list(home_cols.keys())]
        .rename(columns=home_cols)
        .reset_index(drop=True)
    )
    away_feats = (
        team_long[team_long["is_home"] == 0][["game_id"] + list(away_cols.keys())]
        .rename(columns=away_cols)
        .reset_index(drop=True)
    )

    return home_feats.merge(away_feats, on="game_id", how="inner")


def compute_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    df["ppg_diff_q1"] = df["home_ppg_q1"] - df["away_ppg_q1"]
    df["margin_diff_q1"] = df["home_margin_q1"] - df["away_margin_q1"]
    df["ppg_diff_fg"] = df["home_ppg_fg"] - df["away_ppg_fg"]
    df["margin_diff_fg"] = df["home_margin_fg"] - df["away_margin_fg"]
    df["win_pct_diff"] = df["home_win_pct"] - df["away_win_pct"]
    df["rest_diff"] = df["home_rest_days"] - df["away_rest_days"]
    df["q1_win_pct_diff"] = df["home_q1_win_pct"] - df["away_q1_win_pct"]
    return df


def add_dynamic_hca(base_df: pd.DataFrame, enriched_df: pd.DataFrame) -> pd.DataFrame:
    fe = FeatureEngineer(lookback=LOOKBACK_GAMES)
    enriched_df["dynamic_hca"] = 0.0

    for idx, row in enriched_df.iterrows():
        game = base_df.loc[base_df["game_id"] == row["game_id"]].iloc[0]
        try:
            dynamic = fe.compute_dynamic_hca(
                base_df,
                home_team=game["home_team"],
                away_team=game["away_team"],
                game_date=game["date"],
                home_rest=row.get("home_rest_days", 3),
            )
        except Exception:
            dynamic = 3.0
        enriched_df.at[idx, "dynamic_hca"] = dynamic
    return enriched_df


def filter_ready_rows(df: pd.DataFrame, base_df: pd.DataFrame) -> pd.DataFrame:
    merged = base_df.merge(df, on="game_id", how="inner", suffixes=("", "_feat"))
    required_cols = [
        "q1_spread_line",
        "q1_total_line",
        "q1_spread_covered",
        "q1_total_over",
        "home_q1_win",
    ]
    missing = [col for col in required_cols if col not in merged.columns]
    if missing:
        raise ValueError(
            f"Training data missing required Q1 columns: {', '.join(missing)}. "
            f"Ensure betting lines ingestion populated these fields."
        )
    for col in required_cols:
        merged = merged[merged[col].notna()]
    return merged


def build_dataset(training_path: Path) -> pd.DataFrame:
    base_df = load_training_data(training_path)
    team_long = build_team_long(base_df)
    feature_df = pivot_features(team_long)
    feature_df = compute_diff_features(feature_df)
    feature_df = add_dynamic_hca(base_df, feature_df)

    enriched = filter_ready_rows(feature_df, base_df)
    enriched["target_spread"] = enriched["q1_spread_covered"].astype(int)
    enriched["target_total"] = enriched["q1_total_over"].astype(int)
    enriched["target_moneyline"] = enriched["home_q1_win"].astype(int)
    columns_order = [
        "game_id",
        "date",
        "home_team",
        "away_team",
        "q1_spread_line",
        "q1_total_line",
        "q1_home_ml",
        "q1_away_ml",
        "q1_spread_covered",
        "q1_total_over",
        "home_q1_win",
        "target_spread",
        "target_total",
        "target_moneyline",
        "home_ppg_q1",
        "away_ppg_q1",
        "home_margin_q1",
        "away_margin_q1",
        "ppg_diff_q1",
        "margin_diff_q1",
        "home_win_pct",
        "away_win_pct",
        "win_pct_diff",
        "home_rest_days",
        "away_rest_days",
        "home_b2b",
        "away_b2b",
        "rest_diff",
        "dynamic_hca",
    ]
    # Ensure all requested columns exist
    for col in columns_order:
        if col not in enriched.columns:
            enriched[col] = np.nan

    enriched = enriched[columns_order]
    return enriched.sort_values("date").reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate first quarter training dataset")
    parser.add_argument(
        "--training-data",
        type=Path,
        default=Path(settings.data_processed_dir) / "training_data.csv",
        help="Path to base training_data.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = build_dataset(args.training_data)

    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(OUTPUT_PARQUET, index=False)
    dataset.to_csv(OUTPUT_CSV, index=False)

    print("=" * 80)
    print(f"[OK] Generated Q1 training dataset with {len(dataset)} games")
    print(f"Saved to:\n  - {OUTPUT_PARQUET}\n  - {OUTPUT_CSV}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
