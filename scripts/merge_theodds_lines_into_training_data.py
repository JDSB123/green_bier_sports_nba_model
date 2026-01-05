#!/usr/bin/env python3
"""
Merge committed The Odds historical lines into an existing training dataset.

This produces a *new* training CSV (does not overwrite by default) with:
  - spread_line / total_line replaced (or filled) from The Odds (FG)
  - fh_spread_line / fh_total_line replaced (or filled) from The Odds (1H)
  - labels recomputed from actual outcomes (no leakage; labels are derived)

This is the bridge between:
  data/historical/the_odds/** (committed raw odds)
and
  data/processed/training_data*.csv (feature-rich, outcome-rich dataset)

Usage:
  python scripts/cache_theodds_lines.py
  python scripts/merge_theodds_lines_into_training_data.py \
      --in data/processed/training_data.csv \
      --lines data/historical/derived/theodds_lines.csv \
      --out data/processed/training_data_theodds.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent


def _make_match_keys(df: pd.DataFrame, date_col: str, home_col: str, away_col: str) -> pd.DataFrame:
    out = df.copy()
    out["_date"] = pd.to_datetime(out[date_col], errors="coerce")
    out["_date_only"] = out["_date"].dt.tz_localize(None).dt.date
    out["_date_alt"] = (out["_date"].dt.tz_localize(None) - pd.Timedelta(days=1)).dt.date
    out["_date_plus"] = (out["_date"].dt.tz_localize(None) + pd.Timedelta(days=1)).dt.date
    out["_team_key"] = (
        out[home_col].astype(str).str.lower().str.strip() + "_" +
        out[away_col].astype(str).str.lower().str.strip()
    )
    out["_match_key"] = out["_date_only"].astype(str) + "_" + out["_team_key"]
    out["_match_key_alt"] = out["_date_alt"].astype(str) + "_" + out["_team_key"]
    out["_match_key_plus"] = out["_date_plus"].astype(str) + "_" + out["_team_key"]
    return out


def _build_lines_lookup(lines_df: pd.DataFrame) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Build lookup keyed by date+teams.
    """
    df = lines_df.copy()
    df["_date"] = pd.to_datetime(df["commence_time"], errors="coerce", utc=True)
    df["_date_only"] = df["_date"].dt.tz_convert(None).dt.date
    df["_team_key"] = (
        df["home_team"].astype(str).str.lower().str.strip() + "_" +
        df["away_team"].astype(str).str.lower().str.strip()
    )
    df["_match_key"] = df["_date_only"].astype(str) + "_" + df["_team_key"]

    cols = ["fg_spread_line", "fg_total_line", "fh_spread_line", "fh_total_line"]
    keep = df[["_match_key", *cols]].dropna(subset=["_match_key"])

    lookup: Dict[str, Dict[str, Optional[float]]] = {}
    for _, row in keep.iterrows():
        lookup[row["_match_key"]] = {c: row.get(c) for c in cols}
    return lookup


def recompute_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute market outcome labels from scores + lines.

    Uses strict, explicit formulas:
      - spread_covered: (home_margin > -spread_line)
      - total_over: (actual_total > total_line)
      - 1h_spread_covered: (actual_1h_margin > -fh_spread_line)
      - 1h_total_over: (actual_1h_total > fh_total_line)
    """
    out = df.copy()

    # FG derived
    if "home_score" in out.columns and "away_score" in out.columns:
        out["actual_margin"] = out["home_score"] - out["away_score"]
        out["actual_total"] = out["home_score"] + out["away_score"]

    if "spread_line" in out.columns:
        out["spread_covered"] = np.where(
            out["spread_line"].notna(),
            (out["actual_margin"] > -out["spread_line"]).astype(int),
            np.nan,
        )

    if "total_line" in out.columns:
        out["total_over"] = np.where(
            out["total_line"].notna(),
            (out["actual_total"] > out["total_line"]).astype(int),
            np.nan,
        )

    # 1H derived from quarters (if present)
    q_cols = ["home_q1", "home_q2", "away_q1", "away_q2"]
    if all(c in out.columns for c in q_cols):
        for c in q_cols:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        out["home_1h_score"] = out["home_q1"].fillna(0) + out["home_q2"].fillna(0)
        out["away_1h_score"] = out["away_q1"].fillna(0) + out["away_q2"].fillna(0)
        out["actual_1h_margin"] = out["home_1h_score"] - out["away_1h_score"]
        out["actual_1h_total"] = out["home_1h_score"] + out["away_1h_score"]

    # IMPORTANT: use REAL fh_* lines only (no approximations here)
    if "fh_spread_line" in out.columns:
        out["1h_spread_line"] = out["fh_spread_line"]
        out["1h_spread_covered"] = np.where(
            out["fh_spread_line"].notna(),
            (out["actual_1h_margin"] > -out["fh_spread_line"]).astype(int),
            np.nan,
        )

    if "fh_total_line" in out.columns:
        out["1h_total_line"] = out["fh_total_line"]
        out["1h_total_over"] = np.where(
            out["fh_total_line"].notna(),
            (out["actual_1h_total"] > out["fh_total_line"]).astype(int),
            np.nan,
        )

    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge The Odds historical lines into training data")
    p.add_argument("--in", dest="in_path", default="data/processed/training_data.csv")
    p.add_argument("--lines", dest="lines_path", default="data/historical/derived/theodds_lines.csv")
    p.add_argument("--out", dest="out_path", default="data/processed/training_data_theodds.csv")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    in_path = PROJECT_ROOT / args.in_path
    lines_path = PROJECT_ROOT / args.lines_path
    out_path = PROJECT_ROOT / args.out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Missing training data: {in_path}")
    if not lines_path.exists():
        raise FileNotFoundError(f"Missing lines cache: {lines_path}")

    train = pd.read_csv(in_path)
    if "date" not in train.columns:
        raise ValueError("training data missing required column: date")
    if not {"home_team", "away_team"}.issubset(train.columns):
        raise ValueError("training data missing home_team/away_team")

    lines = pd.read_csv(lines_path)

    train = _make_match_keys(train, "date", "home_team", "away_team")
    lookup = _build_lines_lookup(lines)

    fg_spread = 0
    fg_total = 0
    fh_spread = 0
    fh_total = 0

    # Ensure destination columns exist
    for col in ["spread_line", "total_line", "fh_spread_line", "fh_total_line"]:
        if col not in train.columns:
            train[col] = np.nan

    for idx, row in train.iterrows():
        key = row["_match_key"]
        alt = row["_match_key_alt"]
        plus = row["_match_key_plus"]
        payload = lookup.get(key) or lookup.get(alt) or lookup.get(plus)
        if not payload:
            continue

        fg_spread_val = payload.get("fg_spread_line")
        fg_total_val = payload.get("fg_total_line")
        fh_spread_val = payload.get("fh_spread_line")
        fh_total_val = payload.get("fh_total_line")

        if pd.notna(fg_spread_val):
            train.at[idx, "spread_line"] = fg_spread_val
            fg_spread += 1
        if pd.notna(fg_total_val):
            train.at[idx, "total_line"] = fg_total_val
            fg_total += 1
        if pd.notna(fh_spread_val):
            train.at[idx, "fh_spread_line"] = fh_spread_val
            fh_spread += 1
        if pd.notna(fh_total_val):
            train.at[idx, "fh_total_line"] = fh_total_val
            fh_total += 1

    # Clean temp columns
    train = train.drop(columns=[c for c in train.columns if c.startswith("_")], errors="ignore")

    # Recompute labels with the updated lines
    train = recompute_labels(train)

    train.to_csv(out_path, index=False)
    print(f"[OK] Wrote: {out_path}")
    print(f"[OK] Matched lines: fg_spread={fg_spread}, fg_total={fg_total}, fh_spread={fh_spread}, fh_total={fh_total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

