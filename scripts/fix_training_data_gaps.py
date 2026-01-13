#!/usr/bin/env python3
"""
Fix training data gaps for complete backtesting coverage.

Addresses:
1. fg_total_actual - compute from home_score + away_score
2. fg_spread_covered, fg_total_over, fg_home_win - add canonical column names
3. rest_days - compute from game schedule
4. Verify all labels are balanced and correct
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DATA_FILE = PROJECT_ROOT / "data" / "processed" / "training_data.csv"
MAX_NULL_FRACTION = 0.01  # fail fast if more than 1% of values are missing after repair


def _get_score_columns(df: pd.DataFrame) -> tuple[str, str]:
    if "home_score" in df.columns and "away_score" in df.columns:
        return "home_score", "away_score"
    if "score_home" in df.columns and "score_away" in df.columns:
        return "score_home", "score_away"
    raise KeyError("No supported score columns found (expected home_score/away_score or score_home/score_away)")


def _get_line_column(df: pd.DataFrame, preferred: str, fallback: str) -> str:
    if preferred in df.columns:
        return preferred
    if fallback in df.columns:
        return fallback
    raise KeyError(f"No supported line column found (expected {preferred} or {fallback})")


def _validate_not_too_many_nulls(series: pd.Series, name: str, max_null_fraction: float = MAX_NULL_FRACTION) -> None:
    null_fraction = float(series.isna().mean())
    print(f"      {name} null fraction: {null_fraction:.3%}")
    if null_fraction > max_null_fraction:
        raise ValueError(
            f"{name} has {null_fraction:.2%} nulls, exceeds threshold {max_null_fraction:.2%}. Check source data columns."
        )


def main(
    data_file: str | Path | None = None,
    overwrite_legacy_labels: bool = False,
    max_null_fraction: float = MAX_NULL_FRACTION,
):
    print("=" * 70)
    print("FIXING TRAINING DATA GAPS")
    print("=" * 70)
    
    # Load data
    print("\n[1/6] Loading training data...")
    data_path = Path(data_file) if data_file is not None else DEFAULT_DATA_FILE
    df = pd.read_csv(data_path, low_memory=False)
    df["game_date"] = pd.to_datetime(df["game_date"], format="mixed")
    print(f"      Loaded {len(df)} games")
    original_len = len(df)
    
    # ==== FIX 1: fg_total_actual ====
    print("\n[2/6] Fixing fg_total_actual...")
    score_home_col, score_away_col = _get_score_columns(df)
    print(f"      Using score columns: {score_home_col}, {score_away_col}")
    before = df["fg_total_actual"].notna().sum() if "fg_total_actual" in df.columns else 0
    df["fg_total_actual"] = pd.to_numeric(df[score_home_col], errors="coerce") + pd.to_numeric(df[score_away_col], errors="coerce")
    after = df["fg_total_actual"].notna().sum()
    print(f"      Before: {before}, After: {after}")
    _validate_not_too_many_nulls(df["fg_total_actual"], "fg_total_actual", max_null_fraction)
    
    # ==== FIX 2: FG margin ====
    print("\n[3/6] Fixing fg_margin...")
    before = df["fg_margin"].notna().sum() if "fg_margin" in df.columns else 0
    df["fg_margin"] = pd.to_numeric(df[score_home_col], errors="coerce") - pd.to_numeric(df[score_away_col], errors="coerce")
    after = df["fg_margin"].notna().sum()
    print(f"      Before: {before}, After: {after}")
    _validate_not_too_many_nulls(df["fg_margin"], "fg_margin", max_null_fraction)
    
    # ==== FIX 3: FG Labels (canonical names) ====
    print("\n[4/6] Adding canonical FG label columns...")
    
    # Labels should be computed from actual scores and lines.
    # Copying legacy label columns (spread_covered/total_over/home_win) is unsafe because those
    # can be stale or corrupted for partial seasons.

    fg_spread_line_col = _get_line_column(df, "fg_spread_line", "spread_line")
    fg_total_line_col = _get_line_column(df, "fg_total_line", "total_line")
    print(f"      Using line columns: spread={fg_spread_line_col}, total={fg_total_line_col}")

    # fg_spread_covered: (home margin + spread_line) > 0 means home covered
    df["fg_spread_covered"] = np.where(
        df[fg_spread_line_col].notna() & df["fg_margin"].notna(),
        (df["fg_margin"] + pd.to_numeric(df[fg_spread_line_col], errors="coerce") > 0).astype(float),
        np.nan,
    )
    print(f"      fg_spread_covered: computed ({df['fg_spread_covered'].notna().sum()})")
    
    # fg_total_over: actual > line
    df["fg_total_over"] = np.where(
        df[fg_total_line_col].notna() & df["fg_total_actual"].notna(),
        (df["fg_total_actual"] > pd.to_numeric(df[fg_total_line_col], errors="coerce")).astype(float),
        np.nan,
    )
    print(f"      fg_total_over: computed ({df['fg_total_over'].notna().sum()})")
    
    # fg_home_win: margin > 0
    df["fg_home_win"] = np.where(
        df["fg_margin"].notna(),
        (df["fg_margin"] > 0).astype(float),
        np.nan,
    )
    print(f"      fg_home_win: computed ({df['fg_home_win'].notna().sum()})")

    # Also repair legacy labels for compatibility if present/expected elsewhere.
    if overwrite_legacy_labels:
        df["home_win"] = df["fg_home_win"]
        df["spread_covered"] = df["fg_spread_covered"]
        df["total_over"] = df["fg_total_over"]
        print("      Legacy labels overwritten (home_win, spread_covered, total_over)")
    
    # ==== FIX 4: Rest days ====
    print("\n[5/6] Computing rest days...")
    df = df.sort_values(["game_date"]).reset_index(drop=True)
    
    # Track last game date for each team
    last_game = {}
    home_rest = []
    away_rest = []
    
    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        date = row["game_date"]
        
        # Home rest
        if home in last_game:
            rest_h = (date - last_game[home]).days
        else:
            # If the dataset window starts mid-season, the true rest is unknown.
            # Strict mode: do not fabricate rest; keep as NaN.
            rest_h = np.nan
        home_rest.append(rest_h)
        
        # Away rest
        if away in last_game:
            rest_a = (date - last_game[away]).days
        else:
            rest_a = np.nan
        away_rest.append(rest_a)
        
        # Update last game
        last_game[home] = date
        last_game[away] = date
    
    df["home_rest_days"] = home_rest
    df["away_rest_days"] = away_rest
    
    # Cap extreme values (start of season, etc.)
    df["home_rest_days"] = df["home_rest_days"].clip(upper=10)
    df["away_rest_days"] = df["away_rest_days"].clip(upper=10)
    
    print(f"      home_rest_days: {df['home_rest_days'].notna().sum()}")
    print(f"      away_rest_days: {df['away_rest_days'].notna().sum()}")

    if len(df) != original_len:
        raise AssertionError(f"Row count changed from {original_len} to {len(df)} during processing")
    
    # ==== SAVE ====
    print("\n[6/6] Saving...")
    df.to_csv(data_path, index=False)
    print(f"      Saved to {data_path}")
    
    # ==== VERIFICATION ====
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    
    print("\nFG Labels:")
    for col in ["fg_spread_covered", "fg_total_over", "fg_home_win"]:
        ct = df[col].notna().sum()
        dist = df[col].value_counts(dropna=True).to_dict()
        print(f"  {col}: {ct}/{len(df)} dist={dist}")
    
    print("\n1H Labels:")
    for col in ["1h_spread_covered", "1h_total_over", "1h_home_win"]:
        if col in df.columns:
            ct = df[col].notna().sum()
            dist = df[col].value_counts(dropna=True).to_dict()
            print(f"  {col}: {ct}/{len(df)} dist={dist}")
    
    print("\nRest Days:")
    print(f"  home_rest_days: mean={df['home_rest_days'].mean():.1f}, median={df['home_rest_days'].median():.1f}")
    print(f"  away_rest_days: mean={df['away_rest_days'].mean():.1f}, median={df['away_rest_days'].median():.1f}")
    
    print("\n2025-26 Check:")
    df26 = df[df["game_date"] >= "2025-10-01"]
    for col in ["fg_spread_covered", "fg_total_over", "1h_spread_covered", "1h_total_over"]:
        if col in df.columns:
            ct = df26[col].notna().sum()
            print(f"  {col}: {ct}/{len(df26)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-file",
        default=None,
        help=f"Path to training CSV to repair (default: {DEFAULT_DATA_FILE})",
    )
    parser.add_argument(
        "--overwrite-legacy-labels",
        action="store_true",
        help="Also rewrite legacy label columns (home_win, spread_covered, total_over)",
    )
    parser.add_argument(
        "--max-null-fraction",
        type=float,
        default=MAX_NULL_FRACTION,
        help=f"Maximum allowed null fraction after repairs before failing (default: {MAX_NULL_FRACTION})",
    )
    args = parser.parse_args()
    main(args.data_file, args.overwrite_legacy_labels, args.max_null_fraction)
