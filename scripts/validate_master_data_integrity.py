#!/usr/bin/env python3
"""
Validate Training Data Integrity

This script is the SINGLE SOURCE OF TRUTH validator for training_data.csv.
Run this before any backtest or model training to ensure data quality.

Checks performed:
1. Feature independence (predicted_* != betting lines)
2. Label correctness (matches actual scores)
3. Coverage requirements met
4. No data leakage
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def validate_feature_independence(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Ensure predicted features are NOT copied from betting lines."""
    errors = []

    # Check predicted_total != total_line
    total_line = pd.to_numeric(df["total_line"], errors="coerce")
    pred_total = pd.to_numeric(df["predicted_total"], errors="coerce")
    if (pred_total == total_line).all():
        errors.append(
            "CRITICAL: predicted_total == total_line for ALL rows (data corruption)")
    elif (pred_total == total_line).sum() > len(df) * 0.1:
        errors.append(
            f"WARNING: predicted_total == total_line for {(pred_total == total_line).sum()} rows")

    # Check predicted_margin != -spread_line
    spread_line = pd.to_numeric(df["spread_line"], errors="coerce")
    pred_margin = pd.to_numeric(df["predicted_margin"], errors="coerce")
    if (pred_margin == -spread_line).all():
        errors.append(
            "CRITICAL: predicted_margin == -spread_line for ALL rows (data corruption)")
    elif (pred_margin == -spread_line).sum() > len(df) * 0.1:
        errors.append(
            f"WARNING: predicted_margin == -spread_line for {(pred_margin == -spread_line).sum()} rows")

    # Check spread_vs_predicted has variance
    svp = pd.to_numeric(df["spread_vs_predicted"], errors="coerce").dropna()
    if len(svp.unique()) < 10:
        errors.append(
            f"CRITICAL: spread_vs_predicted has only {len(svp.unique())} unique values (should have variance)")

    # Check total_vs_predicted has variance
    tvp = pd.to_numeric(df["total_vs_predicted"], errors="coerce").dropna()
    if len(tvp.unique()) < 10:
        errors.append(
            f"CRITICAL: total_vs_predicted has only {len(tvp.unique())} unique values (should have variance)")

    return len(errors) == 0, errors


def validate_labels(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Ensure labels are correctly computed from actual scores."""
    errors = []

    # FG spread: home_score - away_score + spread_line > 0 means home covered
    home_score = pd.to_numeric(df["home_score"], errors="coerce")
    away_score = pd.to_numeric(df["away_score"], errors="coerce")
    spread_line = pd.to_numeric(df["spread_line"], errors="coerce")

    computed_fg_spread = ((home_score - away_score) +
                          spread_line > 0).astype(int)
    stored_fg_spread = pd.to_numeric(df["fg_spread_covered"], errors="coerce")

    mask = stored_fg_spread.notna() & computed_fg_spread.notna()
    mismatch = (stored_fg_spread[mask] != computed_fg_spread[mask]).sum()
    if mismatch > 0:
        errors.append(
            f"WARNING: fg_spread_covered mismatches: {mismatch} rows")

    # FG total: home_score + away_score > total_line means over hit
    total_line = pd.to_numeric(df["total_line"], errors="coerce")
    computed_fg_total = ((home_score + away_score) > total_line).astype(int)
    stored_fg_total = pd.to_numeric(df["fg_total_over"], errors="coerce")

    mask = stored_fg_total.notna() & computed_fg_total.notna()
    mismatch = (stored_fg_total[mask] != computed_fg_total[mask]).sum()
    if mismatch > 0:
        errors.append(f"WARNING: fg_total_over mismatches: {mismatch} rows")

    return len(errors) == 0, errors


def validate_coverage(df: pd.DataFrame, coverage_start: str) -> tuple[bool, list[str]]:
    """Ensure minimum coverage requirements are met.

    Coverage is evaluated only for rows on/after `coverage_start` (YYYY-MM-DD)
    to align with our enforced canonical coverage window.
    """
    errors = []

    if "game_date" not in df.columns:
        return False, ["MISSING COLUMN: game_date"]

    dates = pd.to_datetime(df["game_date"], errors="coerce", format="mixed")
    start = pd.to_datetime(coverage_start, errors="coerce")
    if pd.isna(start):
        return False, [f"INVALID ARG: coverage_start='{coverage_start}' (expected YYYY-MM-DD)"]

    window = df.loc[dates.notna() & (dates >= start)].copy()
    if window.empty:
        return False, [f"NO DATA: no rows on/after {coverage_start} to evaluate coverage"]

    # Informational note (kept in errors list output block as a non-failing line)
    errors.append(
        f"INFO: Coverage window rows={len(window):,}/{len(df):,} (start={coverage_start})")

    requirements = {
        # Betting lines
        "fg_spread_line": 95.0,
        "fg_total_line": 95.0,
        "fg_ml_home": 95.0,
        "1h_spread_line": 90.0,
        "1h_total_line": 90.0,
        # Features
        "predicted_margin": 90.0,
        "predicted_total": 90.0,
        "home_elo": 95.0,
        "away_elo": 95.0,
        # Actuals
        "home_score": 95.0,
        "away_score": 95.0,
    }

    for col, threshold in requirements.items():
        if col not in df.columns:
            errors.append(f"MISSING COLUMN: {col}")
            continue
        coverage = window[col].notna().mean() * 100
        if coverage < threshold:
            errors.append(
                f"LOW COVERAGE: {col} = {coverage:.1f}% (need {threshold}%)")

    # Ignore the INFO line for pass/fail evaluation.
    failing = [e for e in errors if not e.startswith("INFO:")]
    return len(failing) == 0, errors


def validate_no_leakage(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Ensure no data leakage from future information."""
    errors = []

    # Check that predicted features are computed from historical stats, not game results
    # The correlation between predicted_total and actual_total should be moderate, not perfect
    actual_total = df["home_score"] + df["away_score"]
    pred_total = pd.to_numeric(df["predicted_total"], errors="coerce")

    correlation = pred_total.corr(actual_total)
    if correlation > 0.7:
        errors.append(
            f"POTENTIAL LEAKAGE: predicted_total correlates {correlation:.2f} with actual total (too high)")

    # predicted_margin vs actual margin
    actual_margin = df["home_score"] - df["away_score"]
    pred_margin = pd.to_numeric(df["predicted_margin"], errors="coerce")

    correlation = pred_margin.corr(actual_margin)
    if correlation > 0.7:
        errors.append(
            f"POTENTIAL LEAKAGE: predicted_margin correlates {correlation:.2f} with actual margin (too high)")

    return len(errors) == 0, errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate training data integrity")
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "training_data.csv",
        help="Path to training data",
    )
    parser.add_argument(
        "--coverage-start",
        default="2023-05-01",
        help="Only enforce coverage thresholds on/after this date (YYYY-MM-DD)",
    )
    parser.add_argument("--strict", action="store_true",
                        help="Fail on any warning")
    args = parser.parse_args()

    print("=" * 70)
    print("MASTER DATA INTEGRITY VALIDATION")
    print("=" * 70)
    print(f"File: {args.data}")
    print()

    # Load data
    if not args.data.exists():
        print(f"ERROR: File not found: {args.data}")
        return 1

    df = pd.read_csv(args.data, low_memory=False)
    print(f"Loaded {len(df):,} games")
    print()

    all_passed = True
    all_errors = []

    # Run validations
    validations = [
        ("Feature Independence", validate_feature_independence),
        ("Label Correctness", validate_labels),
        ("Coverage Requirements", lambda d: validate_coverage(d, args.coverage_start)),
        ("No Data Leakage", validate_no_leakage),
    ]

    for name, validator in validations:
        print(f"[{name}]")
        passed, errors = validator(df)
        if passed:
            print("  ✓ PASSED")
        else:
            all_passed = False
            for err in errors:
                print(f"  ✗ {err}")
                all_errors.append(err)
        print()

    # Summary
    print("=" * 70)
    if all_passed:
        print("✓ ALL VALIDATIONS PASSED - Data is ready for backtesting/training")
        print()
        print("Key Statistics:")
        print(f"  Games: {len(df):,}")
        print(
            f"  Date Range: {df['game_date'].min()} to {df['game_date'].max()}")
        print(
            f"  predicted_total unique values: {df['predicted_total'].nunique():,}")
        print(
            f"  predicted_margin unique values: {df['predicted_margin'].nunique():,}")
        print(
            f"  spread_vs_predicted range: [{df['spread_vs_predicted'].min():.1f}, {df['spread_vs_predicted'].max():.1f}]")
        print(
            f"  total_vs_predicted range: [{df['total_vs_predicted'].min():.1f}, {df['total_vs_predicted'].max():.1f}]")
        return 0
    else:
        print("✗ VALIDATION FAILED")
        print()
        print("Errors found:")
        for err in all_errors:
            print(f"  - {err}")
        return 1 if args.strict or any("CRITICAL" in e for e in all_errors) else 0


if __name__ == "__main__":
    sys.exit(main())
