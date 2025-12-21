#!/usr/bin/env python3
"""
Validate training data against the data manifest.

Checks:
1. File exists and matches checksum
2. Schema validation (required columns)
3. Row count within expected range
4. Data quality checks (no nulls in required fields, valid ranges)

Usage:
    python scripts/validate_training_data.py
    python scripts/validate_training_data.py --strict
    python scripts/validate_training_data.py --update-manifest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Use ASCII-safe symbols for Windows compatibility
OK = "[OK]"
FAIL = "[FAIL]"
WARN = "[WARN]"

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "processed"
TRAINING_DATA_PATH = DATA_DIR / "training_data.csv"
MANIFEST_PATH = DATA_DIR / "data_manifest.json"

# Required columns for each model type
REQUIRED_COLUMNS = {
    "identifiers": ["game_id", "date", "season", "home_team", "away_team"],
    "scores": ["home_score", "away_score"],
    "fg_labels": ["home_win", "spread_covered", "total_over"],
    "fg_lines": ["spread_line", "total_line"],
    "fg_features": [
        "home_ppg", "home_papg", "home_avg_margin", "home_win_pct",
        "away_ppg", "away_papg", "away_avg_margin", "away_win_pct",
    ],
    "1h_labels": ["1h_spread_covered", "1h_total_over"],
    "1h_lines": ["1h_spread_line", "1h_total_line"],
    "q1_labels": ["q1_spread_covered", "q1_total_over"],
    "q1_lines": ["q1_spread_line", "q1_total_line"],
}

# Valid ranges for numeric columns
VALID_RANGES = {
    "home_score": (50, 200),
    "away_score": (50, 200),
    "spread_line": (-30, 30),
    "total_line": (180, 280),
    "home_ppg": (80, 140),
    "away_ppg": (80, 140),
    "home_win_pct": (0, 1),
    "away_win_pct": (0, 1),
}


def calculate_checksum(file_path: Path) -> str:
    """Calculate SHA256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_manifest() -> dict | None:
    """Load the data manifest."""
    if not MANIFEST_PATH.exists():
        return None
    return json.loads(MANIFEST_PATH.read_text())


def save_manifest(manifest: dict) -> None:
    """Save the data manifest."""
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def validate_file_exists() -> tuple[bool, str]:
    """Check if training data file exists."""
    if TRAINING_DATA_PATH.exists():
        return True, f"{OK} File exists: {TRAINING_DATA_PATH}"
    return False, f"{FAIL} File not found: {TRAINING_DATA_PATH}"


def validate_checksum(manifest: dict | None) -> tuple[bool, str]:
    """Validate file checksum against manifest."""
    if not manifest:
        return True, f"{WARN} No manifest to check against"

    expected = manifest.get("training_data", {}).get("sha256")
    if not expected:
        return True, f"{WARN} No checksum in manifest"

    actual = calculate_checksum(TRAINING_DATA_PATH)
    if actual == expected:
        return True, f"{OK} Checksum matches: {actual[:16]}..."
    return False, f"{FAIL} Checksum mismatch!\n  Expected: {expected[:16]}...\n  Actual:   {actual[:16]}..."


def validate_schema(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Validate required columns exist."""
    messages = []
    all_valid = True

    for group, columns in REQUIRED_COLUMNS.items():
        missing = [c for c in columns if c not in df.columns]
        if missing:
            all_valid = False
            messages.append(f"{FAIL} Missing {group} columns: {missing}")
        else:
            messages.append(f"{OK} {group}: {len(columns)} columns present")

    return all_valid, messages


def validate_row_count(df: pd.DataFrame, manifest: dict | None) -> tuple[bool, str]:
    """Validate row count."""
    actual = len(df)

    if manifest:
        expected = manifest.get("training_data", {}).get("rows", 0)
        if expected > 0:
            diff = actual - expected
            if abs(diff) <= 100:  # Allow some variance
                return True, f"{OK} Row count: {actual} (expected ~{expected}, diff: {diff:+d})"
            return False, f"{FAIL} Row count: {actual} (expected ~{expected}, diff: {diff:+d})"

    # No manifest, just check reasonable range
    if 1000 <= actual <= 10000:
        return True, f"{OK} Row count: {actual} (reasonable range)"
    return False, f"{WARN} Row count: {actual} (outside typical range 1000-10000)"


def validate_nulls(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Check for nulls in required columns."""
    messages = []
    all_valid = True

    critical_cols = REQUIRED_COLUMNS["identifiers"] + REQUIRED_COLUMNS["fg_labels"]

    for col in critical_cols:
        if col in df.columns:
            null_count = df[col].isna().sum()
            if null_count > 0:
                all_valid = False
                messages.append(f"{FAIL} {col}: {null_count} null values")

    if all_valid:
        messages.append(f"{OK} No nulls in critical columns")

    return all_valid, messages


def validate_ranges(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Validate numeric columns are within expected ranges."""
    messages = []
    all_valid = True

    for col, (min_val, max_val) in VALID_RANGES.items():
        if col not in df.columns:
            continue

        below = (df[col] < min_val).sum()
        above = (df[col] > max_val).sum()

        if below > 0 or above > 0:
            # Warning, not failure (could be outliers)
            messages.append(f"{WARN} {col}: {below} below {min_val}, {above} above {max_val}")
        else:
            pass  # Don't clutter output with all passing

    if not messages:
        messages.append(f"{OK} All numeric columns within expected ranges")
        return True, messages

    return all_valid, messages


def validate_dates(df: pd.DataFrame) -> tuple[bool, str]:
    """Validate date column."""
    if "date" not in df.columns:
        return False, f"{FAIL} No date column"

    try:
        dates = pd.to_datetime(df["date"])
        min_date = dates.min()
        max_date = dates.max()

        # Check reasonable range (2020 onwards)
        if min_date.year < 2020:
            return False, f"{WARN} Dates start too early: {min_date}"

        return True, f"{OK} Date range: {min_date.date()} to {max_date.date()}"
    except Exception as e:
        return False, f"{FAIL} Date parsing error: {e}"


def validate_labels(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Validate label columns have valid values (0/1)."""
    messages = []
    all_valid = True

    label_cols = (
        REQUIRED_COLUMNS["fg_labels"]
        + REQUIRED_COLUMNS["1h_labels"]
        + REQUIRED_COLUMNS["q1_labels"]
    )

    for col in label_cols:
        if col not in df.columns:
            continue

        unique = df[col].dropna().unique()
        invalid = [v for v in unique if v not in [0, 1, True, False]]

        if invalid:
            all_valid = False
            messages.append(f"{FAIL} {col}: invalid values {invalid}")

    if all_valid:
        messages.append(f"{OK} All label columns have valid 0/1 values")

    return all_valid, messages


def run_validation(strict: bool = False) -> tuple[bool, dict]:
    """Run all validations."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "file": str(TRAINING_DATA_PATH),
        "checks": {},
        "passed": True,
    }

    print("=" * 60)
    print("TRAINING DATA VALIDATION")
    print("=" * 60)

    # Check file exists
    passed, msg = validate_file_exists()
    results["checks"]["file_exists"] = passed
    print(f"\n{msg}")
    if not passed:
        results["passed"] = False
        return False, results

    # Load manifest
    manifest = load_manifest()
    if manifest:
        print(f"{OK} Manifest loaded: v{manifest.get('version', 'unknown')}")
    else:
        print("{WARN} No manifest found")

    # Check checksum
    passed, msg = validate_checksum(manifest)
    results["checks"]["checksum"] = passed
    print(f"\n{msg}")
    if not passed and strict:
        results["passed"] = False

    # Load data
    print("\nLoading training data...")
    df = pd.read_csv(TRAINING_DATA_PATH)
    print(f"{OK} Loaded {len(df)} rows, {len(df.columns)} columns")

    # Schema validation
    print("\n--- Schema Validation ---")
    passed, messages = validate_schema(df)
    results["checks"]["schema"] = passed
    for msg in messages:
        print(f"  {msg}")
    if not passed:
        results["passed"] = False

    # Row count
    passed, msg = validate_row_count(df, manifest)
    results["checks"]["row_count"] = passed
    print(f"\n{msg}")
    if not passed and strict:
        results["passed"] = False

    # Null checks
    print("\n--- Null Checks ---")
    passed, messages = validate_nulls(df)
    results["checks"]["nulls"] = passed
    for msg in messages:
        print(f"  {msg}")
    if not passed:
        results["passed"] = False

    # Range validation
    print("\n--- Range Validation ---")
    passed, messages = validate_ranges(df)
    results["checks"]["ranges"] = passed
    for msg in messages:
        print(f"  {msg}")

    # Date validation
    passed, msg = validate_dates(df)
    results["checks"]["dates"] = passed
    print(f"\n{msg}")
    if not passed:
        results["passed"] = False

    # Label validation
    print("\n--- Label Validation ---")
    passed, messages = validate_labels(df)
    results["checks"]["labels"] = passed
    for msg in messages:
        print(f"  {msg}")
    if not passed:
        results["passed"] = False

    # Summary
    print("\n" + "=" * 60)
    if results["passed"]:
        print(f"{OK} VALIDATION PASSED")
    else:
        print(f"{FAIL} VALIDATION FAILED")
    print("=" * 60)

    return results["passed"], results


def update_manifest() -> None:
    """Update the manifest with current file stats."""
    if not TRAINING_DATA_PATH.exists():
        print("Error: Training data file not found")
        return

    df = pd.read_csv(TRAINING_DATA_PATH)
    dates = pd.to_datetime(df["date"])

    manifest = load_manifest() or {"version": "1.0.0"}

    manifest["training_data"] = {
        "file": "training_data.csv",
        "sha256": calculate_checksum(TRAINING_DATA_PATH),
        "rows": len(df),
        "columns": len(df.columns),
        "date_range": {
            "start": str(dates.min().date()),
            "end": str(dates.max().date()),
        },
        "last_updated": datetime.now().isoformat(),
    }

    # Update git commit if available
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        if result.returncode == 0:
            manifest["git_commit"] = result.stdout.strip()
    except Exception:
        pass

    save_manifest(manifest)
    print(f"{OK} Manifest updated: {MANIFEST_PATH}")
    print(f"  Rows: {manifest['training_data']['rows']}")
    print(f"  Checksum: {manifest['training_data']['sha256'][:16]}...")


def main():
    parser = argparse.ArgumentParser(description="Validate training data")
    parser.add_argument("--strict", action="store_true", help="Fail on warnings too")
    parser.add_argument("--update-manifest", action="store_true", help="Update manifest with current stats")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")

    args = parser.parse_args()

    if args.update_manifest:
        update_manifest()
        return

    passed, results = run_validation(strict=args.strict)

    if args.json:
        print(json.dumps(results, indent=2))

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
