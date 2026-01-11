#!/usr/bin/env python3
"""
Upload Quality-Checked Training Data to Azure Blob Storage

This script serves as the SINGLE SOURCE OF TRUTH gatekeeper for training data.
It validates data quality, generates a manifest, and uploads to Azure Blob Storage.

Quality Gates:
1. Schema validation (all required columns present)
2. Coverage checks (injury, odds, features)
3. Data integrity (no nulls in critical fields, valid ranges)
4. Minimum threshold checks (games, coverage %)
5. Manifest generation with checksums

Usage:
    python scripts/upload_training_data_to_azure.py
    python scripts/upload_training_data_to_azure.py --dry-run  # Validate only
    python scripts/upload_training_data_to_azure.py --force    # Skip confirmations
    python scripts/upload_training_data_to_azure.py --version v1.0.0  # Custom version tag
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
STORAGE_ACCOUNT = "nbagbsvstrg"
CONTAINER = "nbahistoricaldata"
BLOB_PREFIX = "training_data"  # training_data/v1.0.0/training_data_complete.csv

# Data paths
DATA_DIR = PROJECT_ROOT / "data" / "processed"
TRAINING_DATA_PATHS = [
    DATA_DIR / "training_data_complete_2023_with_injuries.csv",
    DATA_DIR / "training_data_complete_2023.csv",
]

# Quality thresholds
MIN_GAMES = 3000
MIN_FEATURES = 50
MIN_INJURY_COVERAGE = 0.80  # 80%
MIN_ODDS_COVERAGE = 0.90    # 90%

# Required columns by category
REQUIRED_COLUMNS = {
    "identifiers": ["game_id", "game_date", "season", "home_team", "away_team"],
    "scores": ["home_score", "away_score"],
    "labels": ["home_win", "spread_covered", "total_over"],
    "fg_lines": ["spread_line", "total_line"],
    "elo": ["home_elo", "away_elo", "elo_diff"],
    "rolling": ["home_ppg", "away_ppg"],
}

# Injury columns (if present)
INJURY_COLUMNS = [
    "home_injury_impact",
    "away_injury_impact",
    "home_injury_star_out",
    "away_injury_star_out",
    "injury_imbalance",
]

# Status symbols
OK = "[✓]"
FAIL = "[✗]"
WARN = "[!]"
INFO = "[i]"


def calculate_checksum(file_path: Path) -> str:
    """Calculate SHA256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def find_training_data() -> Path | None:
    """Find the most recent training data file."""
    for path in TRAINING_DATA_PATHS:
        if path.exists():
            return path
    return None


class QualityChecker:
    """Quality validation for training data."""

    def __init__(self, df: pd.DataFrame, file_path: Path):
        self.df = df
        self.file_path = file_path
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.metrics: dict[str, Any] = {}

    def run_all_checks(self) -> bool:
        """Run all quality checks. Returns True if passed."""
        print("\n" + "=" * 60)
        print("QUALITY CHECKS")
        print("=" * 60)

        checks = [
            ("Row Count", self._check_row_count),
            ("Schema", self._check_schema),
            ("Feature Count", self._check_feature_count),
            ("Critical Nulls", self._check_critical_nulls),
            ("Score Ranges", self._check_score_ranges),
            ("Odds Coverage", self._check_odds_coverage),
            ("Injury Coverage", self._check_injury_coverage),
            ("Date Range", self._check_date_range),
            ("Season Balance", self._check_season_balance),
        ]

        all_passed = True
        for name, check_fn in checks:
            try:
                passed, message = check_fn()
                status = OK if passed else (FAIL if not passed else WARN)
                print(f"  {status} {name}: {message}")
                if not passed and "warn" not in message.lower():
                    all_passed = False
            except Exception as e:
                print(f"  {FAIL} {name}: Error - {e}")
                all_passed = False

        return all_passed

    def _check_row_count(self) -> tuple[bool, str]:
        """Check minimum row count."""
        count = len(self.df)
        self.metrics["total_games"] = count
        if count >= MIN_GAMES:
            return True, f"{count:,} games (min: {MIN_GAMES:,})"
        return False, f"Only {count:,} games (need {MIN_GAMES:,})"

    def _check_schema(self) -> tuple[bool, str]:
        """Check required columns exist."""
        missing = []
        for group, cols in REQUIRED_COLUMNS.items():
            for col in cols:
                if col not in self.df.columns:
                    missing.append(f"{group}.{col}")

        if not missing:
            return True, "All required columns present"
        return False, f"Missing: {', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}"

    def _check_feature_count(self) -> tuple[bool, str]:
        """Check total feature count."""
        count = len(self.df.columns)
        self.metrics["total_columns"] = count
        if count >= MIN_FEATURES:
            return True, f"{count} columns (min: {MIN_FEATURES})"
        return False, f"Only {count} columns (need {MIN_FEATURES})"

    def _check_critical_nulls(self) -> tuple[bool, str]:
        """Check for nulls in critical columns."""
        critical = ["game_id", "home_team", "away_team", "home_score", "away_score"]
        null_counts = {}
        for col in critical:
            if col in self.df.columns:
                nulls = self.df[col].isna().sum()
                if nulls > 0:
                    null_counts[col] = nulls

        if not null_counts:
            return True, "No nulls in critical columns"
        return False, f"Nulls found: {null_counts}"

    def _check_score_ranges(self) -> tuple[bool, str]:
        """Check score ranges are valid."""
        issues = []
        for col in ["home_score", "away_score"]:
            if col in self.df.columns:
                below_50 = (self.df[col] < 50).sum()
                above_200 = (self.df[col] > 200).sum()
                if below_50 > 0:
                    issues.append(f"{col} < 50: {below_50}")
                if above_200 > 0:
                    issues.append(f"{col} > 200: {above_200}")

        if not issues:
            return True, "Scores in valid range (50-200)"
        return False, f"Out of range: {'; '.join(issues)}"

    def _check_odds_coverage(self) -> tuple[bool, str]:
        """Check odds coverage percentage."""
        odds_cols = ["spread_line", "total_line"]
        coverage = {}
        for col in odds_cols:
            if col in self.df.columns:
                non_null = self.df[col].notna().sum()
                coverage[col] = non_null / len(self.df)

        if not coverage:
            return False, "No odds columns found"

        avg_coverage = sum(coverage.values()) / len(coverage)
        self.metrics["odds_coverage"] = avg_coverage
        
        if avg_coverage >= MIN_ODDS_COVERAGE:
            return True, f"{avg_coverage:.1%} coverage"
        return False, f"Only {avg_coverage:.1%} (need {MIN_ODDS_COVERAGE:.0%})"

    def _check_injury_coverage(self) -> tuple[bool, str]:
        """Check injury data coverage."""
        injury_cols_present = [c for c in INJURY_COLUMNS if c in self.df.columns]
        if not injury_cols_present:
            self.metrics["injury_coverage"] = 0
            return False, "No injury columns found"

        # Use home_injury_impact as proxy
        if "home_injury_impact" in self.df.columns:
            non_null = self.df["home_injury_impact"].notna().sum()
            coverage = non_null / len(self.df)
        else:
            coverage = 0

        self.metrics["injury_coverage"] = coverage
        if coverage >= MIN_INJURY_COVERAGE:
            return True, f"{coverage:.1%} coverage"
        # Warning, not failure (injury data is supplemental)
        self.warnings.append(f"Injury coverage low: {coverage:.1%}")
        return True, f"{coverage:.1%} coverage (warn: below {MIN_INJURY_COVERAGE:.0%})"

    def _check_date_range(self) -> tuple[bool, str]:
        """Check date range is reasonable."""
        if "game_date" not in self.df.columns:
            return False, "No game_date column"

        dates = pd.to_datetime(self.df["game_date"], format="mixed", utc=True)
        min_date = dates.min()
        max_date = dates.max()

        self.metrics["date_range"] = {"min": str(min_date.date()), "max": str(max_date.date())}

        # Check reasonable range
        if min_date.year >= 2022:
            return True, f"{min_date.date()} to {max_date.date()}"
        return False, f"Starts too early: {min_date.date()}"

    def _check_season_balance(self) -> tuple[bool, str]:
        """Check season distribution."""
        if "season" not in self.df.columns:
            return True, "No season column (skipped)"

        season_counts = self.df["season"].value_counts().to_dict()
        self.metrics["season_counts"] = season_counts

        # Check no season has < 10% of games
        total = len(self.df)
        small_seasons = [s for s, c in season_counts.items() if c < total * 0.1]

        if not small_seasons:
            seasons_str = ", ".join([f"{s}: {c}" for s, c in sorted(season_counts.items())])
            return True, seasons_str
        return True, f"Unbalanced: {small_seasons} (warn only)"

    def get_manifest(self, version: str) -> dict[str, Any]:
        """Generate quality manifest."""
        return {
            "version": version,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "file_name": self.file_path.name,
            "sha256": calculate_checksum(self.file_path),
            "file_size_mb": round(self.file_path.stat().st_size / (1024 * 1024), 2),
            "quality_metrics": self.metrics,
            "quality_thresholds": {
                "min_games": MIN_GAMES,
                "min_features": MIN_FEATURES,
                "min_injury_coverage": MIN_INJURY_COVERAGE,
                "min_odds_coverage": MIN_ODDS_COVERAGE,
            },
            "warnings": self.warnings,
            "schema": {
                "total_columns": len(self.df.columns),
                "columns": list(self.df.columns),
            },
        }


def upload_to_azure(
    file_path: Path,
    manifest: dict,
    version: str,
    dry_run: bool = False
) -> bool:
    """Upload training data and manifest to Azure Blob Storage."""
    print("\n" + "=" * 60)
    print("AZURE UPLOAD")
    print("=" * 60)

    # Blob paths
    data_blob = f"{BLOB_PREFIX}/{version}/{file_path.name}"
    manifest_blob = f"{BLOB_PREFIX}/{version}/manifest.json"
    latest_data_blob = f"{BLOB_PREFIX}/latest/{file_path.name}"
    latest_manifest_blob = f"{BLOB_PREFIX}/latest/manifest.json"

    print(f"  {INFO} Storage Account: {STORAGE_ACCOUNT}")
    print(f"  {INFO} Container: {CONTAINER}")
    print(f"  {INFO} Version: {version}")
    print(f"  {INFO} Data Blob: {data_blob}")
    print(f"  {INFO} Manifest Blob: {manifest_blob}")

    if dry_run:
        print(f"\n  {WARN} DRY RUN - No upload performed")
        return True

    # Save manifest locally first
    manifest_path = file_path.parent / "training_data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\n  {OK} Saved manifest locally: {manifest_path}")

    try:
        # Upload versioned data
        print(f"\n  Uploading {file_path.name}...")
        cmd = (
            f'az storage blob upload --account-name {STORAGE_ACCOUNT} '
            f'--container-name {CONTAINER} --name "{data_blob}" '
            f'--file "{file_path}" --overwrite true --auth-mode key'
        )
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        if result.returncode != 0:
            print(f"  {FAIL} Upload failed: {result.stderr}")
            return False
        print(f"  {OK} Uploaded: {data_blob}")

        # Upload manifest
        print(f"  Uploading manifest.json...")
        cmd = (
            f'az storage blob upload --account-name {STORAGE_ACCOUNT} '
            f'--container-name {CONTAINER} --name "{manifest_blob}" '
            f'--file "{manifest_path}" --overwrite true --auth-mode key'
        )
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        if result.returncode != 0:
            print(f"  {FAIL} Manifest upload failed: {result.stderr}")
            return False
        print(f"  {OK} Uploaded: {manifest_blob}")

        # Copy to latest/
        print(f"\n  Updating latest/...")
        for src, dst in [(data_blob, latest_data_blob), (manifest_blob, latest_manifest_blob)]:
            cmd = (
                f'az storage blob copy start --account-name {STORAGE_ACCOUNT} '
                f'--destination-container {CONTAINER} --destination-blob "{dst}" '
                f'--source-uri "https://{STORAGE_ACCOUNT}.blob.core.windows.net/{CONTAINER}/{src}" '
                f'--auth-mode key'
            )
            result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
            if result.returncode != 0:
                print(f"  {WARN} Copy to latest failed: {result.stderr}")
            else:
                print(f"  {OK} Copied to: {dst}")

        return True

    except Exception as e:
        print(f"  {FAIL} Upload error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Upload quality-checked training data to Azure")
    parser.add_argument("--dry-run", action="store_true", help="Validate only, no upload")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompts")
    parser.add_argument("--version", default=None, help="Version tag (default: vYYYYMMDD)")
    parser.add_argument("--file", type=Path, default=None, help="Specific file to upload")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print(" TRAINING DATA QUALITY GATE & AZURE UPLOAD")
    print("=" * 60)

    # Find training data
    if args.file:
        file_path = args.file
    else:
        file_path = find_training_data()

    if not file_path or not file_path.exists():
        print(f"\n{FAIL} Training data not found!")
        print(f"  Checked: {TRAINING_DATA_PATHS}")
        sys.exit(1)

    print(f"\n{INFO} File: {file_path}")
    print(f"{INFO} Size: {file_path.stat().st_size / (1024*1024):.1f} MB")

    # Load data
    print(f"\n{INFO} Loading data...")
    df = pd.read_csv(file_path)
    print(f"{INFO} Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Run quality checks
    checker = QualityChecker(df, file_path)
    passed = checker.run_all_checks()

    # Generate version
    version = args.version or f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Generate manifest
    manifest = checker.get_manifest(version)

    print("\n" + "=" * 60)
    print("QUALITY SUMMARY")
    print("=" * 60)
    print(f"  Total Games: {manifest['quality_metrics'].get('total_games', 'N/A'):,}")
    print(f"  Total Columns: {manifest['quality_metrics'].get('total_columns', 'N/A')}")
    print(f"  Odds Coverage: {manifest['quality_metrics'].get('odds_coverage', 0):.1%}")
    print(f"  Injury Coverage: {manifest['quality_metrics'].get('injury_coverage', 0):.1%}")
    print(f"  Date Range: {manifest['quality_metrics'].get('date_range', {})}")
    print(f"  SHA256: {manifest['sha256'][:16]}...")

    if checker.warnings:
        print(f"\n{WARN} Warnings:")
        for w in checker.warnings:
            print(f"    - {w}")

    if not passed:
        print(f"\n{FAIL} QUALITY CHECK FAILED - Upload blocked")
        sys.exit(1)

    print(f"\n{OK} QUALITY CHECK PASSED")

    # Confirm upload
    if not args.dry_run and not args.force:
        response = input(f"\nUpload to Azure Blob Storage as {version}? [y/N]: ")
        if response.lower() != "y":
            print("Upload cancelled.")
            sys.exit(0)

    # Upload
    success = upload_to_azure(file_path, manifest, version, args.dry_run)

    if success:
        print("\n" + "=" * 60)
        if args.dry_run:
            print(f"{OK} DRY RUN COMPLETE - Data validated, ready for upload")
        else:
            print(f"{OK} UPLOAD COMPLETE")
            print(f"\nSingle Source of Truth:")
            print(f"  https://{STORAGE_ACCOUNT}.blob.core.windows.net/{CONTAINER}/{BLOB_PREFIX}/{version}/")
            print(f"  https://{STORAGE_ACCOUNT}.blob.core.windows.net/{CONTAINER}/{BLOB_PREFIX}/latest/")
        print("=" * 60)
    else:
        print(f"\n{FAIL} Upload failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
