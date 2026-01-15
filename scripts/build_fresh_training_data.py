#!/usr/bin/env python3
"""Ensure canonical training data exists (no rebuilds).

This script does NOT merge or rebuild raw data. It only verifies that the
canonical dataset is present (2023+), optionally copying it from a provided
source path.

Design principles:
- Use the unified odds endpoint: `fetch_odds()` (NO direct `fetch_historical_odds`).
- Use standardized team name handling: `normalize_team_to_espn()`.
- Keep output deterministic and explicit; no hidden rebuilds.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Optional

# Single source imports (required by architecture tests)
from src.ingestion.the_odds import fetch_odds  # noqa: F401
from src.ingestion.standardize import normalize_team_to_espn  # noqa: F401

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def _copy_source(source: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, output)


def ensure_training_data(output: Path, source: Optional[Path]) -> Path:
    """Ensure canonical training data exists at output; optionally copy from source."""
    if source is not None:
        if not source.exists():
            raise FileNotFoundError(f"Source training data not found: {source}")
        _copy_source(source, output)
        return output

    if output.exists():
        return output

    raise FileNotFoundError(
        "Canonical training data not found. Provide --source or place "
        f"{output} before running backtests."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default=None,
        help="Optional source CSV to copy into the canonical training_data.csv path",
    )
    parser.add_argument(
        "--output",
        default=str(PROCESSED_DIR / "training_data.csv"),
        help="Canonical training data output path",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    source_path = Path(args.source) if args.source else None
    out = ensure_training_data(output=output_path, source=source_path)
    print(f"[OK] Canonical training data ready: {out}")
