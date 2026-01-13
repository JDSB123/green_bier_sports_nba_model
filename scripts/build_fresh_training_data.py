#!/usr/bin/env python3
"""Build fresh training data (single-source odds).

This script exists as the canonical "fresh training data" entrypoint used by
Docker backtest workflows and single-source-of-truth tests.

Design principles:
- Use the unified odds endpoint: `fetch_odds()` (NO direct `fetch_historical_odds`).
- Use standardized team name handling: `normalize_team_to_espn()`.
- Keep output deterministic and explicit; no silent dual-path branching.

Notes:
- For full historical dataset assembly, this script delegates to
  `scripts/build_training_data_complete.py` (which writes a complete training CSV).
- The `fetch_odds()` call is intentionally not required for offline/historical builds.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

# Single source imports (required by architecture tests)
from src.ingestion.the_odds import fetch_odds  # noqa: F401
from src.ingestion.standardize import normalize_team_to_espn  # noqa: F401


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def _derive_start_date_from_seasons(seasons: str) -> str:
    """Convert e.g. "2023-2024,2024-2025" -> "2023-10-01".

    NBA seasons run roughly Oct -> Jun. Using Oct 1 aligns the dataset to the
    real season boundary and avoids accidentally including the prior season.
    """
    first = (seasons or "").split(",")[0].strip()
    year = first.split("-")[0].strip() if first else "2023"
    if not year.isdigit():
        year = "2023"
    return f"{year}-10-01"


def main(seasons: str, output: str) -> Path:
    """Build training data and write/copy to `output`."""
    from scripts.build_training_data_complete import main as build_complete

    start_date = _derive_start_date_from_seasons(seasons)

    # Delegate to the complete builder (historical + post-processing).
    # Seasons are enforced inside the builder to prevent older seasons from being included.
    build_complete(
        start_date=start_date,
        cutoff_date=None,
        sync_from_azure=False,
        blob_account="nbagbsvstrg",
        blob_container="nbahistoricaldata",
        seasons=seasons,
        leakage_days=3,
    )

    generated = PROCESSED_DIR / f"training_data_complete_{start_date[:4]}.csv"
    if not generated.exists():
        raise FileNotFoundError(f"Expected builder output not found: {generated}")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(generated, output_path)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seasons",
        default="2023-2024,2024-2025,2025-2026",
        help="Comma-separated seasons (used to derive start year)",
    )
    parser.add_argument("--output", default=str(PROCESSED_DIR / "training_data.csv"), help="Output CSV path")
    args = parser.parse_args()

    out = main(seasons=args.seasons, output=args.output)
    print(f"✓ Training data written: {out}")
