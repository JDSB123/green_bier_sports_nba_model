#!/usr/bin/env python3
"""
Build training datasets directly from ingestion outputs.

Reads:
- data/processed/odds_the_odds.csv (normalized odds & lines)
- data/processed/game_outcomes.csv (API-Basketball results)

Outputs:
- data/processed/training_data.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.modeling.dataset import DatasetBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description="Build training data from ingestion tables.")
    parser.add_argument(
        "--odds",
        type=Path,
        default=Path(settings.data_processed_dir) / "odds_the_odds.csv",
        help="Path to normalized odds CSV.",
    )
    parser.add_argument(
        "--outcomes",
        type=Path,
        default=Path(settings.data_processed_dir) / "game_outcomes.csv",
        help="Path to normalized game outcomes CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(settings.data_processed_dir) / "training_data.csv",
        help="Destination CSV for training data.",
    )
    args = parser.parse_args()

    builder = DatasetBuilder()
    df = builder.build_training_dataset(
        odds_path=str(args.odds),
        outcomes_path=str(args.outcomes),
        output_path=str(args.output),
    )

    if df.empty:
        raise SystemExit(
            "Training dataset is empty. Ensure odds/outcome CSVs exist and contain overlapping games."
        )

    print(f"Built training dataset with {len(df):,} games at {args.output}")

    # Quick sanity stats
    try:
        date_min = df["game_date"].min()
        date_max = df["game_date"].max()
        spread_hits = 0
        spread_series = df.get("spread_covered")
        if hasattr(spread_series, "sum"):
            spread_hits = int(spread_series.sum())
        print(
            f"Date range: {date_min} → {date_max} | Spread targets: {spread_hits}/{len(df)}"
        )
    except Exception:
        pass


if __name__ == "__main__":
    main()

