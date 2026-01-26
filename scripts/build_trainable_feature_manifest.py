#!/usr/bin/env python3
"""
Build a trainable feature manifest from training data + inference-time feature set.

Default behavior:
- Uses unified inference-time features (src/modeling/unified_features.py)
- Adds line-context features used by prediction engine
- Excludes leaky/label/metadata columns
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Set

import pandas as pd

from src.config import PROJECT_ROOT, settings
from src.modeling.unified_features import (
    LEAKY_FEATURES_BLACKLIST,
    MODEL_CONFIGS,
    UNIFIED_FEATURE_NAMES,
)

EXTRA_CONTEXT_FEATURES: Set[str] = {
    # Period-specific line context injected by prediction engine
    "fg_spread_line",
    "fg_total_line",
    "1h_spread_line",
    "1h_total_line",
    "fh_spread_line",
    "fh_total_line",
    "spread_vs_predicted_1h",
    "total_vs_predicted_1h",
    "fh_spread_vs_predicted",
    "fh_total_vs_predicted",
}

# Columns that are metadata/non-features in training data
METADATA_COLUMNS: Set[str] = {
    "season",
    "date",
    "regular",
    "playoffs",
    "home",
    "away",
    "game_date",
    "home_team",
    "away_team",
    "match_key",
    "whos_favored",
}

# Known leaky/outcome columns in training data (beyond blacklist)
EXTRA_LEAKY_COLUMNS: Set[str] = {
    # Final scores / actual outcomes
    "score_home",
    "score_away",
    "home_score",
    "away_score",
    "fg_margin",
    "1h_margin",
    "fg_total_actual",
    "1h_total_actual",
    "home_1h",
    "away_1h",
    "home_2h",
    "away_2h",
    # Quarter scores for this game
    "home_q1",
    "away_q1",
    "home_q2",
    "away_q2",
    "home_q3",
    "away_q3",
    "home_q4",
    "away_q4",
    "q2_home",
    "q2_away",
    "q3_home",
    "q3_away",
    "q4_home",
    "q4_away",
    "ot_home",
    "ot_away",
}


def _load_training_df(path: Path, nrows: int) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows, low_memory=False)


def _as_sorted_list(items: Iterable[str]) -> List[str]:
    return sorted({str(x) for x in items})


def build_manifest(
    training_path: Path,
    include_non_unified: bool,
    sample_rows: int,
    drop_constant: bool,
) -> dict:
    df = _load_training_df(training_path, nrows=sample_rows)
    all_cols = list(df.columns)
    numeric_cols = set(df.select_dtypes(include="number").columns)

    label_cols = {cfg["label_col"] for cfg in MODEL_CONFIGS.values()}
    leaky = set(LEAKY_FEATURES_BLACKLIST) | label_cols | EXTRA_LEAKY_COLUMNS

    inference_features = set(UNIFIED_FEATURE_NAMES) | EXTRA_CONTEXT_FEATURES

    if include_non_unified:
        candidate = set(all_cols)
    else:
        candidate = inference_features & set(all_cols)

    # Remove metadata/leaky/non-numeric columns
    candidate -= METADATA_COLUMNS
    candidate -= leaky
    candidate &= numeric_cols

    # Drop constant columns (sample-based)
    constant_cols: Set[str] = set()
    for col in candidate:
        series = df[col]
        if series.nunique(dropna=True) <= 1:
            constant_cols.add(col)
    if drop_constant:
        candidate -= constant_cols

    missing_in_training = inference_features - set(all_cols)
    non_numeric = set(all_cols) - numeric_cols

    manifest_features = _as_sorted_list(candidate)

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_data_path": str(training_path),
        "sample_rows": sample_rows,
        "feature_count": len(manifest_features),
        "features": manifest_features,
        "notes": {
            "include_non_unified": include_non_unified,
            "drop_constant": drop_constant,
        },
        "excluded": {
            "leaky_or_labels": _as_sorted_list(leaky & set(all_cols)),
            "metadata": _as_sorted_list(METADATA_COLUMNS & set(all_cols)),
            "non_numeric": _as_sorted_list(non_numeric & set(all_cols)),
            "constant": _as_sorted_list(constant_cols),
        },
        "inference_features_missing_in_training": _as_sorted_list(missing_in_training),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build trainable feature manifest from training data",
    )
    parser.add_argument(
        "--training-file",
        type=str,
        default=str(Path(settings.data_processed_dir) / "training_data.csv"),
        help="Path to training data CSV (default: data/processed/training_data.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "models" / "production" / "trainable_features.json"),
        help="Output path for manifest JSON",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=5000,
        help="Rows to sample for dtype/constant detection (default: 5000)",
    )
    parser.add_argument(
        "--include-non-unified",
        action="store_true",
        help="Include numeric columns not in the unified inference feature set",
    )
    parser.add_argument(
        "--drop-constant",
        action="store_true",
        help="Drop constant columns from the manifest",
    )
    args = parser.parse_args()

    training_path = Path(args.training_file)
    if not training_path.exists():
        raise SystemExit(f"Training file not found: {training_path}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(
        training_path=training_path,
        include_non_unified=bool(args.include_non_unified),
        sample_rows=int(args.sample_rows),
        drop_constant=bool(args.drop_constant),
    )

    output_path.write_text(json.dumps(manifest, indent=2))
    print(f"[OK] Wrote trainable feature manifest to {output_path}")
    print(f"     Features: {manifest['feature_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
