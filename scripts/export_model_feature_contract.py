#!/usr/bin/env python3
"""
Export model feature contract from production joblib files.

Writes models/production/model_features.json as the prediction-time
single source of truth for required feature lists.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.model_features import export_model_feature_contract


def main() -> int:
    parser = argparse.ArgumentParser(description="Export model feature contract JSON.")
    parser.add_argument(
        "--models-dir",
        default="models/production",
        help="Path to models directory (default: models/production)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output path (default: <models-dir>/model_features.json)",
    )
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    out_path = Path(args.out) if args.out else None
    written = export_model_feature_contract(models_dir=models_dir, out_path=out_path)
    print(f"[OK] Wrote model feature contract: {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
