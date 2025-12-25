#!/usr/bin/env python3
"""
Extract and log feature importance from production models.

Usage:
    python scripts/extract_feature_importance.py
    python scripts/extract_feature_importance.py --market fg_moneyline
    python scripts/extract_feature_importance.py --output json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODELS_DIR = PROJECT_ROOT / "models" / "production"
OUTPUT_PATH = PROJECT_ROOT / "models" / "production" / "feature_importance.json"

# Model files mapping
MODEL_FILES = {
    "fg_spread": "fg_spread_model.joblib",
    "fg_total": "fg_total_model.joblib",
    "fg_moneyline": "fg_moneyline_model.joblib",
    "1h_spread": "1h_spread_model.pkl",
    "1h_total": "1h_total_model.pkl",
    "1h_moneyline": "1h_moneyline_model.pkl",
    "q1_spread": "q1_spread_model.joblib",
    "q1_total": "q1_total_model.joblib",
    "q1_moneyline": "q1_moneyline_model.joblib",
}

FEATURE_FILES = {
    "1h_spread": "1h_spread_features.pkl",
    "1h_total": "1h_total_features.pkl",
    "1h_moneyline": "1h_moneyline_features.pkl",
}


def load_model(market: str) -> tuple[Any, list[str]]:
    """Load a model and its feature columns."""
    model_file = MODEL_FILES.get(market)
    if not model_file:
        raise ValueError(f"Unknown market: {market}")

    model_path = MODELS_DIR / model_file
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model_data = joblib.load(model_path)

    # Extract model - handle different formats
    if isinstance(model_data, dict):
        model = model_data.get("pipeline") or model_data.get("model")
        features = model_data.get("feature_columns") or model_data.get("model_columns", [])
    else:
        model = model_data
        features = []

    # Try loading features from separate file if empty
    if not features and market in FEATURE_FILES:
        features_path = MODELS_DIR / FEATURE_FILES[market]
        if features_path.exists():
            features = joblib.load(features_path)

    return model, features


def extract_importance(model: Any, features: list[str]) -> dict[str, float]:
    """Extract feature importance from a model."""
    importance = {}

    # Try different methods to get feature importance
    try:
        # For sklearn pipelines, get the final estimator
        if hasattr(model, "named_steps"):
            # Pipeline - get classifier
            if "classifier" in model.named_steps:
                estimator = model.named_steps["classifier"]
            elif "calibrated" in model.named_steps:
                estimator = model.named_steps["calibrated"]
                if hasattr(estimator, "estimator"):
                    estimator = estimator.estimator
            else:
                estimator = list(model.named_steps.values())[-1]
        else:
            estimator = model

        # For CalibratedClassifierCV, get base estimator
        if hasattr(estimator, "calibrated_classifiers_"):
            # Average across calibrated classifiers
            all_coefs = []
            for cc in estimator.calibrated_classifiers_:
                if hasattr(cc, "estimator") and hasattr(cc.estimator, "coef_"):
                    all_coefs.append(np.abs(cc.estimator.coef_[0]))
            if all_coefs:
                avg_coef = np.mean(all_coefs, axis=0)
                for i, feat in enumerate(features):
                    if i < len(avg_coef):
                        importance[feat] = float(avg_coef[i])
                return importance

        # For logistic regression
        if hasattr(estimator, "coef_"):
            coef = np.abs(estimator.coef_[0])
            for i, feat in enumerate(features):
                if i < len(coef):
                    importance[feat] = float(coef[i])

        # For tree-based models
        elif hasattr(estimator, "feature_importances_"):
            fi = estimator.feature_importances_
            for i, feat in enumerate(features):
                if i < len(fi):
                    importance[feat] = float(fi[i])

    except Exception as e:
        print(f"Warning: Could not extract importance: {e}")

    return importance


def normalize_importance(importance: dict[str, float]) -> dict[str, float]:
    """Normalize importance values to sum to 1."""
    if not importance:
        return {}

    total = sum(importance.values())
    if total == 0:
        return importance

    return {k: round(v / total, 4) for k, v in importance.items()}


def rank_features(importance: dict[str, float]) -> list[tuple[str, float]]:
    """Rank features by importance."""
    return sorted(importance.items(), key=lambda x: x[1], reverse=True)


def extract_all_markets() -> dict:
    """Extract feature importance for all markets."""
    results = {}

    for market in MODEL_FILES:
        try:
            model, features = load_model(market)
            if not features:
                print(f"  {market}: No features found, skipping")
                continue

            raw_importance = extract_importance(model, features)
            if not raw_importance:
                print(f"  {market}: Could not extract importance")
                continue

            normalized = normalize_importance(raw_importance)
            ranked = rank_features(normalized)

            results[market] = {
                "features": features,
                "importance": dict(ranked),
                "top_3": [f[0] for f in ranked[:3]],
                "feature_count": len(features),
            }

            print(f"  {market}: {len(features)} features extracted")

        except Exception as e:
            print(f"  {market}: Error - {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Extract feature importance from models")
    parser.add_argument("--market", type=str, help="Specific market to analyze")
    parser.add_argument("--output", choices=["table", "json"], default="table", help="Output format")
    parser.add_argument("--save", action="store_true", help="Save to feature_importance.json")

    args = parser.parse_args()

    print("Extracting feature importance from production models...\n")

    if args.market:
        # Single market
        model, features = load_model(args.market)
        raw_importance = extract_importance(model, features)
        normalized = normalize_importance(raw_importance)
        ranked = rank_features(normalized)

        if args.output == "json":
            print(json.dumps({"market": args.market, "importance": dict(ranked)}, indent=2))
        else:
            print(f"\n{args.market} Feature Importance:")
            print("-" * 50)
            for feat, imp in ranked:
                bar = "█" * int(imp * 40)
                print(f"  {feat:<30} {imp:.2%} {bar}")
    else:
        # All markets
        results = extract_all_markets()

        if args.output == "json" or args.save:
            output_data = {
                "version": "6.4.0",
                "extracted_at": __import__("datetime").datetime.now().isoformat(),
                "markets": results,
            }

            if args.save:
                OUTPUT_PATH.write_text(json.dumps(output_data, indent=2))
                print(f"\nSaved to {OUTPUT_PATH}")

            if args.output == "json":
                print(json.dumps(output_data, indent=2))
        else:
            # Table output
            print("\n" + "=" * 70)
            print("FEATURE IMPORTANCE SUMMARY")
            print("=" * 70)

            for market, data in results.items():
                print(f"\n{market}:")
                print(f"  Features: {data['feature_count']}")
                print(f"  Top 3: {', '.join(data['top_3'])}")
                print("  Importance:")
                for feat, imp in list(data["importance"].items())[:5]:
                    bar = "█" * int(imp * 30)
                    print(f"    {feat:<28} {imp:.1%} {bar}")


if __name__ == "__main__":
    main()
