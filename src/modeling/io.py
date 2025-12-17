"""
Model I/O helpers: safe persistence for sklearn pipelines and model metadata.

Provides `save_model` and `load_model` which use `joblib` and maintain a
simple `manifest.json` in the same directory to record model metadata.
"""
from __future__ import annotations
import json
import os
from typing import Any, Dict, Optional
from datetime import datetime, UTC
import joblib


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def save_model(
    payload: Dict[str, Any], 
    path: str, 
    manifest_name: str = "manifest.json",
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a model payload (pipeline + metadata) to disk using joblib.

    Payload should be a dict with keys like 'pipeline', 'meta', and
    'feature_columns'.

    Model versioning:
    - version: Semantic version string (e.g., "1.1.0")
    - features_hash: Hash of feature column names for compatibility checking
    
    Args:
        payload: Model payload dict
        path: Path to save model
        manifest_name: Name of manifest file
        metrics: Optional dict with evaluation metrics (accuracy, roi, etc.)
    """
    _ensure_dir(path)

    # Add features hash for compatibility checking
    feature_cols = payload.get("feature_columns", [])
    features_hash = hash(tuple(sorted(feature_cols))) if feature_cols else None

    # Write artifact
    joblib.dump(payload, path)

    # Update manifest
    manifest_path = os.path.join(os.path.dirname(path), manifest_name)
    meta = payload.get("meta", {})
    entry = {
        "path": os.path.basename(path),
        "saved_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "version": meta.get("version", "1.0.0"),
        "model_type": meta.get("model_type", "unknown"),
        "n_features": len(feature_cols),
        "features_hash": features_hash,
        "meta": meta,
    }
    
    # Add metrics if provided
    if metrics:
        entry["metrics"] = {
            "train_accuracy": metrics.get("train_accuracy"),
            "test_accuracy": metrics.get("test_accuracy"),
            "train_roi": metrics.get("train_roi"),
            "test_roi": metrics.get("test_roi"),
            "cv_accuracy_mean": metrics.get("cv_accuracy_mean"),
            "cv_accuracy_std": metrics.get("cv_accuracy_std"),
            "brier_score": metrics.get("brier_score"),
            "calibration_error": metrics.get("calibration_error"),
            "n_train_samples": metrics.get("n_train_samples"),
            "n_test_samples": metrics.get("n_test_samples"),
        }
        # Remove None values
        entry["metrics"] = {k: v for k, v in entry["metrics"].items() if v is not None}

    try:
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        else:
            manifest = {"models": [], "latest": {}}

        manifest.setdefault("models", []).append(entry)

        # Track latest model per type
        model_name = os.path.basename(path).replace(".joblib", "")
        model_name = model_name.replace(".pkl", "")
        manifest.setdefault("latest", {})[model_name] = entry

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    except Exception:
        # Don't block saving model if manifest update fails
        pass


def load_model(path: str) -> Dict[str, Any]:
    """Load a model artifact saved by `save_model`.

    Returns the original payload dict.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)
