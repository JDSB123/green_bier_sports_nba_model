#!/usr/bin/env python3
"""Sync model metadata artifacts to the current VERSION and active models."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models" / "production"

MODEL_PACK_PATH = MODELS_DIR / "model_pack.json"
MANIFEST_PATH = MODELS_DIR / "manifest.json"
MODEL_FEATURES_PATH = MODELS_DIR / "model_features.json"
FEATURE_IMPORTANCE_PATH = MODELS_DIR / "feature_importance.json"
VERSION_PATH = PROJECT_ROOT / "VERSION"


@dataclass
class ActiveModelInfo:
    version_id: str
    model_type: str
    trained_at: Optional[str]
    metrics: Dict[str, Any]
    file_path: Optional[str]


def _run_git(args: List[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _latest_entry_for_version(entries: List[Dict[str, Any]], version_id: str) -> Optional[Dict[str, Any]]:
    matches = [entry for entry in entries if entry.get("version") == version_id]
    if not matches:
        return None
    matches.sort(key=lambda item: item.get("trained_at") or "")
    return matches[-1]


def _collect_active_models(manifest: Dict[str, Any]) -> Dict[str, ActiveModelInfo]:
    active_models = manifest.get("active_models", {})
    entries = manifest.get("versions", [])
    result: Dict[str, ActiveModelInfo] = {}

    for market_key, version_id in active_models.items():
        entry = _latest_entry_for_version(entries, version_id)
        if not entry:
            result[market_key] = ActiveModelInfo(
                version_id=version_id,
                model_type=market_key,
                trained_at=None,
                metrics={},
                file_path=None,
            )
            continue
        result[market_key] = ActiveModelInfo(
            version_id=version_id,
            model_type=entry.get("model_type", market_key),
            trained_at=entry.get("trained_at"),
            metrics=entry.get("metrics", {}),
            file_path=entry.get("file_path"),
        )

    return result


def _parse_iso_dt(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _sync_model_pack() -> None:
    if not MODEL_PACK_PATH.exists():
        raise FileNotFoundError(f"Missing model pack at {MODEL_PACK_PATH}")
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing manifest at {MANIFEST_PATH}")
    if not MODEL_FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing model features at {MODEL_FEATURES_PATH}")

    version = VERSION_PATH.read_text(encoding="utf-8").strip()
    git_commit = _run_git(["rev-parse", "HEAD"])

    model_pack = _read_json(MODEL_PACK_PATH)
    manifest = _read_json(MANIFEST_PATH)
    model_features = _read_json(MODEL_FEATURES_PATH)
    active_models = _collect_active_models(manifest)

    model_pack["version"] = version
    model_pack["git_tag"] = version
    model_pack["git_commit"] = git_commit

    # Update created_at to newest trained_at from active models
    trained_dates = [
        _parse_iso_dt(info.trained_at) for info in active_models.values() if info.trained_at
    ]
    trained_dates = [dt for dt in trained_dates if dt is not None]
    if trained_dates:
        latest_dt = max(trained_dates)
        model_pack["created_at"] = latest_dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    model_pack["last_reviewed"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    model_pack["release_notes"] = (
        f"{version} - Metadata sync, strict market-line enforcement, normalized odds outcomes, "
        "and sharp-weighted filters aligned to production min-edge thresholds. "
        "Includes 63-feature retrained FG/1H models."
    )

    # Update deployment ACR tag
    deployment = model_pack.get("deployment", {})
    if isinstance(deployment, dict):
        deployment["acr"] = f"nbagbsacr.azurecr.io/nba-gbsv-api:{version}"
        model_pack["deployment"] = deployment

    # Update per-market status features and retrain date
    market_status = model_pack.get("market_status", {})
    for market_key, status in market_status.items():
        if not isinstance(status, dict):
            continue
        features = model_features.get("markets", {}).get(market_key)
        if isinstance(features, list):
            status["features"] = len(features)
        active_info = active_models.get(market_key)
        if active_info and active_info.trained_at:
            status["retrained"] = active_info.trained_at.split("T")[0]
            if active_info.file_path:
                status["model_file"] = active_info.file_path
        market_status[market_key] = status
    model_pack["market_status"] = market_status

    _write_json(MODEL_PACK_PATH, model_pack)


def _extract_coefficients(model_payload: Dict[str, Any]) -> Optional[np.ndarray]:
    pipeline = model_payload.get("pipeline")
    if pipeline is None or not hasattr(pipeline, "calibrated_classifiers_"):
        return None

    coefs = []
    for calibrated in pipeline.calibrated_classifiers_:
        estimator = getattr(calibrated, "estimator", None)
        if estimator is None:
            continue
        if hasattr(estimator, "named_steps") and "est" in estimator.named_steps:
            base = estimator.named_steps["est"]
            if hasattr(base, "coef_"):
                coefs.append(np.abs(base.coef_.ravel()))
    if not coefs:
        return None
    return np.mean(coefs, axis=0)


def _sync_feature_importance() -> None:
    version = VERSION_PATH.read_text(encoding="utf-8").strip()
    markets = {
        "fg_spread": "fg_spread_model.joblib",
        "fg_total": "fg_total_model.joblib",
        "1h_spread": "1h_spread_model.joblib",
        "1h_total": "1h_total_model.joblib",
    }

    output: Dict[str, Any] = {
        "version": version,
        "extracted_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "markets": {},
    }

    for market_key, filename in markets.items():
        model_path = MODELS_DIR / filename
        if not model_path.exists():
            continue
        model_payload = joblib.load(model_path)
        feature_columns = model_payload.get("feature_columns") or []
        importance = _extract_coefficients(model_payload)
        if importance is None:
            importance_list: List[float] = []
        else:
            importance_list = [float(x) for x in importance]

        top_3 = []
        if importance_list and feature_columns:
            pairs = list(zip(feature_columns, importance_list))
            pairs.sort(key=lambda item: item[1], reverse=True)
            top_3 = [name for name, _ in pairs[:3]]

        output["markets"][market_key] = {
            "features": feature_columns,
            "importance": importance_list,
            "top_3": top_3,
            "feature_count": len(feature_columns),
        }

    _write_json(FEATURE_IMPORTANCE_PATH, output)


def main() -> None:
    _sync_model_pack()
    _sync_feature_importance()
    print("Synced model_pack.json and feature_importance.json")


if __name__ == "__main__":
    main()
