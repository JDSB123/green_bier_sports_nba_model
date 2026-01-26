from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import joblib

from src.modeling.unified_features import MODEL_CONFIGS
from src.utils.version import resolve_version

CONTRACT_FILENAME = "model_features.json"


def _contract_path(models_dir: Path) -> Path:
    return models_dir / CONTRACT_FILENAME


@lru_cache(maxsize=4)
def load_model_feature_contract(models_dir: Path) -> Optional[Dict]:
    """Load the model feature contract if it exists."""
    path = _contract_path(models_dir)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _load_joblib_features(models_dir: Path, market_key: str) -> List[str]:
    config = MODEL_CONFIGS.get(market_key)
    if not config:
        return []

    model_path = models_dir / config["model_file"]
    if not model_path.exists():
        return []

    data = joblib.load(model_path)
    features = data.get("feature_columns") or data.get("model_columns") or []
    if features:
        return list(features)

    model = data.get("pipeline") or data.get("model")
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    return []


def get_market_features(models_dir: Path, market_key: str) -> List[str]:
    """Return the required feature list for a market (prefers contract)."""
    contract = load_model_feature_contract(models_dir)
    if contract:
        markets = contract.get("markets", {})
        features = markets.get(market_key) or []
        if features:
            return list(features)

    return _load_joblib_features(models_dir, market_key)


def get_union_features(models_dir: Path) -> List[str]:
    """Return the union of all market features (prefers contract)."""
    contract = load_model_feature_contract(models_dir)
    if contract:
        union = contract.get("union") or []
        if union:
            return list(union)

    union_set = set()
    for market_key in MODEL_CONFIGS.keys():
        union_set.update(get_market_features(models_dir, market_key))
    return sorted(union_set)


def export_model_feature_contract(models_dir: Path, out_path: Optional[Path] = None) -> Path:
    """Generate and write the model feature contract JSON."""
    markets: Dict[str, List[str]] = {}
    errors = []

    for market_key in MODEL_CONFIGS.keys():
        features = _load_joblib_features(models_dir, market_key)
        if not features:
            errors.append(market_key)
        markets[market_key] = list(features)

    if errors:
        missing = ", ".join(errors)
        raise FileNotFoundError(f"Missing or empty feature columns for: {missing}")

    union = sorted({feat for feats in markets.values() for feat in feats})

    payload = {
        "version": resolve_version(),
        "generated_at_utc": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "models_dir": str(models_dir),
        "markets": markets,
        "union": union,
    }

    path = out_path or _contract_path(models_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    # Invalidate cache for subsequent reads in the same process
    load_model_feature_contract.cache_clear()
    return path
