from __future__ import annotations

import inspect
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import joblib

from src.config import settings
from src.modeling.period_features import MODEL_CONFIGS
from src.utils.logging import get_logger
from src.utils.markets import get_expected_markets, get_market_catalog
from src.utils.security import validate_required_api_keys

logger = get_logger(__name__)


class StartupIntegrityError(RuntimeError):
    """Raised when startup integrity checks fail."""


@dataclass
class StartupIntegrityReport:
    errors: List[str]
    warnings: List[str]


def _collect_builder_feature_keys() -> Set[str]:
    from src.features.rich_features import RichFeatureBuilder

    source = inspect.getsource(RichFeatureBuilder.build_game_features)
    keys = set()
    keys.update(re.findall(r'"([a-z0-9_]+)"\s*:', source))
    keys.update(re.findall(r"features\[['\"]([a-z0-9_]+)['\"]\]", source))
    return keys


def _load_model_features(models_dir: Path, market_key: str) -> Tuple[List[str], Optional[str]]:
    config = MODEL_CONFIGS.get(market_key)
    if not config:
        return [], f"Unknown market key in MODEL_CONFIGS: {market_key}"

    model_path = models_dir / config["model_file"]
    features: List[str] = []

    if model_path.suffix == ".joblib" and model_path.exists():
        data = joblib.load(model_path)
        if isinstance(data, dict):
            features = data.get("feature_columns", []) or []
        else:
            return [], f"{market_key}: joblib payload missing feature_columns"
    elif model_path.suffix == ".pkl":
        features_file = config.get("features_file")
        if features_file:
            features_path = models_dir / features_file
            if features_path.exists():
                with open(features_path, "rb") as f:
                    features = pickle.load(f)
            else:
                return [], f"{market_key}: features file missing ({features_file})"
        elif model_path.exists():
            return [], f"{market_key}: no features file configured for {model_path.name}"
        else:
            return [], f"{market_key}: model file missing ({model_path.name})"
    else:
        if not model_path.exists():
            return [], f"{market_key}: model file missing ({model_path.name})"

    if not features:
        features = config.get("features", []) or []

    return list(features), None


def _validate_feature_alignment(models_dir: Path, expected_markets: List[str]) -> StartupIntegrityReport:
    errors: List[str] = []
    warnings: List[str] = []

    builder_keys = _collect_builder_feature_keys()
    if not builder_keys:
        errors.append("Feature builder keys not detected (RichFeatureBuilder.build_game_features).")
        return StartupIntegrityReport(errors=errors, warnings=warnings)

    for market_key in expected_markets:
        features, err = _load_model_features(models_dir, market_key)
        if err:
            errors.append(err)
            continue
        missing = sorted(set(features) - builder_keys)
        if missing:
            preview = ", ".join(missing[:8])
            suffix = f" (+{len(missing) - 8} more)" if len(missing) > 8 else ""
            errors.append(f"{market_key}: missing {len(missing)} feature aliases ({preview}{suffix})")

    return StartupIntegrityReport(errors=errors, warnings=warnings)


def _validate_api_auth_config() -> Optional[str]:
    require_auth = os.getenv("REQUIRE_API_AUTH", "true").lower() == "true"
    service_key = os.getenv("SERVICE_API_KEY", "").strip()
    if require_auth and not service_key:
        return "REQUIRE_API_AUTH=true but SERVICE_API_KEY is empty"
    return None


def _validate_filter_thresholds() -> List[str]:
    """
    Validate that filter thresholds are explicitly set (not using defaults).

    Returns list of warnings for any thresholds using defaults.
    """
    warnings = []

    # Check if env vars are explicitly set vs using defaults
    filter_env_vars = [
        ("FILTER_SPREAD_MIN_CONFIDENCE", "0.62"),
        ("FILTER_SPREAD_MIN_EDGE", "2.0"),
        ("FILTER_TOTAL_MIN_CONFIDENCE", "0.72"),
        ("FILTER_TOTAL_MIN_EDGE", "3.0"),
        ("FILTER_1H_SPREAD_MIN_CONFIDENCE", "0.68"),
        ("FILTER_1H_SPREAD_MIN_EDGE", "1.5"),
        ("FILTER_1H_TOTAL_MIN_CONFIDENCE", "0.66"),
        ("FILTER_1H_TOTAL_MIN_EDGE", "2.0"),
    ]

    for env_var, default_value in filter_env_vars:
        env_value = os.getenv(env_var)
        if env_value is None:
            warnings.append(f"{env_var} not set - using default {default_value}. Consider setting explicitly for production.")

    return warnings


def run_startup_integrity_checks(project_root: Path, models_dir: Path) -> None:
    errors: List[str] = []
    warnings: List[str] = []

    # Required API keys (non-empty)
    api_result = validate_required_api_keys()
    errors.extend(api_result.errors)
    warnings.extend(api_result.warnings)

    api_auth_error = _validate_api_auth_config()
    if api_auth_error:
        errors.append(api_auth_error)

    # Filter thresholds validation (warn if using defaults)
    filter_warnings = _validate_filter_thresholds()
    warnings.extend(filter_warnings)

    # Enabled market list alignment
    expected_markets = get_expected_markets()
    catalog = get_market_catalog(project_root, settings.data_processed_dir)
    if catalog.source != "model_pack":
        errors.append("model_pack.json not found - cannot validate enabled market list")
    else:
        enabled = set(catalog.markets)
        expected = set(expected_markets)
        if enabled != expected:
            missing = sorted(expected - enabled)
            extra = sorted(enabled - expected)
            if missing:
                errors.append(f"model_pack missing markets: {', '.join(missing)}")
            if extra:
                errors.append(f"model_pack has unexpected markets: {', '.join(extra)}")

    # Feature alias alignment
    feature_report = _validate_feature_alignment(models_dir, expected_markets)
    errors.extend(feature_report.errors)
    warnings.extend(feature_report.warnings)

    if errors:
        lines = ["STARTUP INTEGRITY CHECK FAILED", "-" * 40, "ERRORS:"]
        lines.extend([f"- {err}" for err in errors])
        if warnings:
            lines.append("WARNINGS:")
            lines.extend([f"- {warn}" for warn in warnings])
        raise StartupIntegrityError("\n".join(lines))

    if warnings:
        for warn in warnings:
            logger.warning(f"Startup integrity warning: {warn}")
