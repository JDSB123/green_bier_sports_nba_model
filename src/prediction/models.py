"""
Model loading and management for predictions.

NBA v33.0.11.0: 4 independent markets (1H + FG spreads/totals)
"""
from pathlib import Path
from typing import Tuple, Any, List
import joblib
from src.modeling.model_tracker import ModelTracker


def load_spread_model(models_dir: Path) -> Tuple[Any, List[str]]:
    """
    Load FG spreads model with feature columns.

    Args:
        models_dir: Path to models directory

    Returns:
        Tuple of (model, feature_columns)

    Raises:
        FileNotFoundError: If model not found
    """
    # Try to get active version from tracker
    tracker = ModelTracker()
    spreads_version = tracker.get_active_version("spreads")
    model_path = None

    if spreads_version:
        info = tracker.get_version_info(spreads_version)
        if info and info.get("file_path"):
            candidate = models_dir / info["file_path"]
            if candidate.exists():
                model_path = candidate

    # Fallback to standard names
    if model_path is None:
        model_path = models_dir / "fg_spread_model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Spreads model not found in {models_dir}")

    model_data = joblib.load(model_path)

    # Support both formats
    model = model_data.get("pipeline") or model_data.get("model")
    feature_cols = model_data.get(
        "feature_columns") or model_data.get("model_columns", [])

    return model, feature_cols


def load_total_model(models_dir: Path) -> Tuple[Any, List[str]]:
    """
    Load FG totals model with feature columns.

    Args:
        models_dir: Path to models directory

    Returns:
        Tuple of (model, feature_columns)

    Raises:
        FileNotFoundError: If model not found
    """
    # Try to get active version from tracker
    tracker = ModelTracker()
    totals_version = tracker.get_active_version("totals")
    model_path = None

    if totals_version:
        info = tracker.get_version_info(totals_version)
        if info and info.get("file_path"):
            candidate = models_dir / info["file_path"]
            if candidate.exists():
                model_path = candidate

    # Fallback to standard names
    if model_path is None:
        model_path = models_dir / "fg_total_model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Totals model not found in {models_dir}")

    model_data = joblib.load(model_path)

    # Support both formats
    model = model_data.get("pipeline") or model_data.get("model")
    feature_cols = model_data.get(
        "feature_columns") or model_data.get("model_columns", [])

    return model, feature_cols


def load_first_half_spread_model(models_dir: Path) -> Tuple[Any, List[str]]:
    """
    Load 1H spread model with feature columns.

    Args:
        models_dir: Path to models directory

    Returns:
        Tuple of (model, feature_columns)

    Raises:
        FileNotFoundError: If model not found
    """
    # Standardized .joblib format (combined model + features)
    model_path = models_dir / "1h_spread_model.joblib"

    # Fallback to legacy .pkl format for backwards compatibility
    legacy_model_path = models_dir / "1h_spread_model.pkl"
    legacy_features_path = models_dir / "1h_spread_features.pkl"

    if model_path.exists():
        # New combined format
        model_data = joblib.load(model_path)
        model = model_data.get("pipeline") or model_data.get("model")
        feature_cols = model_data.get(
            "feature_columns") or model_data.get("model_columns", [])
        return model, feature_cols
    elif legacy_model_path.exists():
        # Legacy separate files format
        model = joblib.load(legacy_model_path)
        feature_cols = joblib.load(
            legacy_features_path) if legacy_features_path.exists() else []
        return model, feature_cols
    else:
        raise FileNotFoundError(
            f"1H spread model not found at {model_path} or {legacy_model_path}. "
            f"Run: python scripts/model_train_all.py --market 1h"
        )


def load_first_half_total_model(models_dir: Path) -> Tuple[Any, List[str]]:
    """
    Load 1H total model with feature columns.

    Args:
        models_dir: Path to models directory

    Returns:
        Tuple of (model, feature_columns)

    Raises:
        FileNotFoundError: If model not found
    """
    # Standardized .joblib format (combined model + features)
    model_path = models_dir / "1h_total_model.joblib"

    # Fallback to legacy .pkl format for backwards compatibility
    legacy_model_path = models_dir / "1h_total_model.pkl"
    legacy_features_path = models_dir / "1h_total_features.pkl"

    if model_path.exists():
        # New combined format
        model_data = joblib.load(model_path)
        model = model_data.get("pipeline") or model_data.get("model")
        feature_cols = model_data.get(
            "feature_columns") or model_data.get("model_columns", [])
        return model, feature_cols
    elif legacy_model_path.exists():
        # Legacy separate files format
        model = joblib.load(legacy_model_path)
        feature_cols = joblib.load(
            legacy_features_path) if legacy_features_path.exists() else []
        return model, feature_cols
    else:
        raise FileNotFoundError(
            f"1H total model not found at {model_path} or {legacy_model_path}. "
            f"Run: python scripts/model_train_all.py --market 1h"
        )
