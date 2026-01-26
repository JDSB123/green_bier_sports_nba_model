"""
NBA prediction modeling package.

Key modules:
- unified_features: Single source of truth for all feature definitions
- features: FeatureEngineer for training data generation
- models: Model classes (SpreadsModel, TotalsModel, etc.)
- dataset: DatasetBuilder for training pipeline
"""

from src.modeling.dataset import DatasetBuilder
from src.modeling.features import FeatureEngineer
from src.modeling.models import SpreadsModel, TotalsModel
from src.modeling.unified_features import (
    FEATURE_DEFAULTS,
    MODEL_CONFIGS,
    MODEL_REGISTRY,
    PERIOD_SCALING,
    UNIFIED_FEATURE_NAMES,
    get_model_features,
)

# Calibration and interpretability (optional)
try:
    from src.modeling.calibration import CalibrationMetrics, ModelCalibrator
    from src.modeling.interpretability import (
        analyze_model,
        get_linear_coefficients,
        get_tree_importance,
    )
except ImportError:
    ModelCalibrator = None
    CalibrationMetrics = None
    analyze_model = None
    get_linear_coefficients = None
    get_tree_importance = None

__all__ = [
    # Core
    "FeatureEngineer",
    "DatasetBuilder",
    "SpreadsModel",
    "TotalsModel",
    # Features (from unified_features)
    "UNIFIED_FEATURE_NAMES",
    "MODEL_REGISTRY",
    "MODEL_CONFIGS",
    "PERIOD_SCALING",
    "FEATURE_DEFAULTS",
    "get_model_features",
    # Calibration (optional)
    "ModelCalibrator",
    "CalibrationMetrics",
    "analyze_model",
    "get_linear_coefficients",
    "get_tree_importance",
]
