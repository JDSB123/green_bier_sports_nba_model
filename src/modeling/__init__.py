"""NBA prediction modeling package."""
from src.modeling.features import FeatureEngineer
from src.modeling.dataset import DatasetBuilder
from src.modeling.models import SpreadsModel, TotalsModel, MoneylineModel

# New modules for v1.2.0
try:
    from src.modeling.calibration import ModelCalibrator, CalibrationMetrics
    from src.modeling.interpretability import (
        analyze_model,
        get_linear_coefficients,
        get_tree_importance,
    )
except ImportError:
    # Graceful fallback if dependencies missing
    ModelCalibrator = None
    CalibrationMetrics = None
    analyze_model = None
    get_linear_coefficients = None
    get_tree_importance = None

__all__ = [
    "FeatureEngineer",
    "DatasetBuilder",
    "SpreadsModel",
    "TotalsModel",
    "MoneylineModel",
    "ModelCalibrator",
    "CalibrationMetrics",
    "analyze_model",
    "get_linear_coefficients",
    "get_tree_importance",
]
