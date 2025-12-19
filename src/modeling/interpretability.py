"""
Model interpretability utilities for NBA prediction models.

Provides:
- Coefficient analysis for linear models
- Feature importance for tree models
- SHAP values (when available)
- Feature correlation analysis
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.inspection import permutation_importance
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# SHAP is optional
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


@dataclass
class FeatureImportance:
    """Container for feature importance results."""
    feature: str
    importance: float
    std: Optional[float] = None  # For permutation importance
    direction: Optional[str] = None  # "positive" or "negative" for linear models


def get_linear_coefficients(
    model: Any,
    feature_names: List[str],
    scaler: Optional[Any] = None,
) -> List[FeatureImportance]:
    """
    Extract and interpret coefficients from linear models.
    
    For logistic regression: coefficients represent log-odds change per unit feature.
    For ridge regression: coefficients represent target change per unit feature.
    
    Args:
        model: Fitted linear model (LogisticRegression or Ridge)
        feature_names: Names of features in order
        scaler: Optional StandardScaler for scaling interpretation
        
    Returns:
        List of FeatureImportance sorted by absolute importance
    """
    if not hasattr(model, "coef_"):
        raise ValueError("Model does not have coefficients (not a linear model)")
    
    coefs = model.coef_.ravel()
    
    if len(coefs) != len(feature_names):
        raise ValueError(
            f"Coefficient count ({len(coefs)}) doesn't match feature count ({len(feature_names)})"
        )
    
    # If we have a scaler, adjust coefficients to original scale
    if scaler is not None and hasattr(scaler, "scale_"):
        # Coefficient in original scale = coef / scale
        coefs = coefs / scaler.scale_
    
    results = []
    for name, coef in zip(feature_names, coefs):
        results.append(FeatureImportance(
            feature=name,
            importance=abs(coef),
            direction="positive" if coef > 0 else "negative",
        ))
    
    # Sort by absolute importance
    results.sort(key=lambda x: x.importance, reverse=True)
    return results


def get_tree_importance(
    model: Any,
    feature_names: List[str],
) -> List[FeatureImportance]:
    """
    Extract feature importance from tree-based models.
    
    Args:
        model: Fitted tree model (GradientBoostingClassifier, RandomForest, etc.)
        feature_names: Names of features in order
        
    Returns:
        List of FeatureImportance sorted by importance
    """
    if not hasattr(model, "feature_importances_"):
        raise ValueError("Model does not have feature_importances_ (not a tree model)")
    
    importances = model.feature_importances_
    
    if len(importances) != len(feature_names):
        raise ValueError(
            f"Importance count ({len(importances)}) doesn't match feature count ({len(feature_names)})"
        )
    
    results = []
    for name, imp in zip(feature_names, importances):
        results.append(FeatureImportance(
            feature=name,
            importance=imp,
        ))
    
    results.sort(key=lambda x: x.importance, reverse=True)
    return results


def get_permutation_importance(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: List[str],
    n_repeats: int = 10,
    random_state: int = 42,
) -> List[FeatureImportance]:
    """
    Calculate permutation importance (model-agnostic).
    
    Measures how much performance drops when a feature is randomly shuffled.
    More reliable than built-in importance for detecting true predictive power.
    
    Args:
        model: Fitted model with predict or predict_proba
        X: Feature DataFrame
        y: Target Series
        feature_names: Names of features
        n_repeats: Number of times to permute each feature
        random_state: Random seed
        
    Returns:
        List of FeatureImportance sorted by importance
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn required for permutation importance")
    
    # Ensure X has the right columns in order
    X_subset = X[feature_names].copy()
    
    result = permutation_importance(
        model, X_subset, y, 
        n_repeats=n_repeats, 
        random_state=random_state,
        n_jobs=-1,
    )
    
    results = []
    for name, imp_mean, imp_std in zip(
        feature_names, 
        result.importances_mean, 
        result.importances_std
    ):
        results.append(FeatureImportance(
            feature=name,
            importance=imp_mean,
            std=imp_std,
        ))
    
    results.sort(key=lambda x: x.importance, reverse=True)
    return results


def get_shap_importance(
    model: Any,
    X: pd.DataFrame,
    feature_names: List[str],
    max_samples: int = 100,
) -> Tuple[List[FeatureImportance], Optional[Any]]:
    """
    Calculate SHAP values for model interpretability.
    
    SHAP provides both global importance and per-prediction explanations.
    
    Args:
        model: Fitted model
        X: Feature DataFrame
        feature_names: Names of features
        max_samples: Max samples for SHAP calculation (for speed)
        
    Returns:
        Tuple of (importance list, shap_values object for plotting)
    """
    if not SHAP_AVAILABLE:
        raise ImportError("shap package required. Install with: pip install shap")
    
    # Sample data if too large
    if len(X) > max_samples:
        X_sample = X.sample(n=max_samples, random_state=42)
    else:
        X_sample = X
    
    X_subset = X_sample[feature_names].copy()
    
    # Choose appropriate explainer
    if hasattr(model, "feature_importances_"):
        # Tree-based model
        explainer = shap.TreeExplainer(model)
    else:
        # General model
        explainer = shap.Explainer(model.predict_proba, X_subset)
    
    shap_values = explainer(X_subset)
    
    # Calculate mean absolute SHAP values
    if hasattr(shap_values, "values"):
        values = np.abs(shap_values.values)
        if values.ndim == 3:
            # Multi-class: take class 1 (positive class)
            values = values[:, :, 1]
        mean_importance = values.mean(axis=0)
    else:
        mean_importance = np.abs(shap_values).mean(axis=0)
    
    results = []
    for name, imp in zip(feature_names, mean_importance):
        results.append(FeatureImportance(
            feature=name,
            importance=float(imp),
        ))
    
    results.sort(key=lambda x: x.importance, reverse=True)
    return results, shap_values


def analyze_model(
    model: Any,
    feature_names: List[str],
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    scaler: Optional[Any] = None,
    use_shap: bool = False,
) -> Dict[str, Any]:
    """
    Comprehensive model analysis with appropriate methods.
    
    Automatically chooses the right analysis based on model type.
    
    Args:
        model: Fitted model
        feature_names: Names of features
        X_val: Validation data (for permutation importance)
        y_val: Validation targets
        scaler: StandardScaler if model was fit on scaled data
        use_shap: Whether to compute SHAP values
        
    Returns:
        Dict with analysis results
    """
    results = {
        "model_type": type(model).__name__,
        "n_features": len(feature_names),
        "feature_importance": None,
        "importance_method": None,
        "top_features": None,
        "shap_values": None,
    }
    
    # Try coefficient analysis for linear models
    if hasattr(model, "coef_"):
        try:
            importance = get_linear_coefficients(model, feature_names, scaler)
            results["feature_importance"] = [
                {"feature": f.feature, "importance": f.importance, "direction": f.direction}
                for f in importance
            ]
            results["importance_method"] = "coefficients"
            results["top_features"] = [f.feature for f in importance[:10]]
        except Exception as e:
            results["coefficient_error"] = str(e)
    
    # Try tree importance
    elif hasattr(model, "feature_importances_"):
        try:
            importance = get_tree_importance(model, feature_names)
            results["feature_importance"] = [
                {"feature": f.feature, "importance": f.importance}
                for f in importance
            ]
            results["importance_method"] = "tree_importance"
            results["top_features"] = [f.feature for f in importance[:10]]
        except Exception as e:
            results["tree_importance_error"] = str(e)
    
    # Permutation importance (if validation data available)
    if X_val is not None and y_val is not None:
        try:
            perm_importance = get_permutation_importance(
                model, X_val, y_val, feature_names, n_repeats=5
            )
            results["permutation_importance"] = [
                {"feature": f.feature, "importance": f.importance, "std": f.std}
                for f in perm_importance
            ]
        except Exception as e:
            results["permutation_error"] = str(e)
    
    # SHAP values
    if use_shap and X_val is not None and SHAP_AVAILABLE:
        try:
            shap_importance, shap_values = get_shap_importance(
                model, X_val, feature_names, max_samples=100
            )
            results["shap_importance"] = [
                {"feature": f.feature, "importance": f.importance}
                for f in shap_importance
            ]
            results["shap_values"] = shap_values  # For plotting
        except Exception as e:
            results["shap_error"] = str(e)
    
    return results


def print_importance_report(
    importance_list: List[Dict[str, Any]],
    name: str = "Model",
    top_n: int = 15,
) -> None:
    """Pretty print feature importance."""
    print(f"\n{'=' * 60}")
    print(f"  {name} Feature Importance (Top {top_n})")
    print(f"{'=' * 60}")
    
    if not importance_list:
        print("  No importance data available")
        return
    
    # Find max importance for scaling
    max_imp = max(f["importance"] for f in importance_list)
    
    for i, feat in enumerate(importance_list[:top_n], 1):
        bar_len = int(feat["importance"] / max_imp * 30) if max_imp > 0 else 0
        bar = "█" * bar_len
        
        direction_str = ""
        if "direction" in feat and feat["direction"]:
            direction_str = f" ({feat['direction'][0].upper()})"
        
        std_str = ""
        if "std" in feat and feat["std"] is not None:
            std_str = f" ±{feat['std']:.4f}"
        
        print(f"  {i:2d}. {feat['feature']:30s} {feat['importance']:.4f}{std_str}{direction_str}")
        print(f"      {bar}")
    
    print(f"{'=' * 60}\n")


def get_feature_correlations(
    df: pd.DataFrame,
    feature_names: List[str],
    target_col: str,
) -> pd.DataFrame:
    """
    Calculate correlation between features and target.
    
    Useful for identifying potentially redundant features.
    
    Args:
        df: DataFrame with features and target
        feature_names: List of feature column names
        target_col: Name of target column
        
    Returns:
        DataFrame with correlation analysis
    """
    available_features = [f for f in feature_names if f in df.columns]
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in DataFrame")
    
    correlations = []
    for feat in available_features:
        try:
            corr = df[feat].corr(df[target_col])
            correlations.append({
                "feature": feat,
                "correlation": corr,
                "abs_correlation": abs(corr),
            })
        except Exception:
            continue
    
    result = pd.DataFrame(correlations)
    result = result.sort_values("abs_correlation", ascending=False)
    return result

