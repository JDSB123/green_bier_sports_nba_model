"""Tests for interpretability module."""
import pytest
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from src.modeling.interpretability import (
    FeatureImportance,
    get_linear_coefficients,
    get_tree_importance,
    get_permutation_importance,
    analyze_model,
    print_importance_report,
    get_feature_correlations,
)


class TestLinearCoefficients:
    """Tests for linear model coefficient analysis."""
    
    def test_logistic_coefficients(self):
        """Test coefficient extraction from logistic regression."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
        
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        
        feature_names = ["feature_a", "feature_b", "feature_c"]
        importance = get_linear_coefficients(model, feature_names)
        
        assert len(importance) == 3
        assert all(isinstance(f, FeatureImportance) for f in importance)
        assert importance[0].importance >= importance[1].importance  # Sorted by abs value
        assert importance[0].direction in ["positive", "negative"]
    
    def test_ridge_coefficients(self):
        """Test coefficient extraction from ridge regression."""
        np.random.seed(42)
        X = np.random.randn(100, 4)
        y = X[:, 0] * 2 + X[:, 1] * 0.5 + np.random.randn(100) * 0.1
        
        model = Ridge(alpha=1.0)
        model.fit(X, y)
        
        feature_names = ["f1", "f2", "f3", "f4"]
        importance = get_linear_coefficients(model, feature_names)
        
        assert len(importance) == 4
        # f1 should have highest importance (coefficient ~2)
        top_feature = importance[0]
        assert top_feature.feature == "f1" or top_feature.importance > 1.5
    
    def test_coefficients_with_scaler(self):
        """Test coefficient adjustment for scaled features."""
        np.random.seed(42)
        X = np.random.randn(100, 2) * np.array([10, 0.1])  # Different scales
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = LogisticRegression(max_iter=1000)
        model.fit(X_scaled, y)
        
        feature_names = ["large_scale", "small_scale"]
        
        # Without scaler adjustment
        imp_raw = get_linear_coefficients(model, feature_names)
        
        # With scaler adjustment
        imp_scaled = get_linear_coefficients(model, feature_names, scaler=scaler)
        
        # Adjusted coefficients should be different
        assert imp_raw[0].importance != imp_scaled[0].importance


class TestTreeImportance:
    """Tests for tree model importance."""
    
    def test_gradient_boosting_importance(self):
        """Test importance from gradient boosting."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = (X[:, 0] + X[:, 1] * 0.5 > 0).astype(int)
        
        model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
        model.fit(X, y)
        
        feature_names = [f"feat_{i}" for i in range(5)]
        importance = get_tree_importance(model, feature_names)
        
        assert len(importance) == 5
        assert sum(f.importance for f in importance) > 0
        # First two features should be most important
        top_2 = {importance[0].feature, importance[1].feature}
        assert "feat_0" in top_2 or "feat_1" in top_2


class TestPermutationImportance:
    """Tests for permutation importance."""
    
    def test_permutation_importance_basic(self):
        """Test basic permutation importance calculation."""
        np.random.seed(42)
        X = pd.DataFrame({
            "important": np.random.randn(50),
            "noise": np.random.randn(50),
        })
        y = pd.Series((X["important"] > 0).astype(int))
        
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        
        importance = get_permutation_importance(
            model, X, y, 
            feature_names=["important", "noise"],
            n_repeats=3,
        )
        
        assert len(importance) == 2
        assert importance[0].std is not None


class TestAnalyzeModel:
    """Tests for comprehensive model analysis."""
    
    def test_analyze_linear_model(self):
        """Test analysis of linear model."""
        np.random.seed(42)
        X = pd.DataFrame({
            "f1": np.random.randn(100),
            "f2": np.random.randn(100),
        })
        y = pd.Series((X["f1"] > 0).astype(int))
        
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        
        result = analyze_model(
            model, 
            feature_names=["f1", "f2"],
            X_val=X,
            y_val=y,
        )
        
        assert result["model_type"] == "LogisticRegression"
        assert result["n_features"] == 2
        assert result["importance_method"] == "coefficients"
        assert result["feature_importance"] is not None
    
    def test_analyze_tree_model(self):
        """Test analysis of tree model."""
        np.random.seed(42)
        X = pd.DataFrame({
            "f1": np.random.randn(100),
            "f2": np.random.randn(100),
        })
        y = pd.Series((X["f1"] > 0).astype(int))
        
        model = GradientBoostingClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        result = analyze_model(model, feature_names=["f1", "f2"])
        
        assert result["importance_method"] == "tree_importance"


class TestFeatureCorrelations:
    """Tests for feature correlation analysis."""
    
    def test_feature_correlations(self):
        """Test correlation calculation."""
        df = pd.DataFrame({
            "target": [0, 1, 0, 1, 0, 1],
            "good_feat": [0.1, 0.9, 0.2, 0.8, 0.1, 0.9],
            "varying_feat": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],  # Some variance but less correlation
        })
        
        result = get_feature_correlations(
            df, 
            feature_names=["good_feat", "varying_feat"],
            target_col="target",
        )
        
        assert len(result) == 2
        assert result.iloc[0]["feature"] == "good_feat"  # Higher correlation
        # Both should have valid (non-NaN) correlations
        assert not pd.isna(result.iloc[0]["correlation"])
        assert not pd.isna(result.iloc[1]["correlation"])


class TestPrintReport:
    """Tests for report printing."""
    
    def test_print_importance_report(self, capsys):
        """Test importance report printing."""
        importance = [
            {"feature": "best_feature", "importance": 0.5, "direction": "positive"},
            {"feature": "ok_feature", "importance": 0.3, "direction": "negative"},
        ]
        
        print_importance_report(importance, name="Test", top_n=5)
        
        captured = capsys.readouterr()
        assert "Test" in captured.out
        assert "best_feature" in captured.out

