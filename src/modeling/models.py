"""
Machine learning models for NBA spreads and totals predictions.

Implements baseline and advanced models for betting predictions.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not installed. Run: pip install scikit-learn")

from . import io


@dataclass
class ModelMetrics:
    """Container for model evaluation metrics."""
    accuracy: float = 0.0
    log_loss: float = 0.0
    mse: float = 0.0
    mae: float = 0.0
    roi: float = 0.0  # Return on investment (flat -110 by default)
    cover_rate: float = 0.0  # % of bets that cover / win
    brier: float = 0.0  # Brier score for probabilistic calibration


class BaseModel(ABC):
    """Base class for prediction models."""

    def __init__(self, name: str):
        self.name = name
        self.model = None
        # `pipeline` is an sklearn Pipeline bundling preprocessing + estimator
        self.pipeline = None
        self.feature_columns: List[str] = []
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseModel":
        """Fit the model on training data."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities (for classification)."""
        pass

    def _prepare_features(self, X: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for model input.

        Raises:
            ValueError: If any feature column is missing or if imputation fails
        """
        # Check for missing feature columns
        missing_cols = [col for col in self.feature_columns if col not in X.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required feature columns: {sorted(missing_cols)}. "
                f"Available columns: {sorted(X.columns.tolist())}"
            )

        # Select only feature columns
        X_features = X[self.feature_columns].copy()

        # Check for NaN values and log imputation
        nan_counts = X_features.isna().sum()
        features_with_nan = nan_counts[nan_counts > 0]

        if len(features_with_nan) > 0:
            total_values = len(X_features)
            logger.warning(f"Found NaN values in {len(features_with_nan)} features during prediction")

            for feature, count in features_with_nan.items():
                pct_missing = (count / total_values) * 100
                logger.warning(f"  - {feature}: {count}/{total_values} ({pct_missing:.1f}%) missing")

                if pct_missing > 50:
                    raise ValueError(
                        f"Feature '{feature}' has {pct_missing:.1f}% missing values (>{50}% threshold). "
                        f"Data quality insufficient for prediction."
                    )

            # Fill missing values with median per column
            medians = X_features.median()

            # Check if median computation failed (all NaN column)
            features_with_no_median = medians[medians.isna()].index.tolist()
            if features_with_no_median:
                raise ValueError(
                    f"Cannot compute median for features (all values are NaN): {features_with_no_median}"
                )

            X_features = X_features.fillna(medians)
            logger.info(f"Imputed NaN values using median for {len(features_with_nan)} features")

        return X_features

    def save(self, path: str) -> None:
        """Save model to disk."""
        payload = {
            "pipeline": self.pipeline,
            "model": self.model,
            "feature_columns": self.feature_columns,
            "name": self.name,
            "meta": {
                "model_type": getattr(self, "model_type", None),
            },
        }
        io.save_model(payload, path)

    def load(self, path: str) -> "BaseModel":
        """Load model from disk."""
        data = io.load_model(path)
        self.pipeline = data.get("pipeline")
        self.model = data.get("model")
        self.feature_columns = data.get("feature_columns", [])
        self.name = data.get("name", self.name)
        self.is_fitted = True
        return self


class SpreadsModel(BaseModel):
    """
    Model for predicting spread outcomes.

    Predicts whether the home team will cover the spread.
    Can also predict the expected margin for finding value bets.
    """

    DEFAULT_FEATURES = [
        # Team performance
        "home_ppg", "home_papg", "home_win_pct", "home_avg_margin",
        "away_ppg", "away_papg", "away_win_pct", "away_avg_margin",
        # Rest
        "home_rest_days", "away_rest_days", "rest_advantage",
        "home_b2b", "away_b2b",
        # Dynamic Home Court Advantage (context-adjusted)
        "dynamic_hca",
        # Head-to-head
        "h2h_win_pct", "h2h_avg_margin",
        # Derived
        "win_pct_diff", "ppg_diff", "predicted_margin",
        # *** LINE FEATURES *** (market information)
        "spread_line",  # The actual spread line
        "spread_vs_predicted",  # Model vs market disagreement
        "spread_opening_line",
        "spread_movement",  # Line movement
        "spread_line_std",  # Book disagreement
        # ATS performance
        "home_ats_pct", "away_ats_pct",
        # RLM and sharp signals (when available)
        "is_rlm_spread", "sharp_side_spread",
        "spread_public_home_pct", "spread_ticket_money_diff",
        # Injury impact (when available)
        "home_injury_spread_impact", "away_injury_spread_impact",
        "injury_spread_diff", "home_star_out", "away_star_out",
        "predicted_margin_adj",  # Injury-adjusted prediction
    ]

    def __init__(
        self,
        name: str = "spreads_model",
        model_type: str = "logistic",  # "logistic", "gradient_boosting", or "regression"
        feature_columns: Optional[List[str]] = None,
        use_calibration: bool = True,  # Wrap model with probability calibration
    ):
        super().__init__(name)
        self.model_type = model_type
        self.feature_columns = feature_columns or self.DEFAULT_FEATURES
        self.use_calibration = use_calibration

        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required. Install with: pip install scikit-learn")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SpreadsModel":
        """
        Fit spreads prediction model.

        Args:
            X: Feature DataFrame
            y: Target (1 = covered, 0 = didn't cover) or margin values
        """
        # Filter to available features
        available_features = [f for f in self.feature_columns if f in X.columns]
        self.feature_columns = available_features

        X_features = X[self.feature_columns].copy()
        X_features = X_features.fillna(X_features.median())

        # Initialize estimator
        if self.model_type == "logistic":
            estimator = LogisticRegression(max_iter=1000, random_state=42)
        elif self.model_type == "gradient_boosting":
            estimator = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
            )
        elif self.model_type == "regression":
            estimator = Ridge(alpha=1.0)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        # Build pipeline: scaler -> estimator
        pipeline = Pipeline([("scaler", StandardScaler()), ("est", estimator)])

        # Apply probability calibration for classification models
        if self.use_calibration and self.model_type in ["logistic", "gradient_boosting"]:
            logger.info(f"Applying isotonic calibration to {self.model_type} model")
            calibrated_model = CalibratedClassifierCV(
                pipeline,
                method='isotonic',  # Isotonic regression for non-parametric calibration
                cv=5,  # 5-fold cross-validation for calibration
            )
            calibrated_model.fit(X_features, y)
            self.pipeline = calibrated_model
        else:
            pipeline.fit(X_features, y)
            self.pipeline = pipeline

        self.model = estimator
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict spread outcomes (1 = cover, 0 = no cover) or margin."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        X_features = self._prepare_features(X)
        return self.pipeline.predict(X_features)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability of covering the spread."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        X_features = self._prepare_features(X)

        if self.model_type == "regression":
            # For regression, convert margin prediction to probability
            margin_pred = self.pipeline.predict(X_features)
            # Rough conversion: assume std of ~10 points
            proba = 1 / (1 + np.exp(-margin_pred / 5))
            return np.column_stack([1 - proba, proba])

        # Classification: delegate to pipeline
        if hasattr(self.pipeline, "predict_proba"):
            return self.pipeline.predict_proba(X_features)
        # Fallback: use estimator directly
        return self.model.predict_proba(self.pipeline.named_steps["scaler"].transform(X_features))

    def evaluate(
        self,
        X: pd.DataFrame,
        y_true: pd.Series,
        spread_lines: Optional[pd.Series] = None,
    ) -> ModelMetrics:
        """Evaluate model performance."""
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]

        metrics = ModelMetrics()

        if self.model_type == "regression":
            metrics.mse = mean_squared_error(y_true, y_pred)
            metrics.mae = mean_absolute_error(y_true, y_pred)
        else:
            y_true_arr = y_true.values
            metrics.accuracy = accuracy_score(y_true_arr, y_pred)
            metrics.log_loss = log_loss(y_true_arr, y_proba)
            metrics.cover_rate = float(y_pred.mean())
            # Brier score for calibration
            metrics.brier = float(np.mean((y_proba - y_true_arr) ** 2))
            # Simple flat-bet ROI assuming -110 on all spread bets
            n = len(y_true_arr)
            if n > 0:
                correct = (y_pred == y_true_arr).sum()
                profit = correct * (100.0 / 110.0) - (n - correct)
                metrics.roi = profit / n

        return metrics


class TotalsModel(BaseModel):
    """
    Model for predicting over/under outcomes.

    Predicts whether the game total will go over or under the line.
    Can also predict the expected total for finding value bets.
    """

    DEFAULT_FEATURES = [
        # Team performance
        "home_ppg", "home_papg", "home_total_ppg",
        "away_ppg", "away_papg", "away_total_ppg",
        # Rest
        "home_rest_days", "away_rest_days",
        "home_b2b", "away_b2b",
        # Dynamic Home Court Advantage (affects pace/scoring)
        "dynamic_hca",
        # Derived
        "predicted_total",
        # *** LINE FEATURES *** (market information)
        "total_line",  # The actual total line
        "total_vs_predicted",  # Model vs market disagreement
        "total_opening_line",
        "total_movement",  # Line movement
        "total_line_std",  # Book disagreement
        # Over/under tendencies
        "home_over_pct", "away_over_pct",
        # RLM and sharp signals (when available)
        "is_rlm_total", "sharp_side_total",
        "over_public_pct", "total_ticket_money_diff",
        # Injury impact (when available)
        "home_injury_total_impact", "away_injury_total_impact",
        "injury_total_diff",
        "predicted_total_adj",  # Injury-adjusted prediction
    ]

    def __init__(
        self,
        name: str = "totals_model",
        model_type: str = "logistic",  # "logistic", "gradient_boosting", or "regression"
        feature_columns: Optional[List[str]] = None,
        use_calibration: bool = True,  # Wrap model with probability calibration
    ):
        super().__init__(name)
        self.model_type = model_type
        self.feature_columns = feature_columns or self.DEFAULT_FEATURES
        self.use_calibration = use_calibration

        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required. Install with: pip install scikit-learn")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TotalsModel":
        """
        Fit totals prediction model.

        Args:
            X: Feature DataFrame
            y: Target (1 = over, 0 = under) or total score values
        """
        available_features = [f for f in self.feature_columns if f in X.columns]
        self.feature_columns = available_features

        X_features = X[self.feature_columns].copy()
        X_features = X_features.fillna(X_features.median())
        # Initialize estimator
        if self.model_type == "logistic":
            estimator = LogisticRegression(max_iter=1000, random_state=42)
        elif self.model_type == "gradient_boosting":
            estimator = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
            )
        elif self.model_type == "regression":
            estimator = Ridge(alpha=1.0)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        pipeline = Pipeline([("scaler", StandardScaler()), ("est", estimator)])

        # Apply probability calibration for classification models
        if self.use_calibration and self.model_type in ["logistic", "gradient_boosting"]:
            logger.info(f"Applying isotonic calibration to {self.model_type} model")
            calibrated_model = CalibratedClassifierCV(
                pipeline,
                method='isotonic',  # Isotonic regression for non-parametric calibration
                cv=5,  # 5-fold cross-validation for calibration
            )
            calibrated_model.fit(X_features, y)
            self.pipeline = calibrated_model
        else:
            pipeline.fit(X_features, y)
            self.pipeline = pipeline

        self.model = estimator
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict totals outcomes (1 = over, 0 = under) or total score."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        X_features = self._prepare_features(X)
        return self.pipeline.predict(X_features)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability of going over."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        X_features = self._prepare_features(X)

        if self.model_type == "regression":
            total_pred = self.pipeline.predict(X_features)
            # Rough conversion with assumed std of ~15 points
            proba = 1 / (1 + np.exp(-total_pred / 7.5))
            return np.column_stack([1 - proba, proba])

        if hasattr(self.pipeline, "predict_proba"):
            return self.pipeline.predict_proba(X_features)
        return self.model.predict_proba(self.pipeline.named_steps["scaler"].transform(X_features))

    def evaluate(
        self,
        X: pd.DataFrame,
        y_true: pd.Series,
        total_lines: Optional[pd.Series] = None,
    ) -> ModelMetrics:
        """Evaluate model performance."""
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]

        metrics = ModelMetrics()

        if self.model_type == "regression":
            metrics.mse = mean_squared_error(y_true, y_pred)
            metrics.mae = mean_absolute_error(y_true, y_pred)
        else:
            y_true_arr = y_true.values
            metrics.accuracy = accuracy_score(y_true_arr, y_pred)
            metrics.log_loss = log_loss(y_true_arr, y_proba)
            metrics.brier = float(np.mean((y_proba - y_true_arr) ** 2))
            # Flat-bet ROI for totals (assume -110 on all bets)
            n = len(y_true_arr)
            if n > 0:
                correct = (y_pred == y_true_arr).sum()
                profit = correct * (100.0 / 110.0) - (n - correct)
                metrics.roi = profit / n

        return metrics


class MoneylineModel(BaseModel):
    """
    Model for predicting full-game moneyline (home win probability).

    Predicts whether the home team will win the game
    (1 = home win, 0 = away win).
    
    Enhanced with:
    - Probability calibration (CalibratedClassifierCV)
    - Moneyline-specific features (Elo, Pythagorean, momentum)
    - Strength of schedule features
    """

    DEFAULT_FEATURES = [
        # Team performance
        "home_win_pct", "away_win_pct", "win_pct_diff",
        "home_margin", "away_margin", "margin_diff",
        "home_ppg", "home_papg", "away_ppg", "away_papg",
        # Context
        "home_rest", "away_rest", "rest_diff",
        "home_b2b", "away_b2b",
        "dynamic_hca",
        # H2H
        "h2h_margin", "h2h_home_win_pct", "h2h_games",
        # Moneyline-specific (from compute_moneyline_features)
        "ml_win_prob_diff", "ml_elo_diff", "ml_pythagorean_diff",
        "ml_momentum_diff", "ml_estimated_home_prob", "ml_h2h_factor",
        # Strength of schedule
        "home_sos_rating", "away_sos_rating", "sos_diff",
        # Derived
        "predicted_margin", "net_rating_diff",
    ]

    def __init__(
        self,
        name: str = "moneyline_model",
        model_type: str = "logistic",
        feature_columns: Optional[List[str]] = None,
        use_calibration: bool = True,
    ):
        super().__init__(name)
        self.model_type = model_type
        self.feature_columns = feature_columns or self.DEFAULT_FEATURES
        self.use_calibration = use_calibration

        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn required. Install with: pip install scikit-learn"
            )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MoneylineModel":
        available_features = [
            f for f in self.feature_columns if f in X.columns
        ]
        self.feature_columns = available_features

        X_features = X[self.feature_columns].copy()
        X_features = X_features.fillna(X_features.median())

        # Initialize estimator
        if self.model_type == "logistic":
            estimator = LogisticRegression(max_iter=1000, random_state=42)
        elif self.model_type == "gradient_boosting":
            estimator = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        pipeline = Pipeline([("scaler", StandardScaler()), ("est", estimator)])

        # Apply probability calibration
        if self.use_calibration:
            logger.info(f"Applying isotonic calibration to {self.model_type} moneyline model")
            calibrated_model = CalibratedClassifierCV(
                pipeline,
                method='isotonic',
                cv=5,
            )
            calibrated_model.fit(X_features, y)
            self.pipeline = calibrated_model
        else:
            pipeline.fit(X_features, y)
            self.pipeline = pipeline

        self.model = estimator
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        X_features = self._prepare_features(X)
        return self.pipeline.predict(X_features)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        X_features = self._prepare_features(X)

        if hasattr(self.pipeline, "predict_proba"):
            return self.pipeline.predict_proba(X_features)
        return self.model.predict_proba(
            self.pipeline.named_steps["scaler"].transform(X_features)
        )

    def evaluate(
        self,
        X: pd.DataFrame,
        y_true: pd.Series,
        ml_odds: Optional[pd.DataFrame] = None,
    ) -> ModelMetrics:
        """
        Evaluate model performance.
        
        Args:
            X: Feature DataFrame
            y_true: True labels (1 = home win, 0 = away win)
            ml_odds: Optional DataFrame with home_ml_odds and away_ml_odds
                     for proper ROI calculation
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]

        metrics = ModelMetrics()
        y_true_arr = y_true.values
        metrics.accuracy = accuracy_score(y_true_arr, y_pred)
        metrics.log_loss = log_loss(y_true_arr, y_proba)
        metrics.brier = float(np.mean((y_proba - y_true_arr) ** 2))
        
        # Calculate ROI
        n = len(y_true_arr)
        if n > 0:
            if ml_odds is not None and "home_ml_odds" in ml_odds.columns:
                # Calculate proper ROI using actual ML odds
                profit = 0.0
                for i, (pred, actual) in enumerate(zip(y_pred, y_true_arr)):
                    if pred == 1:  # Bet on home
                        odds = ml_odds.iloc[i]["home_ml_odds"]
                    else:  # Bet on away
                        odds = ml_odds.iloc[i]["away_ml_odds"]
                    
                    if pred == actual:  # Won
                        if odds > 0:
                            profit += odds / 100
                        else:
                            profit += 100 / abs(odds)
                    else:  # Lost
                        profit -= 1
                
                metrics.roi = profit / n
            else:
                # Simple ROI assuming -110 odds
                correct = (y_pred == y_true_arr).sum()
                profit = correct * (100.0 / 110.0) - (n - correct)
                metrics.roi = profit / n
        
        return metrics


class FirstHalfMixin:
    """Mixin to indicate this model targets first-half outcomes.

    This mixin does not change implementation but provides a semantic marker
    and a place to override defaults in the future.
    """
    pass


class FirstHalfSpreadsModel(FirstHalfMixin, SpreadsModel):
    """First-half spreads model (home covers first-half spread)."""
    def __init__(
        self,
        name: str = "fh_spreads_model",
        model_type: str = "logistic",
        feature_columns: Optional[List[str]] = None,
        use_calibration: bool = True,
    ):
        super().__init__(
            name=name,
            model_type=model_type,
            feature_columns=feature_columns,
            use_calibration=use_calibration,
        )


class FirstHalfTotalsModel(FirstHalfMixin, TotalsModel):
    """First-half totals model (first-half over/under)."""
    def __init__(
        self,
        name: str = "fh_totals_model",
        model_type: str = "logistic",
        feature_columns: Optional[List[str]] = None,
        use_calibration: bool = True,
    ):
        super().__init__(
            name=name,
            model_type=model_type,
            feature_columns=feature_columns,
            use_calibration=use_calibration,
        )


class FirstHalfMoneylineModel(FirstHalfMixin, MoneylineModel):
    """
    First-half moneyline model (home leading at half).
    
    Uses 1H-specific features when available, with calibration.
    """
    
    DEFAULT_FEATURES = [
        # 1H-specific features (when available from 1H training data)
        "home_ppg_1h", "away_ppg_1h",
        "home_margin_1h", "away_margin_1h",
        "ppg_diff_1h", "margin_diff_1h",
        # FG features scaled for 1H context
        "home_win_pct", "away_win_pct", "win_pct_diff",
        "home_margin", "away_margin", "margin_diff",
        # Context (same for 1H)
        "home_rest", "away_rest", "rest_diff",
        "home_b2b", "away_b2b",
        # Scaled HCA for 1H (~1.5 pts instead of 3)
        "dynamic_hca",
        # H2H
        "h2h_margin", "h2h_home_win_pct",
        # Moneyline features
        "ml_win_prob_diff", "ml_elo_diff", "ml_pythagorean_diff",
        "ml_momentum_diff", "ml_estimated_home_prob",
        # Derived
        "predicted_margin",
    ]
    
    def __init__(
        self,
        name: str = "fh_moneyline_model",
        model_type: str = "logistic",
        feature_columns: Optional[List[str]] = None,
        use_calibration: bool = True,
    ):
        # Use 1H-specific defaults if no features provided
        if feature_columns is None:
            feature_columns = self.DEFAULT_FEATURES
        super().__init__(
            name=name,
            model_type=model_type,
            feature_columns=feature_columns,
            use_calibration=use_calibration,
        )


class TeamTotalsModel(BaseModel):
    """Model for predicting team-specific totals (home/away team totals over/under).

    Note: Training will only run if per-team targets (e.g., `home_team_over`, `away_team_over`)
    are present in the training dataset. This class follows the same Pipeline pattern.
    """

    DEFAULT_FEATURES = [
        "home_ppg", "home_papg", "home_total_ppg",
        "away_ppg", "away_papg", "away_total_ppg",
        "predicted_total", "predicted_margin", "ppg_diff",
    ]

    def __init__(
        self,
        name: str = "team_totals_model",
        model_type: str = "logistic",
        feature_columns: Optional[List[str]] = None,
        use_calibration: bool = True,  # Wrap model with probability calibration
    ):
        super().__init__(name)
        self.model_type = model_type
        self.feature_columns = feature_columns or self.DEFAULT_FEATURES
        self.use_calibration = use_calibration

        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required. Install with: pip install scikit-learn")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TeamTotalsModel":
        available_features = [f for f in self.feature_columns if f in X.columns]
        self.feature_columns = available_features

        X_features = X[self.feature_columns].copy()
        X_features = X_features.fillna(X_features.median())

        # Initialize estimator
        if self.model_type == "logistic":
            estimator = LogisticRegression(max_iter=1000, random_state=42)
        elif self.model_type == "gradient_boosting":
            estimator = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        pipeline = Pipeline([("scaler", StandardScaler()), ("est", estimator)])

        # Apply probability calibration for classification models
        if self.use_calibration:
            logger.info(f"Applying isotonic calibration to {self.model_type} team totals model")
            calibrated_model = CalibratedClassifierCV(
                pipeline,
                method='isotonic',  # Isotonic regression for non-parametric calibration
                cv=5,  # 5-fold cross-validation for calibration
            )
            calibrated_model.fit(X_features, y)
            self.pipeline = calibrated_model
        else:
            pipeline.fit(X_features, y)
            self.pipeline = pipeline

        self.model = estimator
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        X_features = self._prepare_features(X)
        return self.pipeline.predict(X_features)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        X_features = self._prepare_features(X)
        if hasattr(self.pipeline, "predict_proba"):
            return self.pipeline.predict_proba(X_features)
        return self.model.predict_proba(self.pipeline.named_steps["scaler"].transform(X_features))

    def evaluate(
        self,
        X: pd.DataFrame,
        y_true: pd.Series,
        team_total_lines: Optional[pd.Series] = None,
    ) -> ModelMetrics:
        """
        Evaluate model performance.
        
        Args:
            X: Feature DataFrame
            y_true: True labels (1 = over, 0 = under)
            team_total_lines: Optional Series with team total lines for ROI calculation
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]

        metrics = ModelMetrics()
        y_true_arr = y_true.values
        metrics.accuracy = accuracy_score(y_true_arr, y_pred)
        metrics.log_loss = log_loss(y_true_arr, y_proba)
        metrics.brier = float(np.mean((y_proba - y_true_arr) ** 2))
        
        # Flat-bet ROI for team totals (assume -110 on all bets)
        n = len(y_true_arr)
        if n > 0:
            correct = (y_pred == y_true_arr).sum()
            profit = correct * (100.0 / 110.0) - (n - correct)
            metrics.roi = profit / n

        return metrics


class EnsembleModel:
    """
    Ensemble of multiple models for more robust predictions.

    Combines predictions from multiple base models.
    """

    def __init__(self, models: List[BaseModel], weights: Optional[List[float]] = None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Weighted average of model probabilities."""
        probas = []
        for model, weight in zip(self.models, self.weights):
            probas.append(model.predict_proba(X) * weight)
        return np.sum(probas, axis=0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using ensemble."""
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)


def find_value_bets(
    predictions_df: pd.DataFrame,
    min_edge: float = 0.05,
    min_confidence: float = 0.55,
) -> pd.DataFrame:
    """
    Find value betting opportunities.

    Args:
        predictions_df: DataFrame with model predictions and implied odds
        min_edge: Minimum edge over implied probability (e.g., 0.05 = 5%)
        min_confidence: Minimum model confidence

    Returns:
        DataFrame of value bet opportunities
    """
    value_bets = []

    for _, row in predictions_df.iterrows():
        # For spreads
        if "spread_prob" in row and "spread_implied_prob" in row:
            edge = row["spread_prob"] - row["spread_implied_prob"]
            if edge >= min_edge and row["spread_prob"] >= min_confidence:
                value_bets.append({
                    "game": f"{row.get('home_team', 'Home')} vs {row.get('away_team', 'Away')}",
                    "bet_type": "spread",
                    "prediction": "cover" if row["spread_prob"] > 0.5 else "no cover",
                    "model_prob": row["spread_prob"],
                    "implied_prob": row["spread_implied_prob"],
                    "edge": edge,
                    "line": row.get("spread_line", None),
                })

        # For totals
        if "total_prob" in row and "total_implied_prob" in row:
            edge = row["total_prob"] - row["total_implied_prob"]
            if edge >= min_edge and row["total_prob"] >= min_confidence:
                value_bets.append({
                    "game": f"{row.get('home_team', 'Home')} vs {row.get('away_team', 'Away')}",
                    "bet_type": "total",
                    "prediction": "over" if row["total_prob"] > 0.5 else "under",
                    "model_prob": row["total_prob"],
                    "implied_prob": row["total_implied_prob"],
                    "edge": edge,
                    "line": row.get("total_line", None),
                })

        # First-half spreads
        if "fh_spread_prob" in row and "fh_spread_implied_prob" in row:
            edge = row["fh_spread_prob"] - row["fh_spread_implied_prob"]
            if edge >= min_edge and row["fh_spread_prob"] >= min_confidence:
                value_bets.append({
                    "game": f"{row.get('home_team', 'Home')} vs {row.get('away_team', 'Away')}",
                    "bet_type": "first_half_spread",
                    "prediction": "cover" if row["fh_spread_prob"] > 0.5 else "no cover",
                    "model_prob": row["fh_spread_prob"],
                    "implied_prob": row["fh_spread_implied_prob"],
                    "edge": edge,
                    "line": row.get("fh_spread_line", None),
                })

        # First-half totals
        if "fh_total_prob" in row and "fh_total_implied_prob" in row:
            edge = row["fh_total_prob"] - row["fh_total_implied_prob"]
            if edge >= min_edge and row["fh_total_prob"] >= min_confidence:
                value_bets.append({
                    "game": f"{row.get('home_team', 'Home')} vs {row.get('away_team', 'Away')}",
                    "bet_type": "first_half_total",
                    "prediction": "over" if row["fh_total_prob"] > 0.5 else "under",
                    "model_prob": row["fh_total_prob"],
                    "implied_prob": row["fh_total_implied_prob"],
                    "edge": edge,
                    "line": row.get("fh_total_line", None),
                })

        # First-half moneyline (team leading at half)
        if "fh_moneyline_prob" in row and "fh_moneyline_implied_prob" in row:
            edge = row["fh_moneyline_prob"] - row["fh_moneyline_implied_prob"]
            if edge >= min_edge and row["fh_moneyline_prob"] >= min_confidence:
                value_bets.append({
                    "game": f"{row.get('home_team', 'Home')} vs {row.get('away_team', 'Away')}",
                    "bet_type": "first_half_moneyline",
                    "prediction": "home" if row["fh_moneyline_prob"] > 0.5 else "away",
                    "model_prob": row["fh_moneyline_prob"],
                    "implied_prob": row["fh_moneyline_implied_prob"],
                    "edge": edge,
                    "line": row.get("fh_moneyline_price", None),
                })

        # Team totals (home/away)
        if "home_team_total_prob" in row and "home_team_total_implied_prob" in row:
            edge = row["home_team_total_prob"] - row["home_team_total_implied_prob"]
            if edge >= min_edge and row["home_team_total_prob"] >= min_confidence:
                value_bets.append({
                    "game": f"{row.get('home_team', 'Home')} vs {row.get('away_team', 'Away')}",
                    "bet_type": "home_team_total",
                    "prediction": "over" if row["home_team_total_prob"] > 0.5 else "under",
                    "model_prob": row["home_team_total_prob"],
                    "implied_prob": row["home_team_total_implied_prob"],
                    "edge": edge,
                    "line": row.get("home_team_total_line", None),
                })

        if "away_team_total_prob" in row and "away_team_total_implied_prob" in row:
            edge = row["away_team_total_prob"] - row["away_team_total_implied_prob"]
            if edge >= min_edge and row["away_team_total_prob"] >= min_confidence:
                value_bets.append({
                    "game": f"{row.get('home_team', 'Home')} vs {row.get('away_team', 'Away')}",
                    "bet_type": "away_team_total",
                    "prediction": "over" if row["away_team_total_prob"] > 0.5 else "under",
                    "model_prob": row["away_team_total_prob"],
                    "implied_prob": row["away_team_total_implied_prob"],
                    "edge": edge,
                    "line": row.get("away_team_total_line", None),
                })

    return pd.DataFrame(value_bets)
    
