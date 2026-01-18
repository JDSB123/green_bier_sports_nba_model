"""
Hyperparameter tuning utilities for NBA prediction models.

Provides:
- GridSearchCV wrapper for scikit-learn models
- Optuna integration for Bayesian optimization
- Custom cross-validation with TimeSeriesSplit
- Model comparison utilities
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple, Callable
import numpy as np
import pandas as pd

try:
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.impute import KNNImputer
    from sklearn.metrics import accuracy_score, log_loss, make_scorer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


@dataclass
class TuningResult:
    """Container for hyperparameter tuning results."""
    best_params: Dict[str, Any]
    best_score: float
    cv_results: Optional[Dict[str, Any]] = None
    all_params_tested: Optional[List[Dict[str, Any]]] = None
    tuning_method: str = "grid_search"


# =============================================================================
# PARAMETER GRIDS
# =============================================================================

LOGISTIC_PARAM_GRID = {
    "est__C": [0.01, 0.1, 1.0, 10.0],
    "est__penalty": ["l2"],
    "est__solver": ["lbfgs", "saga"],
    "est__max_iter": [1000],
    "est__class_weight": [None, "balanced"],
}

GRADIENT_BOOSTING_PARAM_GRID = {
    "est__n_estimators": [50, 100, 200],
    "est__max_depth": [3, 4, 5, 6],
    "est__learning_rate": [0.05, 0.1, 0.2],
    "est__min_samples_split": [2, 5, 10],
    "est__min_samples_leaf": [1, 2, 4],
    "est__subsample": [0.8, 1.0],
}

RANDOM_FOREST_PARAM_GRID = {
    "est__n_estimators": [50, 100, 200],
    "est__max_depth": [5, 10, 15, None],
    "est__min_samples_split": [2, 5, 10],
    "est__min_samples_leaf": [1, 2, 4],
    "est__max_features": ["sqrt", "log2", None],
}

RIDGE_PARAM_GRID = {
    "est__alpha": [0.1, 1.0, 10.0, 100.0],
}


def get_param_grid(model_type: str) -> Dict[str, List[Any]]:
    """Get parameter grid for a model type."""
    grids = {
        "logistic": LOGISTIC_PARAM_GRID,
        "gradient_boosting": GRADIENT_BOOSTING_PARAM_GRID,
        "random_forest": RANDOM_FOREST_PARAM_GRID,
        "regression": RIDGE_PARAM_GRID,
    }
    return grids.get(model_type, {})


# =============================================================================
# CUSTOM SCORERS
# =============================================================================

def roi_scorer(y_true: np.ndarray, y_pred: np.ndarray, odds: float = -110) -> float:
    """
    Calculate ROI as a scoring metric.

    ROI = (profit / total_bet) * 100

    At -110 odds:
    - Win: +0.909 units
    - Lose: -1.0 units
    """
    if len(y_true) == 0:
        return 0.0

    correct = (y_pred == y_true).sum()
    total = len(y_true)

    if odds > 0:
        win_amount = odds / 100
    else:
        win_amount = 100 / abs(odds)

    profit = correct * win_amount - (total - correct) * 1.0
    return profit / total


def betting_profit_scorer(y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.55) -> float:
    """
    Score based on simulated betting profit.

    Only bet when model probability > threshold.
    """
    bets_made = 0
    profit = 0.0

    for i, prob in enumerate(y_pred_proba):
        if prob >= threshold:
            bets_made += 1
            if y_true[i] == 1:
                profit += 0.909  # Win at -110
            else:
                profit -= 1.0
        elif prob <= (1 - threshold):
            bets_made += 1
            if y_true[i] == 0:
                profit += 0.909
            else:
                profit -= 1.0

    if bets_made == 0:
        return 0.0

    return profit / bets_made


# =============================================================================
# GRID SEARCH TUNING
# =============================================================================

def tune_with_grid_search(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "logistic",
    feature_columns: Optional[List[str]] = None,
    n_splits: int = 5,
    scoring: str = "accuracy",
    n_jobs: int = -1,
    verbose: int = 1,
) -> TuningResult:
    """
    Tune hyperparameters using GridSearchCV with TimeSeriesSplit.

    Args:
        X: Feature DataFrame
        y: Target Series
        model_type: Type of model ("logistic", "gradient_boosting", etc.)
        feature_columns: Columns to use (uses all if None)
        n_splits: Number of CV splits
        scoring: Scoring metric
        n_jobs: Parallel jobs (-1 for all cores)
        verbose: Verbosity level

    Returns:
        TuningResult with best parameters and scores
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn required for grid search")

    # Select features
    if feature_columns:
        available = [f for f in feature_columns if f in X.columns]
        X_train = X[available].copy()
    else:
        X_train = X.copy()

    # Updated: Let the pipeline handle imputation with KNN
    # X_train = X_train.fillna(X_train.median())  <-- REMOVED

    # Build pipeline
    if model_type == "logistic":
        estimator = LogisticRegression(random_state=42)
    elif model_type == "gradient_boosting":
        estimator = GradientBoostingClassifier(random_state=42)
    elif model_type == "random_forest":
        estimator = RandomForestClassifier(random_state=42)
    elif model_type == "regression":
        estimator = Ridge()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    pipeline = Pipeline([
        ("imputer", KNNImputer(n_neighbors=5)),
        ("scaler", StandardScaler()),
        ("est", estimator),
    ])

    # Get parameter grid
    param_grid = get_param_grid(model_type)

    if not param_grid:
        return TuningResult(
            best_params={},
            best_score=0.0,
            tuning_method="grid_search",
        )

    # TimeSeriesSplit for proper temporal validation
    cv = TimeSeriesSplit(n_splits=n_splits)

    # Custom scorer if needed
    if scoring == "roi":
        scorer = make_scorer(roi_scorer, greater_is_better=True)
    else:
        scorer = scoring

    # Run grid search
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring=scorer,
            n_jobs=n_jobs,
            verbose=verbose,
            refit=True,
        )

        grid_search.fit(X_train, y)

    return TuningResult(
        best_params=grid_search.best_params_,
        best_score=grid_search.best_score_,
        cv_results=grid_search.cv_results_,
        all_params_tested=[grid_search.cv_results_["params"][i]
                           for i in range(len(grid_search.cv_results_["params"]))],
        tuning_method="grid_search",
    )


def tune_with_randomized_search(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "gradient_boosting",
    feature_columns: Optional[List[str]] = None,
    n_splits: int = 5,
    n_iter: int = 50,
    scoring: str = "accuracy",
    n_jobs: int = -1,
    verbose: int = 1,
) -> TuningResult:
    """
    Tune hyperparameters using RandomizedSearchCV (faster for large grids).

    Args:
        X: Feature DataFrame
        y: Target Series
        model_type: Type of model
        feature_columns: Columns to use
        n_splits: Number of CV splits
        n_iter: Number of random parameter combinations to try
        scoring: Scoring metric
        n_jobs: Parallel jobs
        verbose: Verbosity level

    Returns:
        TuningResult with best parameters
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn required for randomized search")

    if feature_columns:
        available = [f for f in feature_columns if f in X.columns]
        X_train = X[available].copy()
    else:
        X_train = X.copy()

    # Updated: Let the pipeline handle imputation with KNN
    # X_train = X_train.fillna(X_train.median()) <-- REMOVED

    # Build pipeline
    if model_type == "logistic":
        estimator = LogisticRegression(random_state=42)
    elif model_type == "gradient_boosting":
        estimator = GradientBoostingClassifier(random_state=42)
    elif model_type == "random_forest":
        estimator = RandomForestClassifier(random_state=42)
    else:
        estimator = LogisticRegression(random_state=42)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("est", estimator),
    ])

    param_grid = get_param_grid(model_type)
    cv = TimeSeriesSplit(n_splits=n_splits)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        random_search = RandomizedSearchCV(
            pipeline,
            param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=42,
            refit=True,
        )

        random_search.fit(X_train, y)

    return TuningResult(
        best_params=random_search.best_params_,
        best_score=random_search.best_score_,
        cv_results=random_search.cv_results_,
        tuning_method="randomized_search",
    )


# =============================================================================
# OPTUNA TUNING (Bayesian Optimization)
# =============================================================================

def tune_with_optuna(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "gradient_boosting",
    feature_columns: Optional[List[str]] = None,
    n_splits: int = 5,
    n_trials: int = 100,
    scoring: str = "accuracy",
    timeout: Optional[int] = None,
    verbose: bool = False,
) -> TuningResult:
    """
    Tune hyperparameters using Optuna (Bayesian optimization).

    More efficient than grid search for complex parameter spaces.

    Args:
        X: Feature DataFrame
        y: Target Series
        model_type: Type of model
        feature_columns: Columns to use
        n_splits: Number of CV splits
        n_trials: Number of optimization trials
        scoring: Scoring metric
        timeout: Max time in seconds
        verbose: Show optimization progress

    Returns:
        TuningResult with best parameters
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna required. Install with: pip install optuna")

    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn required")

    if feature_columns:
        available = [f for f in feature_columns if f in X.columns]
        X_train = X[available].copy()
    else:
        X_train = X.copy()

    X_train = X_train.fillna(X_train.median())
    cv = TimeSeriesSplit(n_splits=n_splits)

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective function."""

        if model_type == "logistic":
            params = {
                "C": trial.suggest_float("C", 0.01, 100, log=True),
                "solver": trial.suggest_categorical("solver", ["lbfgs", "saga"]),
                "max_iter": 1000,
            }
            estimator = LogisticRegression(**params, random_state=42)

        elif model_type == "gradient_boosting":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            }
            estimator = GradientBoostingClassifier(**params, random_state=42)

        elif model_type == "random_forest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            }
            estimator = RandomForestClassifier(**params, random_state=42)

        else:
            params = {
                "alpha": trial.suggest_float("alpha", 0.01, 100, log=True),
            }
            estimator = Ridge(**params)

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("est", estimator),
        ])

        # Cross-validation
        scores = []
        for train_idx, val_idx in cv.split(X_train):
            X_cv_train = X_train.iloc[train_idx]
            X_cv_val = X_train.iloc[val_idx]
            y_cv_train = y.iloc[train_idx]
            y_cv_val = y.iloc[val_idx]

            pipeline.fit(X_cv_train, y_cv_train)

            if scoring == "accuracy":
                y_pred = pipeline.predict(X_cv_val)
                score = accuracy_score(y_cv_val, y_pred)
            elif scoring == "neg_log_loss":
                y_proba = pipeline.predict_proba(X_cv_val)
                score = -log_loss(y_cv_val, y_proba)
            elif scoring == "roi":
                y_pred = pipeline.predict(X_cv_val)
                score = roi_scorer(y_cv_val.values, y_pred)
            else:
                y_pred = pipeline.predict(X_cv_val)
                score = accuracy_score(y_cv_val, y_pred)

            scores.append(score)

            # Pruning for early stopping
            trial.report(np.mean(scores), len(scores) - 1)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(scores)

    # Create study
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=2)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    # Suppress Optuna logs unless verbose
    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=verbose,
    )

    return TuningResult(
        best_params=study.best_params,
        best_score=study.best_value,
        all_params_tested=[
            t.params for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
        tuning_method="optuna",
    )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def auto_tune(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "gradient_boosting",
    feature_columns: Optional[List[str]] = None,
    method: str = "auto",
    n_splits: int = 5,
    verbose: bool = True,
) -> TuningResult:
    """
    Automatically tune hyperparameters using the best available method.

    Args:
        X: Feature DataFrame
        y: Target Series
        model_type: Type of model
        feature_columns: Columns to use
        method: "grid", "random", "optuna", or "auto"
        n_splits: Number of CV splits
        verbose: Show progress

    Returns:
        TuningResult with best parameters
    """
    if method == "auto":
        # Use Optuna if available (best for complex spaces)
        if OPTUNA_AVAILABLE and model_type in ["gradient_boosting", "random_forest"]:
            method = "optuna"
        elif len(get_param_grid(model_type)) > 100:
            method = "random"
        else:
            method = "grid"

    if verbose:
        print(f"Tuning {model_type} using {method} search...")

    if method == "optuna":
        return tune_with_optuna(
            X, y, model_type, feature_columns,
            n_splits=n_splits, n_trials=50, verbose=verbose
        )
    elif method == "random":
        return tune_with_randomized_search(
            X, y, model_type, feature_columns,
            n_splits=n_splits, n_iter=50, verbose=1 if verbose else 0
        )
    else:
        return tune_with_grid_search(
            X, y, model_type, feature_columns,
            n_splits=n_splits, verbose=1 if verbose else 0
        )


def print_tuning_report(result: TuningResult, model_type: str = "model") -> None:
    """Pretty print tuning results."""
    print(f"\n{'='*60}")
    print(f"Hyperparameter Tuning Results: {model_type}")
    print(f"{'='*60}")
    print(f"Method: {result.tuning_method}")
    print(f"Best Score: {result.best_score:.4f}")
    print(f"\nBest Parameters:")
    for param, value in result.best_params.items():
        # Clean up pipeline param names
        clean_param = param.replace("est__", "")
        print(f"  {clean_param}: {value}")
    print(f"{'='*60}\n")
