"""
Model calibration utilities for NBA prediction models.

Provides:
- Platt scaling (sigmoid calibration)
- Isotonic regression calibration
- Calibration curve analysis
- Brier score and reliability metrics
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

try:
    from sklearn.calibration import CalibratedClassifierCV, calibration_curve
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, log_loss
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class CalibrationMetrics:
    """Container for calibration evaluation metrics."""
    brier_score: float  # Lower is better (0-1)
    log_loss: float  # Lower is better
    expected_calibration_error: float  # ECE - lower is better
    max_calibration_error: float  # MCE - lower is better
    reliability_diagram: Dict[str, List[float]]  # For plotting
    n_samples: int


class ModelCalibrator:
    """
    Calibrates model probabilities using Platt scaling or isotonic regression.
    
    The market is efficient, so a well-calibrated model should have:
    - Predictions of 60% actually winning ~60% of the time
    - Low Brier score and ECE (Expected Calibration Error)
    """
    
    def __init__(self, method: str = "isotonic"):
        """
        Initialize calibrator.
        
        Args:
            method: "isotonic" (non-parametric) or "sigmoid" (Platt scaling)
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for calibration")
        
        self.method = method
        self.calibrator = None
        self.is_fitted = False
        
    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> "ModelCalibrator":
        """
        Fit calibration model on validation data.
        
        Args:
            y_prob: Predicted probabilities (0-1)
            y_true: Actual outcomes (0 or 1)
        """
        y_prob = np.asarray(y_prob).ravel()
        y_true = np.asarray(y_true).ravel()
        
        # Remove NaN values
        valid_mask = ~(np.isnan(y_prob) | np.isnan(y_true))
        y_prob = y_prob[valid_mask]
        y_true = y_true[valid_mask]
        
        if len(y_prob) < 10:
            raise ValueError("Need at least 10 samples for calibration")
        
        if self.method == "isotonic":
            # Isotonic regression - monotonic, non-parametric
            self.calibrator = IsotonicRegression(
                y_min=0.01, 
                y_max=0.99, 
                out_of_bounds="clip"
            )
            self.calibrator.fit(y_prob, y_true)
        else:
            # Platt scaling - fit logistic regression on log-odds
            # Transform to log-odds space
            eps = 1e-7
            y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
            log_odds = np.log(y_prob_clipped / (1 - y_prob_clipped)).reshape(-1, 1)
            
            self.calibrator = LogisticRegression(solver="lbfgs", max_iter=1000)
            self.calibrator.fit(log_odds, y_true)
        
        self.is_fitted = True
        return self
    
    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        """
        Apply calibration to raw probabilities.
        
        Args:
            y_prob: Raw predicted probabilities
            
        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")
        
        y_prob = np.asarray(y_prob).ravel()
        
        if self.method == "isotonic":
            return self.calibrator.predict(y_prob)
        else:
            # Platt scaling
            eps = 1e-7
            y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
            log_odds = np.log(y_prob_clipped / (1 - y_prob_clipped)).reshape(-1, 1)
            return self.calibrator.predict_proba(log_odds)[:, 1]
    
    def evaluate(
        self, 
        y_prob: np.ndarray, 
        y_true: np.ndarray,
        n_bins: int = 10
    ) -> CalibrationMetrics:
        """
        Evaluate calibration quality.
        
        Args:
            y_prob: Predicted probabilities
            y_true: Actual outcomes
            n_bins: Number of bins for calibration curve
            
        Returns:
            CalibrationMetrics with various calibration scores
        """
        y_prob = np.asarray(y_prob).ravel()
        y_true = np.asarray(y_true).ravel()
        
        # Remove NaN values
        valid_mask = ~(np.isnan(y_prob) | np.isnan(y_true))
        y_prob = y_prob[valid_mask]
        y_true = y_true[valid_mask]
        
        if len(y_prob) == 0:
            raise ValueError("No valid samples for calibration evaluation")
        
        # Brier score
        brier = brier_score_loss(y_true, y_prob)
        
        # Log loss
        eps = 1e-7
        y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
        logloss = log_loss(y_true, y_prob_clipped)
        
        # Calibration curve
        prob_true, prob_pred = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy="uniform"
        )
        
        # Expected Calibration Error (ECE)
        # Weighted average of |accuracy - confidence| per bin
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        mce = 0.0
        
        for i in range(n_bins):
            bin_mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
            if i == n_bins - 1:
                bin_mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
            
            bin_count = bin_mask.sum()
            if bin_count > 0:
                bin_accuracy = y_true[bin_mask].mean()
                bin_confidence = y_prob[bin_mask].mean()
                bin_error = abs(bin_accuracy - bin_confidence)
                
                ece += (bin_count / len(y_prob)) * bin_error
                mce = max(mce, bin_error)
        
        return CalibrationMetrics(
            brier_score=brier,
            log_loss=logloss,
            expected_calibration_error=ece,
            max_calibration_error=mce,
            reliability_diagram={
                "mean_predicted_prob": prob_pred.tolist(),
                "fraction_of_positives": prob_true.tolist(),
            },
            n_samples=len(y_prob),
        )


def validate_calibration_assumption(
    edge_per_point: float,
    historical_results: pd.DataFrame,
    edge_column: str = "spread_edge",
    outcome_column: str = "spread_covered",
) -> Dict[str, Any]:
    """
    Validate the assumption that X points of edge = Y% probability.
    
    The model uses hardcoded: 1 pt edge ≈ 2.5% probability shift.
    This function validates that assumption against historical data.
    
    Args:
        edge_per_point: The assumed probability shift per point of edge
        historical_results: DataFrame with edge and outcome columns
        edge_column: Name of edge column
        outcome_column: Name of outcome column (0/1)
        
    Returns:
        Dict with validation metrics
    """
    df = historical_results.dropna(subset=[edge_column, outcome_column]).copy()
    
    if len(df) < 50:
        return {"error": "Insufficient data for calibration validation"}
    
    # Bin by edge magnitude
    df["edge_bin"] = pd.cut(
        df[edge_column], 
        bins=[-float("inf"), -4, -2, 0, 2, 4, float("inf")],
        labels=["<-4", "-4 to -2", "-2 to 0", "0 to 2", "2 to 4", ">4"]
    )
    
    results = []
    for bin_label, group in df.groupby("edge_bin", observed=True):
        if len(group) < 5:
            continue
        
        avg_edge = group[edge_column].mean()
        actual_win_rate = group[outcome_column].mean()
        
        # Expected win rate using the assumed edge_per_point
        expected_win_rate = 0.5 + avg_edge * edge_per_point
        expected_win_rate = max(0.3, min(0.7, expected_win_rate))
        
        results.append({
            "edge_bin": str(bin_label),
            "n_games": len(group),
            "avg_edge": avg_edge,
            "actual_win_rate": actual_win_rate,
            "expected_win_rate": expected_win_rate,
            "calibration_error": actual_win_rate - expected_win_rate,
        })
    
    # Calculate optimal edge_per_point from data
    if len(df) > 0:
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            df[edge_column], df[outcome_column]
        )
        optimal_edge_per_point = slope
    else:
        optimal_edge_per_point = edge_per_point
    
    return {
        "assumed_edge_per_point": edge_per_point,
        "optimal_edge_per_point": optimal_edge_per_point,
        "calibration_factor": optimal_edge_per_point / edge_per_point if edge_per_point != 0 else 1.0,
        "bin_analysis": results,
        "recommendation": (
            f"Adjust probability formula to use {optimal_edge_per_point:.4f} per point "
            f"(currently {edge_per_point:.4f})"
            if abs(optimal_edge_per_point - edge_per_point) > 0.005
            else "Current calibration looks good"
        ),
    }


def print_calibration_report(metrics: CalibrationMetrics, name: str = "Model") -> None:
    """Pretty print calibration metrics."""
    print(f"\n{'=' * 50}")
    print(f"  {name} Calibration Report")
    print(f"{'=' * 50}")
    print(f"  Samples:                    {metrics.n_samples:,}")
    print(f"  Brier Score:                {metrics.brier_score:.4f} (lower is better)")
    print(f"  Log Loss:                   {metrics.log_loss:.4f}")
    print(f"  Expected Calibration Error: {metrics.expected_calibration_error:.4f}")
    print(f"  Max Calibration Error:      {metrics.max_calibration_error:.4f}")
    print(f"{'=' * 50}")
    
    # Simple ASCII reliability diagram
    print("\n  Reliability Diagram (pred vs actual):")
    diagram = metrics.reliability_diagram
    pred = diagram["mean_predicted_prob"]
    actual = diagram["fraction_of_positives"]
    
    for p, a in zip(pred, actual):
        bar_len = int(a * 40)
        expected_marker = int(p * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        # Mark expected position
        marker = " " * expected_marker + "│"
        print(f"  {p:.2f}: [{bar}] {a:.2f}")
    
    print(f"{'=' * 50}\n")

