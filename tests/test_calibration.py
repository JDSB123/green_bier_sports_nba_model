"""Tests for calibration module."""
import pytest
import numpy as np

from src.modeling.calibration import (
    ModelCalibrator,
    CalibrationMetrics,
    validate_calibration_assumption,
    print_calibration_report,
)


class TestModelCalibrator:
    """Tests for ModelCalibrator class."""
    
    def test_isotonic_calibrator_fit_predict(self):
        """Test isotonic calibration fit and predict."""
        calibrator = ModelCalibrator(method="isotonic")
        
        # Create synthetic data with known miscalibration
        np.random.seed(42)
        y_prob = np.random.uniform(0.3, 0.7, 100)
        # Actuals are more extreme than probabilities suggest
        y_true = (y_prob + np.random.normal(0, 0.2, 100) > 0.5).astype(int)
        
        calibrator.fit(y_prob, y_true)
        assert calibrator.is_fitted
        
        # Calibrate should return values in [0, 1]
        calibrated = calibrator.calibrate(y_prob)
        assert calibrated.min() >= 0
        assert calibrated.max() <= 1
    
    def test_sigmoid_calibrator_fit_predict(self):
        """Test sigmoid (Platt) calibration."""
        calibrator = ModelCalibrator(method="sigmoid")
        
        np.random.seed(42)
        y_prob = np.random.uniform(0.2, 0.8, 100)
        y_true = (y_prob > 0.5).astype(int)
        
        calibrator.fit(y_prob, y_true)
        calibrated = calibrator.calibrate(y_prob)
        
        assert len(calibrated) == len(y_prob)
    
    def test_calibrator_evaluate(self):
        """Test calibration evaluation metrics."""
        calibrator = ModelCalibrator(method="isotonic")
        
        np.random.seed(42)
        y_prob = np.random.uniform(0.3, 0.7, 100)
        y_true = (y_prob > 0.5).astype(int)
        
        metrics = calibrator.evaluate(y_prob, y_true)
        
        assert isinstance(metrics, CalibrationMetrics)
        assert 0 <= metrics.brier_score <= 1
        assert metrics.expected_calibration_error >= 0
        assert metrics.max_calibration_error >= 0
        assert metrics.n_samples == 100
        assert "mean_predicted_prob" in metrics.reliability_diagram
    
    def test_calibrator_requires_fit(self):
        """Test that calibrate raises if not fitted."""
        calibrator = ModelCalibrator()
        
        with pytest.raises(ValueError, match="not fitted"):
            calibrator.calibrate(np.array([0.5, 0.6]))
    
    def test_calibrator_handles_nan(self):
        """Test that calibrator handles NaN values in input."""
        calibrator = ModelCalibrator()
        
        # Create enough samples with some NaN values
        np.random.seed(42)
        y_prob = np.concatenate([np.random.uniform(0.3, 0.7, 20), [np.nan, np.nan]])
        y_true = np.concatenate([np.random.randint(0, 2, 20).astype(float), [np.nan, 1.0]])
        
        # Should not raise - NaN values are filtered out
        calibrator.fit(y_prob, y_true)
        assert calibrator.is_fitted


class TestCalibrationValidation:
    """Tests for calibration validation functions."""
    
    def test_validate_calibration_assumption(self):
        """Test calibration assumption validation."""
        import pandas as pd
        
        np.random.seed(42)
        n = 200
        
        # Create synthetic results with known relationship
        edges = np.random.uniform(-5, 5, n)
        # True relationship: 3% per point
        true_probs = 0.5 + edges * 0.03
        true_probs = np.clip(true_probs, 0.1, 0.9)
        outcomes = (np.random.random(n) < true_probs).astype(int)
        
        df = pd.DataFrame({
            "spread_edge": edges,
            "spread_covered": outcomes,
        })
        
        result = validate_calibration_assumption(
            edge_per_point=0.025,  # Our assumed value
            historical_results=df,
        )
        
        assert "optimal_edge_per_point" in result
        assert "calibration_factor" in result
        assert "bin_analysis" in result
        
        # Should detect that true relationship is ~0.03, not 0.025
        assert abs(result["optimal_edge_per_point"] - 0.03) < 0.02


class TestPrintCalibrationReport:
    """Tests for report printing."""
    
    def test_print_report_doesnt_crash(self, capsys):
        """Test that print_calibration_report runs without error."""
        metrics = CalibrationMetrics(
            brier_score=0.25,
            log_loss=0.693,
            expected_calibration_error=0.05,
            max_calibration_error=0.1,
            reliability_diagram={
                "mean_predicted_prob": [0.3, 0.5, 0.7],
                "fraction_of_positives": [0.25, 0.55, 0.65],
            },
            n_samples=100,
        )
        
        print_calibration_report(metrics, name="Test Model")
        
        captured = capsys.readouterr()
        assert "Test Model" in captured.out
        assert "Brier Score" in captured.out

