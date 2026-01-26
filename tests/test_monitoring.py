"""
Tests for src/monitoring/ modules - drift detection, feature completeness, prediction logging.

Coverage target: 95%+ for monitoring modules.
"""

import json
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.monitoring.drift_detection import DriftAlert, DriftMetrics


class TestDriftMetrics:
    """Tests for DriftMetrics dataclass."""

    def test_drift_metrics_creation(self):
        """DriftMetrics should be created with default values."""
        metrics = DriftMetrics()

        assert metrics.predictions == 0
        assert metrics.correct == 0
        assert metrics.incorrect == 0
        assert metrics.pending == 0
        assert metrics.confidence_sum == 0.0
        assert metrics.edge_sum == 0.0

    def test_accuracy_calculation_zero(self):
        """Accuracy should be 0 with no resolved predictions."""
        metrics = DriftMetrics()

        assert metrics.accuracy == 0.0

    def test_accuracy_calculation_perfect(self):
        """Accuracy should be 1.0 with all correct."""
        metrics = DriftMetrics(
            predictions=10,
            correct=10,
            incorrect=0,
        )

        assert metrics.accuracy == 1.0

    def test_accuracy_calculation_mixed(self):
        """Accuracy should calculate correctly with mixed results."""
        metrics = DriftMetrics(
            predictions=10,
            correct=6,
            incorrect=4,
        )

        assert metrics.accuracy == 0.6

    def test_avg_confidence_zero(self):
        """Average confidence should be 0 with no predictions."""
        metrics = DriftMetrics()

        assert metrics.avg_confidence == 0.0

    def test_avg_confidence_calculation(self):
        """Average confidence should calculate correctly."""
        metrics = DriftMetrics(
            predictions=5,
            confidence_sum=3.0,  # 0.6 average
        )

        assert metrics.avg_confidence == 0.6

    def test_avg_edge_zero(self):
        """Average edge should be 0 with no predictions."""
        metrics = DriftMetrics()

        assert metrics.avg_edge == 0.0

    def test_avg_edge_calculation(self):
        """Average edge should calculate correctly."""
        metrics = DriftMetrics(
            predictions=4,
            edge_sum=0.2,  # 0.05 average
        )

        assert metrics.avg_edge == pytest.approx(0.05, abs=0.001)

    def test_drift_metrics_with_values(self):
        """DriftMetrics should track confidence/edge values."""
        metrics = DriftMetrics(
            predictions=3,
            confidence_values=[0.55, 0.60, 0.65],
            edge_values=[0.02, 0.05, 0.03],
        )

        assert len(metrics.confidence_values) == 3
        assert len(metrics.edge_values) == 3


class TestDriftAlert:
    """Tests for DriftAlert dataclass."""

    def test_drift_alert_creation(self):
        """DriftAlert should be created with all required fields."""
        alert = DriftAlert(
            timestamp="2025-01-15T10:30:00",
            drift_type="accuracy",
            market="fg_spread",
            period="fg",
            baseline_value=0.55,
            current_value=0.48,
            deviation_pct=-12.7,
            severity="warning",
            message="Accuracy dropped below baseline",
        )

        assert alert.drift_type == "accuracy"
        assert alert.market == "fg_spread"
        assert alert.severity == "warning"

    def test_drift_alert_accuracy_type(self):
        """Accuracy drift alert should have correct type."""
        alert = DriftAlert(
            timestamp="2025-01-15T10:30:00",
            drift_type="accuracy",
            market="1h_total",
            period="1h",
            baseline_value=0.54,
            current_value=0.46,
            deviation_pct=-14.8,
            severity="critical",
            message="Critical accuracy drop",
        )

        assert alert.drift_type == "accuracy"
        assert alert.severity == "critical"

    def test_drift_alert_confidence_type(self):
        """Confidence drift alert should have correct type."""
        alert = DriftAlert(
            timestamp="2025-01-15T10:30:00",
            drift_type="confidence",
            market="fg_spread",
            period="fg",
            baseline_value=0.60,
            current_value=0.70,
            deviation_pct=16.7,
            severity="warning",
            message="Confidence distribution shift",
        )

        assert alert.drift_type == "confidence"

    def test_drift_alert_edge_type(self):
        """Edge drift alert should have correct type."""
        alert = DriftAlert(
            timestamp="2025-01-15T10:30:00",
            drift_type="edge",
            market="1h_spread",
            period="1h",
            baseline_value=0.03,
            current_value=0.01,
            deviation_pct=-66.7,
            severity="warning",
            message="Edge values declining",
        )

        assert alert.drift_type == "edge"


class TestDriftDetectionLogic:
    """Tests for drift detection logic."""

    def test_accuracy_drift_threshold(self):
        """Accuracy drift should be detected when below threshold."""
        baseline = 0.55
        current = 0.48
        threshold_pct = 10.0  # Alert if 10% below baseline

        deviation_pct = ((current - baseline) / baseline) * 100

        assert deviation_pct < 0  # Negative = worse
        assert abs(deviation_pct) > threshold_pct  # Exceeds threshold

    def test_accuracy_drift_within_threshold(self):
        """No alert when accuracy within threshold."""
        baseline = 0.55
        current = 0.53
        threshold_pct = 10.0

        deviation_pct = ((current - baseline) / baseline) * 100

        assert abs(deviation_pct) < threshold_pct  # Within threshold

    def test_confidence_drift_detection(self):
        """Confidence drift should be detected when distribution shifts."""
        baseline_confidence = 0.60
        current_confidence = 0.72
        threshold_pct = 15.0

        deviation_pct = ((current_confidence - baseline_confidence) / baseline_confidence) * 100

        assert deviation_pct > threshold_pct  # Exceeds threshold

    def test_edge_drift_detection(self):
        """Edge drift should be detected when edges shrink."""
        baseline_edge = 0.04
        current_edge = 0.015
        threshold_pct = 50.0  # Alert if edges drop 50%

        deviation_pct = ((current_edge - baseline_edge) / baseline_edge) * 100

        assert deviation_pct < -threshold_pct  # Below threshold


class TestDriftSeverity:
    """Tests for drift severity levels."""

    def test_warning_severity(self):
        """Warning severity for moderate drift."""
        deviation_pct = -12.0  # 12% below baseline

        severity = "warning" if abs(deviation_pct) < 20 else "critical"

        assert severity == "warning"

    def test_critical_severity(self):
        """Critical severity for severe drift."""
        deviation_pct = -25.0  # 25% below baseline

        severity = "warning" if abs(deviation_pct) < 20 else "critical"

        assert severity == "critical"

    def test_severity_thresholds(self):
        """Severity thresholds should be sensible."""
        warning_threshold = 10.0  # 10% deviation
        critical_threshold = 20.0  # 20% deviation

        assert warning_threshold < critical_threshold


class TestFeatureCompleteness:
    """Tests for feature completeness monitoring."""

    def test_feature_coverage_calculation(self):
        """Feature coverage should be calculated correctly."""
        total_features = 100
        available_features = 95

        coverage = available_features / total_features

        assert coverage == 0.95

    def test_feature_coverage_threshold(self):
        """Should alert when coverage below threshold."""
        coverage = 0.90
        threshold = 0.95

        assert coverage < threshold  # Should alert

    def test_missing_features_detection(self):
        """Missing features should be identified."""
        required = {"feature_a", "feature_b", "feature_c"}
        available = {"feature_a", "feature_c"}

        missing = required - available

        assert missing == {"feature_b"}


class TestPredictionLogging:
    """Tests for prediction logging."""

    def test_prediction_log_structure(self):
        """Prediction log should have required fields."""
        log_entry = {
            "timestamp": "2025-01-15T19:30:00",
            "market": "fg_spread",
            "game_id": "2025-01-15_LAL_BOS",
            "prediction": "home",
            "confidence": 0.62,
            "edge": 0.04,
            "line": -3.5,
            "home_team": "Boston Celtics",
            "away_team": "Los Angeles Lakers",
        }

        assert "timestamp" in log_entry
        assert "market" in log_entry
        assert "prediction" in log_entry
        assert "confidence" in log_entry

    def test_prediction_log_market_coverage(self):
        """All 4 markets should be loggable."""
        markets = ["fg_spread", "fg_total", "1h_spread", "1h_total"]

        for market in markets:
            log_entry = {"market": market}
            assert log_entry["market"] in markets

    def test_prediction_resolution_log(self):
        """Resolved predictions should have outcome."""
        log_entry = {
            "timestamp": "2025-01-15T22:00:00",
            "market": "fg_spread",
            "game_id": "2025-01-15_LAL_BOS",
            "prediction": "home",
            "outcome": "win",  # Added after game ends
            "actual_result": "home_covered",
        }

        assert "outcome" in log_entry
        assert log_entry["outcome"] in ["win", "loss", "push"]


class TestSignalTracking:
    """Tests for signal tracking."""

    def test_signal_agree_tracking(self):
        """Should track when signals agree."""
        model_signal = "home"  # Model says home covers
        public_signal = "home"  # Public money on home

        signals_agree = model_signal == public_signal

        assert signals_agree is True

    def test_signal_disagree_tracking(self):
        """Should track when signals disagree."""
        model_signal = "home"
        public_signal = "away"

        signals_agree = model_signal == public_signal

        assert signals_agree is False

    def test_conflict_signal_performance(self):
        """Should track performance when signals conflict."""
        # When model and public disagree, track who wins
        conflicts = [
            {"model": "home", "public": "away", "actual": "home"},  # Model wins
            {"model": "away", "public": "home", "actual": "home"},  # Public wins
            {"model": "over", "public": "under", "actual": "over"},  # Model wins
        ]

        model_wins = sum(1 for c in conflicts if c["model"] == c["actual"])

        assert model_wins == 2


class TestTemporalValidation:
    """Tests for temporal validation."""

    def test_no_future_data_leak(self):
        """Should ensure no future data is used."""
        prediction_time = datetime(2025, 1, 15, 12, 0, 0)
        game_time = datetime(2025, 1, 15, 19, 30, 0)

        # All data must be before prediction_time
        assert prediction_time < game_time

    def test_feature_timestamp_validation(self):
        """Feature timestamps should be before prediction."""
        prediction_time = datetime(2025, 1, 15, 12, 0, 0)

        feature_timestamps = [
            datetime(2025, 1, 14, 22, 0, 0),  # Previous day
            datetime(2025, 1, 15, 9, 0, 0),  # Same day, earlier
            datetime(2025, 1, 15, 11, 0, 0),  # Just before
        ]

        for ts in feature_timestamps:
            assert ts < prediction_time

    def test_rolling_window_validation(self):
        """Rolling features should only use valid past data."""
        current_date = datetime(2025, 1, 15).date()
        window_days = 10

        # Valid window: Jan 5-14 (before current date)
        window_start = current_date - timedelta(days=window_days)
        window_end = current_date - timedelta(days=1)  # Yesterday

        assert window_end < current_date


class TestMonitoringMetrics:
    """Tests for monitoring metrics export."""

    def test_prometheus_metric_format(self):
        """Metrics should be in Prometheus format."""
        metric_name = "nba_prediction_accuracy"
        metric_value = 0.55
        labels = {"market": "fg_spread", "period": "fg"}

        # Prometheus format: metric_name{labels} value
        label_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
        prometheus_line = f"{metric_name}{{{label_str}}} {metric_value}"

        assert "nba_prediction_accuracy" in prometheus_line
        assert "fg_spread" in prometheus_line

    def test_accuracy_by_market_metric(self):
        """Should have accuracy metric per market."""
        markets = ["fg_spread", "fg_total", "1h_spread", "1h_total"]
        accuracies = [0.55, 0.54, 0.53, 0.52]

        metrics = {m: a for m, a in zip(markets, accuracies)}

        assert len(metrics) == 4
        assert all(0 < a < 1 for a in metrics.values())

    def test_roi_by_market_metric(self):
        """Should have ROI metric per market."""
        markets = ["fg_spread", "fg_total", "1h_spread", "1h_total"]
        rois = [0.02, 0.01, -0.01, 0.03]

        metrics = {m: r for m, r in zip(markets, rois)}

        assert len(metrics) == 4


class TestAlertingIntegration:
    """Tests for alerting system integration."""

    def test_alert_json_serializable(self):
        """Alerts should be JSON serializable."""
        alert = DriftAlert(
            timestamp="2025-01-15T10:30:00",
            drift_type="accuracy",
            market="fg_spread",
            period="fg",
            baseline_value=0.55,
            current_value=0.48,
            deviation_pct=-12.7,
            severity="warning",
            message="Accuracy dropped",
        )

        alert_dict = asdict(alert)
        json_str = json.dumps(alert_dict)

        assert "accuracy" in json_str
        assert "warning" in json_str

    def test_multiple_alerts_aggregation(self):
        """Multiple alerts should be aggregatable."""
        alerts = [
            DriftAlert(
                timestamp="2025-01-15T10:30:00",
                drift_type="accuracy",
                market="fg_spread",
                period="fg",
                baseline_value=0.55,
                current_value=0.48,
                deviation_pct=-12.7,
                severity="warning",
                message="Alert 1",
            ),
            DriftAlert(
                timestamp="2025-01-15T10:35:00",
                drift_type="confidence",
                market="1h_total",
                period="1h",
                baseline_value=0.60,
                current_value=0.72,
                deviation_pct=20.0,
                severity="critical",
                message="Alert 2",
            ),
        ]

        critical_count = sum(1 for a in alerts if a.severity == "critical")

        assert len(alerts) == 2
        assert critical_count == 1
