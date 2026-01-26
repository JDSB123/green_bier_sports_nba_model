"""
Model Drift Detection Module.

Monitors prediction accuracy over time to detect model degradation.
Alerts when performance drops significantly from baseline.

DRIFT TYPES DETECTED:
1. Accuracy drift - prediction accuracy falling below baseline
2. Confidence drift - model confidence changing significantly
3. Edge drift - calculated edges deviating from expected distribution
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import mean, stdev
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class DriftMetrics:
    """Metrics for drift detection."""

    # Accuracy tracking
    predictions: int = 0
    correct: int = 0
    incorrect: int = 0
    pending: int = 0

    # Confidence distribution
    confidence_sum: float = 0.0
    confidence_values: List[float] = field(default_factory=list)

    # Edge distribution
    edge_sum: float = 0.0
    edge_values: List[float] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        """Calculate accuracy from resolved predictions."""
        resolved = self.correct + self.incorrect
        if resolved == 0:
            return 0.0
        return self.correct / resolved

    @property
    def avg_confidence(self) -> float:
        """Average confidence score."""
        if self.predictions == 0:
            return 0.0
        return self.confidence_sum / self.predictions

    @property
    def avg_edge(self) -> float:
        """Average edge value."""
        if self.predictions == 0:
            return 0.0
        return self.edge_sum / self.predictions


@dataclass
class DriftAlert:
    """Alert for detected drift."""

    timestamp: str
    drift_type: str  # "accuracy", "confidence", "edge"
    market: str
    period: str
    baseline_value: float
    current_value: float
    deviation_pct: float
    severity: str  # "warning", "critical"
    message: str


class ModelDriftDetector:
    """
    Detects model drift by comparing current performance to baseline.

    BASELINE VALUES (from backtest):
    - FG Spread: 60.6% accuracy
    - FG Total: 59.2% accuracy
    - 1H Spread: 55.9% accuracy
    - 1H Total: 58.1% accuracy
    - Q1 Markets: 55-58% accuracy
    """

    # Baseline accuracy from backtesting
    BASELINE_ACCURACY = {
        "fg_spread": 0.606,
        "fg_total": 0.592,
        "1h_spread": 0.559,
        "1h_total": 0.581,
        "q1_spread": 0.560,
        "q1_total": 0.575,
    }

    # Thresholds for drift detection
    WARNING_THRESHOLD = 0.05  # 5% drop triggers warning
    CRITICAL_THRESHOLD = 0.10  # 10% drop triggers critical alert
    MIN_SAMPLE_SIZE = 30  # Minimum predictions before checking drift

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize drift detector.

        Args:
            output_dir: Directory to save drift reports
        """
        self.output_dir = output_dir
        self._lock = Lock()

        # Metrics by market
        self.metrics: Dict[str, DriftMetrics] = defaultdict(DriftMetrics)

        # Rolling window metrics (last 7 days)
        self.daily_metrics: Dict[str, Dict[str, DriftMetrics]] = defaultdict(
            lambda: defaultdict(DriftMetrics)
        )

        # Alerts
        self.alerts: List[DriftAlert] = []

    def record_prediction(
        self,
        market: str,
        confidence: float,
        edge: float,
        game_date: Optional[str] = None,
    ) -> None:
        """
        Record a new prediction for drift monitoring.

        Args:
            market: Market key (e.g., "fg_spread")
            confidence: Model confidence (0-1)
            edge: Calculated edge
            game_date: Date of the game (optional)
        """
        with self._lock:
            m = self.metrics[market]
            m.predictions += 1
            m.pending += 1
            m.confidence_sum += confidence
            m.confidence_values.append(confidence)
            m.edge_sum += abs(edge)
            m.edge_values.append(abs(edge))

            # Track daily
            if game_date:
                dm = self.daily_metrics[game_date][market]
                dm.predictions += 1
                dm.pending += 1
                dm.confidence_sum += confidence
                dm.edge_sum += abs(edge)

    def record_outcome(
        self,
        market: str,
        is_correct: bool,
        game_date: Optional[str] = None,
    ) -> None:
        """
        Record the outcome of a prediction.

        Args:
            market: Market key
            is_correct: Whether the prediction was correct
            game_date: Date of the game (optional)
        """
        with self._lock:
            m = self.metrics[market]
            if m.pending > 0:
                m.pending -= 1

            if is_correct:
                m.correct += 1
            else:
                m.incorrect += 1

            # Track daily
            if game_date:
                dm = self.daily_metrics[game_date][market]
                if dm.pending > 0:
                    dm.pending -= 1
                if is_correct:
                    dm.correct += 1
                else:
                    dm.incorrect += 1

            # Check for drift
            self._check_drift(market)

    def _check_drift(self, market: str) -> None:
        """Check for drift in a specific market."""
        m = self.metrics[market]
        resolved = m.correct + m.incorrect

        if resolved < self.MIN_SAMPLE_SIZE:
            return

        baseline = self.BASELINE_ACCURACY.get(market)
        if baseline is None:
            return

        current = m.accuracy
        deviation = baseline - current

        if deviation >= self.CRITICAL_THRESHOLD:
            self._create_alert(
                drift_type="accuracy",
                market=market,
                period=market.split("_")[0],
                baseline_value=baseline,
                current_value=current,
                deviation_pct=deviation,
                severity="critical",
            )
        elif deviation >= self.WARNING_THRESHOLD:
            self._create_alert(
                drift_type="accuracy",
                market=market,
                period=market.split("_")[0],
                baseline_value=baseline,
                current_value=current,
                deviation_pct=deviation,
                severity="warning",
            )

    def _create_alert(
        self,
        drift_type: str,
        market: str,
        period: str,
        baseline_value: float,
        current_value: float,
        deviation_pct: float,
        severity: str,
    ) -> None:
        """Create and log a drift alert."""
        message = (
            f"{severity.upper()} DRIFT: {market} accuracy dropped from "
            f"{baseline_value:.1%} to {current_value:.1%} "
            f"({deviation_pct:+.1%} deviation)"
        )

        alert = DriftAlert(
            timestamp=datetime.utcnow().isoformat(),
            drift_type=drift_type,
            market=market,
            period=period,
            baseline_value=baseline_value,
            current_value=current_value,
            deviation_pct=deviation_pct,
            severity=severity,
            message=message,
        )

        self.alerts.append(alert)

        if severity == "critical":
            logger.error(message)
        else:
            logger.warning(message)

    def get_market_stats(self, market: str) -> Dict[str, Any]:
        """Get statistics for a specific market."""
        with self._lock:
            m = self.metrics[market]
            baseline = self.BASELINE_ACCURACY.get(market, 0.55)
            resolved = m.correct + m.incorrect

            return {
                "predictions": m.predictions,
                "correct": m.correct,
                "incorrect": m.incorrect,
                "pending": m.pending,
                "accuracy": round(m.accuracy, 4) if resolved > 0 else None,
                "baseline_accuracy": baseline,
                "drift": round(baseline - m.accuracy, 4) if resolved > 0 else None,
                "avg_confidence": round(m.avg_confidence, 4) if m.predictions > 0 else None,
                "avg_edge": round(m.avg_edge, 4) if m.predictions > 0 else None,
            }

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all markets."""
        with self._lock:
            return {market: self.get_market_stats(market) for market in self.metrics}

    def get_recent_alerts(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get N most recent drift alerts."""
        with self._lock:
            return [
                {
                    "timestamp": a.timestamp,
                    "severity": a.severity,
                    "market": a.market,
                    "drift_type": a.drift_type,
                    "baseline": a.baseline_value,
                    "current": a.current_value,
                    "deviation": a.deviation_pct,
                    "message": a.message,
                }
                for a in self.alerts[-n:]
            ]

    def get_rolling_accuracy(self, market: str, days: int = 7) -> Dict[str, float]:
        """Get rolling accuracy for the last N days."""
        with self._lock:
            cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            results = {}

            for date_str, markets in self.daily_metrics.items():
                if date_str >= cutoff and market in markets:
                    dm = markets[market]
                    resolved = dm.correct + dm.incorrect
                    if resolved > 0:
                        results[date_str] = dm.correct / resolved

            return results

    def is_drifting(self, market: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a market is currently experiencing drift.

        Returns:
            Tuple of (is_drifting, severity or None)
        """
        with self._lock:
            m = self.metrics[market]
            resolved = m.correct + m.incorrect

            if resolved < self.MIN_SAMPLE_SIZE:
                return False, None

            baseline = self.BASELINE_ACCURACY.get(market)
            if baseline is None:
                return False, None

            deviation = baseline - m.accuracy

            if deviation >= self.CRITICAL_THRESHOLD:
                return True, "critical"
            elif deviation >= self.WARNING_THRESHOLD:
                return True, "warning"

            return False, None

    def save_report(self, output_path: Optional[Path] = None) -> None:
        """Save drift detection report to file."""
        path = output_path or (self.output_dir / "drift_report.json" if self.output_dir else None)
        if not path:
            return

        path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "baseline_accuracy": self.BASELINE_ACCURACY,
            "market_stats": self.get_all_stats(),
            "recent_alerts": self.get_recent_alerts(20),
            "drifting_markets": [market for market in self.metrics if self.is_drifting(market)[0]],
        }

        with open(path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Saved drift report to {path}")

    def reset(self) -> None:
        """Reset all metrics and alerts."""
        with self._lock:
            self.metrics.clear()
            self.daily_metrics.clear()
            self.alerts.clear()


# Global drift detector instance
_drift_detector: Optional[ModelDriftDetector] = None


def get_drift_detector() -> ModelDriftDetector:
    """Get global drift detector instance."""
    global _drift_detector
    if _drift_detector is None:
        from src.config import settings

        output_dir = Path(settings.data_processed_dir) / "monitoring"
        _drift_detector = ModelDriftDetector(output_dir=output_dir)
    return _drift_detector
