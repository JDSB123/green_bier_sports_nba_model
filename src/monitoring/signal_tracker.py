"""
Signal Agreement Rate Tracker.

Tracks when the classifier (ML model) and point prediction signals agree or disagree.
High disagreement rates may indicate model calibration issues.

DUAL-SIGNAL AGREEMENT (v6.5):
- Signal 1: Classifier - ML model trained on historical patterns
- Signal 2: Point Prediction - Quantitative prediction vs market line

Both signals must agree for a bet to pass the filter.
"""

from __future__ import annotations

import logging
import json
from datetime import datetime, date
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class SignalAgreementStats:
    """Statistics for signal agreement tracking."""
    total_predictions: int = 0
    agreements: int = 0
    disagreements: int = 0
    # By market
    by_market: Dict[str, Dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: {"agree": 0, "disagree": 0}))
    # By period
    by_period: Dict[str, Dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: {"agree": 0, "disagree": 0}))
    # Recent disagreements for debugging
    recent_disagreements: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def agreement_rate(self) -> float:
        """Calculate overall agreement rate."""
        if self.total_predictions == 0:
            return 0.0
        return self.agreements / self.total_predictions


@dataclass
class SignalRecord:
    """Record of a single signal comparison."""
    timestamp: str
    game_date: str
    home_team: str
    away_team: str
    market: str
    period: str
    classifier_side: str
    prediction_side: str
    signals_agree: bool
    confidence: float
    edge: float
    passes_filter: bool


class SignalAgreementTracker:
    """
    Tracks agreement between classifier and point prediction signals.

    When agreement rate drops below threshold, logs warning for investigation.
    """

    WARNING_THRESHOLD = 0.50  # Alert if agreement rate drops below 50%
    MAX_RECENT_DISAGREEMENTS = 100  # Keep last N disagreements for debugging

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize signal tracker.

        Args:
            output_dir: Directory to save signal logs (optional)
        """
        self.output_dir = output_dir
        self.stats = SignalAgreementStats()
        self._lock = Lock()
        self._records: List[SignalRecord] = []

    def record(
        self,
        game_date: str,
        home_team: str,
        away_team: str,
        market: str,  # e.g., "spread", "total"
        period: str,  # e.g., "q1", "1h", "fg"
        classifier_side: str,
        prediction_side: str,
        confidence: float = 0.0,
        edge: float = 0.0,
        passes_filter: bool = False,
    ) -> None:
        """
        Record a signal comparison.

        Args:
            game_date: Date of the game
            home_team: Home team name
            away_team: Away team name
            market: Market type (spread, total)
            period: Period (q1, 1h, fg)
            classifier_side: Side chosen by classifier (home/away or over/under)
            prediction_side: Side chosen by point prediction
            confidence: Model confidence
            edge: Calculated edge
            passes_filter: Whether the prediction passed the filter
        """
        signals_agree = classifier_side == prediction_side

        record = SignalRecord(
            timestamp=datetime.utcnow().isoformat(),
            game_date=game_date,
            home_team=home_team,
            away_team=away_team,
            market=market,
            period=period,
            classifier_side=classifier_side,
            prediction_side=prediction_side,
            signals_agree=signals_agree,
            confidence=confidence,
            edge=edge,
            passes_filter=passes_filter,
        )

        with self._lock:
            self._records.append(record)
            self.stats.total_predictions += 1

            market_key = f"{period}_{market}"

            if signals_agree:
                self.stats.agreements += 1
                self.stats.by_market[market_key]["agree"] += 1
                self.stats.by_period[period]["agree"] += 1
            else:
                self.stats.disagreements += 1
                self.stats.by_market[market_key]["disagree"] += 1
                self.stats.by_period[period]["disagree"] += 1

                # Track recent disagreements
                disagreement_info = {
                    "timestamp": record.timestamp,
                    "game_date": game_date,
                    "matchup": f"{away_team} @ {home_team}",
                    "market": market_key,
                    "classifier": classifier_side,
                    "prediction": prediction_side,
                    "confidence": confidence,
                    "edge": edge,
                }
                self.stats.recent_disagreements.append(disagreement_info)

                # Keep only recent disagreements
                if len(self.stats.recent_disagreements) > self.MAX_RECENT_DISAGREEMENTS:
                    self.stats.recent_disagreements = self.stats.recent_disagreements[-self.MAX_RECENT_DISAGREEMENTS:]

            # Check for warning threshold
            if self.stats.total_predictions >= 20:  # Minimum sample size
                rate = self.stats.agreement_rate
                if rate < self.WARNING_THRESHOLD:
                    logger.warning(
                        f"LOW SIGNAL AGREEMENT RATE: {rate:.1%} "
                        f"({self.stats.agreements}/{self.stats.total_predictions}). "
                        f"Investigate model calibration."
                    )

    def get_stats(self) -> Dict[str, Any]:
        """Get current signal agreement statistics."""
        with self._lock:
            return {
                "total_predictions": self.stats.total_predictions,
                "agreements": self.stats.agreements,
                "disagreements": self.stats.disagreements,
                "agreement_rate": round(self.stats.agreement_rate, 4),
                "by_market": dict(self.stats.by_market),
                "by_period": dict(self.stats.by_period),
                "recent_disagreements_count": len(self.stats.recent_disagreements),
            }

    def get_market_rates(self) -> Dict[str, float]:
        """Get agreement rate by market."""
        with self._lock:
            rates = {}
            for market, counts in self.stats.by_market.items():
                total = counts["agree"] + counts["disagree"]
                if total > 0:
                    rates[market] = counts["agree"] / total
            return rates

    def get_period_rates(self) -> Dict[str, float]:
        """Get agreement rate by period."""
        with self._lock:
            rates = {}
            for period, counts in self.stats.by_period.items():
                total = counts["agree"] + counts["disagree"]
                if total > 0:
                    rates[period] = counts["agree"] / total
            return rates

    def get_recent_disagreements(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get N most recent disagreements for debugging."""
        with self._lock:
            return self.stats.recent_disagreements[-n:]

    def save_report(self, output_path: Optional[Path] = None) -> None:
        """Save signal agreement report to file."""
        path = output_path or (self.output_dir / "signal_agreement_report.json" if self.output_dir else None)
        if not path:
            return

        path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "summary": self.get_stats(),
            "market_rates": self.get_market_rates(),
            "period_rates": self.get_period_rates(),
            "recent_disagreements": self.stats.recent_disagreements,
        }

        with open(path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Saved signal agreement report to {path}")

    def reset(self) -> None:
        """Reset all statistics."""
        with self._lock:
            self.stats = SignalAgreementStats()
            self._records.clear()


# Global signal tracker instance
_signal_tracker: Optional[SignalAgreementTracker] = None


def get_signal_tracker() -> SignalAgreementTracker:
    """Get global signal agreement tracker instance."""
    global _signal_tracker
    if _signal_tracker is None:
        from src.config import settings
        output_dir = Path(settings.data_processed_dir) / "monitoring"
        _signal_tracker = SignalAgreementTracker(output_dir=output_dir)
    return _signal_tracker
