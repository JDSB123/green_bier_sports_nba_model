"""
Feature Completeness Tracker.

Tracks which features are present/missing for each prediction.
Helps identify data pipeline issues and monitor data quality.
"""

from __future__ import annotations

import logging
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class FeatureCompletenessStats:
    """Statistics for feature completeness tracking."""
    total_predictions: int = 0
    predictions_with_missing: int = 0
    # Missing feature counts
    missing_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    # Per-market completeness
    by_market: Dict[str, Dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: {"complete": 0, "incomplete": 0}))
    # Feature presence rates
    feature_presence: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    # Recent incomplete predictions
    recent_incomplete: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class CompletenessRecord:
    """Record of a single feature completeness check."""
    timestamp: str
    game_date: str
    home_team: str
    away_team: str
    market: str
    total_features: int
    present_features: int
    missing_features: List[str]
    completeness_pct: float


class FeatureCompletenessTracker:
    """
    Tracks feature completeness for predictions.

    Alerts when feature completeness drops below threshold.
    """

    WARNING_THRESHOLD = 0.95  # Alert if completeness drops below 95%
    MAX_RECENT_INCOMPLETE = 50  # Keep last N incomplete predictions

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize feature completeness tracker.

        Args:
            output_dir: Directory to save completeness logs
        """
        self.output_dir = output_dir
        self.stats = FeatureCompletenessStats()
        self._lock = Lock()
        self._records: List[CompletenessRecord] = []

    def record(
        self,
        game_date: str,
        home_team: str,
        away_team: str,
        market: str,
        required_features: List[str],
        present_features: Set[str],
    ) -> float:
        """
        Record feature completeness for a prediction.

        Args:
            game_date: Date of the game
            home_team: Home team name
            away_team: Away team name
            market: Market type (e.g., "fg_spread")
            required_features: List of required feature names
            present_features: Set of features that were present

        Returns:
            Completeness percentage (0-1)
        """
        required_set = set(required_features)
        missing = required_set - present_features
        missing_list = sorted(missing)

        total = len(required_features)
        present_count = total - len(missing)
        completeness_pct = present_count / total if total > 0 else 1.0

        record = CompletenessRecord(
            timestamp=datetime.utcnow().isoformat(),
            game_date=game_date,
            home_team=home_team,
            away_team=away_team,
            market=market,
            total_features=total,
            present_features=present_count,
            missing_features=missing_list,
            completeness_pct=completeness_pct,
        )

        with self._lock:
            self._records.append(record)
            self.stats.total_predictions += 1

            # Track feature presence
            for feature in present_features:
                if feature in required_set:
                    self.stats.feature_presence[feature] += 1

            if missing:
                self.stats.predictions_with_missing += 1

                # Track missing feature counts
                for feature in missing_list:
                    self.stats.missing_counts[feature] += 1

                # Track by market
                self.stats.by_market[market]["incomplete"] += 1

                # Track recent incomplete
                incomplete_info = {
                    "timestamp": record.timestamp,
                    "game_date": game_date,
                    "matchup": f"{away_team} @ {home_team}",
                    "market": market,
                    "completeness": f"{completeness_pct:.1%}",
                    "missing_count": len(missing_list),
                    "missing_features": missing_list[:10],  # First 10
                }
                self.stats.recent_incomplete.append(incomplete_info)

                # Keep only recent
                if len(self.stats.recent_incomplete) > self.MAX_RECENT_INCOMPLETE:
                    self.stats.recent_incomplete = self.stats.recent_incomplete[-self.MAX_RECENT_INCOMPLETE:]

                # Log warning if completeness is low
                if completeness_pct < self.WARNING_THRESHOLD:
                    logger.warning(
                        f"LOW FEATURE COMPLETENESS: {completeness_pct:.1%} for {market} "
                        f"({away_team} @ {home_team}). Missing: {missing_list[:5]}..."
                    )
            else:
                self.stats.by_market[market]["complete"] += 1

        return completeness_pct

    def get_stats(self) -> Dict[str, Any]:
        """Get current feature completeness statistics."""
        with self._lock:
            overall_rate = 1.0
            if self.stats.total_predictions > 0:
                overall_rate = 1 - (self.stats.predictions_with_missing / self.stats.total_predictions)

            return {
                "total_predictions": self.stats.total_predictions,
                "predictions_with_missing": self.stats.predictions_with_missing,
                "overall_completeness_rate": round(overall_rate, 4),
                "top_missing_features": dict(
                    sorted(self.stats.missing_counts.items(), key=lambda x: -x[1])[:20]
                ),
                "by_market": dict(self.stats.by_market),
                "recent_incomplete_count": len(self.stats.recent_incomplete),
            }

    def get_feature_presence_rates(self) -> Dict[str, float]:
        """Get presence rate for each feature."""
        with self._lock:
            if self.stats.total_predictions == 0:
                return {}
            return {
                feature: count / self.stats.total_predictions
                for feature, count in self.stats.feature_presence.items()
            }

    def get_market_completeness(self) -> Dict[str, float]:
        """Get completeness rate by market."""
        with self._lock:
            rates = {}
            for market, counts in self.stats.by_market.items():
                total = counts["complete"] + counts["incomplete"]
                if total > 0:
                    rates[market] = counts["complete"] / total
            return rates

    def get_most_missing_features(self, n: int = 20) -> List[tuple]:
        """Get N most frequently missing features."""
        with self._lock:
            return sorted(
                self.stats.missing_counts.items(),
                key=lambda x: -x[1]
            )[:n]

    def get_recent_incomplete(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get N most recent incomplete predictions."""
        with self._lock:
            return self.stats.recent_incomplete[-n:]

    def save_report(self, output_path: Optional[Path] = None) -> None:
        """Save feature completeness report to file."""
        path = output_path or (self.output_dir / "feature_completeness_report.json" if self.output_dir else None)
        if not path:
            return

        path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "summary": self.get_stats(),
            "market_completeness": self.get_market_completeness(),
            "most_missing_features": self.get_most_missing_features(30),
            "recent_incomplete": self.stats.recent_incomplete,
        }

        with open(path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Saved feature completeness report to {path}")

    def reset(self) -> None:
        """Reset all statistics."""
        with self._lock:
            self.stats = FeatureCompletenessStats()
            self._records.clear()


# Global tracker instance
_feature_tracker: Optional[FeatureCompletenessTracker] = None


def get_feature_tracker() -> FeatureCompletenessTracker:
    """Get global feature completeness tracker instance."""
    global _feature_tracker
    if _feature_tracker is None:
        from src.config import settings
        output_dir = Path(settings.data_processed_dir) / "monitoring"
        _feature_tracker = FeatureCompletenessTracker(output_dir=output_dir)
    return _feature_tracker
