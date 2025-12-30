"""
Prediction Audit Logger.

Provides comprehensive logging of all predictions for:
- Debugging and troubleshooting
- Performance tracking
- Compliance and audit trails
- Post-hoc analysis
"""

from __future__ import annotations

import logging
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path
from threading import Lock
import gzip

logger = logging.getLogger(__name__)

# Model version from central source
_MODEL_VERSION = os.getenv("NBA_MODEL_VERSION", "NBA_v33.0.8.0")


@dataclass
class PredictionRecord:
    """Complete record of a single prediction."""
    # Identifiers
    prediction_id: str
    request_id: Optional[str]
    timestamp: str

    # Game info
    game_date: str
    home_team: str
    away_team: str
    commence_time: Optional[str]

    # Market info
    period: str  # q1, 1h, fg
    market: str  # spread, total
    market_key: str  # Combined: fg_spread, q1_total, etc.

    # Lines
    line: Optional[float]  # Spread line or total line

    # Prediction output
    classifier_side: str
    prediction_side: str
    signals_agree: bool
    bet_side: Optional[str]
    confidence: float
    edge: float
    raw_edge: float
    passes_filter: bool
    filter_reason: Optional[str]

    # Model info
    model_version: str = field(default_factory=lambda: _MODEL_VERSION)

    # Feature completeness
    features_present: int = 0
    features_required: int = 0
    missing_features: List[str] = field(default_factory=list)

    # Outcome (filled later)
    outcome: Optional[str] = None  # "win", "loss", "push", None
    actual_score_home: Optional[int] = None
    actual_score_away: Optional[int] = None


class PredictionLogger:
    """
    Logs all predictions with full audit trail.

    Features:
    - Structured JSON logging
    - Automatic log rotation
    - Compression of old logs
    - Query interface for analysis
    """

    MAX_RECORDS_IN_MEMORY = 1000
    FLUSH_INTERVAL = 100  # Flush every N records

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize prediction logger.

        Args:
            output_dir: Directory to store prediction logs
        """
        self.output_dir = output_dir or Path("data/processed/prediction_logs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._lock = Lock()
        self._records: List[PredictionRecord] = []
        self._record_count = 0
        self._current_date: Optional[str] = None
        self._current_file: Optional[Path] = None

    def log(
        self,
        request_id: Optional[str],
        game_date: str,
        home_team: str,
        away_team: str,
        period: str,
        market: str,
        prediction_result: Dict[str, Any],
        line: Optional[float] = None,
        commence_time: Optional[str] = None,
        features_present: int = 0,
        features_required: int = 0,
        missing_features: Optional[List[str]] = None,
    ) -> str:
        """
        Log a prediction with full details.

        Args:
            request_id: Request ID for tracing
            game_date: Date of the game
            home_team: Home team name
            away_team: Away team name
            period: Period (q1, 1h, fg)
            market: Market type (spread, total)
            prediction_result: Prediction output dictionary
            line: Spread or total line
            commence_time: Game start time
            features_present: Number of features present
            features_required: Number of required features
            missing_features: List of missing feature names

        Returns:
            Prediction ID
        """
        prediction_id = f"{game_date}_{home_team[:3]}_{away_team[:3]}_{period}_{market}_{datetime.utcnow().strftime('%H%M%S%f')}"

        # Extract prediction details based on market type
        classifier_side = prediction_result.get("classifier_side", "unknown")
        prediction_side = prediction_result.get("prediction_side", "unknown")
        bet_side = prediction_result.get("bet_side")
        edge = prediction_result.get("edge", 0)
        raw_edge = prediction_result.get("raw_edge", 0)

        record = PredictionRecord(
            prediction_id=prediction_id,
            request_id=request_id,
            timestamp=datetime.utcnow().isoformat(),
            game_date=game_date,
            home_team=home_team,
            away_team=away_team,
            commence_time=commence_time,
            period=period,
            market=market,
            market_key=f"{period}_{market}",
            line=line,
            classifier_side=classifier_side,
            prediction_side=prediction_side,
            signals_agree=prediction_result.get("signals_agree", classifier_side == prediction_side),
            bet_side=bet_side,
            confidence=prediction_result.get("confidence", 0.0),
            edge=abs(edge) if edge else 0.0,
            raw_edge=raw_edge if raw_edge else 0.0,
            passes_filter=prediction_result.get("passes_filter", False),
            filter_reason=prediction_result.get("filter_reason"),
            features_present=features_present,
            features_required=features_required,
            missing_features=missing_features or [],
        )

        with self._lock:
            self._records.append(record)
            self._record_count += 1

            # Flush periodically
            if len(self._records) >= self.FLUSH_INTERVAL:
                self._flush()

            # Keep memory bounded
            if len(self._records) > self.MAX_RECORDS_IN_MEMORY:
                self._records = self._records[-self.MAX_RECORDS_IN_MEMORY:]

        return prediction_id

    def _flush(self) -> None:
        """Flush records to file."""
        if not self._records:
            return

        today = datetime.utcnow().strftime("%Y-%m-%d")

        # Rotate file daily
        if self._current_date != today:
            self._current_date = today
            self._current_file = self.output_dir / f"predictions_{today}.jsonl"

        # Append to file
        try:
            with open(self._current_file, "a") as f:
                for record in self._records:
                    f.write(json.dumps(asdict(record), default=str) + "\n")
            self._records.clear()
        except Exception as e:
            logger.error(f"Failed to flush prediction logs: {e}")

    def update_outcome(
        self,
        prediction_id: str,
        outcome: str,
        actual_score_home: Optional[int] = None,
        actual_score_away: Optional[int] = None,
    ) -> None:
        """
        Update a prediction with the actual outcome.

        Args:
            prediction_id: ID of the prediction to update
            outcome: "win", "loss", or "push"
            actual_score_home: Actual home score
            actual_score_away: Actual away score
        """
        # This would update the record in storage
        # For now, log the update
        logger.info(f"Prediction {prediction_id} outcome: {outcome} (scores: {actual_score_home}-{actual_score_away})")

    def get_recent_predictions(self, n: int = 50) -> List[Dict[str, Any]]:
        """Get N most recent predictions from memory."""
        with self._lock:
            return [asdict(r) for r in self._records[-n:]]

    def get_predictions_by_date(self, game_date: str) -> List[Dict[str, Any]]:
        """Get all predictions for a specific date from file."""
        log_file = self.output_dir / f"predictions_{game_date}.jsonl"
        if not log_file.exists():
            return []

        predictions = []
        try:
            with open(log_file, "r") as f:
                for line in f:
                    predictions.append(json.loads(line))
        except Exception as e:
            logger.error(f"Failed to read predictions for {game_date}: {e}")

        return predictions

    def get_stats_by_market(self) -> Dict[str, Dict[str, int]]:
        """Get prediction counts by market."""
        with self._lock:
            stats = {}
            for record in self._records:
                key = record.market_key
                if key not in stats:
                    stats[key] = {"total": 0, "passed_filter": 0, "signals_agree": 0}
                stats[key]["total"] += 1
                if record.passes_filter:
                    stats[key]["passed_filter"] += 1
                if record.signals_agree:
                    stats[key]["signals_agree"] += 1
            return stats

    def compress_old_logs(self, days_old: int = 7) -> int:
        """Compress log files older than N days."""
        cutoff = datetime.utcnow().date()
        compressed_count = 0

        for log_file in self.output_dir.glob("predictions_*.jsonl"):
            try:
                # Extract date from filename
                date_str = log_file.stem.replace("predictions_", "")
                file_date = datetime.strptime(date_str, "%Y-%m-%d").date()

                if (cutoff - file_date).days > days_old:
                    # Compress
                    gz_path = log_file.with_suffix(".jsonl.gz")
                    with open(log_file, "rb") as f_in:
                        with gzip.open(gz_path, "wb") as f_out:
                            f_out.writelines(f_in)
                    log_file.unlink()
                    compressed_count += 1
            except Exception as e:
                logger.warning(f"Failed to compress {log_file}: {e}")

        return compressed_count

    def close(self) -> None:
        """Flush remaining records and close."""
        with self._lock:
            self._flush()


# Global prediction logger instance
_prediction_logger: Optional[PredictionLogger] = None


def get_prediction_logger() -> PredictionLogger:
    """Get global prediction logger instance."""
    global _prediction_logger
    if _prediction_logger is None:
        from src.config import settings
        output_dir = Path(settings.data_processed_dir) / "prediction_logs"
        _prediction_logger = PredictionLogger(output_dir=output_dir)
    return _prediction_logger
