"""
Temporal Validation Module - Feature Leakage Prevention.

Ensures all features are computed from data strictly BEFORE the game date.
This prevents temporal leakage where future information contaminates training.

CRITICAL: This module validates that:
1. Rolling stats only use games before the current game
2. H2H features only use historical matchups
3. No post-game data (scores, results) leaks into pre-game features
"""

from __future__ import annotations

import logging
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class TemporalLeakageError(ValueError):
    """Raised when temporal leakage is detected in feature computation."""

    def __init__(
        self,
        feature_name: str,
        game_date: date,
        data_date: date,
        details: str = "",
    ):
        self.feature_name = feature_name
        self.game_date = game_date
        self.data_date = data_date
        self.details = details

        message = (
            f"TEMPORAL LEAKAGE DETECTED: Feature '{feature_name}' uses data from {data_date} "
            f"for game on {game_date}. {details}"
        )
        super().__init__(message)


@dataclass
class TemporalValidationResult:
    """Result of temporal validation check."""
    is_valid: bool
    game_date: date
    feature_name: str
    data_dates: List[date] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)
    warning_count: int = 0


class TemporalValidator:
    """
    Validates that features are computed from strictly historical data.

    WALK-FORWARD VALIDATION RULES:
    - For a game on date D, all features must use data from dates < D
    - Game outcomes (scores) from date D must NOT be used
    - Rolling stats must be computed from games that FINISHED before D
    - H2H features must only include games that FINISHED before D
    """

    # Features that MUST be strictly historical (game-outcome dependent)
    STRICT_TEMPORAL_FEATURES = {
        # Rolling averages
        "home_ppg", "away_ppg",
        "home_papg", "away_papg",
        "home_avg_margin", "away_avg_margin",
        "home_win_pct", "away_win_pct",
        # Period-specific rolling
        "home_q1_ppg", "away_q1_ppg",
        "home_q1_papg", "away_q1_papg",
        "home_1h_ppg", "away_1h_ppg",
        "home_1h_papg", "away_1h_papg",
        # Form
        "home_form", "away_form",
        "home_last_3_margin", "away_last_3_margin",
        # H2H
        "h2h_win_pct", "h2h_avg_margin",
        # Streaks
        "home_streak", "away_streak",
    }

    # Features that can use same-day data (pre-game info)
    SAME_DAY_ALLOWED = {
        # Lines are set before game starts
        "spread_line", "total_line", "home_ml_odds", "away_ml_odds",
        "fh_spread_line", "fh_total_line",
        "q1_spread_line", "q1_total_line",
        # Injury reports (pre-game)
        "home_injury_impact", "away_injury_impact",
        # Rest days (computed from schedule)
        "home_rest_days", "away_rest_days",
        "home_b2b", "away_b2b",
    }

    def __init__(self, strict_mode: bool = True):
        """
        Initialize temporal validator.

        Args:
            strict_mode: If True, raises TemporalLeakageError on violations.
                        If False, logs warnings and continues.
        """
        self.strict_mode = strict_mode
        self._violation_log: List[Dict[str, Any]] = []

    def validate_feature_date(
        self,
        feature_name: str,
        game_date: date,
        data_date: date,
    ) -> bool:
        """
        Validate that a feature's data date is before the game date.

        Args:
            feature_name: Name of the feature being validated
            game_date: Date of the game being predicted
            data_date: Date of the data used to compute the feature

        Returns:
            True if valid (data_date < game_date)

        Raises:
            TemporalLeakageError: If strict_mode and data_date >= game_date
        """
        # Same-day data is allowed for certain features
        if feature_name in self.SAME_DAY_ALLOWED:
            return True

        # For strict temporal features, data must be from before game date
        if data_date >= game_date:
            violation = {
                "feature": feature_name,
                "game_date": str(game_date),
                "data_date": str(data_date),
                "timestamp": datetime.utcnow().isoformat(),
            }
            self._violation_log.append(violation)

            if self.strict_mode and feature_name in self.STRICT_TEMPORAL_FEATURES:
                raise TemporalLeakageError(
                    feature_name=feature_name,
                    game_date=game_date,
                    data_date=data_date,
                    details="Feature uses game outcome data from same day or future."
                )
            else:
                logger.warning(
                    f"Temporal warning: Feature '{feature_name}' uses data from {data_date} "
                    f"for game on {game_date}"
                )
            return False

        return True

    def validate_rolling_window(
        self,
        game_date: date,
        window_games: List[Dict[str, Any]],
        window_name: str = "rolling",
    ) -> TemporalValidationResult:
        """
        Validate that all games in a rolling window are before the target game.

        Args:
            game_date: Date of the game being predicted
            window_games: List of games used in the rolling window
            window_name: Name for error messages

        Returns:
            TemporalValidationResult with validation details
        """
        violations = []
        data_dates = []

        for game in window_games:
            game_end_date = None

            # Try to extract game date from various formats
            if "date" in game:
                game_end_date = self._parse_date(game["date"])
            elif "game_date" in game:
                game_end_date = self._parse_date(game["game_date"])

            if game_end_date:
                data_dates.append(game_end_date)

                if game_end_date >= game_date:
                    violations.append(
                        f"Game from {game_end_date} used in {window_name} window for {game_date}"
                    )

        is_valid = len(violations) == 0

        if not is_valid and self.strict_mode:
            raise TemporalLeakageError(
                feature_name=window_name,
                game_date=game_date,
                data_date=max(data_dates) if data_dates else game_date,
                details=f"Found {len(violations)} future games in window: {violations[:3]}"
            )

        return TemporalValidationResult(
            is_valid=is_valid,
            game_date=game_date,
            feature_name=window_name,
            data_dates=data_dates,
            violations=violations,
            warning_count=len(violations) if not is_valid else 0,
        )

    def validate_h2h_history(
        self,
        game_date: date,
        h2h_games: List[Dict[str, Any]],
    ) -> TemporalValidationResult:
        """
        Validate that H2H history only includes games before the current matchup.

        Args:
            game_date: Date of the game being predicted
            h2h_games: List of historical H2H games

        Returns:
            TemporalValidationResult with validation details
        """
        return self.validate_rolling_window(game_date, h2h_games, "h2h_history")

    def get_violations(self) -> List[Dict[str, Any]]:
        """Get list of all temporal violations detected."""
        return self._violation_log.copy()

    def clear_violations(self) -> None:
        """Clear the violation log."""
        self._violation_log.clear()

    def save_violations(self, output_path: Path) -> None:
        """Save violation log to file for debugging."""
        if self._violation_log:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(self._violation_log, f, indent=2)
            logger.info(f"Saved {len(self._violation_log)} temporal violations to {output_path}")

    @staticmethod
    def _parse_date(date_value: Any) -> Optional[date]:
        """Parse date from various formats."""
        if isinstance(date_value, date):
            return date_value
        if isinstance(date_value, datetime):
            return date_value.date()
        if isinstance(date_value, str):
            try:
                # Try ISO format
                if "T" in date_value:
                    return datetime.fromisoformat(date_value.replace("Z", "+00:00")).date()
                else:
                    return datetime.strptime(date_value, "%Y-%m-%d").date()
            except ValueError:
                pass
        return None


# Global validator instance
_temporal_validator = TemporalValidator(strict_mode=False)  # Default to warn mode


def validate_feature_temporality(
    feature_name: str,
    game_date: date,
    data_date: date,
    strict: bool = False,
) -> bool:
    """
    Convenience function to validate feature temporality.

    Args:
        feature_name: Name of the feature
        game_date: Date of the game being predicted
        data_date: Date of the data used
        strict: If True, raises on violation

    Returns:
        True if valid
    """
    validator = TemporalValidator(strict_mode=strict)
    return validator.validate_feature_date(feature_name, game_date, data_date)
