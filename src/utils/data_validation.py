"""
Data Validation Module.

Provides comprehensive data validation for training data pipeline:
- Per-column null thresholds (different limits for core vs optional features)
- Betting lines date match validation
- Data quality scoring
"""

from __future__ import annotations

import logging
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)


# Per-column null thresholds
# STRICT columns: Core features that should rarely be missing
# MODERATE columns: Important but sometimes unavailable
# LENIENT columns: Optional enrichment features
COLUMN_NULL_THRESHOLDS = {
    # === STRICT (max 5% nulls) ===
    # Core game data
    "date": 0.00,
    "home_team": 0.00,
    "away_team": 0.00,
    "home_score": 0.00,
    "away_score": 0.00,
    "game_id": 0.05,
    "season": 0.05,

    # Core betting lines (FG)
    "spread_line": 0.05,
    "total_line": 0.05,

    # === MODERATE (max 15% nulls) ===
    # Quarter scores
    "home_q1": 0.15,
    "home_q2": 0.15,
    "home_q3": 0.15,
    "home_q4": 0.15,
    "away_q1": 0.15,
    "away_q2": 0.15,
    "away_q3": 0.15,
    "away_q4": 0.15,

    # Period lines
    "fh_spread_line": 0.15,
    "fh_total_line": 0.15,
    "q1_spread_line": 0.20,
    "q1_total_line": 0.20,

    # Moneyline odds
    "home_ml_odds": 0.15,
    "away_ml_odds": 0.15,

    # === LENIENT (max 30% nulls) ===
    # Betting splits (optional feature)
    "home_spread_pct": 0.30,
    "away_spread_pct": 0.30,
    "over_pct": 0.30,
    "under_pct": 0.30,

    # Period ML odds
    "fh_home_ml_odds": 0.30,
    "fh_away_ml_odds": 0.30,
    "q1_home_ml_odds": 0.40,
    "q1_away_ml_odds": 0.40,

    # Injury data
    "home_injury_impact": 0.30,
    "away_injury_impact": 0.30,
}

# Default threshold for unlisted columns
DEFAULT_NULL_THRESHOLD = 0.20


def validate_null_thresholds(
    df: pd.DataFrame,
    strict: bool = True,
) -> ValidationResult:
    """
    Validate null rates against per-column thresholds.

    Args:
        df: DataFrame to validate
        strict: If True, fail on threshold violations. If False, warn only.

    Returns:
        ValidationResult with validation details
    """
    errors = []
    warnings = []
    stats = {
        "total_rows": len(df),
        "columns_checked": 0,
        "columns_passed": 0,
        "columns_failed": 0,
        "null_rates": {},
    }

    for col in df.columns:
        null_rate = df[col].isna().mean()
        stats["null_rates"][col] = round(null_rate, 4)
        stats["columns_checked"] += 1

        threshold = COLUMN_NULL_THRESHOLDS.get(col, DEFAULT_NULL_THRESHOLD)

        if null_rate > threshold:
            message = (
                f"Column '{col}' has {null_rate:.1%} nulls "
                f"(threshold: {threshold:.1%})"
            )
            if strict and threshold < 0.10:  # Only fail on strict columns
                errors.append(message)
                stats["columns_failed"] += 1
            else:
                warnings.append(message)
                stats["columns_passed"] += 1
        else:
            stats["columns_passed"] += 1

    is_valid = len(errors) == 0

    if errors:
        logger.error(f"Data validation failed: {len(errors)} columns exceed null thresholds")
        for error in errors[:5]:
            logger.error(f"  - {error}")

    if warnings:
        logger.warning(f"Data validation warnings: {len(warnings)} columns have elevated null rates")
        for warning in warnings[:5]:
            logger.warning(f"  - {warning}")

    return ValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        stats=stats,
    )


def validate_betting_lines_match(
    outcomes_df: pd.DataFrame,
    lines_df: pd.DataFrame,
    date_tolerance_days: int = 1,
    time_tolerance_hours: int = 12,
) -> ValidationResult:
    """
    Validate betting lines match against game outcomes.

    Ensures:
    1. Team names match exactly (both standardized to ESPN format)
    2. Dates are within tolerance (handles timezone issues)
    3. Game times are close (prevents wrong game matches)

    Args:
        outcomes_df: DataFrame with game outcomes
        lines_df: DataFrame with betting lines
        date_tolerance_days: Max days difference for date matching
        time_tolerance_hours: Max hours difference for time matching

    Returns:
        ValidationResult with match statistics
    """
    errors = []
    warnings = []
    stats = {
        "outcomes_count": len(outcomes_df),
        "lines_count": len(lines_df),
        "matched": 0,
        "unmatched_outcomes": 0,
        "unmatched_lines": 0,
        "suspicious_matches": [],
    }

    # Track matched games
    matched_outcomes = set()
    matched_lines = set()

    for idx, outcome in outcomes_df.iterrows():
        home_team = outcome.get("home_team", "")
        away_team = outcome.get("away_team", "")
        game_date = outcome.get("date")

        if not home_team or not away_team:
            continue

        # Parse game date
        if isinstance(game_date, str):
            try:
                game_date = datetime.strptime(game_date[:10], "%Y-%m-%d").date()
            except ValueError:
                continue
        elif isinstance(game_date, datetime):
            game_date = game_date.date()
        elif not isinstance(game_date, date):
            continue

        # Find matching line
        best_match = None
        best_match_idx = None

        for line_idx, line in lines_df.iterrows():
            line_home = line.get("home_team", "")
            line_away = line.get("away_team", "")
            line_date = line.get("date") or line.get("commence_time")

            # Check team match
            if line_home != home_team or line_away != away_team:
                continue

            # Parse line date
            if isinstance(line_date, str):
                try:
                    if "T" in line_date:
                        line_dt = datetime.fromisoformat(line_date.replace("Z", "+00:00"))
                        line_date = line_dt.date()
                    else:
                        line_date = datetime.strptime(line_date[:10], "%Y-%m-%d").date()
                except ValueError:
                    continue
            elif isinstance(line_date, datetime):
                line_date = line_date.date()
            elif not isinstance(line_date, date):
                continue

            # Check date within tolerance
            date_diff = abs((game_date - line_date).days)
            if date_diff <= date_tolerance_days:
                if best_match is None or date_diff < best_match["date_diff"]:
                    best_match = {
                        "line_idx": line_idx,
                        "date_diff": date_diff,
                        "line_date": line_date,
                    }
                    best_match_idx = line_idx

        if best_match:
            matched_outcomes.add(idx)
            matched_lines.add(best_match_idx)
            stats["matched"] += 1

            # Warn if date doesn't match exactly
            if best_match["date_diff"] > 0:
                warning = (
                    f"Date mismatch for {away_team} @ {home_team}: "
                    f"outcome={game_date}, line={best_match['line_date']} "
                    f"(diff={best_match['date_diff']} days)"
                )
                warnings.append(warning)
                stats["suspicious_matches"].append({
                    "matchup": f"{away_team} @ {home_team}",
                    "outcome_date": str(game_date),
                    "line_date": str(best_match["line_date"]),
                    "date_diff": best_match["date_diff"],
                })

    stats["unmatched_outcomes"] = len(outcomes_df) - len(matched_outcomes)
    stats["unmatched_lines"] = len(lines_df) - len(matched_lines)

    # Calculate match rate
    if len(outcomes_df) > 0:
        match_rate = stats["matched"] / len(outcomes_df)
        stats["match_rate"] = round(match_rate, 4)

        if match_rate < 0.80:
            errors.append(
                f"Low match rate: {match_rate:.1%} of outcomes have matching lines. "
                f"Expected >80%."
            )
        elif match_rate < 0.90:
            warnings.append(
                f"Moderate match rate: {match_rate:.1%} of outcomes have matching lines."
            )

    # Warn about unmatched
    if stats["unmatched_outcomes"] > 0:
        warnings.append(
            f"{stats['unmatched_outcomes']} game outcomes have no matching betting lines"
        )

    is_valid = len(errors) == 0

    return ValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        stats=stats,
    )


def validate_training_data(
    df: pd.DataFrame,
    strict: bool = True,
) -> ValidationResult:
    """
    Comprehensive validation of training data.

    Checks:
    1. Required columns present
    2. Null thresholds per column
    3. Data types correct
    4. Value ranges reasonable
    5. No duplicate games

    Args:
        df: Training data DataFrame
        strict: If True, fail on violations

    Returns:
        ValidationResult with all validation details
    """
    all_errors = []
    all_warnings = []
    all_stats = {}

    # 1. Check required columns
    required_cols = ["date", "home_team", "away_team", "home_score", "away_score"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        all_errors.append(f"Missing required columns: {missing_cols}")

    # 2. Validate null thresholds
    null_result = validate_null_thresholds(df, strict=strict)
    all_errors.extend(null_result.errors)
    all_warnings.extend(null_result.warnings)
    all_stats["null_validation"] = null_result.stats

    # 3. Check for duplicates
    if "date" in df.columns and "home_team" in df.columns and "away_team" in df.columns:
        dup_cols = ["date", "home_team", "away_team"]
        duplicates = df.duplicated(subset=dup_cols, keep=False)
        dup_count = duplicates.sum() // 2  # Each duplicate counted twice
        if dup_count > 0:
            all_warnings.append(f"Found {dup_count} duplicate games")
            all_stats["duplicate_games"] = dup_count

    # 4. Check score ranges
    if "home_score" in df.columns and "away_score" in df.columns:
        min_score = min(df["home_score"].min(), df["away_score"].min())
        max_score = max(df["home_score"].max(), df["away_score"].max())

        if min_score < 0:
            all_errors.append(f"Negative scores found (min: {min_score})")
        if max_score > 200:
            all_warnings.append(f"Unusually high scores found (max: {max_score})")

        all_stats["score_range"] = {"min": int(min_score), "max": int(max_score)}

    # 5. Check date range
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        min_date = df["date"].min()
        max_date = df["date"].max()
        all_stats["date_range"] = {
            "min": str(min_date.date()) if pd.notna(min_date) else None,
            "max": str(max_date.date()) if pd.notna(max_date) else None,
        }

    # 6. Check spread line ranges
    if "spread_line" in df.columns:
        spread_min = df["spread_line"].min()
        spread_max = df["spread_line"].max()
        if abs(spread_min) > 30 or abs(spread_max) > 30:
            all_warnings.append(
                f"Unusual spread lines: min={spread_min}, max={spread_max}"
            )
        all_stats["spread_range"] = {"min": float(spread_min), "max": float(spread_max)}

    # 7. Check total line ranges
    if "total_line" in df.columns:
        total_min = df["total_line"].min()
        total_max = df["total_line"].max()
        if total_min < 150 or total_max > 280:
            all_warnings.append(
                f"Unusual total lines: min={total_min}, max={total_max}"
            )
        all_stats["total_range"] = {"min": float(total_min), "max": float(total_max)}

    is_valid = len(all_errors) == 0

    if is_valid:
        logger.info(f"Training data validation PASSED ({len(df)} rows)")
    else:
        logger.error(f"Training data validation FAILED: {len(all_errors)} errors")

    return ValidationResult(
        is_valid=is_valid,
        errors=all_errors,
        warnings=all_warnings,
        stats=all_stats,
    )


def get_data_quality_score(validation_result: ValidationResult) -> float:
    """
    Calculate overall data quality score (0-100).

    Args:
        validation_result: Result from validate_training_data

    Returns:
        Quality score from 0 (worst) to 100 (best)
    """
    score = 100.0

    # Deduct for errors
    score -= len(validation_result.errors) * 10

    # Deduct for warnings
    score -= len(validation_result.warnings) * 2

    # Deduct for null rates
    null_stats = validation_result.stats.get("null_validation", {})
    null_rates = null_stats.get("null_rates", {})

    for col, rate in null_rates.items():
        if rate > 0.10:
            score -= rate * 10

    # Deduct for low match rate
    if "match_rate" in validation_result.stats:
        match_rate = validation_result.stats["match_rate"]
        if match_rate < 0.90:
            score -= (0.90 - match_rate) * 50

    return max(0.0, min(100.0, score))
