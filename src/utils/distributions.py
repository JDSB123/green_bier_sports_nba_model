from __future__ import annotations

import math
from typing import Dict, Optional

from src.modeling.period_features import PERIOD_SCALING


def _normal_cdf(x: float, mean: float, std: float) -> float:
    if std <= 0:
        return 0.5
    z = (x - mean) / (std * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def _pace_factor(features: Dict[str, float]) -> float:
    pace = features.get("expected_pace_factor")
    if pace is None:
        home = features.get("home_pace_factor")
        away = features.get("away_pace_factor")
        if home is not None and away is not None:
            pace = (home + away) / 2.0
    if pace is None:
        pace = 1.0
    return max(0.7, min(1.3, float(pace)))


def _mean(values: list[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def estimate_spread_std(features: Dict[str, float], period: str) -> float:
    base = 12.0
    if period != "fg":
        base = base * math.sqrt(PERIOD_SCALING["1h"]["scoring_pct"])

    suffix = "_1h" if period == "1h" else ""
    std = _mean([
        features.get(f"home_margin_std{suffix}"),
        features.get(f"away_margin_std{suffix}"),
    ])
    if std is None:
        std = base

    scale = math.sqrt(_pace_factor(features))
    return max(4.0, float(std) * scale)


def estimate_total_std(features: Dict[str, float], period: str) -> float:
    base = 15.0
    if period != "fg":
        base = base * math.sqrt(PERIOD_SCALING["1h"]["scoring_pct"])

    std = _mean([
        features.get("home_score_std"),
        features.get("away_score_std"),
    ])
    if std is None:
        std = base

    scale = math.sqrt(_pace_factor(features))
    return max(6.0, float(std) * scale)


def cover_probability(mean_margin: float, spread_line: float, std: float) -> float:
    threshold = -spread_line
    return 1.0 - _normal_cdf(threshold, mean_margin, std)


def over_probability(mean_total: float, total_line: float, std: float) -> float:
    return 1.0 - _normal_cdf(total_line, mean_total, std)
