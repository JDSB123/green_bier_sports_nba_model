"""Prediction signal resolution helpers.

This module defines the contract for reconciling multiple signals into a single,
consistent pick output.

Design goals:
- `bet_side` and `confidence` must ALWAYS refer to the same side.
- Never block a pick solely due to classifier/point-prediction disagreement.

This prevents the class of bugs where the API says "OVER" but the confidence
value corresponds to the UNDER probability (or vice-versa).

FIXED v33.1.0: Added resolve_spread_two_signal for spreads using same logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


TotalSide = Literal["over", "under"]
SpreadSide = Literal["home", "away"]


@dataclass(frozen=True)
class ResolvedTwoWay:
    bet_side: TotalSide
    confidence: float
    classifier_confidence: float
    signals_agree: bool
    passes_filter: bool
    filter_reason: Optional[str]


def _clip01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def resolve_total_two_signal(
    *,
    over_prob: float,
    under_prob: float,
    classifier_side: TotalSide,
    prediction_side: TotalSide,
    edge_abs: float,
    min_confidence: float,
    min_edge: float,
    classifier_extreme: bool,
) -> ResolvedTwoWay:
    """Resolve total pick from classifier + point prediction.

    Contract:
    - bet_side is always the point-prediction side (prediction_side)
    - confidence is the probability of bet_side (NOT max(probabilities))
    - never reject solely due to classifier disagreement
    """
    if classifier_side not in ("over", "under"):
        raise ValueError(f"Invalid classifier_side: {classifier_side}")
    if prediction_side not in ("over", "under"):
        raise ValueError(f"Invalid prediction_side: {prediction_side}")

    over_prob = _clip01(over_prob)
    under_prob = _clip01(under_prob)

    bet_side: TotalSide = prediction_side
    signals_agree = classifier_side == prediction_side

    classifier_confidence = max(over_prob, under_prob)
    confidence = over_prob if bet_side == "over" else under_prob

    filter_reason: Optional[str] = None

    passes_filter = confidence >= min_confidence and edge_abs >= min_edge
    if not passes_filter:
        if confidence < min_confidence:
            filter_reason = f"Low confidence: {confidence:.1%}"
        else:
            filter_reason = f"Low edge: {edge_abs:.1f} pts (min: {min_edge})"

    return ResolvedTwoWay(
        bet_side=bet_side,
        confidence=confidence,
        classifier_confidence=classifier_confidence,
        signals_agree=signals_agree,
        passes_filter=passes_filter,
        filter_reason=filter_reason,
    )


def resolve_spread_two_signal(
    *,
    home_cover_prob: float,
    away_cover_prob: float,
    classifier_side: SpreadSide,
    prediction_side: SpreadSide,
    edge_abs: float,
    min_confidence: float,
    min_edge: float,
    classifier_extreme: bool,
) -> ResolvedTwoWay:
    """Resolve spread pick from classifier + point prediction.

    Contract (IDENTICAL to totals):
    - bet_side is always the point-prediction side (prediction_side)
    - confidence is the probability of bet_side (NOT max(probabilities))
    - never reject solely due to classifier disagreement

    Args:
        home_cover_prob: Classifier probability home covers
        away_cover_prob: Classifier probability away covers
        classifier_side: What the ML classifier says ("home" or "away")
        prediction_side: What the point prediction says ("home" or "away")
        edge_abs: Absolute value of edge in points
        min_confidence: Minimum confidence threshold
        min_edge: Minimum edge threshold
        classifier_extreme: Unused for filtering (kept for diagnostics)

    Returns:
        ResolvedTwoWay with bet_side and confidence referring to the same side
    """
    if classifier_side not in ("home", "away"):
        raise ValueError(f"Invalid classifier_side: {classifier_side}")
    if prediction_side not in ("home", "away"):
        raise ValueError(f"Invalid prediction_side: {prediction_side}")

    home_cover_prob = _clip01(home_cover_prob)
    away_cover_prob = _clip01(away_cover_prob)

    # BET SIDE: Always use edge-based prediction (more robust than classifier)
    bet_side: SpreadSide = prediction_side
    signals_agree = classifier_side == prediction_side

    # CONFIDENCE: Probability of the bet_side (edge-based)
    # This ensures bet_side and confidence always refer to the same side
    classifier_confidence = max(home_cover_prob, away_cover_prob)
    confidence = home_cover_prob if bet_side == "home" else away_cover_prob

    filter_reason: Optional[str] = None

    passes_filter = confidence >= min_confidence and edge_abs >= min_edge
    if not passes_filter:
        if confidence < min_confidence:
            filter_reason = f"Low confidence: {confidence:.1%}"
        else:
            filter_reason = f"Low edge: {edge_abs:.1f} pts (min: {min_edge})"

    return ResolvedTwoWay(
        bet_side=bet_side,
        confidence=confidence,
        classifier_confidence=classifier_confidence,
        signals_agree=signals_agree,
        passes_filter=passes_filter,
        filter_reason=filter_reason,
    )
