"""Prediction signal resolution helpers.

This module defines the contract for reconciling multiple signals into a single,
consistent pick output.

Design goals:
- `bet_side` and `confidence` must ALWAYS refer to the same side.
- If classifier and point-prediction signals disagree (and classifier is not
  obviously broken/extreme), do not emit a bet.
- If classifier is extreme/unreliable, allow an edge-only fallback.

This prevents the class of bugs where the API says "OVER" but the confidence
value corresponds to the UNDER probability (or vice-versa).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


TotalSide = Literal["over", "under"]


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
    - if signals conflict and classifier isn't extreme: reject pick
    - if classifier is extreme: allow edge-only filter
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

    if not signals_agree and not classifier_extreme:
        passes_filter = False
        filter_reason = f"Signal conflict: classifier={classifier_side}, prediction={prediction_side}"
    elif classifier_extreme:
        passes_filter = edge_abs >= min_edge
        if not passes_filter:
            filter_reason = (
                f"Classifier unreliable (extreme prob: over={over_prob:.1%}), low edge"
            )
    else:
        passes_filter = confidence >= min_confidence and edge_abs >= min_edge
        if not passes_filter:
            if confidence < min_confidence:
                filter_reason = f"Low confidence: {confidence:.1%}"
            else:
                filter_reason = f"Low edge: {edge_abs:.1f} pts (min: {min_edge})"

    # Safety invariant: we never allow conflicting signals through unless extreme.
    if passes_filter and (not signals_agree) and (not classifier_extreme):
        raise AssertionError("Invariant violated: conflicting signals passed filter")

    return ResolvedTwoWay(
        bet_side=bet_side,
        confidence=confidence,
        classifier_confidence=classifier_confidence,
        signals_agree=signals_agree,
        passes_filter=passes_filter,
        filter_reason=filter_reason,
    )
