"""
Confidence calculation utilities for prediction models.

Since production models use CalibratedClassifierCV (isotonic regression),
the predict_proba output is already calibrated. Confidence = calibrated probability.

No arbitrary caps. No entropy transformations. The calibration ensures probabilities
reflect actual win rates (e.g., 70% predicted = ~70% actual historical win rate).
"""

from typing import Tuple

import numpy as np


def calculate_confidence_from_probabilities(
    prob_a: float,
    prob_b: float,
    min_confidence: float = 0.5,
    max_confidence: float = 1.0,  # No arbitrary cap - calibration handles this
) -> float:
    """
    Return confidence from calibrated binary classification probabilities.

    IMPORTANT: This function assumes probabilities come from a calibrated model
    (e.g., CalibratedClassifierCV with isotonic regression). The calibrated
    probability IS the confidence - no transformation needed.

    Statistical basis:
    - Isotonic regression calibration ensures predicted probabilities match
      actual win rates in the training data
    - A calibrated 70% probability means the model historically won ~70% of
      games when it predicted 70%
    - No arbitrary cap is needed because calibration naturally constrains
      probabilities based on the model's actual performance ceiling

    Args:
        prob_a: Calibrated probability of class A (0-1)
        prob_b: Calibrated probability of class B (0-1), should sum to ~1.0
        min_confidence: Floor for confidence (default 50% = coin flip)
        max_confidence: Ceiling for confidence (default 1.0 = no cap)

    Returns:
        Confidence value = max(prob_a, prob_b), floored at min_confidence
    """
    # Normalize probabilities to sum to 1.0 (defensive)
    total = prob_a + prob_b
    if total < 0.01:  # Avoid division by zero
        return min_confidence

    prob_a_norm = prob_a / total
    prob_b_norm = prob_b / total

    # Confidence = the calibrated probability of the predicted side
    # This is the statistically correct interpretation of calibrated output
    confidence = max(prob_a_norm, prob_b_norm)

    # Floor at min_confidence (50% = no edge over coin flip)
    # Ceiling at max_confidence (default 1.0 = no arbitrary cap)
    return float(np.clip(confidence, min_confidence, max_confidence))


def calculate_confidence_from_binary_probability(
    winning_prob: float,
    min_confidence: float = 0.5,
    max_confidence: float = 1.0,  # No arbitrary cap
) -> float:
    """
    Return confidence from a calibrated probability (binary case).

    Convenience wrapper for binary classification where probabilities sum to 1.0.

    Args:
        winning_prob: Calibrated probability of the winning class (0-1)
        min_confidence: Floor for confidence (default 50%)
        max_confidence: Ceiling for confidence (default 1.0 = no cap)

    Returns:
        Confidence value = winning_prob, clipped to [min_confidence, max_confidence]
    """
    losing_prob = 1.0 - winning_prob
    return calculate_confidence_from_probabilities(
        winning_prob, losing_prob, min_confidence, max_confidence
    )
