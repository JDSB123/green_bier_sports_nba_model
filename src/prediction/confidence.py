"""
Confidence calculation utilities for prediction models.

Uses information-theoretic measures (entropy) to derive confidence
from model probabilities, accounting for uncertainty.
"""
import numpy as np
from typing import Tuple


def calculate_confidence_from_probabilities(
    prob_a: float,
    prob_b: float,
    min_confidence: float = 0.5,
    max_confidence: float = 0.95,
) -> float:
    """
    Calculate confidence from binary classification probabilities using entropy.
    
    Confidence reflects uncertainty in the prediction, not just raw probability.
    Uses information entropy to temper extreme probabilities:
    - High entropy (prob ~0.5) = uncertain = lower confidence
    - Low entropy (prob ~0.0 or 1.0) = certain = higher confidence (capped)
    
    This prevents 100% confidence even when model outputs 1.0 probability,
    accounting for inherent model uncertainty.
    
    Args:
        prob_a: Probability of class A (0-1)
        prob_b: Probability of class B (0-1), should sum to ~1.0 with prob_a
        min_confidence: Minimum confidence level (default 50%)
        max_confidence: Maximum confidence level (default 95%)
    
    Returns:
        Confidence value between min_confidence and max_confidence
    """
    # Normalize probabilities to sum to 1.0
    total = prob_a + prob_b
    if total < 0.01:  # Avoid division by zero
        return min_confidence
    
    prob_a_norm = prob_a / total
    prob_b_norm = prob_b / total
    
    # Clip to avoid log(0)
    eps = 1e-10
    prob_a_clipped = np.clip(prob_a_norm, eps, 1.0 - eps)
    prob_b_clipped = np.clip(prob_b_norm, eps, 1.0 - eps)
    
    # Calculate entropy: H = -Σ p * log2(p)
    # Range: 0.0 (certain: prob=0 or 1) to 1.0 (uncertain: prob=0.5)
    entropy = -(prob_a_clipped * np.log2(prob_a_clipped) + 
                prob_b_clipped * np.log2(prob_b_clipped))
    
    # Convert entropy to certainty: 0 entropy -> 1.0 certainty, 1.0 entropy -> 0.0 certainty
    certainty = 1.0 - entropy
    
    # Get winning probability (the one we're betting on)
    winning_prob = max(prob_a_norm, prob_b_norm)
    
    # Calculate confidence: combines probability strength with uncertainty
    # When entropy is low (certain) and probability is high -> high confidence
    # When entropy is high (uncertain) -> lower confidence regardless of probability
    
    # Map winning probability from [0.5, 1.0] to [0.0, 1.0]
    prob_strength = (winning_prob - 0.5) * 2.0  # 0.5 -> 0.0, 1.0 -> 1.0
    
    # Confidence = certainty weighted by probability strength
    # This means: even with 100% probability, if entropy is 0, we get high confidence
    # But the max_confidence cap prevents going above 95%
    raw_confidence = 0.5 + (certainty * prob_strength) * 0.5
    
    # Scale from [0.5, 1.0] to [min_confidence, max_confidence]
    # Map 0.5 -> min_confidence, 1.0 -> max_confidence
    confidence = min_confidence + (raw_confidence - 0.5) * 2.0 * (max_confidence - min_confidence)
    
    # Ensure we're within bounds
    return float(np.clip(confidence, min_confidence, max_confidence))


def calculate_confidence_from_binary_probability(
    winning_prob: float,
    min_confidence: float = 0.5,
    max_confidence: float = 0.95,
) -> float:
    """
    Calculate confidence from a single probability (when other class is 1 - prob).
    
    Convenience wrapper for binary classification where probabilities sum to 1.0.
    
    Args:
        winning_prob: Probability of the winning class (0-1)
        min_confidence: Minimum confidence level (default 50%)
        max_confidence: Maximum confidence level (default 95%)
    
    Returns:
        Confidence value between min_confidence and max_confidence
    """
    losing_prob = 1.0 - winning_prob
    return calculate_confidence_from_probabilities(
        winning_prob, losing_prob, min_confidence, max_confidence
    )
