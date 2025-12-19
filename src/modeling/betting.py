"""
Betting utilities for converting model probabilities and odds into
expected value and staking recommendations.

This module is intentionally lightweight so it can be reused by
backtests and live prediction scripts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


def american_to_decimal(odds: float) -> float:
    """Convert American odds to decimal odds.

    Examples:
        -110 -> 1.909...
        +120 -> 2.20
    """
    if odds == 0:
        return 1.0
    if odds < 0:
        return 1.0 + 100.0 / abs(odds)
    return 1.0 + odds / 100.0


def implied_prob_from_american(odds: float) -> float:
    """Convert American odds to implied probability (ignoring vig)."""
    if odds == 0:
        return 0.5
    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    return 100.0 / (odds + 100.0)


def kelly_fraction(prob: float, decimal_odds: float, max_fraction: float = 0.05) -> float:
    """Compute Kelly fraction for a single bet, clipped to max_fraction.

    Args:
        prob: Model win probability for the bet side (0-1).
        decimal_odds: Decimal odds (e.g., 1.91 for -110).
        max_fraction: Upper cap on recommended stake as a fraction of bankroll.
    """
    prob = float(prob)
    if prob <= 0.0 or prob >= 1.0 or decimal_odds <= 1.0:
        return 0.0

    b = decimal_odds - 1.0
    f = (prob * b - (1.0 - prob)) / b
    if f <= 0:
        return 0.0
    return float(min(f, max_fraction))


@dataclass
class BetRecommendation:
    game: str
    bet_type: str
    side: str
    model_prob: float
    implied_prob: float
    edge: float
    decimal_odds: float
    kelly_fraction: float


def annotate_value_bets(
    value_bets_df: pd.DataFrame,
    prob_col: str = "model_prob",
    implied_col: str = "implied_prob",
    odds_col: Optional[str] = None,
    default_american_odds: float = -110.0,
    max_fraction: float = 0.05,
) -> pd.DataFrame:
    """Add EV and Kelly-based stake recommendations to a value-bets DataFrame.

    The input DataFrame is expected to resemble the output of
    `find_value_bets` in `src.modeling.models`.
    """
    df = value_bets_df.copy()

    # Determine decimal odds per row
    if odds_col and odds_col in df.columns:
        dec_odds = df[odds_col].apply(
            lambda x: american_to_decimal(x) if pd.notna(x) and x != 0 else american_to_decimal(default_american_odds)
        )
    else:
        dec_odds = pd.Series(
            american_to_decimal(default_american_odds),
            index=df.index,
            dtype="float64",
        )

    df["decimal_odds"] = dec_odds

    # Expected value edge (model_prob - implied_prob)
    df["edge"] = df[prob_col] - df[implied_col]

    # Kelly fraction for sizing
    df["kelly_fraction"] = [
        kelly_fraction(p, o, max_fraction=max_fraction)
        for p, o in zip(df[prob_col].values, df["decimal_odds"].values)
    ]

    return df












