from __future__ import annotations

from typing import Optional, Tuple


COMMON_JUICES = (-105, -110, -115)


def american_to_implied_prob(odds: Optional[int]) -> Optional[float]:
    """Convert American odds to implied probability."""
    if odds is None:
        return None
    try:
        odds_int = int(odds)
    except (TypeError, ValueError):
        return None
    if odds_int == 0:
        return None
    if odds_int > 0:
        return 100 / (odds_int + 100)
    return abs(odds_int) / (abs(odds_int) + 100)


def devig_two_way(odds_a: Optional[int], odds_b: Optional[int]) -> Tuple[Optional[float], Optional[float]]:
    """Remove vig for a two-way market using proportional normalization."""
    p_a = american_to_implied_prob(odds_a)
    p_b = american_to_implied_prob(odds_b)
    if p_a is None or p_b is None:
        return None, None
    total = p_a + p_b
    if total <= 0:
        return None, None
    return p_a / total, p_b / total


def devig_three_way(
    odds_a: Optional[int],
    odds_b: Optional[int],
    odds_c: Optional[int],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Remove vig for a three-way market using proportional normalization."""
    p_a = american_to_implied_prob(odds_a)
    p_b = american_to_implied_prob(odds_b)
    p_c = american_to_implied_prob(odds_c)
    if p_a is None or p_b is None or p_c is None:
        return None, None, None
    total = p_a + p_b + p_c
    if total <= 0:
        return None, None, None
    return p_a / total, p_b / total, p_c / total


def expected_value(p_model: Optional[float], odds: Optional[int], stake: float = 1.0) -> Optional[float]:
    """Return expected value in stake units (stake=1.0 -> EV% = EV * 100)."""
    if p_model is None or odds is None:
        return None
    if stake <= 0:
        return None
    try:
        odds_int = int(odds)
    except (TypeError, ValueError):
        return None
    if odds_int == 0:
        return None
    win_profit = stake * (odds_int / 100.0) if odds_int > 0 else stake * (100.0 / abs(odds_int))
    return (p_model * win_profit) - ((1 - p_model) * stake)


def kelly_fraction(p_model: Optional[float], odds: Optional[int], fraction: float = 0.5) -> Optional[float]:
    """Return fractional Kelly fraction for a given probability and American odds."""
    if p_model is None or odds is None:
        return None
    try:
        odds_int = int(odds)
    except (TypeError, ValueError):
        return None
    if odds_int == 0:
        return None
    b = (odds_int / 100.0) if odds_int > 0 else (100.0 / abs(odds_int))
    f_star = (p_model * (b + 1) - 1) / b
    if f_star <= 0:
        return 0.0
    return f_star * fraction


def break_even_probability(odds: Optional[int]) -> Optional[float]:
    """Return the break-even probability implied by American odds."""
    return american_to_implied_prob(odds)


def break_even_percent(odds: Optional[int]) -> Optional[float]:
    """Return the break-even percentage implied by American odds."""
    prob = break_even_probability(odds)
    return None if prob is None else prob * 100.0


def break_even_common_juices() -> dict[int, float]:
    """Return break-even percentages for common two-way juices."""
    return {odds: round(break_even_percent(odds) or 0.0, 2) for odds in COMMON_JUICES}
