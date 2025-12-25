"""
Smart filtering for moneyline predictions (Full Game, First Half, First Quarter).

Backtest-validated filtering strategies.
BACKTESTED markets:
  - FG: 68.1% acc, +30.0% ROI
  - 1H: 62.5% acc, +19.3% ROI
  - Q1: 53.0% overall -> HIGH CONF ONLY (58.8% acc at >=60% conf)
"""
from typing import Tuple, Optional


class FGMoneylineFilter:
    """Smart filtering for full game moneyline predictions.

    Backtest Results (232 games, Dec 2025):
        All Bets: 68.1% accuracy, +30.0% ROI
        High Confidence (>=60%): 73.4% accuracy, +40.2% ROI
    """

    def __init__(
        self,
        min_edge_pct: float = 0.05,
        min_confidence: float = 0.55,
    ):
        """
        Initialize full game moneyline filter.

        Args:
            min_edge_pct: Minimum edge over implied probability
            min_confidence: Minimum model confidence required
        """
        self.min_edge_pct = min_edge_pct
        self.min_confidence = min_confidence

    def should_bet(
        self,
        predicted_prob: float,
        implied_prob: float,
    ) -> Tuple[bool, Optional[str]]:
        """
        Determine if a full game moneyline bet should be made.

        Args:
            predicted_prob: Model's predicted win probability
            implied_prob: Implied probability from odds

        Returns:
            Tuple of (should_bet, filter_reason)
        """
        edge = predicted_prob - implied_prob

        # Filter 1: Minimum confidence
        if predicted_prob < self.min_confidence:
            return False, f"Low confidence ({predicted_prob:.1%} < {self.min_confidence:.0%})"

        # Filter 2: Minimum edge over implied
        if edge < self.min_edge_pct:
            return False, f"Insufficient edge ({edge:.1%} < {self.min_edge_pct:.0%})"

        return True, None


class FirstHalfMoneylineFilter:
    """Smart filtering for first half moneyline predictions.

    Backtest Results (232 games, Dec 2025):
        All Bets: 62.5% accuracy, +19.3% ROI
        High Confidence (>=60%): 66.1% accuracy, +26.2% ROI
    """

    def __init__(
        self,
        min_edge_pct: float = 0.05,
        min_confidence: float = 0.55,
    ):
        """
        Initialize first half moneyline filter.

        Args:
            min_edge_pct: Minimum edge over implied probability
            min_confidence: Minimum model confidence required
        """
        self.min_edge_pct = min_edge_pct
        self.min_confidence = min_confidence

    def should_bet(
        self,
        predicted_prob: float,
        implied_prob: float,
    ) -> Tuple[bool, Optional[str]]:
        """
        Determine if a first half moneyline bet should be made.

        Args:
            predicted_prob: Model's predicted win probability
            implied_prob: Implied probability from odds

        Returns:
            Tuple of (should_bet, filter_reason)
        """
        edge = predicted_prob - implied_prob

        # Filter 1: Minimum confidence
        if predicted_prob < self.min_confidence:
            return False, f"Low confidence ({predicted_prob:.1%} < {self.min_confidence:.0%})"

        # Filter 2: Minimum edge over implied
        if edge < self.min_edge_pct:
            return False, f"Insufficient edge ({edge:.1%} < {self.min_edge_pct:.0%})"

        return True, None


class FirstQuarterMoneylineFilter:
    """Smart filtering for first quarter moneyline predictions.

    CRITICAL: Q1 moneyline has marginal overall accuracy (53.0%).
    ONLY USE HIGH CONFIDENCE PICKS (>=60% confidence).

    Backtest Results (232 games, Dec 2025):
        All Bets: 53.0% accuracy, +1.2% ROI (MARGINAL - avoid)
        High Confidence (>=60%): 58.8% accuracy, +12.3% ROI (ACCEPTABLE)

    Default min_confidence is set to 0.60 (stricter than FG/1H).
    """

    def __init__(
        self,
        min_edge_pct: float = 0.05,
        min_confidence: float = 0.60,  # STRICTER: 60% vs 55% for FG/1H
    ):
        """
        Initialize first quarter moneyline filter.

        Args:
            min_edge_pct: Minimum edge over implied probability
            min_confidence: Minimum model confidence required (DEFAULT: 0.60)
        """
        self.min_edge_pct = min_edge_pct
        self.min_confidence = min_confidence

    def should_bet(
        self,
        predicted_prob: float,
        implied_prob: float,
    ) -> Tuple[bool, Optional[str]]:
        """
        Determine if a first quarter moneyline bet should be made.

        IMPORTANT: Q1 requires higher confidence threshold (60%) for profitability.

        Args:
            predicted_prob: Model's predicted win probability
            implied_prob: Implied probability from odds

        Returns:
            Tuple of (should_bet, filter_reason)
        """
        edge = predicted_prob - implied_prob

        # Filter 1: STRICT minimum confidence (60% for Q1)
        if predicted_prob < self.min_confidence:
            return False, f"Q1 requires high confidence ({predicted_prob:.1%} < {self.min_confidence:.0%})"

        # Filter 2: Minimum edge over implied
        if edge < self.min_edge_pct:
            return False, f"Insufficient edge ({edge:.1%} < {self.min_edge_pct:.0%})"

        return True, None
