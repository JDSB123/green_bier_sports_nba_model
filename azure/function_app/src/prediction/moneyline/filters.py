"""
Smart filtering for moneyline predictions (Full Game + First Half).

Backtest-validated filtering strategies.
Only BACKTESTED markets: FG (65.5% acc) and 1H (63.0% acc).
"""
from typing import Tuple, Optional


class FGMoneylineFilter:
    """Smart filtering for full game moneyline predictions.

    Backtest Results (316 games):
        All Bets: 65.5% accuracy, +25.1% ROI
        High Confidence (>=60%): 67.5% accuracy, +28.9% ROI
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

    Backtest Results (303 games):
        All Bets: 63.0% accuracy, +20.3% ROI
        High Confidence (>=60%): 64.5% accuracy, +23.1% ROI
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
