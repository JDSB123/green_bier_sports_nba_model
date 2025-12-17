"""
Smart filtering for moneyline predictions (Full Game + First Half).

Value-based filtering using implied probability from odds.
"""
from typing import Tuple, Optional


class FGMoneylineFilter:
    """
    Smart filtering for full game moneyline predictions.

    Filters based on value (predicted prob vs implied odds prob).
    """

    def __init__(
        self,
        use_filter: bool = True,
        min_edge_pct: float = 0.05,
    ):
        """
        Initialize full game moneyline filter.

        Args:
            use_filter: Whether to apply filtering
            min_edge_pct: Minimum edge required (predicted prob - implied prob)
        """
        self.use_filter = use_filter
        self.min_edge_pct = min_edge_pct

    def should_bet(
        self,
        predicted_prob: float,
        implied_prob: Optional[float] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Determine if a full game moneyline bet should be made.

        Args:
            predicted_prob: Model predicted win probability (0-1)
            implied_prob: Implied probability from moneyline odds (0-1)

        Returns:
            Tuple of (should_bet, filter_reason)
        """
        # If not using filter, bet everything
        if not self.use_filter:
            return True, None

        # If no implied prob provided, can't calculate edge
        if implied_prob is None:
            return True, None

        # Calculate value edge
        edge = predicted_prob - implied_prob

        if edge < self.min_edge_pct:
            return False, f"Insufficient value edge ({edge:.1%} < {self.min_edge_pct:.0%})"

        return True, None


class FirstHalfMoneylineFilter:
    """
    Smart filtering for first half moneyline predictions.

    Like FG moneyline, filters based on value.
    """

    def __init__(
        self,
        use_filter: bool = True,
        min_edge_pct: float = 0.05,
    ):
        """
        Initialize first half moneyline filter.

        Args:
            use_filter: Whether to apply filtering
            min_edge_pct: Minimum edge required
        """
        self.use_filter = use_filter
        self.min_edge_pct = min_edge_pct

    def should_bet(
        self,
        predicted_prob: float,
        implied_prob: Optional[float] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Determine if a first half moneyline bet should be made.

        Args:
            predicted_prob: Model predicted win probability (0-1)
            implied_prob: Implied probability from moneyline odds (0-1)

        Returns:
            Tuple of (should_bet, filter_reason)
        """
        # If not using filter, bet everything
        if not self.use_filter:
            return True, None

        # If no implied prob provided, can't calculate edge
        if implied_prob is None:
            return True, None

        # Calculate value edge
        edge = predicted_prob - implied_prob

        if edge < self.min_edge_pct:
            return False, f"Insufficient value edge ({edge:.1%} < {self.min_edge_pct:.0%})"

        return True, None
