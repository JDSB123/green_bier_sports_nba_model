"""
Smart filtering for spread predictions (Full Game + First Half).

Backtest-validated filtering strategies.
Only BACKTESTED markets: FG (60.6% acc) and 1H (55.9% acc).
"""
from typing import Tuple, Optional


class FGSpreadFilter:
    """Smart filtering for full game spread predictions.

    Backtest Results (422 games):
        Without filter: 54.5% accuracy, +4.1% ROI
        With filter: 60.6% accuracy, +15.7% ROI
    """

    def __init__(
        self,
        filter_small_spreads: bool = True,
        small_spread_min: float = 3.0,
        small_spread_max: float = 6.0,
        min_edge_pct: float = 0.05,
    ):
        """
        Initialize full game spread filter.

        Args:
            filter_small_spreads: Remove 3-6 point spreads (low accuracy zone)
            small_spread_min: Minimum spread to filter
            small_spread_max: Maximum spread to filter
            min_edge_pct: Minimum model edge required (default 5%)
        """
        self.filter_small_spreads = filter_small_spreads
        self.small_spread_min = small_spread_min
        self.small_spread_max = small_spread_max
        self.min_edge_pct = min_edge_pct

    def should_bet(
        self,
        spread_line: float,
        confidence: float,
    ) -> Tuple[bool, Optional[str]]:
        """
        Determine if a full game spread bet should be made.

        Args:
            spread_line: Vegas spread line (absolute value)
            confidence: Model confidence (0-1)

        Returns:
            Tuple of (should_bet, filter_reason)
        """
        spread_abs = abs(spread_line)
        model_edge_pct = abs(confidence - 0.5)

        # Filter 1: Small spreads (3-6 points)
        if self.filter_small_spreads:
            if self.small_spread_min <= spread_abs <= self.small_spread_max:
                return False, f"Small spread ({spread_abs:.1f} pts) - low accuracy zone"

        # Filter 2: Minimum edge
        if model_edge_pct < self.min_edge_pct:
            return False, f"Insufficient edge ({model_edge_pct:.1%} < {self.min_edge_pct:.0%})"

        return True, None


class FirstHalfSpreadFilter:
    """Smart filtering for first half spread predictions.

    Backtest Results (303 games):
        High Confidence: 55.9% accuracy, +6.7% ROI
    """

    def __init__(
        self,
        filter_small_spreads: bool = True,
        small_spread_min: float = 1.5,
        small_spread_max: float = 3.0,
        min_edge_pct: float = 0.05,
    ):
        """
        Initialize first half spread filter.

        Args:
            filter_small_spreads: Remove 1.5-3 point spreads (proportional to FG 3-6)
            small_spread_min: Minimum spread to filter
            small_spread_max: Maximum spread to filter
            min_edge_pct: Minimum model edge required (default 5%)
        """
        self.filter_small_spreads = filter_small_spreads
        self.small_spread_min = small_spread_min
        self.small_spread_max = small_spread_max
        self.min_edge_pct = min_edge_pct

    def should_bet(
        self,
        spread_line: float,
        confidence: float,
    ) -> Tuple[bool, Optional[str]]:
        """
        Determine if a first half spread bet should be made.

        Args:
            spread_line: Vegas 1H spread line (absolute value)
            confidence: Model confidence (0-1)

        Returns:
            Tuple of (should_bet, filter_reason)
        """
        spread_abs = abs(spread_line)
        model_edge_pct = abs(confidence - 0.5)

        # Filter 1: Small spreads (1.5-3 points for 1H)
        if self.filter_small_spreads:
            if self.small_spread_min <= spread_abs <= self.small_spread_max:
                return False, f"Small 1H spread ({spread_abs:.1f} pts) - low accuracy zone"

        # Filter 2: Minimum edge
        if model_edge_pct < self.min_edge_pct:
            return False, f"Insufficient edge ({model_edge_pct:.1%} < {self.min_edge_pct:.0%})"

        return True, None


