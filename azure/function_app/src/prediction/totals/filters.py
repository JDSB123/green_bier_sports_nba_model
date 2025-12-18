"""
Smart filtering for totals predictions (Full Game + First Half).

Backtest-validated filtering strategies.
Only BACKTESTED markets: FG (59.2% acc) and 1H (58.1% acc).
"""
from typing import Tuple, Optional


class FGTotalFilter:
    """Smart filtering for full game totals predictions.

    Backtest Results (422 games):
        Baseline (no filter): 59.2% accuracy, +13.1% ROI <- BEST
        With 5% edge filter: 57.0% accuracy, +8.8% ROI <- WORSE

    Recommendation: Don't filter totals (use_filter=False)
    """

    def __init__(
        self,
        use_filter: bool = False,
        min_edge_pct: float = 0.05,
    ):
        """
        Initialize full game totals filter.

        Args:
            use_filter: Whether to apply filtering (baseline is best!)
            min_edge_pct: Minimum model edge required if filtering
        """
        self.use_filter = use_filter
        self.min_edge_pct = min_edge_pct

    def should_bet(
        self,
        confidence: float,
    ) -> Tuple[bool, Optional[str]]:
        """
        Determine if a full game totals bet should be made.

        Args:
            confidence: Model confidence (0-1)

        Returns:
            Tuple of (should_bet, filter_reason)
        """
        # If not using filter, bet everything
        if not self.use_filter:
            return True, None

        # Otherwise apply minimum edge filter
        model_edge_pct = abs(confidence - 0.5)
        if model_edge_pct < self.min_edge_pct:
            return False, f"Insufficient edge ({model_edge_pct:.1%} < {self.min_edge_pct:.0%})"

        return True, None


class FirstHalfTotalFilter:
    """Smart filtering for first half totals predictions.

    Backtest Results (303 games):
        All Bets: 58.1% accuracy, +10.9% ROI

    Like FG totals, baseline (no filter) performs best.
    """

    def __init__(
        self,
        use_filter: bool = False,
        min_edge_pct: float = 0.05,
    ):
        """
        Initialize first half totals filter.

        Args:
            use_filter: Whether to apply filtering (baseline likely best)
            min_edge_pct: Minimum model edge required if filtering
        """
        self.use_filter = use_filter
        self.min_edge_pct = min_edge_pct

    def should_bet(
        self,
        confidence: float,
    ) -> Tuple[bool, Optional[str]]:
        """
        Determine if a first half totals bet should be made.

        Args:
            confidence: Model confidence (0-1)

        Returns:
            Tuple of (should_bet, filter_reason)
        """
        # If not using filter, bet everything (recommended)
        if not self.use_filter:
            return True, None

        # Otherwise apply minimum edge filter
        model_edge_pct = abs(confidence - 0.5)
        if model_edge_pct < self.min_edge_pct:
            return False, f"Insufficient edge ({model_edge_pct:.1%} < {self.min_edge_pct:.0%})"

        return True, None
