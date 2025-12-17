"""
Smart filtering logic for spreads and totals (full game + first half).

Based on backtest validation:
- FG Spreads: Avoid 3-6 point spreads (42% accuracy), require 5% edge
- FG Totals: Baseline is best (59.2%/+13.1%), light filtering optional
- 1H Markets: Use proportional filtering (typically ~50% of FG lines)
"""
from typing import Dict, Tuple, Optional
import pandas as pd


class SpreadFilter:
    """Smart filtering for spread predictions (backtest validated)."""

    def __init__(
        self,
        filter_small_spreads: bool = True,
        small_spread_min: float = 3.0,
        small_spread_max: float = 6.0,
        min_edge_pct: float = 0.05,
    ):
        """
        Initialize spread filter.

        Args:
            filter_small_spreads: Remove 3-6 point spreads (low accuracy zone)
            small_spread_min: Minimum spread to filter
            small_spread_max: Maximum spread to filter
            min_edge_pct: Minimum model edge required (default 5%)

        Backtest Results (422 games):
            Without filter: 54.5% accuracy, +4.1% ROI
            With filter: 60.6% accuracy, +15.7% ROI
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
        Determine if a spread bet should be made.

        Args:
            spread_line: Vegas spread line (absolute value)
            confidence: Model confidence (0-1)

        Returns:
            Tuple of (should_bet, filter_reason)
            - should_bet: True if bet passes filters
            - filter_reason: Reason if filtered out, None if passes
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


class TotalFilter:
    """Smart filtering for totals predictions."""

    def __init__(
        self,
        use_filter: bool = False,
        min_edge_pct: float = 0.05,
    ):
        """
        Initialize totals filter.

        Args:
            use_filter: Whether to apply filtering (baseline is best!)
            min_edge_pct: Minimum model edge required if filtering

        Backtest Results (422 games):
            Baseline (no filter): 59.2% accuracy, +13.1% ROI ← BEST
            With 5% edge filter: 57.0% accuracy, +8.8% ROI ← WORSE

        Recommendation: Don't filter totals (use_filter=False)
        """
        self.use_filter = use_filter
        self.min_edge_pct = min_edge_pct

    def should_bet(
        self,
        confidence: float,
    ) -> Tuple[bool, Optional[str]]:
        """
        Determine if a totals bet should be made.

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


class FirstHalfSpreadFilter:
    """Smart filtering for first half spread predictions.

    First half spreads are typically ~50% of full game spreads.
    We proportionally adjust filtering thresholds.
    """

    def __init__(
        self,
        filter_small_spreads: bool = True,
        small_spread_min: float = 1.5,  # ~50% of FG 3.0
        small_spread_max: float = 3.0,  # ~50% of FG 6.0
        min_edge_pct: float = 0.05,
    ):
        """
        Initialize first half spread filter.

        Args:
            filter_small_spreads: Remove 1.5-3 point spreads (proportional to FG 3-6)
            small_spread_min: Minimum spread to filter
            small_spread_max: Maximum spread to filter
            min_edge_pct: Minimum model edge required (default 5%)

        Note:
            Using FG model for 1H predictions until dedicated 1H model trained.
            Filtering thresholds proportionally adjusted (~50% of FG thresholds).
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


class FirstHalfTotalFilter:
    """Smart filtering for first half totals predictions.

    First half totals are typically ~50% of full game totals.
    Like FG totals, baseline (no filter) likely performs best.
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

        Note:
            Using FG totals model for 1H predictions.
            Like FG totals, baseline (no filtering) likely performs best.
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


def filter_predictions(
    predictions_df: pd.DataFrame,
    spread_filter: Optional[SpreadFilter] = None,
    total_filter: Optional[TotalFilter] = None,
) -> pd.DataFrame:
    """
    Apply smart filters to predictions DataFrame.

    Args:
        predictions_df: DataFrame with predictions
        spread_filter: SpreadFilter instance (None = use defaults)
        total_filter: TotalFilter instance (None = use defaults)

    Returns:
        DataFrame with added columns:
            - spread_passes_filter: bool
            - spread_filter_reason: str
            - total_passes_filter: bool
            - total_filter_reason: str
    """
    df = predictions_df.copy()

    # Initialize filters with defaults if not provided
    if spread_filter is None:
        spread_filter = SpreadFilter()
    if total_filter is None:
        total_filter = TotalFilter()

    # Apply spread filter
    spread_results = df.apply(
        lambda row: spread_filter.should_bet(
            spread_line=row.get("spread_line", 0),
            confidence=row.get("spread_confidence", 0.5),
        ),
        axis=1,
    )
    df["spread_passes_filter"] = [r[0] for r in spread_results]
    df["spread_filter_reason"] = [r[1] if r[1] else "" for r in spread_results]

    # Apply totals filter
    total_results = df.apply(
        lambda row: total_filter.should_bet(
            confidence=row.get("total_confidence", 0.5),
        ),
        axis=1,
    )
    df["total_passes_filter"] = [r[0] for r in total_results]
    df["total_filter_reason"] = [r[1] if r[1] else "" for r in total_results]

    return df
