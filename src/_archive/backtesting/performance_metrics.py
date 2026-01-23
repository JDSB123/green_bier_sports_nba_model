"""
Performance metrics for backtesting.

Enterprise-level metrics including ROI, Sharpe, drawdown, Kelly criterion.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MarketPerformance:
    """Comprehensive performance metrics for a single market."""
    
    # Basic metrics
    market: str
    n_bets: int
    accuracy: float
    roi: float  # Return on investment (flat betting)
    total_profit: float  # Total units won/lost
    
    # Confidence intervals (95%)
    accuracy_ci: Tuple[float, float] = (0.0, 0.0)
    roi_ci: Tuple[float, float] = (0.0, 0.0)
    
    # Hypothesis testing
    vs_random_pvalue: float = 1.0  # H0: accuracy = 50%
    vs_vig_pvalue: float = 1.0  # H0: ROI = -4.55% (break-even at -110)
    is_significant: bool = False  # vs random at p < 0.05
    is_profitable: bool = False  # vs vig at p < 0.05
    
    # Risk-adjusted metrics
    sharpe_ratio: float = 0.0  # (mean_return - rf) / std_return
    sortino_ratio: float = 0.0  # Downside deviation only
    calmar_ratio: float = 0.0  # Return / max drawdown
    
    # Drawdown analysis
    max_drawdown: float = 0.0  # Maximum peak-to-trough (units)
    max_drawdown_pct: float = 0.0  # As percentage of peak
    max_drawdown_duration: int = 0  # Games
    avg_drawdown: float = 0.0
    
    # Kelly criterion
    kelly_fraction: float = 0.0  # Optimal bet fraction
    kelly_growth_rate: float = 0.0  # Expected log-growth
    half_kelly_fraction: float = 0.0  # Conservative approach
    
    # Streak analysis
    longest_win_streak: int = 0
    longest_lose_streak: int = 0
    avg_win_streak: float = 0.0
    avg_lose_streak: float = 0.0
    
    # By confidence tier
    high_conf_accuracy: float = 0.0  # >= 60% confidence
    high_conf_roi: float = 0.0
    high_conf_n: int = 0
    
    # Time analysis
    first_half_roi: float = 0.0  # First half of backtest period
    second_half_roi: float = 0.0  # Second half (for consistency check)


class PerformanceMetrics:
    """Calculate comprehensive performance metrics from backtest results."""
    
    # Assumed odds for -110
    VIG_BREAK_EVEN = -0.0455  # ROI needed to break even at -110
    WIN_PAYOUT = 100.0 / 110.0  # ~0.909
    
    def __init__(
        self,
        returns: List[float],
        predictions: Optional[List[int]] = None,
        actuals: Optional[List[int]] = None,
        confidences: Optional[List[float]] = None,
        risk_free_rate: float = 0.0,
    ):
        """
        Initialize with bet returns.
        
        Args:
            returns: List of returns per bet (e.g., 0.909 for win, -1.0 for loss)
            predictions: Predicted outcomes (1 = home/over)
            actuals: Actual outcomes
            confidences: Model confidence per prediction
            risk_free_rate: Risk-free rate for Sharpe calculation
        """
        self.returns = np.array(returns)
        self.predictions = np.array(predictions) if predictions else None
        self.actuals = np.array(actuals) if actuals else None
        self.confidences = np.array(confidences) if confidences else None
        self.risk_free_rate = risk_free_rate
        
        self.n_bets = len(returns)
        self.wins = np.array([1 if r > 0 else 0 for r in returns])
    
    def calculate_all(self, market: str = "unknown") -> MarketPerformance:
        """Calculate all performance metrics."""
        if self.n_bets == 0:
            return MarketPerformance(
                market=market,
                n_bets=0,
                accuracy=0.0,
                roi=0.0,
                total_profit=0.0,
            )
        
        # Basic metrics
        accuracy = self.wins.mean()
        total_profit = self.returns.sum()
        roi = total_profit / self.n_bets
        
        # Confidence intervals
        accuracy_ci = self._bootstrap_ci(self.wins)
        roi_ci = self._bootstrap_ci(self.returns)
        
        # Hypothesis tests
        vs_random_pvalue = self._test_vs_random()
        vs_vig_pvalue = self._test_vs_vig()
        
        # Risk-adjusted metrics
        sharpe = self._sharpe_ratio()
        sortino = self._sortino_ratio()
        
        # Drawdown analysis
        dd = self._drawdown_analysis()
        
        # Kelly criterion
        kelly = self._kelly_criterion()
        
        # Streak analysis
        streaks = self._streak_analysis()
        
        # Confidence tiers
        high_conf = self._high_confidence_metrics()
        
        # Time analysis
        time_split = self._time_split_analysis()
        
        calmar = abs(roi / dd["max_drawdown_pct"]) if dd["max_drawdown_pct"] != 0 else 0
        
        return MarketPerformance(
            market=market,
            n_bets=self.n_bets,
            accuracy=accuracy,
            roi=roi,
            total_profit=total_profit,
            accuracy_ci=accuracy_ci,
            roi_ci=roi_ci,
            vs_random_pvalue=vs_random_pvalue,
            vs_vig_pvalue=vs_vig_pvalue,
            is_significant=vs_random_pvalue < 0.05,
            is_profitable=vs_vig_pvalue < 0.05 and roi > self.VIG_BREAK_EVEN,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=dd["max_drawdown"],
            max_drawdown_pct=dd["max_drawdown_pct"],
            max_drawdown_duration=dd["max_drawdown_duration"],
            avg_drawdown=dd["avg_drawdown"],
            kelly_fraction=kelly["kelly_fraction"],
            kelly_growth_rate=kelly["growth_rate"],
            half_kelly_fraction=kelly["kelly_fraction"] / 2,
            longest_win_streak=streaks["longest_win"],
            longest_lose_streak=streaks["longest_lose"],
            avg_win_streak=streaks["avg_win"],
            avg_lose_streak=streaks["avg_lose"],
            high_conf_accuracy=high_conf["accuracy"],
            high_conf_roi=high_conf["roi"],
            high_conf_n=high_conf["n"],
            first_half_roi=time_split["first_half"],
            second_half_roi=time_split["second_half"],
        )
    
    def _bootstrap_ci(
        self,
        data: np.ndarray,
        n_bootstrap: int = 10000,
        ci: float = 0.95,
    ) -> Tuple[float, float]:
        """Non-parametric bootstrap confidence interval."""
        if len(data) == 0:
            return (0.0, 0.0)
        
        bootstrapped = np.array([
            np.random.choice(data, size=len(data), replace=True).mean()
            for _ in range(n_bootstrap)
        ])
        
        alpha = (1 - ci) / 2
        lower = float(np.percentile(bootstrapped, alpha * 100))
        upper = float(np.percentile(bootstrapped, (1 - alpha) * 100))
        
        return (lower, upper)
    
    def _test_vs_random(self) -> float:
        """
        One-sample t-test: H0 accuracy = 50%.
        
        Returns p-value.
        """
        from scipy import stats
        
        if self.n_bets < 30:
            return 1.0
        
        # Test if win rate is significantly different from 50%
        result = stats.ttest_1samp(self.wins, 0.5)
        return float(result.pvalue)
    
    def _test_vs_vig(self) -> float:
        """
        One-sample t-test: H0 ROI = -4.55% (break-even at -110).
        
        Returns p-value.
        """
        from scipy import stats
        
        if self.n_bets < 30:
            return 1.0
        
        # Test if ROI is significantly greater than break-even
        result = stats.ttest_1samp(self.returns, self.VIG_BREAK_EVEN)
        
        # One-sided test (we only care if ROI > break-even)
        if self.returns.mean() > self.VIG_BREAK_EVEN:
            return float(result.pvalue / 2)
        return 1.0
    
    def _sharpe_ratio(self) -> float:
        """
        Calculate Sharpe ratio.
        
        Sharpe = (mean_return - risk_free) / std_return
        """
        if self.n_bets < 10 or self.returns.std() == 0:
            return 0.0
        
        excess_return = self.returns.mean() - self.risk_free_rate
        return float(excess_return / self.returns.std())
    
    def _sortino_ratio(self) -> float:
        """
        Calculate Sortino ratio (downside deviation only).
        
        More appropriate for betting where we care about losses.
        """
        if self.n_bets < 10:
            return 0.0
        
        # Only consider negative returns for downside deviation
        negative_returns = self.returns[self.returns < 0]
        
        if len(negative_returns) == 0 or negative_returns.std() == 0:
            return float("inf") if self.returns.mean() > 0 else 0.0
        
        downside_std = negative_returns.std()
        excess_return = self.returns.mean() - self.risk_free_rate
        
        return float(excess_return / downside_std)
    
    def _drawdown_analysis(self) -> dict:
        """Analyze drawdowns in cumulative profit."""
        if self.n_bets == 0:
            return {
                "max_drawdown": 0.0,
                "max_drawdown_pct": 0.0,
                "max_drawdown_duration": 0,
                "avg_drawdown": 0.0,
            }
        
        # Cumulative profit
        cumulative = np.cumsum(self.returns)
        
        # Running maximum
        running_max = np.maximum.accumulate(cumulative)
        
        # Drawdown at each point
        drawdowns = running_max - cumulative
        
        # Max drawdown
        max_dd = float(drawdowns.max())
        
        # Max drawdown as percentage of peak
        peak_at_max_dd = running_max[drawdowns.argmax()]
        max_dd_pct = max_dd / peak_at_max_dd if peak_at_max_dd > 0 else 0.0
        
        # Max drawdown duration
        in_drawdown = drawdowns > 0
        duration = 0
        max_duration = 0
        for is_dd in in_drawdown:
            if is_dd:
                duration += 1
                max_duration = max(max_duration, duration)
            else:
                duration = 0
        
        # Average drawdown
        avg_dd = float(drawdowns.mean())
        
        return {
            "max_drawdown": max_dd,
            "max_drawdown_pct": float(max_dd_pct),
            "max_drawdown_duration": max_duration,
            "avg_drawdown": avg_dd,
        }
    
    def _kelly_criterion(self) -> dict:
        """
        Calculate Kelly criterion for optimal bet sizing.
        
        Kelly fraction = (bp - q) / b
        where:
            b = odds (decimal - 1) = payout ratio
            p = probability of winning
            q = probability of losing = 1 - p
        """
        if self.n_bets == 0:
            return {"kelly_fraction": 0.0, "growth_rate": 0.0}
        
        p = self.wins.mean()  # Empirical win probability
        q = 1 - p
        b = self.WIN_PAYOUT  # Payout for winning
        
        # Kelly fraction
        kelly = (b * p - q) / b
        
        # Expected log growth rate
        if kelly > 0:
            growth = p * np.log(1 + kelly * b) + q * np.log(1 - kelly)
        else:
            growth = 0.0
        
        return {
            "kelly_fraction": float(max(0, kelly)),  # Don't bet if negative edge
            "growth_rate": float(growth),
        }
    
    def _streak_analysis(self) -> dict:
        """Analyze win/loss streaks."""
        if self.n_bets == 0:
            return {
                "longest_win": 0,
                "longest_lose": 0,
                "avg_win": 0.0,
                "avg_lose": 0.0,
            }
        
        win_streaks = []
        lose_streaks = []
        current_streak = 0
        current_type = None
        
        for w in self.wins:
            if current_type is None:
                current_type = w
                current_streak = 1
            elif w == current_type:
                current_streak += 1
            else:
                if current_type == 1:
                    win_streaks.append(current_streak)
                else:
                    lose_streaks.append(current_streak)
                current_type = w
                current_streak = 1
        
        # Don't forget the last streak
        if current_type == 1:
            win_streaks.append(current_streak)
        else:
            lose_streaks.append(current_streak)
        
        return {
            "longest_win": max(win_streaks) if win_streaks else 0,
            "longest_lose": max(lose_streaks) if lose_streaks else 0,
            "avg_win": float(np.mean(win_streaks)) if win_streaks else 0.0,
            "avg_lose": float(np.mean(lose_streaks)) if lose_streaks else 0.0,
        }
    
    def _high_confidence_metrics(self, threshold: float = 0.60) -> dict:
        """Metrics for high-confidence predictions only."""
        if self.confidences is None or len(self.confidences) == 0:
            return {"accuracy": 0.0, "roi": 0.0, "n": 0}
        
        mask = self.confidences >= threshold
        n = mask.sum()
        
        if n == 0:
            return {"accuracy": 0.0, "roi": 0.0, "n": 0}
        
        high_wins = self.wins[mask]
        high_returns = self.returns[mask]
        
        return {
            "accuracy": float(high_wins.mean()),
            "roi": float(high_returns.mean()),
            "n": int(n),
        }
    
    def _time_split_analysis(self) -> dict:
        """Compare first half vs second half of backtest."""
        if self.n_bets < 20:
            return {"first_half": 0.0, "second_half": 0.0}
        
        mid = self.n_bets // 2
        
        first_half = self.returns[:mid]
        second_half = self.returns[mid:]
        
        return {
            "first_half": float(first_half.mean()),
            "second_half": float(second_half.mean()),
        }
