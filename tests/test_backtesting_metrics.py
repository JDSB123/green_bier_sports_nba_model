"""
Tests for src/backtesting/ modules.

These tests cover the backtest engine, performance metrics, and related utilities.
Coverage target: 95%+ for backtesting modules.
"""
import pytest
import numpy as np
from datetime import datetime

from src.backtesting.performance_metrics import PerformanceMetrics, MarketPerformance


class TestPerformanceMetricsBasics:
    """Tests for PerformanceMetrics class - basic calculations."""

    def test_empty_returns(self):
        """Empty returns should handle gracefully."""
        metrics = PerformanceMetrics(returns=[])
        result = metrics.calculate_all(market="test")
        
        assert result.n_bets == 0
        assert result.accuracy == 0.0
        assert result.roi == 0.0

    def test_roi_calculation_winning(self):
        """ROI should be positive for winning bets."""
        # All wins at -110 odds (win 0.909 per unit)
        returns = [0.909, 0.909, 0.909, 0.909, 0.909]
        metrics = PerformanceMetrics(returns=returns)
        result = metrics.calculate_all(market="test")
        
        assert result.roi > 0
        assert result.accuracy == 1.0

    def test_roi_calculation_losing(self):
        """ROI should be negative for losing bets."""
        # All losses (lose 1 unit per bet)
        returns = [-1.0, -1.0, -1.0, -1.0, -1.0]
        metrics = PerformanceMetrics(returns=returns)
        result = metrics.calculate_all(market="test")
        
        assert result.roi < 0
        assert result.accuracy == 0.0

    def test_roi_mixed_results(self):
        """Mixed results should calculate correctly."""
        # 3 wins, 2 losses at -110
        returns = [0.909, 0.909, 0.909, -1.0, -1.0]
        metrics = PerformanceMetrics(returns=returns)
        result = metrics.calculate_all(market="test")
        
        # 3 * 0.909 - 2 * 1.0 = 2.727 - 2.0 = 0.727
        assert abs(result.total_profit - 0.727) < 0.01
        assert result.accuracy == 0.6  # 3/5 wins

    def test_accuracy_calculation(self):
        """Accuracy should be wins / total based on returns."""
        # 4 wins (positive returns), 1 loss (negative return)
        returns = [0.909, 0.909, 0.909, 0.909, -1.0]
        
        metrics = PerformanceMetrics(returns=returns)
        result = metrics.calculate_all(market="test")
        
        assert result.accuracy == 0.8  # 4 out of 5

    def test_win_streak_detection(self):
        """Longest win/loss streaks should be detected."""
        # 3 wins, 2 losses, 1 win (longest win = 3, longest loss = 2)
        returns = [0.909, 0.909, 0.909, -1.0, -1.0, 0.909]
        metrics = PerformanceMetrics(returns=returns)
        result = metrics.calculate_all(market="test")
        
        assert result.longest_win_streak == 3
        assert result.longest_lose_streak == 2


class TestPerformanceMetricsAdvanced:
    """Tests for advanced metrics like Sharpe, Sortino, drawdown."""

    def test_sharpe_ratio_positive(self):
        """Sharpe ratio should be positive for consistent wins."""
        # Many consistent small wins with some variability
        returns = [0.05, 0.06, 0.04, 0.07, 0.05, 0.08, 0.03] * 15  # 105 bets
        metrics = PerformanceMetrics(returns=returns)
        result = metrics.calculate_all(market="test")
        
        # With positive mean and reasonable std, Sharpe should be positive
        assert result.sharpe_ratio > 0

    def test_max_drawdown_calculation(self):
        """Max drawdown should capture worst peak-to-trough."""
        # Win, win, lose, lose, lose, win (drawdown in middle)
        returns = [0.909, 0.909, -1.0, -1.0, -1.0, 0.909]
        metrics = PerformanceMetrics(returns=returns)
        result = metrics.calculate_all(market="test")
        
        # After 2 wins: cumsum = 1.818
        # After 3 losses: cumsum = 1.818 - 3 = -1.182
        # Max drawdown = 1.818 - (-1.182) = 3.0
        assert result.max_drawdown == pytest.approx(3.0, abs=0.1)

    def test_kelly_fraction_positive_edge(self):
        """Kelly fraction should be positive for profitable system."""
        # 60% win rate at -110 (odds of 0.909)
        # We need enough samples for the calculation
        wins = [0.909] * 60
        losses = [-1.0] * 40
        returns = wins + losses
        np.random.shuffle(np.array(returns))  # Mix them up
        
        metrics = PerformanceMetrics(returns=list(returns))
        result = metrics.calculate_all(market="test")
        
        # With positive edge, Kelly should be positive
        # Kelly = (p * b - q) / b where p=0.6, q=0.4, b=0.909
        # Kelly = (0.6 * 0.909 - 0.4) / 0.909 = 0.16
        assert result.kelly_fraction > 0


class TestMarketPerformance:
    """Tests for MarketPerformance dataclass."""

    def test_market_performance_creation(self):
        """MarketPerformance should be created with required fields."""
        perf = MarketPerformance(
            market="fg_spread",
            n_bets=100,
            accuracy=0.55,
            roi=0.02,
            total_profit=2.0,
        )
        
        assert perf.market == "fg_spread"
        assert perf.n_bets == 100
        assert perf.accuracy == 0.55
        assert perf.roi == 0.02
        assert perf.total_profit == 2.0

    def test_market_performance_defaults(self):
        """Default values should be sensible."""
        perf = MarketPerformance(
            market="1h_total",
            n_bets=50,
            accuracy=0.52,
            roi=-0.03,
            total_profit=-1.5,
        )
        
        assert perf.accuracy_ci == (0.0, 0.0)
        assert perf.is_significant is False
        assert perf.is_profitable is False
        assert perf.sharpe_ratio == 0.0

    def test_market_performance_with_all_fields(self):
        """All fields should be settable."""
        perf = MarketPerformance(
            market="fg_total",
            n_bets=200,
            accuracy=0.58,
            roi=0.08,
            total_profit=16.0,
            accuracy_ci=(0.52, 0.64),
            roi_ci=(0.02, 0.14),
            vs_random_pvalue=0.01,
            vs_vig_pvalue=0.03,
            is_significant=True,
            is_profitable=True,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            calmar_ratio=1.2,
            max_drawdown=8.0,
            max_drawdown_pct=0.15,
            max_drawdown_duration=25,
            avg_drawdown=3.0,
            kelly_fraction=0.10,
            kelly_growth_rate=0.02,
            half_kelly_fraction=0.05,
            longest_win_streak=8,
            longest_lose_streak=5,
            avg_win_streak=2.5,
            avg_lose_streak=1.8,
            high_conf_accuracy=0.65,
            high_conf_roi=0.12,
            high_conf_n=50,
            first_half_roi=0.06,
            second_half_roi=0.10,
        )
        
        assert perf.is_significant is True
        assert perf.is_profitable is True
        assert perf.sharpe_ratio == 1.5


class TestVigCalculations:
    """Tests for vig/juice break-even calculations."""

    def test_vig_break_even_at_minus_110(self):
        """Break-even ROI at -110 should be about -4.55%."""
        # At -110 odds, you need 52.38% win rate to break even
        # ROI at 50% win rate = (0.5 * 0.909) - (0.5 * 1.0) = -0.0455
        
        assert PerformanceMetrics.VIG_BREAK_EVEN == pytest.approx(-0.0455, abs=0.001)

    def test_win_payout_at_minus_110(self):
        """Win payout at -110 should be 100/110 = 0.909."""
        assert PerformanceMetrics.WIN_PAYOUT == pytest.approx(0.909, abs=0.001)


class TestStatisticalSignificance:
    """Tests for statistical significance calculations."""

    def test_sample_size_for_significance(self):
        """Need sufficient sample size for statistical significance."""
        # With 100 bets at 55% accuracy, is it significant?
        # Using binomial test: p(X >= 55 | n=100, p=0.5)
        from scipy import stats
        
        n_bets = 100
        wins = 55
        
        # One-sided binomial test
        pvalue = 1 - stats.binom.cdf(wins - 1, n_bets, 0.5)
        
        # 55/100 is not quite significant at p < 0.05
        # Need about 58-59% for significance with n=100
        assert pvalue > 0.05  # Not significant

    def test_large_sample_significance(self):
        """Large sample with small edge should be significant."""
        from scipy import stats
        
        n_bets = 1000
        wins = 540  # 54% accuracy
        
        pvalue = 1 - stats.binom.cdf(wins - 1, n_bets, 0.5)
        
        # 540/1000 = 54% should be significant with n=1000
        assert pvalue < 0.05


class TestConfidenceTiers:
    """Tests for confidence tier analysis."""

    def test_high_confidence_tier(self):
        """High confidence bets should be trackable."""
        confidences = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
        returns = [0.909, -1.0, 0.909, 0.909, 0.909, 0.909]
        
        metrics = PerformanceMetrics(
            returns=returns,
            confidences=confidences,
        )
        
        # High confidence (>= 60%)
        high_conf_mask = np.array(confidences) >= 0.60
        high_conf_returns = np.array(returns)[high_conf_mask]
        
        assert len(high_conf_returns) == 4
        assert sum(high_conf_returns > 0) == 4  # All high conf bets won

    def test_low_confidence_filtering(self):
        """Low confidence bets should be filterable."""
        confidences = [0.51, 0.52, 0.53, 0.54, 0.55]
        
        min_conf = 0.55
        filtered = [c for c in confidences if c >= min_conf]
        
        assert len(filtered) == 1


class TestEdgeCases:
    """Edge case tests for performance metrics."""

    def test_single_bet_win(self):
        """Single winning bet should work."""
        returns = [0.909]
        metrics = PerformanceMetrics(returns=returns)
        result = metrics.calculate_all(market="test")
        
        assert result.n_bets == 1
        assert result.accuracy == 1.0

    def test_single_bet_loss(self):
        """Single losing bet should work."""
        returns = [-1.0]
        metrics = PerformanceMetrics(returns=returns)
        result = metrics.calculate_all(market="test")
        
        assert result.n_bets == 1
        assert result.accuracy == 0.0

    def test_all_wins(self):
        """100% win rate should work."""
        returns = [0.909] * 100
        metrics = PerformanceMetrics(returns=returns)
        result = metrics.calculate_all(market="test")
        
        assert result.accuracy == 1.0
        assert result.total_profit > 0
        assert result.longest_win_streak == 100

    def test_all_losses(self):
        """0% win rate should work."""
        returns = [-1.0] * 100
        metrics = PerformanceMetrics(returns=returns)
        result = metrics.calculate_all(market="test")
        
        assert result.accuracy == 0.0
        assert result.total_profit == -100.0
        assert result.longest_lose_streak == 100

    def test_numpy_array_input(self):
        """Should accept numpy arrays."""
        returns = np.array([0.909, -1.0, 0.909, -1.0])
        metrics = PerformanceMetrics(returns=returns)
        result = metrics.calculate_all(market="test")
        
        assert result.n_bets == 4

    def test_mixed_types(self):
        """Should handle int/float mix."""
        returns = [1, -1, 0.909, -1.0]
        metrics = PerformanceMetrics(returns=returns)
        result = metrics.calculate_all(market="test")
        
        assert result.n_bets == 4
