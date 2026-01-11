"""
Enterprise statistical validation for backtesting.

Implements hypothesis tests, confidence intervals, and statistical significance checks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class HypothesisTestResult:
    """Result of a statistical hypothesis test."""
    test_name: str
    null_hypothesis: str
    alternative_hypothesis: str
    statistic: float
    p_value: float
    is_significant: bool  # at alpha = 0.05
    conclusion: str
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None


@dataclass
class StatisticalSummary:
    """Complete statistical summary for a market."""
    market: str
    n_samples: int
    
    # Central tendency
    mean_accuracy: float
    median_accuracy: float
    mean_roi: float
    median_roi: float
    
    # Dispersion
    std_accuracy: float
    std_roi: float
    iqr_roi: float  # Interquartile range
    
    # Distribution shape
    skewness: float
    kurtosis: float
    is_normal: bool  # Shapiro-Wilk test result
    normality_pvalue: float
    
    # Tests
    tests: List[HypothesisTestResult]
    
    # Overall assessment
    is_statistically_significant: bool
    is_economically_significant: bool  # ROI > break-even
    recommendation: str


class StatisticalValidator:
    """
    Enterprise-level statistical validation.
    
    Performs comprehensive hypothesis testing and provides
    statistically rigorous conclusions about model performance.
    """
    
    # Break-even ROI at -110 odds
    BREAK_EVEN_ROI = -0.0455
    
    # Minimum win rate needed to be profitable at -110
    MIN_PROFITABLE_WIN_RATE = 0.5238  # 110 / (110 + 100)
    
    def __init__(
        self,
        alpha: float = 0.05,  # Significance level
        min_samples: int = 30,  # Minimum for valid inference
    ):
        """
        Initialize validator.
        
        Args:
            alpha: Significance level for hypothesis tests
            min_samples: Minimum sample size for valid inference
        """
        self.alpha = alpha
        self.min_samples = min_samples
    
    def validate_market(
        self,
        wins: np.ndarray,
        returns: np.ndarray,
        market: str = "unknown",
    ) -> StatisticalSummary:
        """
        Perform comprehensive statistical validation for a market.
        
        Args:
            wins: Array of win/loss outcomes (1 = win, 0 = loss)
            returns: Array of returns per bet
            market: Market name for reporting
            
        Returns:
            StatisticalSummary with all test results
        """
        n = len(wins)
        
        if n < self.min_samples:
            logger.warning(
                f"Market {market}: Only {n} samples, need {self.min_samples} for valid inference"
            )
        
        # Calculate basic statistics
        mean_acc = float(wins.mean())
        median_acc = float(np.median(wins))
        std_acc = float(wins.std())
        
        mean_roi = float(returns.mean())
        median_roi = float(np.median(returns))
        std_roi = float(returns.std())
        iqr_roi = float(np.percentile(returns, 75) - np.percentile(returns, 25))
        
        # Distribution shape
        skewness = float(stats.skew(returns)) if n > 10 else 0.0
        kurtosis = float(stats.kurtosis(returns)) if n > 10 else 0.0
        
        # Normality test
        if n >= 20:
            _, normality_p = stats.shapiro(returns[:min(n, 5000)])  # Shapiro-Wilk limit
            is_normal = normality_p > self.alpha
        else:
            normality_p = 1.0
            is_normal = True
        
        # Run all hypothesis tests
        tests = []
        
        # Test 1: Accuracy vs 50%
        test_random = self._test_vs_random(wins)
        tests.append(test_random)
        
        # Test 2: Accuracy vs profitable threshold
        test_profitable = self._test_vs_profitable(wins)
        tests.append(test_profitable)
        
        # Test 3: ROI vs break-even
        test_roi = self._test_roi_vs_breakeven(returns)
        tests.append(test_roi)
        
        # Test 4: ROI > 0
        test_positive = self._test_roi_positive(returns)
        tests.append(test_positive)
        
        # Test 5: Binomial exact test
        test_binomial = self._binomial_test(wins)
        tests.append(test_binomial)
        
        # Test 6: Runs test for independence
        test_runs = self._runs_test(wins)
        tests.append(test_runs)
        
        # Overall assessment
        is_stat_sig = test_random.is_significant and test_roi.is_significant
        is_econ_sig = mean_roi > self.BREAK_EVEN_ROI
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            mean_acc, mean_roi, is_stat_sig, is_econ_sig, n
        )
        
        return StatisticalSummary(
            market=market,
            n_samples=n,
            mean_accuracy=mean_acc,
            median_accuracy=median_acc,
            mean_roi=mean_roi,
            median_roi=median_roi,
            std_accuracy=std_acc,
            std_roi=std_roi,
            iqr_roi=iqr_roi,
            skewness=skewness,
            kurtosis=kurtosis,
            is_normal=is_normal,
            normality_pvalue=normality_p,
            tests=tests,
            is_statistically_significant=is_stat_sig,
            is_economically_significant=is_econ_sig,
            recommendation=recommendation,
        )
    
    def _test_vs_random(self, wins: np.ndarray) -> HypothesisTestResult:
        """
        One-sample t-test: H0: win_rate = 50%
        """
        n = len(wins)
        
        if n < 10:
            return HypothesisTestResult(
                test_name="t-test vs random",
                null_hypothesis="Win rate = 50%",
                alternative_hypothesis="Win rate ≠ 50%",
                statistic=0.0,
                p_value=1.0,
                is_significant=False,
                conclusion="Insufficient samples",
            )
        
        result = stats.ttest_1samp(wins, 0.5)
        
        # Effect size (Cohen's d)
        effect_size = (wins.mean() - 0.5) / wins.std() if wins.std() > 0 else 0
        
        # Confidence interval for win rate
        se = wins.std() / np.sqrt(n)
        ci = (wins.mean() - 1.96 * se, wins.mean() + 1.96 * se)
        
        is_sig = result.pvalue < self.alpha
        
        if is_sig:
            if wins.mean() > 0.5:
                conclusion = f"Win rate ({wins.mean():.1%}) significantly > 50%"
            else:
                conclusion = f"Win rate ({wins.mean():.1%}) significantly < 50%"
        else:
            conclusion = f"Win rate ({wins.mean():.1%}) not significantly different from 50%"
        
        return HypothesisTestResult(
            test_name="t-test vs random",
            null_hypothesis="Win rate = 50%",
            alternative_hypothesis="Win rate ≠ 50%",
            statistic=float(result.statistic),
            p_value=float(result.pvalue),
            is_significant=is_sig,
            conclusion=conclusion,
            effect_size=float(effect_size),
            confidence_interval=(float(ci[0]), float(ci[1])),
        )
    
    def _test_vs_profitable(self, wins: np.ndarray) -> HypothesisTestResult:
        """
        One-sample t-test: H0: win_rate <= 52.38% (break-even at -110)
        """
        n = len(wins)
        
        if n < 10:
            return HypothesisTestResult(
                test_name="t-test vs profitable threshold",
                null_hypothesis=f"Win rate <= {self.MIN_PROFITABLE_WIN_RATE:.1%}",
                alternative_hypothesis=f"Win rate > {self.MIN_PROFITABLE_WIN_RATE:.1%}",
                statistic=0.0,
                p_value=1.0,
                is_significant=False,
                conclusion="Insufficient samples",
            )
        
        result = stats.ttest_1samp(wins, self.MIN_PROFITABLE_WIN_RATE)
        
        # One-sided test (we only care if win rate > threshold)
        if wins.mean() > self.MIN_PROFITABLE_WIN_RATE:
            p_value = result.pvalue / 2
        else:
            p_value = 1 - result.pvalue / 2
        
        is_sig = p_value < self.alpha and wins.mean() > self.MIN_PROFITABLE_WIN_RATE
        
        if is_sig:
            conclusion = f"Win rate ({wins.mean():.1%}) significantly > profitable threshold"
        else:
            conclusion = f"Win rate ({wins.mean():.1%}) not significantly > profitable threshold"
        
        return HypothesisTestResult(
            test_name="t-test vs profitable threshold",
            null_hypothesis=f"Win rate <= {self.MIN_PROFITABLE_WIN_RATE:.1%}",
            alternative_hypothesis=f"Win rate > {self.MIN_PROFITABLE_WIN_RATE:.1%}",
            statistic=float(result.statistic),
            p_value=float(p_value),
            is_significant=is_sig,
            conclusion=conclusion,
        )
    
    def _test_roi_vs_breakeven(self, returns: np.ndarray) -> HypothesisTestResult:
        """
        One-sample t-test: H0: ROI <= -4.55% (break-even)
        """
        n = len(returns)
        
        if n < 10:
            return HypothesisTestResult(
                test_name="t-test ROI vs break-even",
                null_hypothesis=f"ROI <= {self.BREAK_EVEN_ROI:.2%}",
                alternative_hypothesis=f"ROI > {self.BREAK_EVEN_ROI:.2%}",
                statistic=0.0,
                p_value=1.0,
                is_significant=False,
                conclusion="Insufficient samples",
            )
        
        result = stats.ttest_1samp(returns, self.BREAK_EVEN_ROI)
        
        # One-sided test
        mean_roi = returns.mean()
        if mean_roi > self.BREAK_EVEN_ROI:
            p_value = result.pvalue / 2
        else:
            p_value = 1 - result.pvalue / 2
        
        is_sig = p_value < self.alpha and mean_roi > self.BREAK_EVEN_ROI
        
        # Confidence interval
        se = returns.std() / np.sqrt(n)
        ci = (mean_roi - 1.96 * se, mean_roi + 1.96 * se)
        
        if is_sig:
            conclusion = f"ROI ({mean_roi:.1%}) significantly > break-even"
        else:
            conclusion = f"ROI ({mean_roi:.1%}) not significantly > break-even"
        
        return HypothesisTestResult(
            test_name="t-test ROI vs break-even",
            null_hypothesis=f"ROI <= {self.BREAK_EVEN_ROI:.2%}",
            alternative_hypothesis=f"ROI > {self.BREAK_EVEN_ROI:.2%}",
            statistic=float(result.statistic),
            p_value=float(p_value),
            is_significant=is_sig,
            conclusion=conclusion,
            confidence_interval=(float(ci[0]), float(ci[1])),
        )
    
    def _test_roi_positive(self, returns: np.ndarray) -> HypothesisTestResult:
        """
        One-sample t-test: H0: ROI <= 0
        """
        n = len(returns)
        
        if n < 10:
            return HypothesisTestResult(
                test_name="t-test ROI positive",
                null_hypothesis="ROI <= 0",
                alternative_hypothesis="ROI > 0",
                statistic=0.0,
                p_value=1.0,
                is_significant=False,
                conclusion="Insufficient samples",
            )
        
        result = stats.ttest_1samp(returns, 0)
        
        mean_roi = returns.mean()
        if mean_roi > 0:
            p_value = result.pvalue / 2
        else:
            p_value = 1 - result.pvalue / 2
        
        is_sig = p_value < self.alpha and mean_roi > 0
        
        if is_sig:
            conclusion = f"ROI ({mean_roi:.1%}) significantly positive"
        else:
            conclusion = f"ROI ({mean_roi:.1%}) not significantly positive"
        
        return HypothesisTestResult(
            test_name="t-test ROI positive",
            null_hypothesis="ROI <= 0",
            alternative_hypothesis="ROI > 0",
            statistic=float(result.statistic),
            p_value=float(p_value),
            is_significant=is_sig,
            conclusion=conclusion,
        )
    
    def _binomial_test(self, wins: np.ndarray) -> HypothesisTestResult:
        """
        Exact binomial test: H0: p = 0.5
        
        More accurate than t-test for small samples.
        """
        n = len(wins)
        k = int(wins.sum())
        
        result = stats.binomtest(k, n, 0.5, alternative="two-sided")
        
        is_sig = result.pvalue < self.alpha
        
        ci = result.proportion_ci(confidence_level=0.95)
        
        if is_sig:
            conclusion = f"Win rate ({k}/{n} = {k/n:.1%}) significantly ≠ 50%"
        else:
            conclusion = f"Win rate ({k}/{n} = {k/n:.1%}) consistent with random"
        
        return HypothesisTestResult(
            test_name="Binomial exact test",
            null_hypothesis="Win probability = 50%",
            alternative_hypothesis="Win probability ≠ 50%",
            statistic=float(k),
            p_value=float(result.pvalue),
            is_significant=is_sig,
            conclusion=conclusion,
            confidence_interval=(float(ci.low), float(ci.high)),
        )
    
    def _runs_test(self, wins: np.ndarray) -> HypothesisTestResult:
        """
        Wald-Wolfowitz runs test for independence/randomness.
        
        Tests if wins/losses are independent (not streaky).
        """
        n = len(wins)
        
        if n < 20:
            return HypothesisTestResult(
                test_name="Runs test",
                null_hypothesis="Outcomes are independent",
                alternative_hypothesis="Outcomes show patterns",
                statistic=0.0,
                p_value=1.0,
                is_significant=False,
                conclusion="Insufficient samples for runs test",
            )
        
        # Count runs (consecutive same outcomes)
        runs = 1
        for i in range(1, n):
            if wins[i] != wins[i-1]:
                runs += 1
        
        # Expected number of runs under independence
        n1 = int(wins.sum())
        n0 = n - n1
        
        if n1 == 0 or n0 == 0:
            return HypothesisTestResult(
                test_name="Runs test",
                null_hypothesis="Outcomes are independent",
                alternative_hypothesis="Outcomes show patterns",
                statistic=float(runs),
                p_value=1.0,
                is_significant=False,
                conclusion="All same outcome - runs test not applicable",
            )
        
        expected_runs = 1 + (2 * n1 * n0) / n
        var_runs = (2 * n1 * n0 * (2 * n1 * n0 - n)) / (n**2 * (n - 1))
        
        if var_runs <= 0:
            return HypothesisTestResult(
                test_name="Runs test",
                null_hypothesis="Outcomes are independent",
                alternative_hypothesis="Outcomes show patterns",
                statistic=float(runs),
                p_value=1.0,
                is_significant=False,
                conclusion="Variance calculation issue",
            )
        
        z = (runs - expected_runs) / np.sqrt(var_runs)
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        is_sig = p_value < self.alpha
        
        if is_sig:
            if runs < expected_runs:
                conclusion = "Significant clustering (streaky outcomes)"
            else:
                conclusion = "Significant alternation (anti-streaky)"
        else:
            conclusion = "Outcomes appear independent (no significant pattern)"
        
        return HypothesisTestResult(
            test_name="Runs test",
            null_hypothesis="Outcomes are independent",
            alternative_hypothesis="Outcomes show patterns",
            statistic=float(z),
            p_value=float(p_value),
            is_significant=is_sig,
            conclusion=conclusion,
        )
    
    def _generate_recommendation(
        self,
        accuracy: float,
        roi: float,
        is_stat_sig: bool,
        is_econ_sig: bool,
        n_samples: int,
    ) -> str:
        """Generate recommendation based on all evidence."""
        if n_samples < self.min_samples:
            return "INSUFFICIENT DATA: Need more samples for reliable conclusions"
        
        if is_stat_sig and is_econ_sig:
            if accuracy >= 0.55 and roi >= 0.05:
                return "STRONG RECOMMEND: Statistically and economically significant positive edge"
            else:
                return "RECOMMEND WITH CAUTION: Significant edge but smaller than ideal"
        
        if is_econ_sig and not is_stat_sig:
            return "MONITOR: Positive ROI but not statistically significant yet"
        
        if is_stat_sig and not is_econ_sig:
            return "NOT RECOMMENDED: Statistically significant but not profitable after vig"
        
        return "NOT RECOMMENDED: No significant edge detected"
    
    def compare_markets(
        self,
        market_results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    ) -> Dict[str, HypothesisTestResult]:
        """
        Compare performance across markets using paired tests.
        
        Args:
            market_results: Dict of market -> (wins, returns)
            
        Returns:
            Dict of comparison test results
        """
        comparisons = {}
        markets = list(market_results.keys())
        
        for i, m1 in enumerate(markets):
            for m2 in markets[i+1:]:
                _, returns1 = market_results[m1]
                _, returns2 = market_results[m2]
                
                # Need same length for paired test
                min_len = min(len(returns1), len(returns2))
                if min_len < 20:
                    continue
                
                r1 = returns1[:min_len]
                r2 = returns2[:min_len]
                
                result = stats.ttest_rel(r1, r2)
                
                diff = r1.mean() - r2.mean()
                is_sig = result.pvalue < self.alpha
                
                if is_sig:
                    if diff > 0:
                        conclusion = f"{m1} significantly outperforms {m2}"
                    else:
                        conclusion = f"{m2} significantly outperforms {m1}"
                else:
                    conclusion = f"No significant difference between {m1} and {m2}"
                
                comparisons[f"{m1}_vs_{m2}"] = HypothesisTestResult(
                    test_name=f"Paired t-test: {m1} vs {m2}",
                    null_hypothesis=f"ROI({m1}) = ROI({m2})",
                    alternative_hypothesis=f"ROI({m1}) ≠ ROI({m2})",
                    statistic=float(result.statistic),
                    p_value=float(result.pvalue),
                    is_significant=is_sig,
                    conclusion=conclusion,
                    effect_size=float(diff),
                )
        
        return comparisons
