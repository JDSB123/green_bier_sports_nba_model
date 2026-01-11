"""
Monte Carlo simulation for backtesting.

Simulates possible outcomes to estimate probability of ruin,
expected bankroll trajectories, and confidence intervals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloResults:
    """Results from Monte Carlo simulation."""
    
    # Configuration
    n_simulations: int
    n_bets: int
    initial_bankroll: float
    bet_size_type: str  # "flat" or "kelly"
    bet_size_value: float  # units or kelly fraction
    
    # Bankroll distribution
    mean_final_bankroll: float
    median_final_bankroll: float
    std_final_bankroll: float
    min_final_bankroll: float
    max_final_bankroll: float
    percentile_5: float
    percentile_25: float
    percentile_75: float
    percentile_95: float
    
    # Risk metrics
    probability_of_ruin: float  # P(bankroll <= 0)
    probability_of_loss: float  # P(final < initial)
    probability_of_profit: float  # P(final > initial)
    probability_double: float  # P(final >= 2 * initial)
    
    # Expected outcomes
    expected_roi: float
    expected_growth_rate: float  # Log growth rate
    
    # Drawdown distribution
    mean_max_drawdown: float
    median_max_drawdown: float
    worst_max_drawdown: float
    
    # Time-to-ruin for paths that go bust
    mean_time_to_ruin: Optional[float]
    
    # Sample paths (for visualization)
    sample_paths: Optional[np.ndarray] = None  # Shape: (n_paths, n_bets)


class MonteCarloSimulator:
    """
    Monte Carlo simulation for betting outcomes.
    
    Uses empirical win rate and returns to simulate possible
    future outcomes and calculate risk metrics.
    """
    
    def __init__(
        self,
        win_probability: float,
        win_payout: float = 100.0 / 110.0,  # ~0.909 for -110 odds
        random_seed: Optional[int] = None,
    ):
        """
        Initialize simulator.
        
        Args:
            win_probability: Probability of winning each bet
            win_payout: Payout ratio for wins (e.g., 0.909 for -110)
            random_seed: Random seed for reproducibility
        """
        self.win_prob = win_probability
        self.win_payout = win_payout
        self.rng = np.random.default_rng(random_seed)
    
    def simulate_flat_betting(
        self,
        n_simulations: int = 10000,
        n_bets: int = 1000,
        initial_bankroll: float = 1000.0,
        bet_size: float = 10.0,  # Fixed bet size per bet
        store_paths: bool = False,
        n_paths_to_store: int = 100,
    ) -> MonteCarloResults:
        """
        Simulate flat betting (fixed bet size).
        
        Args:
            n_simulations: Number of simulations to run
            n_bets: Number of bets per simulation
            initial_bankroll: Starting bankroll
            bet_size: Fixed bet size in units
            store_paths: Whether to store sample paths
            n_paths_to_store: Number of paths to store if storing
            
        Returns:
            MonteCarloResults with simulation outcomes
        """
        # Generate all outcomes at once for efficiency
        # Shape: (n_simulations, n_bets)
        outcomes = self.rng.random((n_simulations, n_bets)) < self.win_prob
        
        # Calculate returns
        returns = np.where(outcomes, bet_size * self.win_payout, -bet_size)
        
        # Cumulative bankroll
        cumulative = initial_bankroll + np.cumsum(returns, axis=1)
        
        # Final bankrolls
        final_bankrolls = cumulative[:, -1]
        
        # Check for ruin (bankroll <= 0 at any point)
        min_bankrolls = np.minimum.accumulate(cumulative, axis=1)
        ruined = (min_bankrolls <= 0).any(axis=1)
        
        # Time to ruin for paths that go bust
        time_to_ruin = []
        for i in range(n_simulations):
            if ruined[i]:
                ruin_idx = np.where(cumulative[i] <= 0)[0]
                if len(ruin_idx) > 0:
                    time_to_ruin.append(ruin_idx[0])
        
        # Max drawdown for each simulation
        running_max = np.maximum.accumulate(cumulative, axis=1)
        drawdowns = running_max - cumulative
        max_drawdowns = drawdowns.max(axis=1)
        
        # Sample paths
        sample_paths = None
        if store_paths:
            path_indices = self.rng.choice(
                n_simulations,
                size=min(n_paths_to_store, n_simulations),
                replace=False
            )
            sample_paths = cumulative[path_indices]
        
        return MonteCarloResults(
            n_simulations=n_simulations,
            n_bets=n_bets,
            initial_bankroll=initial_bankroll,
            bet_size_type="flat",
            bet_size_value=bet_size,
            mean_final_bankroll=float(final_bankrolls.mean()),
            median_final_bankroll=float(np.median(final_bankrolls)),
            std_final_bankroll=float(final_bankrolls.std()),
            min_final_bankroll=float(final_bankrolls.min()),
            max_final_bankroll=float(final_bankrolls.max()),
            percentile_5=float(np.percentile(final_bankrolls, 5)),
            percentile_25=float(np.percentile(final_bankrolls, 25)),
            percentile_75=float(np.percentile(final_bankrolls, 75)),
            percentile_95=float(np.percentile(final_bankrolls, 95)),
            probability_of_ruin=float(ruined.mean()),
            probability_of_loss=float((final_bankrolls < initial_bankroll).mean()),
            probability_of_profit=float((final_bankrolls > initial_bankroll).mean()),
            probability_double=float((final_bankrolls >= 2 * initial_bankroll).mean()),
            expected_roi=float((final_bankrolls.mean() - initial_bankroll) / initial_bankroll),
            expected_growth_rate=float(np.log(final_bankrolls / initial_bankroll).mean()),
            mean_max_drawdown=float(max_drawdowns.mean()),
            median_max_drawdown=float(np.median(max_drawdowns)),
            worst_max_drawdown=float(max_drawdowns.max()),
            mean_time_to_ruin=float(np.mean(time_to_ruin)) if time_to_ruin else None,
            sample_paths=sample_paths,
        )
    
    def simulate_kelly_betting(
        self,
        n_simulations: int = 10000,
        n_bets: int = 1000,
        initial_bankroll: float = 1000.0,
        kelly_fraction: float = 0.25,  # Often use half-Kelly (0.5 * full Kelly)
        min_bet_pct: float = 0.01,  # Minimum 1% of bankroll
        max_bet_pct: float = 0.10,  # Maximum 10% of bankroll
        store_paths: bool = False,
        n_paths_to_store: int = 100,
    ) -> MonteCarloResults:
        """
        Simulate Kelly criterion betting (proportional to bankroll).
        
        Args:
            n_simulations: Number of simulations to run
            n_bets: Number of bets per simulation
            initial_bankroll: Starting bankroll
            kelly_fraction: Kelly fraction to use (e.g., 0.25 for quarter-Kelly)
            min_bet_pct: Minimum bet as percentage of bankroll
            max_bet_pct: Maximum bet as percentage of bankroll
            store_paths: Whether to store sample paths
            n_paths_to_store: Number of paths to store
            
        Returns:
            MonteCarloResults with simulation outcomes
        """
        # Initialize bankrolls
        bankrolls = np.full((n_simulations, n_bets + 1), initial_bankroll)
        ruined = np.zeros(n_simulations, dtype=bool)
        time_to_ruin_list: List[int] = []
        
        # Generate all outcomes
        outcomes = self.rng.random((n_simulations, n_bets)) < self.win_prob
        
        # Simulate bet by bet (necessary for Kelly since bet size depends on bankroll)
        for bet_idx in range(n_bets):
            current_bankroll = bankrolls[:, bet_idx]
            
            # Calculate bet size (Kelly fraction of current bankroll)
            bet_size = current_bankroll * kelly_fraction
            
            # Apply min/max constraints
            bet_size = np.clip(
                bet_size,
                current_bankroll * min_bet_pct,
                current_bankroll * max_bet_pct,
            )
            
            # Calculate returns
            returns = np.where(
                outcomes[:, bet_idx],
                bet_size * self.win_payout,
                -bet_size
            )
            
            # Update bankroll
            bankrolls[:, bet_idx + 1] = current_bankroll + returns
            
            # Mark new ruins
            new_ruins = (bankrolls[:, bet_idx + 1] <= 0) & ~ruined
            if new_ruins.any():
                for sim_idx in np.where(new_ruins)[0]:
                    time_to_ruin_list.append(bet_idx + 1)
                ruined |= new_ruins
        
        # Final bankrolls
        final_bankrolls = bankrolls[:, -1]
        
        # Max drawdown
        running_max = np.maximum.accumulate(bankrolls, axis=1)
        drawdowns = running_max - bankrolls
        max_drawdowns = drawdowns.max(axis=1)
        
        # Sample paths
        sample_paths = None
        if store_paths:
            path_indices = self.rng.choice(
                n_simulations,
                size=min(n_paths_to_store, n_simulations),
                replace=False
            )
            sample_paths = bankrolls[path_indices]
        
        return MonteCarloResults(
            n_simulations=n_simulations,
            n_bets=n_bets,
            initial_bankroll=initial_bankroll,
            bet_size_type="kelly",
            bet_size_value=kelly_fraction,
            mean_final_bankroll=float(final_bankrolls.mean()),
            median_final_bankroll=float(np.median(final_bankrolls)),
            std_final_bankroll=float(final_bankrolls.std()),
            min_final_bankroll=float(final_bankrolls.min()),
            max_final_bankroll=float(final_bankrolls.max()),
            percentile_5=float(np.percentile(final_bankrolls, 5)),
            percentile_25=float(np.percentile(final_bankrolls, 25)),
            percentile_75=float(np.percentile(final_bankrolls, 75)),
            percentile_95=float(np.percentile(final_bankrolls, 95)),
            probability_of_ruin=float(ruined.mean()),
            probability_of_loss=float((final_bankrolls < initial_bankroll).mean()),
            probability_of_profit=float((final_bankrolls > initial_bankroll).mean()),
            probability_double=float((final_bankrolls >= 2 * initial_bankroll).mean()),
            expected_roi=float((final_bankrolls.mean() - initial_bankroll) / initial_bankroll),
            expected_growth_rate=float(
                np.log(np.maximum(final_bankrolls, 0.01) / initial_bankroll).mean()
            ),
            mean_max_drawdown=float(max_drawdowns.mean()),
            median_max_drawdown=float(np.median(max_drawdowns)),
            worst_max_drawdown=float(max_drawdowns.max()),
            mean_time_to_ruin=float(np.mean(time_to_ruin_list)) if time_to_ruin_list else None,
            sample_paths=sample_paths,
        )
    
    def optimal_kelly_fraction(self) -> float:
        """
        Calculate the optimal Kelly fraction.
        
        Kelly = (bp - q) / b
        where b = payout, p = win prob, q = 1 - p
        """
        p = self.win_prob
        q = 1 - p
        b = self.win_payout
        
        kelly = (b * p - q) / b
        return max(0.0, kelly)
    
    def calculate_probability_of_ruin_analytical(
        self,
        initial_bankroll: float,
        bet_size: float,
    ) -> float:
        """
        Analytical approximation of probability of ruin for flat betting.
        
        Uses the Gambler's Ruin formula for biased random walk.
        """
        if bet_size <= 0 or initial_bankroll <= 0:
            return 1.0
        
        p = self.win_prob
        q = 1 - p
        
        # Number of "units" we start with
        n_units = initial_bankroll / bet_size
        
        if abs(p - q) < 0.001:
            # Fair game: P(ruin) = 1 (always go bust eventually)
            return 1.0
        
        if p > q:
            # Favorable game
            # P(ruin) = (q/p)^n
            ruin_prob = (q / p) ** n_units
        else:
            # Unfavorable game: P(ruin) = 1
            ruin_prob = 1.0
        
        return min(1.0, max(0.0, ruin_prob))
    
    def run_sensitivity_analysis(
        self,
        n_simulations: int = 5000,
        n_bets: int = 500,
        initial_bankroll: float = 1000.0,
        win_prob_range: Tuple[float, float] = (0.48, 0.58),
        n_prob_points: int = 11,
    ) -> List[Tuple[float, MonteCarloResults]]:
        """
        Run Monte Carlo for different win probabilities.
        
        Useful for understanding how edge affects outcomes.
        
        Returns:
            List of (win_prob, MonteCarloResults) tuples
        """
        results = []
        
        for win_prob in np.linspace(win_prob_range[0], win_prob_range[1], n_prob_points):
            self.win_prob = win_prob
            mc_result = self.simulate_flat_betting(
                n_simulations=n_simulations,
                n_bets=n_bets,
                initial_bankroll=initial_bankroll,
                bet_size=initial_bankroll * 0.01,  # 1% of bankroll
            )
            results.append((float(win_prob), mc_result))
            
            logger.info(
                f"Win prob {win_prob:.1%}: "
                f"Expected ROI = {mc_result.expected_roi:.1%}, "
                f"P(ruin) = {mc_result.probability_of_ruin:.1%}"
            )
        
        return results
