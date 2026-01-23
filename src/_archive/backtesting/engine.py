"""
Core backtest engine orchestrating all components.

This is the main entry point for running backtests.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import numpy as np
import pandas as pd

from src.backtesting.data_loader import (
    BacktestDataLoader,
    MARKET_CONFIGS,
    MarketConfig,
    StrictModeViolation,
)
from src.backtesting.walk_forward import (
    WalkForwardEngine,
    WalkForwardConfig,
    PredictionResult,
    results_to_dataframe,
)
from src.backtesting.performance_metrics import PerformanceMetrics, MarketPerformance
from src.backtesting.statistical_tests import StatisticalValidator, StatisticalSummary
from src.backtesting.monte_carlo import MonteCarloSimulator, MonteCarloResults
from src.backtesting.report_generator import ReportGenerator

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for a complete backtest run."""
    
    # Data
    data_path: str = "data/processed/training_data.csv"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    
    # Markets
    markets: List[str] = None  # None = all available
    
    # Walk-forward
    min_train_games: int = 500
    test_chunk_size: int = 50
    recency_weight_halflife: int = 100
    retrain_frequency: int = 50
    
    # Model
    model_type: str = "logistic"  # "logistic" or "gradient_boosting"
    use_calibration: bool = True
    
    # Filtering
    min_confidence: float = 0.0  # Minimum confidence to count as a bet
    min_edge: float = 0.0  # Minimum edge vs implied probability
    
    # Monte Carlo
    run_monte_carlo: bool = True
    monte_carlo_simulations: int = 10000
    
    # Output
    output_dir: str = "data/backtest_results"
    version: str = "33.0.15.0"
    
    # Juice/odds configuration (EXPLICIT - no hidden defaults)
    spread_juice: Optional[int] = None  # e.g., -110
    total_juice: Optional[int] = None  # e.g., -110
    
    def __post_init__(self):
        if self.markets is None:
            self.markets = list(MARKET_CONFIGS.keys())


class BacktestEngine:
    """
    Enterprise-grade backtesting engine.
    
    Orchestrates data loading, walk-forward validation, statistical testing,
    Monte Carlo simulation, and report generation.
    """
    
    def __init__(self, config: BacktestConfig):
        """
        Initialize backtest engine.
        
        Args:
            config: Backtest configuration
        """
        self.config = config
        self.data_loader: Optional[BacktestDataLoader] = None
        self.walk_forward: Optional[WalkForwardEngine] = None
        self.validator: Optional[StatisticalValidator] = None
        
        # Results storage
        self.predictions: Dict[str, List[PredictionResult]] = {}
        self.performances: Dict[str, MarketPerformance] = {}
        self.statistics: Dict[str, StatisticalSummary] = {}
        self.monte_carlo_results: Dict[str, MonteCarloResults] = {}
        
        # Create output directory
        self.output_dir = Path(config.output_dir) / config.version
        self.run_dir: Optional[Path] = None
    
    def run(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Run complete backtest pipeline.
        
        Args:
            verbose: Print progress updates
            
        Returns:
            Dict with all results and report paths
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.run_dir = self.output_dir / timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting backtest run: {self.run_dir}")
        
        # Step 1: Load and validate data
        if verbose:
            logger.info("Step 1: Loading data...")
        
        self._load_data()
        
        # Step 2: Run walk-forward backtest for each market
        if verbose:
            logger.info("Step 2: Running walk-forward backtest...")
        
        self._run_walk_forward(verbose)
        
        # Step 3: Calculate performance metrics
        if verbose:
            logger.info("Step 3: Calculating performance metrics...")
        
        self._calculate_metrics()
        
        # Step 4: Statistical validation
        if verbose:
            logger.info("Step 4: Running statistical validation...")
        
        self._run_statistical_tests()
        
        # Step 5: Monte Carlo simulation
        if self.config.run_monte_carlo:
            if verbose:
                logger.info("Step 5: Running Monte Carlo simulations...")
            self._run_monte_carlo()
        
        # Step 6: Generate reports
        if verbose:
            logger.info("Step 6: Generating reports...")
        
        report_paths = self._generate_reports()
        
        # Summary
        if verbose:
            self._print_summary()
        
        return {
            "run_dir": str(self.run_dir),
            "predictions": {k: len(v) for k, v in self.predictions.items()},
            "performances": self.performances,
            "statistics": self.statistics,
            "monte_carlo": self.monte_carlo_results,
            "reports": report_paths,
        }
    
    def _load_data(self) -> None:
        """Load and validate training data."""
        self.data_loader = BacktestDataLoader(
            data_path=self.config.data_path,
            strict_mode=True,
        )
        
        df = self.data_loader.load()
        
        # Log quality report
        quality = self.data_loader.get_quality_report()
        logger.info(f"Loaded {quality.total_games} games ({quality.date_range[0]} to {quality.date_range[1]})")
        
        for warning in quality.warnings:
            logger.warning(warning)
    
    def _get_feature_columns(self, market_key: str) -> List[str]:
        """Get feature columns for a market based on available data."""
        config = MARKET_CONFIGS[market_key]
        
        # Use features that exist in the training data.
        # Feature sets are derived from the canonical processed dataset.
        
        # Rolling stat features (5g, 10g, 20g windows)
        rolling_features = []
        for window in ["5g", "10g", "20g"]:
            for side in ["home", "away"]:
                rolling_features.extend([
                    f"{side}_{window}_score",
                    f"{side}_{window}_1h",
                    f"{side}_{window}_efg_pct",
                    f"{side}_{window}_off_rtg",
                    f"{side}_{window}_def_rtg",
                    f"{side}_{window}_net_rtg",
                ])
        
        # Differential features
        diff_features = [
            "diff_5g_score", "diff_5g_1h",
            "diff_10g_score", "diff_10g_1h",
            "diff_20g_score", "diff_20g_1h",
            "diff_5g_off_rtg", "diff_5g_def_rtg", "diff_5g_net_rtg",
            "diff_10g_off_rtg", "diff_10g_def_rtg", "diff_10g_net_rtg",
        ]
        
        # Rest and situational
        situational_features = [
            "home_rest", "away_rest", "rest_adv",
            "home_b2b", "away_b2b",
            "home_streak", "away_streak", "streak_diff",
        ]
        
        # ELO features
        elo_features = [
            "home_elo", "away_elo", "elo_diff", "elo_prob",
        ]
        
        # Efficiency features
        efficiency_features = [
            "home_off_rtg", "home_def_rtg", "home_net_rtg",
            "away_off_rtg", "away_def_rtg", "away_net_rtg",
            "home_efg_pct", "away_efg_pct",
        ]
        
        base_features = (
            rolling_features + diff_features + situational_features + 
            elo_features + efficiency_features
        )
        
        # Market-specific features
        if config.market_type == "spread":
            if config.period == "1h":
                market_features = [
                    "1h_spread_line",
                    "home_5g_1h", "away_5g_1h",
                    "home_10g_1h", "away_10g_1h",
                    "diff_5g_1h", "diff_10g_1h",
                ]
            else:
                market_features = [
                    "fg_spread_line",
                    "open_spread", "close_spread", "spread_move",
                ]
            
        elif config.market_type == "total":
            if config.period == "1h":
                market_features = [
                    "1h_total_line",
                    "home_5g_1h", "away_5g_1h",
                    "home_10g_1h", "away_10g_1h",
                ]
            else:
                market_features = [
                    "fg_total_line",
                    "open_total", "close_total", "total_move",
                    "pace",
                ]
            
        else:  # moneyline
            market_features = [
                "fg_ml_home", "fg_ml_away" if config.period == "fg" else "1h_ml_home",
            ]
        
        return base_features + market_features
    
    def _get_model_class(self, market_key: str) -> Type:
        """Get appropriate model class for a market."""
        from src.modeling.models import (
            SpreadsModel,
            TotalsModel,
            FirstHalfSpreadsModel,
            FirstHalfTotalsModel,
        )
        
        config = MARKET_CONFIGS[market_key]
        
        if config.period == "fg":
            if config.market_type == "spread":
                return SpreadsModel
            elif config.market_type == "total":
                return TotalsModel
            else:
                return SpreadsModel  # Use spreads model for moneyline
        else:  # 1h
            if config.market_type == "spread":
                return FirstHalfSpreadsModel
            elif config.market_type == "total":
                return FirstHalfTotalsModel
            else:
                return FirstHalfSpreadsModel  # Use spreads model for moneyline
    
    def _run_walk_forward(self, verbose: bool) -> None:
        """Run walk-forward backtest for each market."""
        wf_config = WalkForwardConfig(
            min_train_games=self.config.min_train_games,
            test_chunk_size=self.config.test_chunk_size,
            recency_weight_halflife=self.config.recency_weight_halflife,
            retrain_frequency=self.config.retrain_frequency,
        )
        
        self.walk_forward = WalkForwardEngine(config=wf_config)
        
        for market_key in self.config.markets:
            if market_key not in MARKET_CONFIGS:
                logger.warning(f"Unknown market: {market_key}, skipping")
                continue
            
            config = MARKET_CONFIGS[market_key]
            
            try:
                # Get market-specific data
                df = self.data_loader.get_market_data(
                    market_key,
                    start_date=self.config.start_date,
                    end_date=self.config.end_date,
                )
                
                if len(df) < self.config.min_train_games + 100:
                    logger.warning(f"Insufficient data for {market_key}: {len(df)} games")
                    continue
                
                # Get features and filter to available columns
                feature_cols = self._get_feature_columns(market_key)
                available_features = [f for f in feature_cols if f in df.columns]
                
                if len(available_features) < 5:
                    logger.warning(f"Too few features for {market_key}: {len(available_features)}")
                    continue
                
                # Get model class
                model_class = self._get_model_class(market_key)
                
                # Determine juice for this market type
                if config.market_type == "spread":
                    market_juice = self.config.spread_juice
                elif config.market_type == "total":
                    market_juice = self.config.total_juice
                else:
                    market_juice = None  # Moneyline uses real odds from data
                
                # Run backtest
                results = self.walk_forward.run_backtest(
                    df=df,
                    market_config=config,
                    feature_columns=available_features,
                    model_class=model_class,
                    model_kwargs={
                        "model_type": self.config.model_type,
                        "use_calibration": self.config.use_calibration,
                    },
                    verbose=verbose,
                    configured_juice=market_juice,  # Explicit user-configured juice
                )

                # Apply optional confidence/edge filters to simulate betting thresholds.
                if self.config.min_confidence > 0.0 or self.config.min_edge > 0.0:
                    original_count = len(results)
                    filtered = []
                    for r in results:
                        if self.config.min_confidence > 0.0 and r.confidence < self.config.min_confidence:
                            continue
                        if self.config.min_edge > 0.0:
                            if r.edge is None or r.edge < self.config.min_edge:
                                continue
                        filtered.append(r)
                    results = filtered
                    if verbose:
                        logger.info(
                            f"{market_key}: filtered to {len(results)}/{original_count} bets "
                            f"(min_confidence={self.config.min_confidence}, min_edge={self.config.min_edge})"
                        )
                
                self.predictions[market_key] = results
                
            except Exception as e:
                logger.error(f"Backtest failed for {market_key}: {e}")
                if isinstance(e, StrictModeViolation):
                    raise
    
    def _calculate_metrics(self) -> None:
        """Calculate performance metrics for each market."""
        for market_key, results in self.predictions.items():
            if not results:
                continue
            
            returns = [r.profit for r in results]
            predictions = [r.prediction for r in results]
            actuals = [r.actual for r in results]
            confidences = [r.confidence for r in results]
            
            metrics = PerformanceMetrics(
                returns=returns,
                predictions=predictions,
                actuals=actuals,
                confidences=confidences,
            )
            
            self.performances[market_key] = metrics.calculate_all(market=market_key)
    
    def _run_statistical_tests(self) -> None:
        """Run statistical validation for each market."""
        self.validator = StatisticalValidator()
        
        for market_key, results in self.predictions.items():
            if not results:
                continue
            
            wins = np.array([1 if r.correct else 0 for r in results])
            returns = np.array([r.profit for r in results])
            
            self.statistics[market_key] = self.validator.validate_market(
                wins=wins,
                returns=returns,
                market=market_key,
            )
    
    def _run_monte_carlo(self) -> None:
        """Run Monte Carlo simulations for each market."""
        for market_key, perf in self.performances.items():
            if perf.n_bets < 100:
                continue
            
            win_rate = perf.accuracy
            
            simulator = MonteCarloSimulator(
                win_probability=win_rate,
                random_seed=42,
            )
            
            # Simulate flat betting
            mc_results = simulator.simulate_flat_betting(
                n_simulations=self.config.monte_carlo_simulations,
                n_bets=min(perf.n_bets, 1000),
                initial_bankroll=1000.0,
                bet_size=10.0,
            )
            
            self.monte_carlo_results[market_key] = mc_results
    
    def _generate_reports(self) -> Dict[str, Path]:
        """Generate all reports."""
        generator = ReportGenerator(
            output_dir=self.run_dir,
            version=self.config.version,
        )
        
        # Combine all predictions into DataFrame
        all_results = []
        for results in self.predictions.values():
            all_results.extend(results)
        
        predictions_df = results_to_dataframe(all_results)
        
        return generator.generate_full_report(
            market_performances=self.performances,
            statistical_summaries=self.statistics,
            monte_carlo_results=self.monte_carlo_results if self.config.run_monte_carlo else None,
            predictions_df=predictions_df,
            config={
                "data_path": self.config.data_path,
                "start_date": self.config.start_date,
                "end_date": self.config.end_date,
                "markets": self.config.markets,
                "min_train_games": self.config.min_train_games,
                "model_type": self.config.model_type,
            },
        )
    
    def _print_summary(self) -> None:
        """Print summary of results."""
        print("\n" + "=" * 60)
        print("BACKTEST SUMMARY")
        print("=" * 60)
        
        for market_key, perf in self.performances.items():
            stat = self.statistics.get(market_key)
            sig = "SIG" if stat and stat.is_statistically_significant else "N/S"
            
            print(f"\n{market_key.upper()}")
            print(f"  Bets: {perf.n_bets}")
            print(f"  Accuracy: {perf.accuracy:.1%}")
            print(f"  ROI: {perf.roi:+.1%}")
            print(f"  Sharpe: {perf.sharpe_ratio:.2f}")
            print(f"  Status: {sig}")
            
            if stat:
                print(f"  Recommendation: {stat.recommendation[:50]}...")
        
        print("\n" + "=" * 60)
        print(f"Reports saved to: {self.run_dir}")
        print("=" * 60 + "\n")
