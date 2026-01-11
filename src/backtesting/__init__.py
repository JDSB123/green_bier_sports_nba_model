"""
NBA Backtesting Engine v33.0.15.0

Enterprise-grade backtesting framework for NBA prediction models.
Supports 6 independent markets (FG/1H for Spreads, Totals, Moneylines).

Key Features:
- Hybrid walk-forward validation with recency weighting
- Enterprise statistical validation (bootstrap CI, hypothesis tests, Sharpe, drawdown, Kelly)
- Monte Carlo simulation for bankroll projections
- NO silent fallbacks - fails loudly on bad data
- Reproducible outputs with full audit trail
"""

from src.backtesting.engine import BacktestEngine
from src.backtesting.data_loader import (
    BacktestDataLoader,
    StrictModeViolation,
    DataValidationError,
)
from src.backtesting.walk_forward import WalkForwardEngine
from src.backtesting.statistical_tests import StatisticalValidator
from src.backtesting.performance_metrics import PerformanceMetrics, MarketPerformance
from src.backtesting.monte_carlo import MonteCarloSimulator, MonteCarloResults
from src.backtesting.report_generator import ReportGenerator

__all__ = [
    "BacktestEngine",
    "BacktestDataLoader",
    "StrictModeViolation",
    "DataValidationError",
    "WalkForwardEngine",
    "StatisticalValidator",
    "PerformanceMetrics",
    "MarketPerformance",
    "MonteCarloSimulator",
    "MonteCarloResults",
    "ReportGenerator",
]

__version__ = "33.0.15.0"
