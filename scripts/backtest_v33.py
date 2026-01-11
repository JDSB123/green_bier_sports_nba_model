#!/usr/bin/env python3
"""
NBA Backtest Engine v33.0.15.0

Enterprise-grade backtesting for NBA prediction models.

Features:
- 6 independent markets (FG/1H for Spreads, Totals, Moneylines)
- Hybrid walk-forward validation with recency weighting
- Enterprise statistical validation (bootstrap CI, hypothesis tests, Sharpe, drawdown, Kelly)
- Monte Carlo simulation for bankroll projections
- NO silent fallbacks - fails loudly on bad data
- Reproducible outputs with full audit trail

Usage:
    # Run all markets, full backtest
    python scripts/backtest_v33.py
    
    # Specific markets
    python scripts/backtest_v33.py --markets fg_spread,1h_total
    
    # Date range
    python scripts/backtest_v33.py --start 2024-01-01 --end 2025-06-01
    
    # Custom thresholds
    python scripts/backtest_v33.py --min-confidence 0.55 --min-edge 0.03
    
    # Monte Carlo simulations
    python scripts/backtest_v33.py --monte-carlo 10000
    
    # Quick validation mode (fewer games)
    python scripts/backtest_v33.py --quick

Output:
    data/backtest_results/v33.0.15.0/YYYY-MM-DD_HHMMSS/
        predictions.csv          # All predictions with outcomes
        backtest_summary.json    # Aggregated metrics
        backtest_report.html     # Visual report
        backtest_report.md       # Markdown summary
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtesting.engine import BacktestEngine, BacktestConfig
from src.backtesting.data_loader import MARKET_CONFIGS, StrictModeViolation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="NBA Backtest Engine v33.0.15.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Data options
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/training_data_complete_2023.csv",
        help="Path to training data CSV (default: data/processed/training_data_complete_2023.csv)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD)",
    )
    
    # Market selection
    parser.add_argument(
        "--markets",
        type=str,
        default="all",
        help=(
            "Comma-separated markets or 'all' (default: all). "
            f"Available: {', '.join(MARKET_CONFIGS.keys())}"
        ),
    )
    
    # Walk-forward options
    parser.add_argument(
        "--min-train",
        type=int,
        default=500,
        help="Minimum training games before first prediction (default: 500)",
    )
    parser.add_argument(
        "--test-chunk",
        type=int,
        default=50,
        help="Games per test chunk (default: 50)",
    )
    parser.add_argument(
        "--retrain-freq",
        type=int,
        default=50,
        help="Retrain model every N games (default: 50)",
    )
    parser.add_argument(
        "--halflife",
        type=int,
        default=100,
        help="Recency weight halflife in games (default: 100)",
    )
    
    # Model options
    parser.add_argument(
        "--model-type",
        type=str,
        default="logistic",
        choices=["logistic", "gradient_boosting"],
        help="Model type (default: logistic)",
    )
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Disable probability calibration",
    )
    
    # Filter options
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Minimum confidence threshold (default: 0.0)",
    )
    parser.add_argument(
        "--min-edge",
        type=float,
        default=0.0,
        help="Minimum edge vs implied probability (default: 0.0)",
    )
    
    # Spread/Total juice configuration (REQUIRED - no hidden assumptions)
    parser.add_argument(
        "--spread-juice",
        type=int,
        default=None,
        help="American odds for spread bets (e.g., -110). REQUIRED for spread markets - no default assumed.",
    )
    parser.add_argument(
        "--total-juice",
        type=int,
        default=None,
        help="American odds for total bets (e.g., -110). REQUIRED for total markets - no default assumed.",
    )
    
    # Monte Carlo options
    parser.add_argument(
        "--monte-carlo",
        type=int,
        default=10000,
        help="Number of Monte Carlo simulations (default: 10000, 0 to disable)",
    )
    
    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/backtest_results",
        help="Output directory (default: data/backtest_results)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="33.0.15.0",
        help="Version tag for output (default: 33.0.15.0)",
    )
    
    # Quick mode
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick validation mode (reduced min-train, fewer markets)",
    )
    
    # Verbosity
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Quiet mode (errors only)",
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Configure log level
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Print banner
    print("=" * 60)
    print("NBA BACKTEST ENGINE v33.0.15.0")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Parse markets
    if args.markets.lower() == "all":
        markets = list(MARKET_CONFIGS.keys())
    else:
        markets = [m.strip() for m in args.markets.split(",")]
        # Validate markets
        unknown = [m for m in markets if m not in MARKET_CONFIGS]
        if unknown:
            logger.error(f"Unknown markets: {unknown}")
            logger.info(f"Available markets: {list(MARKET_CONFIGS.keys())}")
            return 1
    
    # Quick mode adjustments
    if args.quick:
        logger.info("Running in QUICK mode")
        args.min_train = 200
        args.test_chunk = 100
        args.monte_carlo = 1000
        # Only run FG markets in quick mode
        markets = [m for m in markets if m.startswith("fg_")]
    
    # Validate juice parameters for spread/total markets
    spread_markets = [m for m in markets if 'spread' in m]
    total_markets = [m for m in markets if 'total' in m]
    
    if spread_markets and args.spread_juice is None:
        logger.error("--spread-juice is REQUIRED for spread markets (e.g., --spread-juice -110)")
        logger.error("NO ASSUMPTIONS: You must explicitly specify the juice/odds for spread bets.")
        return 1
    
    if total_markets and args.total_juice is None:
        logger.error("--total-juice is REQUIRED for total markets (e.g., --total-juice -110)")
        logger.error("NO ASSUMPTIONS: You must explicitly specify the juice/odds for total bets.")
        return 1
    
    # Build configuration
    config = BacktestConfig(
        data_path=args.data,
        start_date=args.start,
        end_date=args.end,
        markets=markets,
        min_train_games=args.min_train,
        test_chunk_size=args.test_chunk,
        recency_weight_halflife=args.halflife,
        retrain_frequency=args.retrain_freq,
        model_type=args.model_type,
        use_calibration=not args.no_calibration,
        min_confidence=args.min_confidence,
        min_edge=args.min_edge,
        run_monte_carlo=args.monte_carlo > 0,
        monte_carlo_simulations=args.monte_carlo,
        output_dir=args.output_dir,
        version=args.version,
        spread_juice=args.spread_juice,
        total_juice=args.total_juice,
    )
    
    # Log configuration
    logger.info(f"Data: {config.data_path}")
    logger.info(f"Markets: {', '.join(config.markets)}")
    logger.info(f"Model: {config.model_type}")
    logger.info(f"Walk-forward: min_train={config.min_train_games}, chunk={config.test_chunk_size}")
    
    # Create and run engine
    engine = BacktestEngine(config)
    
    try:
        results = engine.run(verbose=not args.quiet)
    except StrictModeViolation as e:
        logger.error(f"STRICT MODE VIOLATION: {e}")
        logger.error("Fix data quality issues or adjust validation thresholds.")
        return 1
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Backtest failed: {e}")
        return 1
    
    # Print completion
    print()
    print("=" * 60)
    print("BACKTEST COMPLETE")
    print("=" * 60)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results: {results['run_dir']}")
    
    # Quick summary
    print()
    print("Market Summary:")
    print("-" * 40)
    for market_key, perf in results["performances"].items():
        sig = "SIG" if results["statistics"].get(market_key, {}) else "N/S"
        if hasattr(results["statistics"].get(market_key), "is_statistically_significant"):
            sig = "SIG" if results["statistics"][market_key].is_statistically_significant else "N/S"
        print(f"  {market_key:15s}: {perf.n_bets:4d} bets, {perf.accuracy:5.1%} acc, {perf.roi:+5.1%} ROI [{sig}]")
    
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
