"""
Hybrid Walk-Forward Backtesting Engine.

Implements expanding window with recency weighting for temporal validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.backtesting.data_loader import MARKET_CONFIGS, MarketConfig, StrictModeViolation
from src.prediction.confidence import calculate_confidence_from_binary_probability

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Single prediction with outcome."""
    date: pd.Timestamp
    home_team: str
    away_team: str
    market: str
    period: str
    line: Optional[float]
    odds: Optional[int]  # Actual American odds for the bet side
    model_prob: float
    prediction: int  # 1 = home/over, 0 = away/under
    actual: int
    correct: bool
    profit: float  # Calculated using actual odds
    confidence: float
    edge: Optional[float]  # vs implied probability
    kelly_fraction: Optional[float]
    implied_prob: Optional[float] = None  # Implied probability from odds


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation."""
    min_train_games: int = 500  # Minimum training set size
    test_chunk_size: int = 50  # Games per test chunk
    recency_weight_halflife: int = 100  # Games for 50% weight decay
    retrain_frequency: int = 50  # Retrain every N games
    use_sample_weights: bool = True  # Use recency weighting
    

class WalkForwardEngine:
    """
    Hybrid walk-forward backtesting engine.
    
    Features:
    - Expanding window (uses all historical data up to test point)
    - Recency weighting (more recent games weighted higher)
    - Periodic retraining (configurable frequency)
    - NO data leakage (strict temporal ordering)
    """
    
    def __init__(
        self,
        config: Optional[WalkForwardConfig] = None,
        feature_builder: Optional[Callable] = None,
        model_factory: Optional[Callable] = None,
    ):
        """
        Initialize walk-forward engine.
        
        Args:
            config: Walk-forward configuration
            feature_builder: Function to build features from game data
            model_factory: Factory function to create new model instances
        """
        self.config = config or WalkForwardConfig()
        self.feature_builder = feature_builder
        self.model_factory = model_factory
        
        self._current_model = None
        self._last_train_idx = 0
    
    def compute_sample_weights(
        self,
        train_dates: pd.Series,
        current_date: pd.Timestamp,
    ) -> np.ndarray:
        """
        Compute recency-based sample weights.
        
        More recent games get higher weight using exponential decay.
        Halflife = number of games where weight drops to 50%.
        
        Args:
            train_dates: Series of training game dates
            current_date: Current prediction date
            
        Returns:
            Normalized sample weights
        """
        if not self.config.use_sample_weights:
            return np.ones(len(train_dates)) / len(train_dates)
        
        # Calculate days ago for each game
        days_ago = (current_date - train_dates).dt.days
        
        # Convert games to approximate days (avg 2.5 days between games)
        halflife_days = self.config.recency_weight_halflife * 2.5
        
        # Exponential decay
        weights = 0.5 ** (days_ago / halflife_days)
        
        # Handle negative days (shouldn't happen but be safe)
        weights = np.clip(weights, 0.01, 1.0)
        
        # Normalize to sum to 1
        weights = weights / weights.sum()
        
        return weights.values
    
    def should_retrain(self, current_idx: int) -> bool:
        """Check if model should be retrained at this point."""
        games_since_train = current_idx - self._last_train_idx
        return games_since_train >= self.config.retrain_frequency
    
    def _get_bet_odds(
        self,
        row: pd.Series,
        market_config: MarketConfig,
        prediction: int,
    ) -> Optional[int]:
        """
        Get actual odds for the bet from the data.
        
        NO FALLBACKS - returns None if odds not available.
        
        Args:
            row: Game data row
            market_config: Market configuration
            prediction: 1 = home/over, 0 = away/under
            
        Returns:
            American odds (e.g., -110, +150) or None if unavailable
        """
        period = market_config.period  # "fg" or "1h"
        market_type = market_config.market_type
        
        if market_type == "moneyline":
            # Moneyline: use actual ML odds from data
            if period == "fg":
                home_col, away_col = "fg_ml_home", "fg_ml_away"
            else:
                home_col, away_col = "1h_ml_home", "1h_ml_away"
            
            if prediction == 1:  # Betting home
                odds = row.get(home_col)
            else:  # Betting away
                odds = row.get(away_col)
            
            if pd.isna(odds):
                return None
            return int(odds)
        
        elif market_type == "spread":
            # Use user-configured juice for spread bets
            # This is NOT an assumption - user explicitly provides this value
            configured = getattr(self, '_configured_juice', None)
            # Return user-configured juice, or None if not provided
            return configured
        
        elif market_type == "total":
            # Use user-configured juice for total bets
            # This is NOT an assumption - user explicitly provides this value
            configured = getattr(self, '_configured_juice', None)
            # Return user-configured juice, or None if not provided
            return configured
        
        return None
    
    def _american_to_implied(self, odds: int) -> float:
        """Convert American odds to implied probability."""
        if odds < 0:
            return abs(odds) / (abs(odds) + 100)
        else:
            return 100 / (odds + 100)
    
    def _calculate_profit(self, correct: bool, odds: Optional[int], market_type: str = "") -> float:
        """
        Calculate profit using actual odds.
        
        Args:
            correct: Whether the bet won
            odds: American odds (e.g., -110, +150)
            market_type: Type of market for logging
            
        Returns:
            Profit in units (positive = win, negative = loss)
        """
        
        if odds is None:
            # NO FALLBACK - if no odds, profit is 0 (no bet placed)
            return 0.0
        
        if not correct:
            return -1.0  # Lose 1 unit on any loss
        
        # Calculate win payout based on odds
        if odds < 0:
            # Favorite: risk more to win less
            # e.g., -150 means risk $150 to win $100
            # So 1 unit bet wins: 100 / abs(odds)
            profit = 100.0 / abs(odds)
        else:
            # Underdog: risk less to win more
            # e.g., +150 means risk $100 to win $150
            # So 1 unit bet wins: odds / 100
            profit = odds / 100.0
        
        return profit
    
    def _calculate_kelly(self, win_prob: float, odds: int) -> float:
        """Calculate Kelly fraction for optimal bet sizing."""
        if odds < 0:
            decimal_odds = 1 + (100 / abs(odds))
        else:
            decimal_odds = 1 + (odds / 100)
        
        b = decimal_odds - 1
        p = win_prob
        q = 1 - p
        
        if b <= 0:
            return 0.0
        
        kelly = (b * p - q) / b
        return max(0.0, kelly)
    
    def run_backtest(
        self,
        df: pd.DataFrame,
        market_config: MarketConfig,
        feature_columns: List[str],
        model_class: type,
        model_kwargs: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
        configured_juice: Optional[int] = None,  # User-configured juice for spread/total
    ) -> List[PredictionResult]:
        """
        Run walk-forward backtest for a single market.
        
        Args:
            df: Prepared DataFrame with features and labels
            market_config: Configuration for target market
            feature_columns: List of feature column names
            model_class: Model class to instantiate
            model_kwargs: Additional kwargs for model
            verbose: Print progress
            configured_juice: User-configured juice/odds for spread/total markets (e.g., -110)
            
        Returns:
            List of prediction results
        """
        model_kwargs = model_kwargs or {}
        results: List[PredictionResult] = []
        self._configured_juice = configured_juice  # Store for use in _get_bet_odds
        
        # Ensure data is sorted by date
        df = df.sort_values("date").reset_index(drop=True)
        
        # Filter feature columns to only those that exist in data
        available_features = [f for f in feature_columns if f in df.columns]
        if len(available_features) < len(feature_columns):
            missing = [f for f in feature_columns if f not in df.columns]
            logger.warning(f"Missing {len(missing)} feature columns, using {len(available_features)} available")
        
        feature_columns = available_features
        
        if len(feature_columns) < 5:
            raise StrictModeViolation(
                f"Too few feature columns available: {len(feature_columns)}. Need at least 5."
            )
        
        label_col = market_config.label_column
        if label_col not in df.columns:
            raise StrictModeViolation(f"Missing label column: {label_col}")
        
        # Drop rows where any feature or label is NaN
        required_cols = feature_columns + [label_col, "date"]
        initial_count = len(df)
        df = df.dropna(subset=required_cols).reset_index(drop=True)
        if len(df) < initial_count:
            logger.info(f"Filtered {initial_count - len(df)} rows with NaN in features/labels")
        
        n_games = len(df)
        
        # Validate we have enough data
        if n_games < self.config.min_train_games + self.config.test_chunk_size:
            raise StrictModeViolation(
                f"Insufficient data for backtest: {n_games} games "
                f"(need at least {self.config.min_train_games + self.config.test_chunk_size})"
            )
        
        # Initialize model
        model = model_class(
            name=f"{market_config.market_key}_backtest",
            feature_columns=feature_columns,
            **model_kwargs
        )
        
        # Walk-forward loop
        self._last_train_idx = 0
        chunk_start = self.config.min_train_games
        
        while chunk_start < n_games:
            chunk_end = min(chunk_start + self.config.test_chunk_size, n_games)
            
            # Training data: all games before chunk
            train_df = df.iloc[:chunk_start].copy()
            test_df = df.iloc[chunk_start:chunk_end].copy()
            
            # Get current date for weighting
            current_date = test_df.iloc[0]["date"]
            
            # Compute sample weights
            sample_weights = self.compute_sample_weights(
                train_df["date"],
                current_date,
            )
            
            # Check if we should retrain
            if self.should_retrain(chunk_start) or model.is_fitted is False:
                if verbose and chunk_start % 500 == 0:
                    logger.info(
                        f"Training at game {chunk_start}/{n_games} "
                        f"(train size: {len(train_df)})"
                    )
                
                # Prepare training features
                X_train = train_df[feature_columns]
                y_train = train_df[label_col].astype(int)
                
                # Fit model with sample weights
                try:
                    if hasattr(model, "fit_weighted"):
                        model.fit_weighted(X_train, y_train, sample_weights)
                    else:
                        model.fit(X_train, y_train)
                except Exception as e:
                    logger.error(f"Model training failed at game {chunk_start}: {e}")
                    raise StrictModeViolation(f"Model training failed: {e}")
                
                self._last_train_idx = chunk_start
            
            # Predict on test chunk
            X_test = test_df[feature_columns]
            
            try:
                probas = model.predict_proba(X_test)[:, 1]  # P(home/over)
            except Exception as e:
                logger.error(f"Prediction failed at game {chunk_start}: {e}")
                raise StrictModeViolation(f"Prediction failed: {e}")
            
            # Generate predictions
            for i, (idx, row) in enumerate(test_df.iterrows()):
                prob = float(probas[i])
                pred = 1 if prob >= 0.5 else 0
                actual = int(row[label_col])
                correct = pred == actual
                
                # Confidence = calibrated probability of the predicted side
                bet_prob = prob if pred == 1 else (1 - prob)
                confidence = calculate_confidence_from_binary_probability(bet_prob)
                
                # Get line if available
                line = None
                if market_config.line_column and market_config.line_column in row.index:
                    line = row[market_config.line_column]
                    if pd.isna(line):
                        line = None
                
                # Get actual odds for profit calculation
                # CRITICAL: Use actual market odds, NOT assumed -110
                odds = self._get_bet_odds(row, market_config, pred)
                implied_prob = self._american_to_implied(odds) if odds else None
                
                # Calculate profit using ACTUAL odds
                profit = self._calculate_profit(correct, odds, market_config.market_type)
                
                # Edge vs market
                edge = None
                if implied_prob is not None:
                    bet_prob = prob if pred == 1 else (1 - prob)
                    edge = bet_prob - implied_prob
                
                # Kelly fraction
                kelly = None
                if odds is not None and profit != 0:
                    kelly = self._calculate_kelly(prob if pred == 1 else (1 - prob), odds)
                
                results.append(PredictionResult(
                    date=row["date"],
                    home_team=row.get("home_team", ""),
                    away_team=row.get("away_team", ""),
                    market=market_config.market_key,
                    period=market_config.period,
                    line=float(line) if line is not None else None,
                    odds=int(odds) if odds is not None else None,
                    model_prob=prob,
                    prediction=pred,
                    actual=actual,
                    correct=correct,
                    profit=profit,
                    confidence=confidence,
                    edge=edge,
                    kelly_fraction=kelly,
                    implied_prob=implied_prob,
                ))
            
            chunk_start = chunk_end
        
        if verbose:
            n_correct = sum(1 for r in results if r.correct)
            accuracy = n_correct / len(results) if results else 0
            total_profit = sum(r.profit for r in results)
            logger.info(
                f"{market_config.market_key}: {len(results)} predictions, "
                f"{accuracy:.1%} accuracy, {total_profit:+.1f} units profit"
            )
        
        return results
    
    def run_all_markets(
        self,
        df: pd.DataFrame,
        markets: List[str],
        feature_columns_by_market: Dict[str, List[str]],
        model_class: type,
        model_kwargs: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
    ) -> Dict[str, List[PredictionResult]]:
        """
        Run walk-forward backtest for multiple markets.
        
        Args:
            df: Prepared DataFrame with all features and labels
            markets: List of market keys to backtest
            feature_columns_by_market: Feature columns for each market
            model_class: Model class to instantiate
            model_kwargs: Additional kwargs for model
            verbose: Print progress
            
        Returns:
            Dict mapping market key to list of prediction results
        """
        all_results: Dict[str, List[PredictionResult]] = {}
        
        for market_key in markets:
            if market_key not in MARKET_CONFIGS:
                logger.warning(f"Unknown market: {market_key}, skipping")
                continue
            
            config = MARKET_CONFIGS[market_key]
            feature_cols = feature_columns_by_market.get(market_key, [])
            
            if not feature_cols:
                logger.warning(f"No features defined for {market_key}, skipping")
                continue
            
            # Filter to valid data for this market
            market_df = df[df[config.label_column].notna()].copy()
            if config.line_column:
                market_df = market_df[market_df[config.line_column].notna()]
            
            if len(market_df) < self.config.min_train_games + 100:
                logger.warning(
                    f"Insufficient data for {market_key}: {len(market_df)} games, skipping"
                )
                continue
            
            try:
                results = self.run_backtest(
                    df=market_df,
                    market_config=config,
                    feature_columns=feature_cols,
                    model_class=model_class,
                    model_kwargs=model_kwargs,
                    verbose=verbose,
                )
                all_results[market_key] = results
            except Exception as e:
                logger.error(f"Backtest failed for {market_key}: {e}")
                if isinstance(e, StrictModeViolation):
                    raise
        
        return all_results


def results_to_dataframe(results: List[PredictionResult]) -> pd.DataFrame:
    """Convert prediction results to DataFrame."""
    if not results:
        return pd.DataFrame()
    
    records = []
    for r in results:
        records.append({
            "date": r.date,
            "home_team": r.home_team,
            "away_team": r.away_team,
            "market": r.market,
            "period": r.period,
            "line": r.line,
            "odds": r.odds,
            "implied_prob": r.implied_prob,
            "model_prob": r.model_prob,
            "prediction": r.prediction,
            "actual": r.actual,
            "correct": int(r.correct),
            "profit": r.profit,
            "confidence": r.confidence,
            "edge": r.edge,
            "kelly_fraction": r.kelly_fraction,
        })
    
    return pd.DataFrame(records)
