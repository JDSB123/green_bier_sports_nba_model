"""Time-aware backtesting with proper temporal splits.

Implements walk-forward validation where:
- Training set contains only games before a cutoff date
- Test set contains games after the cutoff
- No future information leakage
"""
import os
import sys
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import settings
from src.modeling.models import SpreadsModel, TotalsModel, MoneylineModel


def time_based_split(df: pd.DataFrame, test_size_days: int = 90):
    """Split data by time, training on earlier games, testing on later."""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    cutoff = df['date'].max() - timedelta(days=test_size_days)
    train = df[df['date'] < cutoff]
    test = df[df['date'] >= cutoff]
    
    return train, test


def rolling_origin_cv(df: pd.DataFrame, n_splits: int = 10):
    """
    Perform rolling-origin cross-validation maintaining temporal order.

    Args:
        df: DataFrame with temporal data (must have 'date' column)
        n_splits: Number of CV folds (default 10, increased from 5 for better validation)

    Returns:
        List of fold results with train/test splits
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(df)):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        
        train_dates = (train['date'].min(), train['date'].max())
        test_dates = (test['date'].min(), test['date'].max())
        
        results.append({
            'fold': fold_idx + 1,
            'train_size': len(train),
            'test_size': len(test),
            'train_dates': train_dates,
            'test_dates': test_dates,
            'train': train,
            'test': test
        })
    
    return results


def backtest_model(model_class, df: pd.DataFrame, target_col: str, model_name: str):
    """Run time-aware backtest for a model."""
    print(f"\n{'='*60}")
    print(f"Backtesting {model_name}")
    print(f"{'='*60}")
    
    train, test = time_based_split(df)
    
    print(f"Train period: {train['date'].min()} to {train['date'].max()}")
    print(f"Test period:  {test['date'].min()} to {test['date'].max()}")
    print(f"Train size: {len(train)} | Test size: {len(test)}")
    
    # Filter to rows with target
    train_valid = train.dropna(subset=[target_col])
    test_valid = test.dropna(subset=[target_col])
    
    if len(train_valid) < 100:
        print(f"⚠ Insufficient training data ({len(train_valid)} rows)")
        return None
    
    # Train model
    model = model_class()
    model.fit(train_valid, train_valid[target_col])
    
    # Evaluate
    train_metrics = model.evaluate(train_valid, train_valid[target_col])
    test_metrics = model.evaluate(test_valid, test_valid[target_col])
    
    print(f"\nTrain Accuracy: {train_metrics.accuracy:.1%}")
    print(f"Test Accuracy:  {test_metrics.accuracy:.1%}")
    
    return {
        'model_name': model_name,
        'train_acc': train_metrics.accuracy,
        'test_acc': test_metrics.accuracy,
        'train_size': len(train_valid),
        'test_size': len(test_valid)
    }


def main():
    print("TIME-AWARE BACKTESTING")
    print("="*60)
    
    data_path = os.path.join(settings.data_processed_dir, "training_data_fh.csv")
    df = pd.read_csv(data_path)
    
    print(f"\nDataset: {len(df)} games")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    results = []
    
    # Backtest full-game models
    if 'spread_covered' in df.columns:
        r = backtest_model(SpreadsModel, df, 'spread_covered', 'Spreads')
        if r:
            results.append(r)
    
    if 'went_over' in df.columns:
        r = backtest_model(TotalsModel, df, 'went_over', 'Totals')
        if r:
            results.append(r)
    
    if 'home_score' in df.columns and 'away_score' in df.columns:
        df_ml = df.copy()
        df_ml['home_win'] = (df_ml['home_score'] > df_ml['away_score']).astype(int)
        r = backtest_model(MoneylineModel, df_ml, 'home_win', 'Moneyline')
        if r:
            results.append(r)
    
    # Summary
    print(f"\n{'='*60}")
    print("BACKTEST SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(f"{r['model_name']:15} | Train: {r['train_acc']:.1%} | Test: {r['test_acc']:.1%}")
    
    print("\n✓ Time-aware backtesting complete (no future leakage)")


if __name__ == "__main__":
    main()
