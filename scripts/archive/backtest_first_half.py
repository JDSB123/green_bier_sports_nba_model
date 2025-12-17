"""
Backtest first half models using walk-forward validation.

Tests 1H spread and total models on historical data with NO LEAKAGE.
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "processed"


def load_first_half_training_data() -> pd.DataFrame:
    """Load 1H training data."""
    data_path = DATA_DIR / "first_half_training_data.csv"
    if not data_path.exists():
        raise FileNotFoundError(
            f"1H training data not found. Run: python scripts/generate_first_half_training_data.py"
        )

    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df


def prepare_features(df: pd.DataFrame):
    """Extract features and targets."""
    exclude_cols = [
        'game_id', 'date', 'home_team', 'away_team',
        '1h_home_score', '1h_away_score', '1h_spread', '1h_total',
        'fg_home_score', 'fg_away_score', 'fg_spread', 'fg_total',
        '1h_spread_line', '1h_total_line',
        '1h_spread_covered', '1h_total_over',
    ]

    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols].fillna(0)
    y_spread = df['1h_spread_covered']
    y_total = df['1h_total_over']

    return feature_cols, X, y_spread, y_total


def walk_forward_backtest(X, y, feature_cols, market_name, min_train=80, retrain_freq=20):
    """
    Walk-forward backtest with no lookahead.

    Args:
        X: Features DataFrame
        y: Targets Series
        feature_cols: List of feature names
        market_name: "1H Spread" or "1H Total"
        min_train: Minimum training games before predictions
        retrain_freq: Retrain every N games

    Returns:
        DataFrame with predictions
    """
    print(f"\n{'='*80}")
    print(f"WALK-FORWARD BACKTEST: {market_name}")
    print(f"{'='*80}")

    results = []
    model = None
    games_since_retrain = 0

    for idx in range(len(X)):
        # Need minimum training data
        if idx < min_train:
            continue

        # Get training data (all games before current)
        X_train = X.iloc[:idx]
        y_train = y.iloc[:idx]

        # Retrain model?
        if model is None or games_since_retrain >= retrain_freq:
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    random_state=42,
                ))
            ])
            model.fit(X_train, y_train)
            games_since_retrain = 0

        # Predict current game
        X_test = X.iloc[[idx]]
        y_pred = model.predict(X_test)[0]
        y_proba = model.predict_proba(X_test)[0, 1]  # P(home covers / over)
        y_actual = y.iloc[idx]

        results.append({
            'idx': idx,
            'predicted': y_pred,
            'probability': y_proba,
            'actual': y_actual,
            'correct': y_pred == y_actual,
        })

        games_since_retrain += 1

        if len(results) % 50 == 0:
            print(f"  Progress: {len(results)}/{len(X) - min_train} predictions")

    results_df = pd.DataFrame(results)

    # Calculate metrics
    accuracy = results_df['correct'].mean()
    n_bets = len(results_df)

    # ROI at -110 odds
    correct = results_df['correct'].sum()
    profit = correct * (100/110) - (n_bets - correct)
    roi = profit / n_bets

    print(f"\nResults:")
    print(f"  Predictions: {n_bets}")
    print(f"  Accuracy: {accuracy:.1%}")
    print(f"  ROI (at -110): {roi:+.1%}")

    # High confidence bets (>60% or <40%)
    high_conf = results_df[
        (results_df['probability'] > 0.6) | (results_df['probability'] < 0.4)
    ]

    if len(high_conf) > 0:
        hc_accuracy = high_conf['correct'].mean()
        hc_correct = high_conf['correct'].sum()
        hc_n = len(high_conf)
        hc_profit = hc_correct * (100/110) - (hc_n - hc_correct)
        hc_roi = hc_profit / hc_n

        print(f"\nHigh Confidence (>60% or <40%):")
        print(f"  Bets: {hc_n} ({hc_n/n_bets:.1%} of total)")
        print(f"  Accuracy: {hc_accuracy:.1%}")
        print(f"  ROI: {hc_roi:+.1%}")

    return results_df, accuracy, roi


def main():
    """Main execution."""
    print("="*80)
    print("FIRST HALF MODELS BACKTEST")
    print("="*80)

    # Load data
    df = load_first_half_training_data()
    print(f"\nLoaded {len(df)} games")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Prepare features
    feature_cols, X, y_spread, y_total = prepare_features(df)
    print(f"Features: {len(feature_cols)}")

    # Backtest 1H spread
    spread_results, spread_acc, spread_roi = walk_forward_backtest(
        X, y_spread, feature_cols, "1H Spread"
    )

    # Backtest 1H total
    total_results, total_acc, total_roi = walk_forward_backtest(
        X, y_total, feature_cols, "1H Total"
    )

    # Summary
    print("\n" + "="*80)
    print("BACKTEST SUMMARY")
    print("="*80)
    print(f"\n1H Spread:")
    print(f"  Accuracy: {spread_acc:.1%}")
    print(f"  ROI: {spread_roi:+.1%}")
    print(f"\n1H Total:")
    print(f"  Accuracy: {total_acc:.1%}")
    print(f"  ROI: {total_roi:+.1%}")

    # Save results
    output_dir = DATA_DIR
    spread_results.to_csv(output_dir / "backtest_1h_spread_results.csv", index=False)
    total_results.to_csv(output_dir / "backtest_1h_total_results.csv", index=False)

    print(f"\n[OK] Saved backtest results to {output_dir}")

    # Verdict
    print("\n" + "="*80)
    print("PRODUCTION READINESS VERDICT")
    print("="*80)

    if spread_acc >= 0.55 and spread_roi > 0.05:
        print("\n✓ 1H SPREAD: PRODUCTION READY")
        print(f"  {spread_acc:.1%} accuracy, {spread_roi:+.1%} ROI")
    else:
        print("\n✗ 1H SPREAD: NOT READY")
        print(f"  {spread_acc:.1%} accuracy, {spread_roi:+.1%} ROI")
        print(f"  Needs >55% accuracy and >5% ROI")

    if total_acc >= 0.55 and total_roi > 0.05:
        print("\n✓ 1H TOTAL: PRODUCTION READY")
        print(f"  {total_acc:.1%} accuracy, {total_roi:+.1%} ROI")
    else:
        print("\n✗ 1H TOTAL: NOT READY")
        print(f"  {total_acc:.1%} accuracy, {total_roi:+.1%} ROI")
        print(f"  Needs >55% accuracy and >5% ROI")


if __name__ == "__main__":
    main()
