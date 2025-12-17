"""
Backtest first half moneyline models using walk-forward validation.

Tests 1H moneyline model (home leading at halftime) on historical data with NO LEAKAGE.

Usage: python scripts/backtest_first_half_moneyline.py
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "processed"


def load_first_half_training_data() -> pd.DataFrame:
    """Load 1H training data."""
    data_path = DATA_DIR / "first_half_training_data.csv"
    if not data_path.exists():
        raise FileNotFoundError(
            f"1H training data not found. Run: python scripts/generate_first_half_training_data_fast.py"
        )

    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Add 1H moneyline target: home team leading at halftime
    df['1h_home_leading'] = (df['1h_home_score'] > df['1h_away_score']).astype(int)
    
    # Handle ties (count as 0 = home not leading)
    # Could also exclude ties, but this is conservative
    
    return df


def prepare_features(df: pd.DataFrame):
    """Extract features and target for moneyline prediction."""
    # Features relevant for predicting who leads at halftime
    feature_cols = [
        # Team performance
        'home_ppg', 'away_ppg', 'home_papg', 'away_papg',
        'home_ortg', 'away_ortg', 'home_drtg', 'away_drtg',
        'home_net_rtg', 'away_net_rtg',
        # Pace
        'home_pace_factor', 'away_pace_factor', 'expected_pace_factor',
        # Margins & predictions
        'predicted_margin', 'predicted_total',
        'home_avg_margin', 'away_avg_margin',
        'ppg_diff', 'win_pct_diff',
        # Win %
        'home_win_pct', 'away_win_pct',
        # Recent form
        'home_l5_win_pct', 'away_l5_win_pct',
        'home_l5_margin', 'away_l5_margin',
        'home_l10_margin', 'away_l10_margin',
        # Rest
        'home_days_rest', 'away_days_rest',
        'home_rest_adj', 'away_rest_adj', 'rest_margin_adj',
        # Form adjustments
        'home_form_adj', 'away_form_adj', 'form_margin_adj',
        # ELO
        'home_elo', 'away_elo', 'elo_diff', 'elo_prob_home',
        # Travel
        'away_travel_distance', 'away_travel_fatigue', 'travel_advantage',
        # Home court
        'home_court_advantage',
    ]
    
    # Filter to available features
    available_features = [col for col in feature_cols if col in df.columns]
    
    X = df[available_features].fillna(0)
    y = df['1h_home_leading']
    
    return available_features, X, y


def walk_forward_backtest(
    df: pd.DataFrame,
    X: pd.DataFrame, 
    y: pd.Series, 
    feature_cols: list,
    model_type: str = "gradient_boosting",
    use_calibration: bool = True,
    min_train: int = 80, 
    retrain_freq: int = 20
):
    """
    Walk-forward backtest with no lookahead.

    Args:
        df: Original DataFrame (for metadata)
        X: Features DataFrame
        y: Targets Series
        feature_cols: List of feature names
        model_type: "gradient_boosting" or "logistic"
        use_calibration: Whether to use probability calibration
        min_train: Minimum training games before predictions
        retrain_freq: Retrain every N games

    Returns:
        DataFrame with predictions
    """
    print(f"\n{'='*80}")
    print(f"WALK-FORWARD BACKTEST: 1H Moneyline ({model_type})")
    print(f"{'='*80}")
    print(f"  Calibration: {'Yes' if use_calibration else 'No'}")
    print(f"  Min training: {min_train} games")
    print(f"  Retrain frequency: {retrain_freq} games")

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
            # Build base model
            if model_type == "gradient_boosting":
                base_model = Pipeline([
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
            else:
                base_model = Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', LogisticRegression(
                        max_iter=1000,
                        random_state=42,
                    ))
                ])
            
            # Apply calibration if requested
            if use_calibration and len(X_train) >= 100:
                model = CalibratedClassifierCV(
                    base_model,
                    method='isotonic',
                    cv=3,  # Fewer folds due to limited data
                )
            else:
                model = base_model
            
            model.fit(X_train, y_train)
            games_since_retrain = 0

        # Predict current game
        X_test = X.iloc[[idx]]
        y_pred = model.predict(X_test)[0]
        y_proba = model.predict_proba(X_test)[0, 1]  # P(home leading)
        y_actual = y.iloc[idx]

        results.append({
            'idx': idx,
            'date': df.iloc[idx]['date'],
            'home_team': df.iloc[idx]['home_team'],
            'away_team': df.iloc[idx]['away_team'],
            '1h_home_score': df.iloc[idx]['1h_home_score'],
            '1h_away_score': df.iloc[idx]['1h_away_score'],
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

    # ROI calculation for moneyline is different - need to simulate odds
    # Assuming -110 both sides for simplicity (conservative estimate)
    correct = results_df['correct'].sum()
    profit = correct * (100/110) - (n_bets - correct)
    roi = profit / n_bets if n_bets > 0 else 0

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
        hc_roi = hc_profit / hc_n if hc_n > 0 else 0

        print(f"\nHigh Confidence (>60% or <40%):")
        print(f"  Bets: {hc_n} ({hc_n/n_bets:.1%} of total)")
        print(f"  Accuracy: {hc_accuracy:.1%}")
        print(f"  ROI: {hc_roi:+.1%}")

    # Very high confidence (>65% or <35%)
    very_high_conf = results_df[
        (results_df['probability'] > 0.65) | (results_df['probability'] < 0.35)
    ]

    if len(very_high_conf) > 0:
        vhc_accuracy = very_high_conf['correct'].mean()
        vhc_correct = very_high_conf['correct'].sum()
        vhc_n = len(very_high_conf)
        vhc_profit = vhc_correct * (100/110) - (vhc_n - vhc_correct)
        vhc_roi = vhc_profit / vhc_n if vhc_n > 0 else 0

        print(f"\nVery High Confidence (>65% or <35%):")
        print(f"  Bets: {vhc_n} ({vhc_n/n_bets:.1%} of total)")
        print(f"  Accuracy: {vhc_accuracy:.1%}")
        print(f"  ROI: {vhc_roi:+.1%}")

    # By predicted side
    home_picks = results_df[results_df['predicted'] == 1]
    away_picks = results_df[results_df['predicted'] == 0]

    if len(home_picks) > 0:
        print(f"\nHome Picks: {len(home_picks)} ({home_picks['correct'].mean():.1%} acc)")
    if len(away_picks) > 0:
        print(f"Away Picks: {len(away_picks)} ({away_picks['correct'].mean():.1%} acc)")

    return results_df, accuracy, roi


def main():
    """Main execution."""
    print("="*80)
    print("FIRST HALF MONEYLINE BACKTEST")
    print("="*80)

    # Load data
    df = load_first_half_training_data()
    print(f"\nLoaded {len(df)} games")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Check target distribution
    home_leading = df['1h_home_leading'].sum()
    ties = ((df['1h_home_score'] == df['1h_away_score']).sum())
    print(f"\n1H Moneyline baseline:")
    print(f"  Home leading at half: {home_leading} ({home_leading/len(df):.1%})")
    print(f"  Away leading at half: {len(df) - home_leading - ties} ({(len(df) - home_leading - ties)/len(df):.1%})")
    print(f"  Ties at half: {ties} ({ties/len(df):.1%})")

    # Prepare features
    feature_cols, X, y = prepare_features(df)
    print(f"\nFeatures: {len(feature_cols)}")

    # Run backtest with different model types
    all_results = {}
    
    for model_type in ["logistic", "gradient_boosting"]:
        for use_cal in [False, True]:
            key = f"{model_type}{'_calibrated' if use_cal else ''}"
            print(f"\n\n{'#'*80}")
            print(f"MODEL: {key.upper()}")
            print(f"{'#'*80}")
            
            results, acc, roi = walk_forward_backtest(
                df, X, y, feature_cols,
                model_type=model_type,
                use_calibration=use_cal,
            )
            all_results[key] = {
                'results': results,
                'accuracy': acc,
                'roi': roi,
            }

    # Summary
    print("\n" + "="*80)
    print("BACKTEST SUMMARY - 1H MONEYLINE")
    print("="*80)
    
    print("\n| Model | Accuracy | ROI | High Conf Acc | High Conf ROI |")
    print("|-------|----------|-----|---------------|---------------|")
    
    for key, data in all_results.items():
        results = data['results']
        acc = data['accuracy']
        roi = data['roi']
        
        # High confidence
        high_conf = results[
            (results['probability'] > 0.6) | (results['probability'] < 0.4)
        ]
        hc_acc = high_conf['correct'].mean() if len(high_conf) > 0 else 0
        hc_correct = high_conf['correct'].sum()
        hc_n = len(high_conf)
        hc_profit = hc_correct * (100/110) - (hc_n - hc_correct) if hc_n > 0 else 0
        hc_roi = hc_profit / hc_n if hc_n > 0 else 0
        
        print(f"| {key:25} | {acc:.1%} | {roi:+.1%} | {hc_acc:.1%} | {hc_roi:+.1%} |")

    # Find best model
    best_key = max(all_results.keys(), key=lambda k: all_results[k]['roi'])
    best = all_results[best_key]
    
    print(f"\nBest Model: {best_key}")
    print(f"  Accuracy: {best['accuracy']:.1%}")
    print(f"  ROI: {best['roi']:+.1%}")

    # Save best results
    output_path = DATA_DIR / "backtest_1h_moneyline_results.csv"
    best['results'].to_csv(output_path, index=False)
    print(f"\n[OK] Saved results to {output_path}")

    # Verdict
    print("\n" + "="*80)
    print("PRODUCTION READINESS VERDICT")
    print("="*80)

    if best['accuracy'] >= 0.55 and best['roi'] > 0.05:
        print("\n[OK] 1H MONEYLINE: PRODUCTION READY")
        print(f"  {best['accuracy']:.1%} accuracy, {best['roi']:+.1%} ROI")
    elif best['accuracy'] >= 0.52 and best['roi'] > 0:
        print("\n[~] 1H MONEYLINE: MARGINALLY READY")
        print(f"  {best['accuracy']:.1%} accuracy, {best['roi']:+.1%} ROI")
        print(f"  Consider using with high-confidence filtering only")
    else:
        print("\n[X] 1H MONEYLINE: NOT READY")
        print(f"  {best['accuracy']:.1%} accuracy, {best['roi']:+.1%} ROI")
        print(f"  Needs >55% accuracy and >5% ROI")

    return all_results


if __name__ == "__main__":
    main()
