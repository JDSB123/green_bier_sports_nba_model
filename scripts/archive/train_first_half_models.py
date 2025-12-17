"""
Train first half spread and total models.

Trains separate models for 1H markets using 1H-specific training data.
These models learn first half scoring dynamics independently from FG models.
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROCESSED_DIR / "models"


def load_first_half_training_data() -> pd.DataFrame:
    """Load 1H training data."""
    data_path = PROCESSED_DIR / "first_half_training_data.csv"
    if not data_path.exists():
        raise FileNotFoundError(
            f"1H training data not found. Run: python scripts/generate_first_half_training_data.py"
        )

    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    return df


def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Prepare features for training.

    Returns:
        (feature_columns, X_spread, y_spread, X_total, y_total)
    """
    # Identify feature columns (exclude metadata and targets)
    exclude_cols = [
        'game_id', 'date', 'home_team', 'away_team',
        '1h_home_score', '1h_away_score', '1h_spread', '1h_total',
        'fg_home_score', 'fg_away_score', 'fg_spread', 'fg_total',
        '1h_spread_line', '1h_total_line',
        '1h_spread_covered', '1h_total_over',
    ]

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Handle NaN values
    X = df[feature_cols].fillna(0)

    # Targets
    y_spread = df['1h_spread_covered']
    y_total = df['1h_total_over']

    return feature_cols, X, y_spread, y_total


def train_first_half_spread_model(X, y, feature_cols):
    """Train 1H spread model."""
    print("\n" + "="*80)
    print("TRAINING FIRST HALF SPREAD MODEL")
    print("="*80)

    # Split data (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False  # No shuffle to maintain time order
    )

    print(f"\nTraining set: {len(X_train)} games")
    print(f"Test set: {len(X_test)} games")

    # Build pipeline
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            min_samples_split=20,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=42,
        ))
    ])

    # Train
    print("\nTraining model...")
    model.fit(X_train, y_train)

    # Evaluate
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]

    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]

    print("\nTraining Performance:")
    print(f"  Accuracy: {accuracy_score(y_train, y_train_pred):.1%}")
    print(f"  ROC AUC: {roc_auc_score(y_train, y_train_proba):.3f}")
    print(f"  Brier Score: {brier_score_loss(y_train, y_train_proba):.3f}")

    print("\nTest Performance:")
    print(f"  Accuracy: {accuracy_score(y_test, y_test_pred):.1%}")
    print(f"  ROC AUC: {roc_auc_score(y_test, y_test_proba):.3f}")
    print(f"  Brier Score: {brier_score_loss(y_test, y_test_proba):.3f}")

    # Feature importance
    clf = model.named_steps['classifier']
    importances = clf.feature_importances_
    feature_importance = sorted(
        zip(feature_cols, importances),
        key=lambda x: x[1],
        reverse=True
    )

    print("\nTop 10 Features:")
    for feat, imp in feature_importance[:10]:
        print(f"  {feat}: {imp:.4f}")

    return model, feature_cols


def train_first_half_total_model(X, y, feature_cols):
    """Train 1H total model."""
    print("\n" + "="*80)
    print("TRAINING FIRST HALF TOTAL MODEL")
    print("="*80)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    print(f"\nTraining set: {len(X_train)} games")
    print(f"Test set: {len(X_test)} games")

    # Build pipeline
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            min_samples_split=20,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=42,
        ))
    ])

    # Train
    print("\nTraining model...")
    model.fit(X_train, y_train)

    # Evaluate
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]

    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]

    print("\nTraining Performance:")
    print(f"  Accuracy: {accuracy_score(y_train, y_train_pred):.1%}")
    print(f"  ROC AUC: {roc_auc_score(y_train, y_train_proba):.3f}")
    print(f"  Brier Score: {brier_score_loss(y_train, y_train_proba):.3f}")

    print("\nTest Performance:")
    print(f"  Accuracy: {accuracy_score(y_test, y_test_pred):.1%}")
    print(f"  ROC AUC: {roc_auc_score(y_test, y_test_proba):.3f}")
    print(f"  Brier Score: {brier_score_loss(y_test, y_test_proba):.3f}")

    # Feature importance
    clf = model.named_steps['classifier']
    importances = clf.feature_importances_
    feature_importance = sorted(
        zip(feature_cols, importances),
        key=lambda x: x[1],
        reverse=True
    )

    print("\nTop 10 Features:")
    for feat, imp in feature_importance[:10]:
        print(f"  {feat}: {imp:.4f}")

    return model, feature_cols


def save_models(spread_model, spread_features, total_model, total_features):
    """Save 1H models."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save 1H spread model
    spread_model_path = MODELS_DIR / "first_half_spread_model.pkl"
    spread_features_path = MODELS_DIR / "first_half_spread_features.pkl"

    joblib.dump(spread_model, spread_model_path)
    joblib.dump(spread_features, spread_features_path)

    print(f"\n[OK] Saved 1H spread model to:")
    print(f"     {spread_model_path}")
    print(f"     {spread_features_path}")

    # Save 1H total model
    total_model_path = MODELS_DIR / "first_half_total_model.pkl"
    total_features_path = MODELS_DIR / "first_half_total_features.pkl"

    joblib.dump(total_model, total_model_path)
    joblib.dump(total_features, total_features_path)

    print(f"\n[OK] Saved 1H total model to:")
    print(f"     {total_model_path}")
    print(f"     {total_features_path}")


def main():
    """Main execution."""
    print("="*80)
    print("FIRST HALF MODEL TRAINING")
    print("="*80)

    # Load training data
    df = load_first_half_training_data()
    print(f"\nLoaded {len(df)} games for training")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Prepare features
    feature_cols, X, y_spread, y_total = prepare_features(df)
    print(f"\nFeatures: {len(feature_cols)} columns")

    # Train 1H spread model
    spread_model, spread_features = train_first_half_spread_model(
        X, y_spread, feature_cols
    )

    # Train 1H total model
    total_model, total_features = train_first_half_total_model(
        X, y_total, feature_cols
    )

    # Save models
    save_models(spread_model, spread_features, total_model, total_features)

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("  1. python scripts/backtest_first_half.py")
    print("  2. Update predictors to use separate FG/1H models")


if __name__ == "__main__":
    main()
