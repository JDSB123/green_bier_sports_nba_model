#!/usr/bin/env python3
"""
Quick test of production models - no feature engineering
"""
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_production_models():
    """Quick test using pre-computed features from training data."""
    print("LOADING DATA...")

    # Load training data (which already has engineered features)
    df = pd.read_csv('data/processed/training_data_complete_2023.csv', low_memory=False)
    df['date'] = pd.to_datetime(df['game_date'] if 'game_date' in df.columns else 'date', errors='coerce')
    df = df.dropna(subset=['date'])
    df = df.sort_values('date').reset_index(drop=True)

    print(f"Loaded {len(df)} games from {df['date'].min()} to {df['date'].max()}")

    # Load production models
    models_dir = Path('models/production')
    markets = {}

    model_configs = [
        ("fg_spread", "fg_spread_model.joblib", "spread_covered", "fg_spread_line", -110),
        ("fg_total", "fg_total_model.joblib", "total_over", "fg_total_line", -110),
    ]

    for market_key, model_file, label_col, line_col, odds in model_configs:
        model_path = models_dir / model_file
        if model_path.exists():
            print(f"Loading {market_key}...")
            data = joblib.load(model_path)
            if isinstance(data, dict):
                model = data.get("pipeline") or data.get("model")
                features = data.get("feature_columns", [])
            else:
                model = data
                features = model.feature_names_in_.tolist() if hasattr(model, 'feature_names_in_') else []

            markets[market_key] = {
                "model": model,
                "features": features,
                "label_col": label_col,
                "line_col": line_col,
                "odds": odds,
            }
            print(f"  {market_key}: {len(features)} features")
        else:
            print(f"  [SKIP] {market_key} - model not found")

    if not markets:
        print("NO MODELS LOADED!")
        return

    print("\nTESTING PREDICTIONS...")

    # Use last 100 games for quick test
    test_df = df.tail(100).copy()
    results = {k: [] for k in markets}

    for i, game in test_df.iterrows():
        if i % 20 == 0:
            print(f"  Processing game {i+1}/100...")

        # Create feature vector (fill missing with 0)
        feature_df = pd.DataFrame([{}])  # Empty features
        for market_key, config in markets.items():
            model_features = config["features"]
            X = feature_df.reindex(columns=model_features, fill_value=0)

            try:
                proba = config["model"].predict_proba(X)[0]
                pred_class = 1 if proba[1] > 0.5 else 0
                confidence = max(proba)

                # Check if we have the outcome
                if config["label_col"] in game and pd.notna(game[config["label_col"]]):
                    actual = int(game[config["label_col"]])
                    won = (pred_class == actual)
                    results[market_key].append({
                        'won': won,
                        'confidence': confidence,
                        'date': str(game['date'].date())
                    })

            except Exception as e:
                logger.debug(f"Prediction failed for {market_key}: {e}")
                continue

    print("\nRESULTS:")
    for market, bets in results.items():
        if bets:
            wins = sum(1 for b in bets if b['won'])
            accuracy = wins / len(bets)
            print(f"  {market}: {len(bets)} bets, {accuracy:.1%} accuracy")
        else:
            print(f"  {market}: No predictions")

if __name__ == "__main__":
    test_production_models()