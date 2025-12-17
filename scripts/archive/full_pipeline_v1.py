#!/usr/bin/env python3
"""
Complete NBA prediction pipeline - runs all steps in sequence.

This script orchestrates the entire workflow:
1. Fetch odds data from The Odds API
2. Fetch injury data
3. Process odds data (extract splits, first-half lines)
4. Generate training data (if needed)
5. Train models (if needed)
6. Generate predictions for upcoming games

Usage:
    python scripts/full_pipeline.py [--skip-odds] [--skip-train] [--date YYYY-MM-DD]
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent


def run_script(script_path, args=None, description=None):
    """Run a Python script and return success status."""
    if description:
        print(f"\n{'='*80}")
        print(f"{description}")
        print(f"{'='*80}")

    cmd = [sys.executable, str(PROJECT_ROOT / script_path)]
    if args:
        cmd.extend(args)

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)

    if result.returncode != 0:
        print(f"[ERROR] Script failed with exit code {result.returncode}")
        return False

    print(f"[OK] {script_path} completed successfully")
    return True


def main():
    parser = argparse.ArgumentParser(description='Run complete NBA prediction pipeline')
    parser.add_argument('--skip-odds', action='store_true', help='Skip odds data fetching')
    parser.add_argument('--skip-train', action='store_true', help='Skip model training')
    parser.add_argument('--date', help='Target date for predictions (YYYY-MM-DD)')
    parser.add_argument('--skip-review', action='store_true', help='Skip pick review step')
    parser.add_argument('--review-date', help='Date to review (defaults to yesterday CST)')
    args = parser.parse_args()

    print("="*80)
    print("NBA V4.0 COMPLETE PREDICTION PIPELINE")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Step 1: Fetch odds data
    if not args.skip_odds:
        if not run_script('scripts/run_the_odds_tomorrow.py',
                         description="STEP 1: Fetching odds data from The Odds API"):
            print("[WARN] Odds fetching failed, continuing with existing data...")
    else:
        print("\n[SKIP] Odds data fetching (--skip-odds)")

    # Step 2: Fetch injury data
    if not run_script('scripts/fetch_injuries.py',
                     description="STEP 2: Fetching NBA injury data"):
        print("[WARN] Injury fetching failed, continuing without injury data...")

    # Step 3: Process odds data
    if not run_script('scripts/process_odds_data.py',
                     description="STEP 3: Processing odds data (splits, first-half lines)"):
        print("[WARN] Odds processing failed, continuing...")

    # Step 4: Archive processed cache
    if not run_script(
        'scripts/archive_processed_cache.py',
        description="STEP 4: Archiving processed cache"
    ):
        print("[WARN] Cache archive failed, continuing...")

    # Step 5: Build training data (if needed)
    training_data_path = PROJECT_ROOT / 'data' / 'processed' / 'training_data.csv'
    if not training_data_path.exists() or not args.skip_train:
        if not run_script('scripts/build_training_dataset.py',
                         description="STEP 5: Building training dataset"):
            print("[ERROR] Training data generation failed")
            sys.exit(1)
    else:
        print("\n[SKIP] Training data generation (file exists)")

    # Step 6: Train models (if needed)
    if not args.skip_train:
        # Train base models
        if not run_script('scripts/train_models.py',
                         description="STEP 6a: Training base models"):
            print("[ERROR] Base model training failed")
            sys.exit(1)
        
        # Train ensemble models
        if not run_script('scripts/train_ensemble_models.py',
                         description="STEP 6b: Training ensemble models"):
            print("[ERROR] Ensemble model training failed")
            sys.exit(1)
    else:
        print("\n[SKIP] Model training (--skip-train)")

    # Step 7: Generate predictions
    predict_args = ['--date', args.date] if args.date else []
    if not run_script('scripts/predict.py', args=predict_args,
                     description="STEP 7: Generating predictions"):
        print("[ERROR] Prediction generation failed")
        sys.exit(1)

    # Step 8: Review previous slate (optional)
    if not args.skip_review:
        review_args = []
        if args.review_date:
            review_args = ['--date', args.review_date]
        if not run_script(
            'scripts/review_predictions.py',
            args=review_args,
            description="STEP 8: Reviewing picks vs results",
        ):
            print("[WARN] Review step failed, continuing...")
    else:
        print("\n[SKIP] Review step (--skip-review)")

    # Summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nOutputs:")
    print(f"  - Predictions: {PROJECT_ROOT / 'data' / 'processed' / 'predictions.csv'}")
    print(f"  - Models: {PROJECT_ROOT / 'data' / 'processed' / 'models' / '*.joblib'}")
    print(f"  - Odds data: {PROJECT_ROOT / 'data' / 'raw' / 'the_odds' / '*'}")
    print(f"  - Pick reviews: {PROJECT_ROOT / 'data' / 'processed' / 'pick_review_*.json'}")
    print("="*80)


if __name__ == '__main__':
    main()
