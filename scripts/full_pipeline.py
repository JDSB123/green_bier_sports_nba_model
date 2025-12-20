#!/usr/bin/env python3
"""
NBA Prediction Pipeline - Using orchestration framework.

Complete end-to-end pipeline with error handling, retries, and logging.

Usage:
    python scripts/full_pipeline.py [--skip-odds] [--skip-train] [--date YYYY-MM-DD]
"""
import asyncio
import argparse
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import Pipeline
from src.utils.logging import get_logger

logger = get_logger(__name__)


def create_nba_pipeline(args) -> Pipeline:
    """Create the NBA prediction pipeline with all tasks.

    Args:
        args: Command line arguments

    Returns:
        Configured Pipeline instance
    """
    pipeline = Pipeline(name="NBA Prediction Pipeline v2")

    # Task 1: Fetch odds data
    if not args.skip_odds:
        async def fetch_odds():
            import subprocess
            result = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "scripts" / "run_the_odds_tomorrow.py")],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                logger.warning(f"Odds fetching stderr: {result.stderr}")
                raise Exception(f"Odds fetching failed: {result.stderr}")
            return result.stdout

        pipeline.add_task(
            name="fetch_odds",
            func=fetch_odds,
            max_retries=2,
            continue_on_failure=True,  # Continue even if odds fetch fails
        )
    
    # Task 2: Fetch injury data
    async def fetch_injuries():
        import subprocess
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "fetch_injuries.py")],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.warning(f"Injury fetching stderr: {result.stderr}")
            raise Exception(f"Injury fetching failed: {result.stderr}")
        return result.stdout

    pipeline.add_task(
        name="fetch_injuries",
        func=fetch_injuries,
        dependencies=["fetch_odds"] if not args.skip_odds else [],
        max_retries=2,
        continue_on_failure=True,  # Continue even if injury fetch fails
    )

    # Task 3: Process odds data
    async def process_odds():
        import subprocess
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "process_odds_data.py")],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.warning(f"Odds processing stderr: {result.stderr}")
            raise Exception(f"Odds processing failed: {result.stderr}")
        return result.stdout

    deps = []
    if not args.skip_odds:
        deps.append("fetch_odds")
    
    pipeline.add_task(
        name="process_odds",
        func=process_odds,
        dependencies=deps,
        max_retries=1,
        continue_on_failure=True,
    )

    # Task 4: Archive processed cache
    async def archive_cache():
        import subprocess
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "archive_processed_cache.py")],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.warning(f"Cache archive stderr: {result.stderr}")
            raise Exception(f"Cache archive failed: {result.stderr}")
        return result.stdout

    pipeline.add_task(
        name="archive_processed_cache",
        func=archive_cache,
        dependencies=["process_odds"],
        max_retries=1,
        continue_on_failure=True,
    )

    # Task 5: Build training data (if needed)
    def should_skip_training_data():
        training_data_path = PROJECT_ROOT / "data" / "processed" / "training_data.csv"
        return training_data_path.exists() and args.skip_train

    async def build_training_dataset():
        import subprocess
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "build_training_dataset.py")],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise Exception(f"Training data build failed: {result.stderr}")
        return result.stdout

    pipeline.add_task(
        name="build_training_dataset",
        func=build_training_dataset,
        dependencies=["archive_processed_cache"],
        skip_if=should_skip_training_data,
        max_retries=1,
        continue_on_failure=False,  # Critical task
    )

    # Task 6: Train base models
    if not args.skip_train:
        async def train_models():
            import subprocess
            result = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "scripts" / "train_models.py")],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise Exception(f"Model training failed: {result.stderr}")
            return result.stdout

        pipeline.add_task(
            name="train_models",
            func=train_models,
            dependencies=["build_training_dataset"],
            max_retries=0,  # Don't retry model training
            continue_on_failure=False,  # Critical task
        )

        # NOTE: Ensemble training removed - script was archived
        # Base models (spreads, totals, moneyline) are sufficient for production

    # Task 7: Generate predictions
    # (becomes Task 8 when training enabled)
    async def generate_predictions():
        import subprocess
        cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "predict.py")]
        if args.date:
            cmd.extend(["--date", args.date])
        
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise Exception(f"Prediction generation failed: {result.stderr}")
        return result.stdout

    pred_deps = ["train_models"] if not args.skip_train else ["build_training_dataset"]

    pipeline.add_task(
        name="generate_predictions",
        func=generate_predictions,
        dependencies=pred_deps,
        max_retries=1,
        continue_on_failure=False,  # Critical task
    )

    if not args.skip_review:
        async def review_predictions():
            import subprocess
            cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "review_predictions.py")]
            if args.review_date:
                cmd.extend(["--date", args.review_date])
            result = subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise Exception(f"Review generation failed: {result.stderr}")
            return result.stdout

        pipeline.add_task(
            name="review_predictions",
            func=review_predictions,
            dependencies=["generate_predictions"],
            max_retries=0,
            continue_on_failure=True,
        )

    return pipeline


async def main():
    parser = argparse.ArgumentParser(description='Run NBA prediction pipeline v2')
    parser.add_argument('--skip-odds', action='store_true', help='Skip odds data fetching')
    parser.add_argument('--skip-train', action='store_true', help='Skip model training')
    parser.add_argument('--date', help='Target date for predictions (YYYY-MM-DD)')
    parser.add_argument('--skip-review', action='store_true', help='Skip pick review step')
    parser.add_argument('--review-date', help='Date to review (defaults to yesterday CST)')
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("NBA V4.0 PREDICTION PIPELINE (Orchestrated)")
    logger.info("=" * 80)
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Create and run pipeline
    pipeline = create_nba_pipeline(args)
    results = await pipeline.run()

    # Check if pipeline succeeded
    from src.pipeline import TaskStatus
    
    failed_tasks = [
        name for name, result in results.items()
        if result.status == TaskStatus.FAILED
    ]

    if failed_tasks:
        logger.error(f"Pipeline completed with failures: {', '.join(failed_tasks)}")
        sys.exit(1)
    else:
        logger.info("Pipeline completed successfully!")
        logger.info(f"Outputs:")
        logger.info(f"  - Predictions: {PROJECT_ROOT / 'data' / 'processed' / 'predictions.csv'}")
        logger.info(f"  - Models: {PROJECT_ROOT / 'data' / 'processed' / 'models' / '*.joblib'}")
        logger.info(f"  - Pick reviews: {PROJECT_ROOT / 'data' / 'processed' / 'pick_review_*.json'}")


if __name__ == "__main__":
    asyncio.run(main())

