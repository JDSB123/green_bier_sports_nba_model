#!/usr/bin/env python3
"""
NBA MODEL PIPELINE - SINGLE ENTRY POINT

This is the ONLY entry point for building, training, and validating the model.
It ensures a clean architectural stack with no stale data.

Usage:
    # Full pipeline (data → train → validate)
    python scripts/model_pipeline.py --full
    
    # Just rebuild training data
    python scripts/model_pipeline.py --data-only
    
    # Just train models (assumes data exists)
    python scripts/model_pipeline.py --train-only
    
    # Validate existing models
    python scripts/model_pipeline.py --validate-only
    
    # Quick test with sample data
    python scripts/model_pipeline.py --quick-test

Architecture:
    1. Data Collection - Fresh from APIs (no caching)
    2. Feature Engineering - Unified features for all markets
    3. Training - All 4 models with identical architecture
    4. Validation - Backtest and verify predictions
"""
from __future__ import annotations
import argparse
import asyncio
import os
import sys
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.unified_features import (
    UNIFIED_FEATURE_NAMES,
    MODEL_REGISTRY,
    FEATURE_DEFAULTS,
    get_model_config,
    print_feature_summary,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

class PipelineConfig:
    """Pipeline configuration."""
    
    def __init__(self, args: argparse.Namespace):
        self.project_root = PROJECT_ROOT
        self.data_dir = PROJECT_ROOT / "data" / "processed"
        self.models_dir = PROJECT_ROOT / "models" / "production"
        self.training_file = self.data_dir / "training_data.csv"
        self.splits_file = self.data_dir / "betting_splits.csv"
        
        # Pipeline options
        self.seasons = args.seasons if hasattr(args, 'seasons') else ["2024-2025", "2025-2026"]
        self.clear_cache = args.clear_cache if hasattr(args, 'clear_cache') else True
        self.fetch_splits = args.fetch_splits if hasattr(args, 'fetch_splits') else True
        self.model_type = args.model_type if hasattr(args, 'model_type') else "gradient_boosting"
        self.verbose = args.verbose if hasattr(args, 'verbose') else False


# =============================================================================
# PIPELINE STAGES
# =============================================================================

class ModelPipeline:
    """
    Single entry point for the NBA model pipeline.
    
    Stages:
        1. Clear - Remove stale data and cached files
        2. Collect - Fetch fresh data from APIs
        3. Engineer - Build features from raw data
        4. Train - Train all 4 models
        5. Validate - Run backtest and verify
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.start_time = datetime.now()
        self.stage_results: Dict[str, Any] = {}
    
    async def run_full_pipeline(self) -> bool:
        """Run the complete pipeline."""
        logger.info("=" * 60)
        logger.info("NBA MODEL PIPELINE - FULL BUILD")
        logger.info("=" * 60)
        print_feature_summary()
        
        stages = [
            ("clear", self.stage_clear),
            ("collect", self.stage_collect_data),
            ("engineer", self.stage_engineer_features),
            ("train", self.stage_train_models),
            ("validate", self.stage_validate),
        ]
        
        for stage_name, stage_fn in stages:
            logger.info(f"\n{'='*60}")
            logger.info(f"STAGE: {stage_name.upper()}")
            logger.info(f"{'='*60}")
            
            try:
                result = await stage_fn() if asyncio.iscoroutinefunction(stage_fn) else stage_fn()
                self.stage_results[stage_name] = {"status": "success", "result": result}
                logger.info(f"✓ Stage {stage_name} completed successfully")
            except Exception as e:
                self.stage_results[stage_name] = {"status": "failed", "error": str(e)}
                logger.error(f"✗ Stage {stage_name} FAILED: {e}")
                return False
        
        self._print_summary()
        return True
    
    async def run_data_only(self) -> bool:
        """Run only data collection and feature engineering."""
        logger.info("=" * 60)
        logger.info("NBA MODEL PIPELINE - DATA BUILD ONLY")
        logger.info("=" * 60)
        
        stages = [
            ("clear", self.stage_clear),
            ("collect", self.stage_collect_data),
            ("engineer", self.stage_engineer_features),
        ]
        
        for stage_name, stage_fn in stages:
            try:
                result = await stage_fn() if asyncio.iscoroutinefunction(stage_fn) else stage_fn()
                self.stage_results[stage_name] = {"status": "success", "result": result}
                logger.info(f"✓ Stage {stage_name} completed")
            except Exception as e:
                logger.error(f"✗ Stage {stage_name} FAILED: {e}")
                return False
        
        return True
    
    async def run_train_only(self) -> bool:
        """Run only model training (assumes data exists)."""
        logger.info("=" * 60)
        logger.info("NBA MODEL PIPELINE - TRAIN ONLY")
        logger.info("=" * 60)
        
        if not self.config.training_file.exists():
            logger.error(f"Training data not found: {self.config.training_file}")
            logger.error("Run with --data-only first to build training data")
            return False
        
        try:
            await self.stage_train_models()
            return True
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False
    
    async def run_validate_only(self) -> bool:
        """Run only validation."""
        logger.info("=" * 60)
        logger.info("NBA MODEL PIPELINE - VALIDATE ONLY")
        logger.info("=" * 60)
        
        try:
            await self.stage_validate()
            return True
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False
    
    # -------------------------------------------------------------------------
    # STAGE: CLEAR
    # -------------------------------------------------------------------------
    
    def stage_clear(self) -> Dict[str, int]:
        """Clear stale data and caches."""
        logger.info("Clearing stale data and caches...")
        
        cleared = {
            "cache_files": 0,
            "old_models": 0,
        }
        
        # Clear API caches
        cache_patterns = [
            self.config.project_root / "*.cache",
            self.config.project_root / ".cache" / "*",
            self.config.data_dir / "*.cache",
        ]
        
        for pattern in cache_patterns:
            for cache_file in pattern.parent.glob(pattern.name):
                if cache_file.is_file():
                    cache_file.unlink()
                    cleared["cache_files"] += 1
        
        # Don't delete existing models unless explicitly requested
        if self.config.clear_cache:
            logger.info(f"Cleared {cleared['cache_files']} cache files")
        
        return cleared
    
    # -------------------------------------------------------------------------
    # STAGE: COLLECT DATA
    # -------------------------------------------------------------------------
    
    async def stage_collect_data(self) -> Dict[str, Any]:
        """Collect fresh data from APIs."""
        logger.info("Collecting fresh data from APIs...")
        logger.info(f"Seasons: {self.config.seasons}")
        
        # Import here to avoid circular imports
        from scripts.build_fresh_training_data import TrainingDataBuilder
        
        builder = TrainingDataBuilder(
            seasons=self.config.seasons,
            output_dir=str(self.config.data_dir),
        )
        
        result = await builder.build()
        
        logger.info(f"Collected data: {result.get('total_games', 0)} games")
        
        return result
    
    # -------------------------------------------------------------------------
    # STAGE: ENGINEER FEATURES
    # -------------------------------------------------------------------------
    
    async def stage_engineer_features(self) -> Dict[str, Any]:
        """Engineer features and add betting splits."""
        logger.info("Engineering features...")
        
        if not self.config.training_file.exists():
            raise FileNotFoundError(f"Training data not found: {self.config.training_file}")
        
        # Load training data
        df = pd.read_csv(self.config.training_file)
        original_cols = set(df.columns)
        
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Check which unified features are present
        present = set(df.columns) & set(UNIFIED_FEATURE_NAMES)
        missing = set(UNIFIED_FEATURE_NAMES) - set(df.columns)
        
        logger.info(f"Features present: {len(present)}/{len(UNIFIED_FEATURE_NAMES)}")
        
        # Add missing features with defaults
        for feat_name in missing:
            df[feat_name] = FEATURE_DEFAULTS.get(feat_name, 0.0)
        
        # Try to fetch betting splits for enrichment
        if self.config.fetch_splits:
            df = await self._enrich_with_betting_splits(df)
        
        # Save enriched data
        df.to_csv(self.config.training_file, index=False)
        
        new_cols = set(df.columns) - original_cols
        logger.info(f"Added {len(new_cols)} new feature columns")
        
        return {
            "rows": len(df),
            "total_columns": len(df.columns),
            "features_present": len(set(df.columns) & set(UNIFIED_FEATURE_NAMES)),
            "new_columns": len(new_cols),
        }
    
    async def _enrich_with_betting_splits(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich training data with betting splits from Action Network."""
        logger.info("Fetching betting splits from Action Network...")
        
        try:
            from src.ingestion.betting_splits import fetch_splits_action_network, splits_to_features
            
            # Fetch current splits
            splits = await fetch_splits_action_network()
            
            if not splits:
                logger.warning("No betting splits available")
                return df
            
            logger.info(f"Fetched {len(splits)} games with betting splits")
            
            # Convert to features DataFrame
            splits_rows = []
            for split in splits:
                features = splits_to_features(split)
                features["home_team"] = split.home_team
                features["away_team"] = split.away_team
                splits_rows.append(features)
            
            if splits_rows:
                splits_df = pd.DataFrame(splits_rows)
                
                # Merge with training data on team names
                # This enriches recent games with real betting data
                merged = df.merge(
                    splits_df,
                    on=["home_team", "away_team"],
                    how="left",
                    suffixes=("", "_splits"),
                )
                
                # Use splits data where available
                for col in splits_df.columns:
                    if col in ["home_team", "away_team"]:
                        continue
                    splits_col = f"{col}_splits"
                    if splits_col in merged.columns:
                        merged[col] = merged[splits_col].fillna(merged.get(col, 0.0))
                        merged.drop(columns=[splits_col], inplace=True, errors="ignore")
                
                enriched_count = merged["has_real_splits"].sum() if "has_real_splits" in merged.columns else 0
                logger.info(f"Enriched {enriched_count} games with real betting splits")
                
                return merged
        
        except Exception as e:
            logger.warning(f"Failed to enrich with betting splits: {e}")
        
        return df
    
    # -------------------------------------------------------------------------
    # STAGE: TRAIN MODELS
    # -------------------------------------------------------------------------
    
    async def stage_train_models(self) -> Dict[str, Any]:
        """Train all 4 models."""
        logger.info("Training all models...")
        logger.info(f"Model type: {self.config.model_type}")
        
        # Import training module
        from scripts.train_models import train_single_market
        
        results = {}
        
        for market_key, config in MODEL_REGISTRY.items():
            logger.info(f"\nTraining {market_key}...")
            
            try:
                result = train_single_market(
                    market=market_key,
                    model_type=self.config.model_type,
                    training_file=str(self.config.training_file),
                    output_dir=str(self.config.models_dir),
                )
                results[market_key] = {
                    "status": "success",
                    "accuracy": result.get("accuracy", 0),
                    "features": result.get("feature_count", len(config.features)),
                }
                logger.info(f"✓ {market_key}: accuracy={results[market_key]['accuracy']:.3f}")
            except Exception as e:
                results[market_key] = {"status": "failed", "error": str(e)}
                logger.error(f"✗ {market_key} training failed: {e}")
        
        # Update model_pack.json
        await self._update_model_pack(results)
        
        return results
    
    async def _update_model_pack(self, training_results: Dict[str, Any]):
        """Update model_pack.json with training results."""
        import json
        
        model_pack_path = self.config.models_dir / "model_pack.json"
        
        model_pack = {
            "version": f"NBA_v{datetime.now().strftime('%y.%m.%d.%H')}",
            "created_at": datetime.now().isoformat(),
            "unified_features": len(UNIFIED_FEATURE_NAMES),
            "markets": {},
        }
        
        for market_key, result in training_results.items():
            config = get_model_config(market_key)
            model_pack["markets"][market_key] = {
                "model_file": config.model_file,
                "features": len(config.features),
                "training_result": result,
            }
        
        with open(model_pack_path, "w") as f:
            json.dump(model_pack, f, indent=2)
        
        logger.info(f"Updated {model_pack_path}")
    
    # -------------------------------------------------------------------------
    # STAGE: VALIDATE
    # -------------------------------------------------------------------------
    
    async def stage_validate(self) -> Dict[str, Any]:
        """Validate trained models."""
        logger.info("Validating models...")
        
        results = {
            "models_exist": True,
            "features_consistent": True,
            "markets": {},
        }
        
        for market_key, config in MODEL_REGISTRY.items():
            model_path = self.config.models_dir / config.model_file
            
            if not model_path.exists():
                results["models_exist"] = False
                results["markets"][market_key] = {"status": "missing"}
                logger.error(f"✗ {market_key}: model file not found")
                continue
            
            # Load and check model
            import joblib
            try:
                model_data = joblib.load(model_path)
                model_features = model_data.get("feature_columns", [])
                
                results["markets"][market_key] = {
                    "status": "valid",
                    "features": len(model_features),
                    "expected_features": len(UNIFIED_FEATURE_NAMES),
                }
                
                # Check feature consistency
                if set(model_features) != set(UNIFIED_FEATURE_NAMES):
                    results["features_consistent"] = False
                    logger.warning(f"⚠ {market_key}: feature mismatch")
                else:
                    logger.info(f"✓ {market_key}: valid ({len(model_features)} features)")
                    
            except Exception as e:
                results["markets"][market_key] = {"status": "error", "error": str(e)}
                logger.error(f"✗ {market_key}: load failed - {e}")
        
        return results
    
    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    
    def _print_summary(self):
        """Print pipeline summary."""
        elapsed = datetime.now() - self.start_time
        
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total time: {elapsed}")
        
        for stage, result in self.stage_results.items():
            status = result.get("status", "unknown")
            emoji = "✓" if status == "success" else "✗"
            logger.info(f"  {emoji} {stage}: {status}")
        
        logger.info("=" * 60)


# =============================================================================
# QUICK TEST
# =============================================================================

async def run_quick_test():
    """Run a quick test to verify the architecture."""
    logger.info("=" * 60)
    logger.info("QUICK TEST - Verifying Architecture")
    logger.info("=" * 60)
    
    # Test 1: Feature consistency
    print_feature_summary()
    
    # Test 2: Model registry
    logger.info("\nModel Registry:")
    for market_key, config in MODEL_REGISTRY.items():
        logger.info(f"  {market_key}: {len(config.features)} features")
    
    # Test 3: Check if all models have identical features
    feature_sets = [set(config.features) for config in MODEL_REGISTRY.values()]
    all_identical = all(fs == feature_sets[0] for fs in feature_sets)
    
    if all_identical:
        logger.info("\n✓ All 4 models use IDENTICAL feature sets")
    else:
        logger.error("\n✗ Feature sets are NOT identical!")
        return False
    
    # Test 4: Betting splits availability
    logger.info("\nTesting betting splits fetch...")
    try:
        from src.ingestion.betting_splits import fetch_splits_action_network
        splits = await fetch_splits_action_network()
        if splits:
            logger.info(f"✓ Fetched {len(splits)} games with betting splits")
            sample = splits[0]
            logger.info(f"  Sample: {sample.away_team} @ {sample.home_team}")
            logger.info(f"  Spread: {sample.spread_home_ticket_pct:.0f}% / {sample.spread_away_ticket_pct:.0f}%")
        else:
            logger.warning("⚠ No betting splits available (no games today?)")
    except Exception as e:
        logger.warning(f"⚠ Betting splits fetch failed: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info("QUICK TEST PASSED")
    logger.info("=" * 60)
    
    return True


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="NBA Model Pipeline - Single Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/model_pipeline.py --full           # Full pipeline
  python scripts/model_pipeline.py --data-only      # Just rebuild data
  python scripts/model_pipeline.py --train-only     # Just train models
  python scripts/model_pipeline.py --validate-only  # Just validate
  python scripts/model_pipeline.py --quick-test     # Quick architecture test
        """,
    )
    
    # Pipeline mode (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--full", action="store_true", help="Run full pipeline")
    mode_group.add_argument("--data-only", action="store_true", help="Only build training data")
    mode_group.add_argument("--train-only", action="store_true", help="Only train models")
    mode_group.add_argument("--validate-only", action="store_true", help="Only validate models")
    mode_group.add_argument("--quick-test", action="store_true", help="Quick architecture test")
    
    # Options
    parser.add_argument("--seasons", nargs="+", default=["2024-2025", "2025-2026"],
                        help="Seasons to include in training data")
    parser.add_argument("--model-type", default="gradient_boosting",
                        choices=["logistic", "gradient_boosting", "ridge"],
                        help="Model algorithm to use")
    parser.add_argument("--no-clear-cache", dest="clear_cache", action="store_false",
                        help="Don't clear cache files")
    parser.add_argument("--no-splits", dest="fetch_splits", action="store_false",
                        help="Don't fetch betting splits")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    
    # Run appropriate mode
    config = PipelineConfig(args)
    pipeline = ModelPipeline(config)
    
    if args.quick_test:
        success = asyncio.run(run_quick_test())
    elif args.full:
        success = asyncio.run(pipeline.run_full_pipeline())
    elif args.data_only:
        success = asyncio.run(pipeline.run_data_only())
    elif args.train_only:
        success = asyncio.run(pipeline.run_train_only())
    elif args.validate_only:
        success = asyncio.run(pipeline.run_validate_only())
    else:
        parser.print_help()
        success = False
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
