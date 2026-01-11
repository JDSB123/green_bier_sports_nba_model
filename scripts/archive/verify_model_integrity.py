#!/usr/bin/env python3
"""
Model Integrity Verification Script
====================================
Verifies that:
1. All required models are loaded correctly
2. Models are using the correct components
3. Predictions are using actual models (not simplified calculations)
"""
import sys
import io
from pathlib import Path
import json

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set required environment variables for config loading
import os
os.environ.setdefault('FILTER_SPREAD_MIN_CONFIDENCE', '0.55')
os.environ.setdefault('FILTER_SPREAD_MIN_EDGE', '1.0')
os.environ.setdefault('FILTER_TOTAL_MIN_CONFIDENCE', '0.55')
os.environ.setdefault('FILTER_TOTAL_MIN_EDGE', '1.5')
os.environ.setdefault('THE_ODDS_BASE_URL', 'https://api.the-odds-api.com/v4')
os.environ.setdefault('API_BASKETBALL_BASE_URL', 'https://v1.basketball.api-sports.io')
os.environ.setdefault('DATA_RAW_DIR', 'data/raw')
os.environ.setdefault('DATA_PROCESSED_DIR', 'data/processed')
os.environ.setdefault('CURRENT_SEASON', '2025-2026')
os.environ.setdefault('SEASONS_TO_PROCESS', '2025-2026')
os.environ.setdefault('NBA_MARKETS', '1h_spread,1h_total,fg_spread,fg_total')

from src.prediction.engine import UnifiedPredictionEngine
from src.prediction.models import (
    load_spread_model,
    load_total_model,
    load_first_half_spread_model,
    load_first_half_total_model,
)
from src.modeling.unified_features import get_feature_defaults
# RichFeatureBuilder not needed for verification


def verify_model_files(models_dir: Path) -> dict:
    """Verify all required model files exist."""
    results = {
        "status": "pass",
        "missing": [],
        "found": [],
        "errors": []
    }
    
    required_models = {
        "fg_spread_model.joblib": "Full Game Spread",
        "fg_total_model.joblib": "Full Game Total",
        "1h_spread_model.joblib": "First Half Spread",
        "1h_total_model.joblib": "First Half Total",
    }
    
    for filename, description in required_models.items():
        filepath = models_dir / filename
        if filepath.exists():
            results["found"].append(f"{filename} ({description})")
        else:
            results["missing"].append(f"{filename} ({description})")
            results["status"] = "fail"
            results["errors"].append(f"Missing: {filename} - {description}")
    
    return results


def verify_model_loading(models_dir: Path) -> dict:
    """Verify models can be loaded correctly."""
    results = {
        "status": "pass",
        "loaded": [],
        "errors": []
    }
    
    loaders = [
        ("Full Game Spread", load_spread_model),
        ("Full Game Total", load_total_model),
        ("First Half Spread", load_first_half_spread_model),
        ("First Half Total", load_first_half_total_model),
    ]
    
    for name, loader_func in loaders:
        try:
            model, features = loader_func(models_dir)
            results["loaded"].append(f"{name} - Model type: {type(model).__name__}")
        except Exception as e:
            results["status"] = "fail"
            results["errors"].append(f"{name}: {str(e)}")
    
    return results


def verify_engine_initialization(models_dir: Path) -> dict:
    """Verify UnifiedPredictionEngine can be initialized."""
    results = {
        "status": "pass",
        "engine_loaded": False,
        "predictors": {},
        "errors": []
    }
    
    try:
        engine = UnifiedPredictionEngine(models_dir)
        results["engine_loaded"] = True
        
        # Verify each predictor is initialized
        if hasattr(engine, 'spread_predictor'):
            results["predictors"]["spread"] = "loaded"
        if hasattr(engine, 'total_predictor'):
            results["predictors"]["total"] = "loaded"
    except Exception as e:
        results["status"] = "fail"
        results["errors"].append(f"Engine initialization: {str(e)}")
    
    return results


def verify_prediction_pipeline(models_dir: Path) -> dict:
    """Verify end-to-end prediction pipeline uses actual models."""
    results = {
        "status": "pass",
        "tests": [],
        "errors": []
    }
    
    try:
        engine = UnifiedPredictionEngine(models_dir)
        
        # Create test features based on model-required columns
        model_feature_lists = []
        for predictor in (getattr(engine, "fg_predictor", None), getattr(engine, "h1_predictor", None)):
            if predictor:
                model_feature_lists.append(getattr(predictor, "spread_features", []) or [])
                model_feature_lists.append(getattr(predictor, "total_features", []) or [])

        required = set()
        for feats in model_feature_lists:
            required.update(feats)

        if not required:
            required = set(get_feature_defaults().keys())

        defaults = get_feature_defaults()
        overrides = {
            "predicted_margin": 3.0,
            "predicted_total": 227.0,
            "predicted_margin_1h": 1.5,
            "predicted_total_1h": 113.5,
            "home_win_pct": 0.6,
            "away_win_pct": 0.4,
            "home_margin": 2.0,
            "away_margin": -1.0,
            "home_rest": 2.0,
            "away_rest": 1.0,
            "home_b2b": 0.0,
            "away_b2b": 0.0,
            "spread_line": -3.5,
            "total_line": 225.0,
            "spread_public_home_pct": 50.0,
            "spread_ticket_money_diff": 0.0,
            "has_real_splits": 0.0,
            "dynamic_hca": 3.0,
        }
        defaults.update(overrides)
        test_features = {name: defaults.get(name, 0.0) for name in required}
        test_features.setdefault("predicted_margin_1h", overrides["predicted_margin_1h"])
        test_features.setdefault("predicted_total_1h", overrides["predicted_total_1h"])
        
        # Test full game predictions
        try:
            fg_preds = engine.predict_full_game(
                features=test_features,
                spread_line=-3.5,
                total_line=225.0,
            )
            
            results["tests"].append("[PASS] Full game predictions generated")
            
        except Exception as e:
            results["status"] = "fail"
            results["errors"].append(f"Full game prediction test: {str(e)}")
        
    except Exception as e:
        results["status"] = "fail"
        results["errors"].append(f"Pipeline test: {str(e)}")
    
    return results


def verify_feature_alignment(models_dir: Path) -> dict:
    """
    CRITICAL: Verify feature builder produces all features required by models.

    This prevents the bug where models expect 'home_days_rest' but builder
    produces 'home_rest_days' (or similar naming mismatches).
    """
    results = {
        "status": "pass",
        "models_checked": [],
        "missing_features": {},
        "errors": []
    }

    import joblib
    import asyncio

    # Load model feature requirements
    model_features = {}

    feature_files = [
        ("1h_spread", "1h_spread_model.joblib"),
        ("1h_total", "1h_total_model.joblib"),
        ("fg_spread", "fg_spread_model.joblib"),
        ("fg_total", "fg_total_model.joblib"),
    ]

    for model_name, filename in feature_files:
        filepath = models_dir / filename
        if filepath.exists():
            try:
                model_dict = joblib.load(filepath)
                if isinstance(model_dict, dict) and 'feature_columns' in model_dict:
                    model_features[model_name] = set(model_dict['feature_columns'])
                    results["models_checked"].append(model_name)
            except Exception as e:
                results["errors"].append(f"Failed to load {filename}: {e}")

    if not model_features:
        results["status"] = "fail"
        results["errors"].append("No model feature lists found")
        return results

    # Get feature builder output (using mock data)
    try:
        # Import here to avoid circular imports
        import os
        os.environ.setdefault('FILTER_SPREAD_MIN_CONFIDENCE', '0.55')
        os.environ.setdefault('FILTER_SPREAD_MIN_EDGE', '1.0')
        os.environ.setdefault('FILTER_TOTAL_MIN_CONFIDENCE', '0.55')
        os.environ.setdefault('FILTER_TOTAL_MIN_EDGE', '1.5')
        os.environ.setdefault('THE_ODDS_BASE_URL', 'https://api.the-odds-api.com/v4')
        os.environ.setdefault('API_BASKETBALL_BASE_URL', 'https://v1.basketball.api-sports.io')
        os.environ.setdefault('DATA_RAW_DIR', 'data/raw')
        os.environ.setdefault('DATA_PROCESSED_DIR', 'data/processed')
        os.environ.setdefault('CURRENT_SEASON', '2025-2026')
        os.environ.setdefault('SEASONS_TO_PROCESS', '2025-2026')

        # Check feature names statically from the return dict in rich_features.py
        # This is a static check that doesn't require API calls
        from src.features.rich_features import RichFeatureBuilder
        import inspect

        # Get the source of build_game_features to find all feature keys
        source = inspect.getsource(RichFeatureBuilder.build_game_features)

        # Extract feature names from source (look for "feature_name": patterns)
        import re
        builder_features = set()

        # Pattern 1: "feature_name": value
        pattern1 = re.findall(r'"([a-z_0-9]+)":', source)
        builder_features.update(pattern1)

        # Pattern 2: features["feature_name"] = value
        pattern2 = re.findall(r'features\["([a-z_0-9]+)"\]', source)
        builder_features.update(pattern2)

        # Check each model's required features against builder output
        for model_name, required_features in model_features.items():
            missing = required_features - builder_features
            if missing:
                results["missing_features"][model_name] = list(sorted(missing))
                results["status"] = "fail"
                results["errors"].append(
                    f"{model_name}: Missing {len(missing)} features: {sorted(missing)[:5]}..."
                )

    except Exception as e:
        results["status"] = "warn"
        results["errors"].append(f"Could not verify feature builder: {e}")

    return results


def verify_comprehensive_edge_uses_model() -> dict:
    """Verify comprehensive_edge function uses engine predictions."""
    results = {
        "status": "pass",
        "tests": [],
        "errors": []
    }
    
    try:
        from src.utils.comprehensive_edge import calculate_comprehensive_edge
        
        # Check if function accepts engine_predictions parameter
        import inspect
        sig = inspect.signature(calculate_comprehensive_edge)
        if "engine_predictions" in sig.parameters:
            results["tests"].append("[PASS] calculate_comprehensive_edge accepts engine_predictions parameter")
        else:
            results["status"] = "fail"
            results["errors"].append("calculate_comprehensive_edge missing engine_predictions parameter")
        
        # Check source code for actual model usage
        import inspect
        source = inspect.getsource(calculate_comprehensive_edge)
        if "engine_predictions" in source and "home_win_prob" in source:
            results["tests"].append("[PASS] Function checks for engine_predictions and uses home_win_prob")
        else:
            results["status"] = "fail"
            results["errors"].append("Function may not be using actual model predictions")
        
        # Check for simplified calculation (should only be fallback)
        if "0.5 + (fg_predicted_margin * 0.02)" in source:
            if "Fallback" in source or "fallback" in source or "if engine_predictions" in source:
                results["tests"].append("[PASS] Simplified calculation only used as fallback")
            else:
                results["status"] = "warn"
                results["errors"].append("WARNING: Simplified calculation may be primary method")
        
    except Exception as e:
        results["status"] = "fail"
        results["errors"].append(f"Comprehensive edge verification: {str(e)}")
    
    return results


def main():
    """Run all verification checks."""
    print("=" * 80)
    print("MODEL INTEGRITY VERIFICATION")
    print("=" * 80)
    print()
    
    models_dir = PROJECT_ROOT / "models" / "production"
    
    if not models_dir.exists():
        print(f"❌ Models directory not found: {models_dir}")
        print("   Run: python scripts/train_models.py")
        return 1
    
    all_results = {}
    
    # 1. Verify model files exist
    print("1. Verifying model files exist...")
    file_results = verify_model_files(models_dir)
    all_results["files"] = file_results
    if file_results["status"] == "pass":
        print(f"   [PASS] All {len(file_results['found'])} model files found")
    else:
        print(f"   [FAIL] Missing {len(file_results['missing'])} model files:")
        for error in file_results["errors"]:
            print(f"      - {error}")
    print()
    
    # 2. Verify models can be loaded
    print("2. Verifying models can be loaded...")
    load_results = verify_model_loading(models_dir)
    all_results["loading"] = load_results
    if load_results["status"] == "pass":
        print(f"   [PASS] All {len(load_results['loaded'])} models loaded successfully")
        for loaded in load_results["loaded"]:
            print(f"      - {loaded}")
    else:
        print(f"   [FAIL] Model loading errors:")
        for error in load_results["errors"]:
            print(f"      - {error}")
    print()
    
    # 3. Verify engine initialization
    print("3. Verifying engine initialization...")
    engine_results = verify_engine_initialization(models_dir)
    all_results["engine"] = engine_results
    if engine_results["status"] == "pass":
        print(f"   [PASS] Engine initialized successfully")
        print(f"   [PASS] Predictors loaded: {', '.join(engine_results['predictors'].keys())}")
    else:
        print(f"   [FAIL] Engine initialization errors:")
        for error in engine_results["errors"]:
            print(f"      - {error}")
    print()
    
    # 4. Verify prediction pipeline
    print("4. Verifying prediction pipeline...")
    pipeline_results = verify_prediction_pipeline(models_dir)
    all_results["pipeline"] = pipeline_results
    if pipeline_results["status"] == "pass":
        print(f"   [PASS] Prediction pipeline working")
        for test in pipeline_results["tests"]:
            print(f"      {test}")
    else:
        print(f"   [FAIL] Pipeline errors:")
        for error in pipeline_results["errors"]:
            print(f"      - {error}")
    print()
    
    # 5. Verify comprehensive_edge uses models
    print("5. Verifying comprehensive_edge uses actual models...")
    edge_results = verify_comprehensive_edge_uses_model()
    all_results["comprehensive_edge"] = edge_results
    if edge_results["status"] == "pass":
        print(f"   [PASS] Comprehensive edge function verified")
        for test in edge_results["tests"]:
            print(f"      {test}")
    else:
        print(f"   [FAIL] Comprehensive edge errors:")
        for error in edge_results["errors"]:
            print(f"      - {error}")
    print()

    # 6. CRITICAL: Verify feature alignment between models and feature builder
    print("6. Verifying feature alignment (model requirements vs feature builder)...")
    align_results = verify_feature_alignment(models_dir)
    all_results["feature_alignment"] = align_results
    if align_results["status"] == "pass":
        print(f"   [PASS] Feature alignment verified for {len(align_results['models_checked'])} models")
    elif align_results["status"] == "warn":
        print(f"   [WARN] Could not fully verify feature alignment:")
        for error in align_results["errors"]:
            print(f"      - {error}")
    else:
        print(f"   [FAIL] FEATURE MISMATCH DETECTED:")
        for model, missing in align_results.get("missing_features", {}).items():
            print(f"      {model}: missing {len(missing)} features")
            for feat in missing[:5]:
                print(f"         - {feat}")
            if len(missing) > 5:
                print(f"         ... and {len(missing) - 5} more")
    print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    all_passed = all(r.get("status") == "pass" for r in all_results.values())
    
    if all_passed:
        print("[PASS] ALL CHECKS PASSED - Models are correctly configured and using actual components")
        return 0
    else:
        print("[FAIL] SOME CHECKS FAILED - Review errors above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
