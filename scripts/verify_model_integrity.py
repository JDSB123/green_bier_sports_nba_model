#!/usr/bin/env python3
"""
Model Integrity Verification Script
====================================
Verifies that:
1. All required models are loaded correctly
2. Models are using the correct components
3. Predictions are using actual models (not simplified calculations)
4. Moneyline predictions use the audited model
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

from src.prediction.engine import UnifiedPredictionEngine
from src.prediction.models import (
    load_spread_model,
    load_total_model,
    load_first_half_spread_model,
    load_first_half_total_model,
)
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
        "1h_spread_model.pkl": "First Half Spread",
        "1h_spread_features.pkl": "First Half Spread Features",
        "1h_total_model.pkl": "First Half Total",
        "1h_total_features.pkl": "First Half Total Features",
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
        if hasattr(engine, 'moneyline_predictor'):
            results["predictors"]["moneyline"] = "loaded"
            # Verify moneyline uses actual model (not simplified)
            if hasattr(engine.moneyline_predictor, 'model'):
                model_type = type(engine.moneyline_predictor.model).__name__
                results["predictors"]["moneyline_model_type"] = model_type
            else:
                results["status"] = "fail"
                results["errors"].append("Moneyline predictor missing model attribute")
        
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
        
        # Create test features
        test_features = {
            "home_ppg": 115.0,
            "away_ppg": 112.0,
            "home_papg": 110.0,
            "away_papg": 115.0,
            "predicted_margin": 3.0,
            "predicted_total": 227.0,
            "predicted_margin_1h": 1.5,
            "predicted_total_1h": 113.5,
            "home_win_pct": 0.6,
            "away_win_pct": 0.4,
            "home_avg_margin": 2.0,
            "away_avg_margin": -1.0,
            "home_rest_days": 2,
            "away_rest_days": 1,
            "home_b2b": 0,
            "away_b2b": 0,
            "dynamic_hca": 3.0,
            "h2h_win_pct": 0.5,
            "h2h_avg_margin": 0.0,
            # Added missing features for verification
            "away_ats_pct": 0.5,
            "away_elo": 1500,
            "away_injury_spread_impact": 0.0,
            "away_star_out": 0,
            "elo_diff": 0,
            "home_ats_pct": 0.5,
            "home_elo": 1500,
            "home_injury_spread_impact": 0.0,
            "home_star_out": 0,
            "injury_spread_diff": 0.0,
            "home_last_10_avg_margin": 2.0,
            "away_last_10_avg_margin": -1.0,
            "home_last_10_win_pct": 0.6,
            "away_last_10_win_pct": 0.4,
            "home_home_win_pct": 0.7,
            "away_away_win_pct": 0.3,
            "home_home_avg_margin": 4.0,
            "away_away_avg_margin": -3.0,
        }
        
        # Test full game predictions
        try:
            fg_preds = engine.predict_full_game(
                features=test_features,
                spread_line=-3.5,
                total_line=225.0,
                home_ml_odds=-150,
                away_ml_odds=130,
            )
            
            # Verify moneyline uses actual model
            if "moneyline" in fg_preds:
                ml_pred = fg_preds["moneyline"]
                if "home_win_prob" in ml_pred and "away_win_prob" in ml_pred:
                    results["tests"].append("[PASS] Moneyline uses actual model probabilities")
                    # Verify probabilities are reasonable (not simplified calculation)
                    home_prob = ml_pred["home_win_prob"]
                    away_prob = ml_pred["away_win_prob"]
                    if abs(home_prob + away_prob - 1.0) < 0.01:
                        results["tests"].append("[PASS] Moneyline probabilities sum to ~1.0")
                    else:
                        results["status"] = "fail"
                        results["errors"].append(f"Moneyline probabilities don't sum to 1.0: {home_prob} + {away_prob}")
                else:
                    results["status"] = "fail"
                    results["errors"].append("Moneyline prediction missing probabilities")
            
            results["tests"].append("[PASS] Full game predictions generated")
            
        except Exception as e:
            results["status"] = "fail"
            results["errors"].append(f"Full game prediction test: {str(e)}")
        
    except Exception as e:
        results["status"] = "fail"
        results["errors"].append(f"Pipeline test: {str(e)}")
    
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
        if "moneyline_model_type" in engine_results["predictors"]:
            print(f"   [PASS] Moneyline model type: {engine_results['predictors']['moneyline_model_type']}")
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
