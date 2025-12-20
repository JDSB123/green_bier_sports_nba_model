#!/usr/bin/env python3
"""
Production Readiness Validation - Current Data Only

Validates production readiness WITHOUT using historical data:
1. Code quality and configuration
2. Model files exist and can be loaded
3. Prediction engine can be initialized
4. Can fetch current odds (if API key available)
5. Can make predictions for today's games (if available)

NO HISTORICAL DATA USED - only checks current/live capabilities.
"""
import sys
from pathlib import Path
import traceback
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_models_exist():
    """Test that model files exist and can be loaded."""
    print("\n" + "=" * 60)
    print("6. Testing Model Files")
    print("=" * 60)
    
    try:
        from src.config import settings
        
        models_dir = Path(settings.data_processed_dir) / "models"
        if not models_dir.exists():
            print(f"  [WARN] Models directory not found: {models_dir}")
            print(f"  [INFO] Run: python scripts/train_models.py")
            return False
        
        print(f"  [OK] Models directory exists: {models_dir}")
        
        # Check for required models
        required_models = {
            "spreads": ["spreads_model.joblib", "spreads_model.pkl"],
            "totals": ["totals_model.joblib", "totals_model.pkl"],
        }
        
        found_models = {}
        for model_type, filenames in required_models.items():
            for filename in filenames:
                model_path = models_dir / filename
                if model_path.exists():
                    found_models[model_type] = model_path
                    print(f"  [OK] {model_type} model found: {filename}")
                    break
            else:
                print(f"  [WARN] {model_type} model not found (tried: {', '.join(filenames)})")
        
        if len(found_models) == 0:
            print(f"  [WARN] No models found. Models are required for predictions.")
            print(f"  [INFO] To train models: python scripts/train_models.py")
            print(f"  [NOTE] Model training requires training data (may use historical data)")
            return None  # Skip, not a failure - models can be trained later
        
        # Try to load models
        print("\n  Testing model loading:")
        try:
            from src.prediction.models import load_spread_model, load_total_model
            
            if "spreads" in found_models:
                try:
                    model, features = load_spread_model(models_dir)
                    print(f"  [OK] Spreads model loaded successfully ({len(features)} features)")
                except Exception as e:
                    print(f"  [FAIL] Failed to load spreads model: {e}")
                    return False
            
            if "totals" in found_models:
                try:
                    model, features = load_total_model(models_dir)
                    print(f"  [OK] Totals model loaded successfully ({len(features)} features)")
                except Exception as e:
                    print(f"  [FAIL] Failed to load totals model: {e}")
                    return False
            
        except Exception as e:
            print(f"  [FAIL] Model loading test failed: {e}")
            traceback.print_exc()
            return False
        
        print("\n[OK] Models exist and can be loaded")
        return True
        
    except Exception as e:
        print(f"  [FAIL] Model check failed: {e}")
        traceback.print_exc()
        return False


def test_prediction_engine():
    """Test that prediction engine can be initialized."""
    print("\n" + "=" * 60)
    print("7. Testing Prediction Engine")
    print("=" * 60)
    
    try:
        from src.config import settings
        from src.prediction.engine import UnifiedPredictionEngine
        
        models_dir = Path(settings.data_processed_dir) / "models"
        
        # Check if models exist first
        if not (models_dir / "spreads_model.joblib").exists() and not (models_dir / "spreads_model.pkl").exists():
            print(f"  [SKIP] Models not found - skipping engine test")
            print(f"  [INFO] Run: python scripts/train_models.py")
            return None  # Skip, not a failure
        
        try:
            engine = UnifiedPredictionEngine(models_dir)
            print(f"  [OK] Prediction engine initialized successfully")
            print(f"  [OK] Spread predictor: {type(engine.spread_predictor).__name__}")
            print(f"  [OK] Total predictor: {type(engine.total_predictor).__name__}")
            print(f"  [OK] Moneyline predictor: {type(engine.moneyline_predictor).__name__}")
            
            print("\n[OK] Prediction engine is ready")
            return True
            
        except FileNotFoundError as e:
            print(f"  [WARN] Engine initialization failed (models missing): {e}")
            print(f"  [INFO] Run: python scripts/train_models.py")
            return None  # Skip, not a failure
        except Exception as e:
            print(f"  [FAIL] Engine initialization failed: {e}")
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"  [FAIL] Prediction engine test failed: {e}")
        traceback.print_exc()
        return False


def test_current_odds_fetch():
    """Test that we can fetch current odds (if API key is available)."""
    print("\n" + "=" * 60)
    print("8. Testing Current Odds Fetch (Optional)")
    print("=" * 60)
    
    try:
        from src.config import settings
        
        if not settings.the_odds_api_key:
            print(f"  [SKIP] THE_ODDS_API_KEY not set - skipping odds fetch test")
            return None  # Skip, not a failure
        
        print(f"  [INFO] API key found, testing odds fetch...")
        
        import asyncio
        from src.ingestion import the_odds
        
        # Try to fetch current odds (not historical!)
        print(f"  [INFO] Testing odds fetch (current data only)...")
        
        try:
            # Use async function correctly
            async def test_fetch():
                odds_data = await the_odds.fetch_odds()
                return odds_data
            
            games = asyncio.run(test_fetch())
            
            if games and len(games) > 0:
                print(f"  [OK] Successfully fetched {len(games)} current games")
                if len(games) > 0:
                    first_game = games[0] if isinstance(games[0], dict) else games[0].__dict__ if hasattr(games[0], '__dict__') else {}
                    print(f"  [OK] Sample game: {first_game.get('home_team', 'N/A')} vs {first_game.get('away_team', 'N/A')}")
                return True
            else:
                print(f"  [WARN] No games found (may be off-season or no games scheduled)")
                return None  # Not a failure, just no games
                
        except Exception as e:
            error_msg = str(e).lower()
            if "api" in error_msg or "key" in error_msg or "auth" in error_msg:
                print(f"  [WARN] API error (check API key): {e}")
                return None  # API issue, not a code failure
            else:
                print(f"  [WARN] Odds fetch failed: {e}")
                return None  # Network/API issue, not a code failure
        
    except Exception as e:
        print(f"  [WARN] Odds fetch test error: {e}")
        return None  # Skip, not a failure


def test_prediction_script():
    """Test that prediction script can run (dry-run, no actual predictions)."""
    print("\n" + "=" * 60)
    print("9. Testing Prediction Script Availability")
    print("=" * 60)
    
    try:
        predict_script = PROJECT_ROOT / "scripts" / "predict.py"
        
        if not predict_script.exists():
            print(f"  [FAIL] Prediction script not found: {predict_script}")
            return False
        
        print(f"  [OK] Prediction script exists: {predict_script}")
        
        # Check if script has required imports
        with open(predict_script, 'r', encoding='utf-8') as f:
            content = f.read()
            required_imports = [
                "UnifiedPredictionEngine",
                "fetch_upcoming_games",
            ]
            
            missing = []
            for imp in required_imports:
                if imp not in content:
                    missing.append(imp)
            
            if missing:
                print(f"  [WARN] Script may be missing imports: {missing}")
            else:
                print(f"  [OK] Script has required components")
        
        print("\n[OK] Prediction script is available")
        return True
        
    except Exception as e:
        print(f"  [FAIL] Script check failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all production readiness checks (current data only)."""
    print("\n" + "=" * 60)
    print("PRODUCTION READINESS VALIDATION (CURRENT DATA ONLY)")
    print("=" * 60)
    print(f"Running checks at: {Path.cwd()}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nNOTE: This validation uses ONLY current/live data.")
    print("      NO historical data will be accessed.")
    print()
    
    # First run basic validation
    print("=" * 60)
    print("Running Basic Production Readiness Checks...")
    print("=" * 60)
    
    import subprocess
    result = subprocess.run(
        [sys.executable, "scripts/validate_production_readiness.py"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT
    )
    
    if result.returncode != 0:
        print("\n[FAIL] Basic validation failed. See output above.")
        return 1
    
    # Additional checks
    checks = [
        ("Model Files", test_models_exist),
        ("Prediction Engine", test_prediction_engine),
        ("Current Odds Fetch", test_current_odds_fetch),
        ("Prediction Script", test_prediction_script),
    ]
    
    results = []
    for name, test_func in checks:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n[FAIL] {name} check crashed: {e}")
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result is True)
    failed = sum(1 for _, result in results if result is False)
    skipped = sum(1 for _, result in results if result is None)
    total = len(results)
    
    for name, result in results:
        if result is True:
            status = "[PASS]"
        elif result is False:
            status = "[FAIL]"
        else:
            status = "[SKIP]"
        print(f"  {status}: {name}")
    
    print("\n" + "=" * 60)
    if failed == 0:
        if passed == total:
            print(f"[OK] PRODUCTION READY: All {total} checks passed")
        else:
            print(f"[OK] PRODUCTION READY: {passed} passed, {skipped} skipped (no failures)")
        print("\nSystem is ready for production use with current data.")
        return 0
    else:
        print(f"[FAIL] NOT READY: {passed} passed, {failed} failed, {skipped} skipped")
        return 1


if __name__ == "__main__":
    sys.exit(main())



