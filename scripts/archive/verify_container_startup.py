#!/usr/bin/env python3
"""
Container Startup Verification Script

Verifies that all required model files exist and are readable before the container starts.
This prevents silent failures and provides clear error messages.
"""
import sys
from pathlib import Path

REQUIRED_MODELS = [
    "spreads_model.joblib",
    "totals_model.joblib",
    "first_half_spread_model.pkl",
    "first_half_spread_features.pkl",
    "first_half_total_model.pkl",
    "first_half_total_features.pkl",
]

def verify_models(models_dir: Path) -> bool:
    """
    Verify all required model files exist and are readable.
    
    Returns:
        True if all models exist, False otherwise
    """
    errors = []
    warnings = []
    
    models_dir = Path(models_dir)
    
    if not models_dir.exists():
        print(f"❌ ERROR: Models directory does not exist: {models_dir}")
        return False
    
    print(f"🔍 Checking models in: {models_dir}")
    print()
    
    # Check each required model
    for model_file in REQUIRED_MODELS:
        model_path = models_dir / model_file
        if not model_path.exists():
            errors.append(f"❌ MISSING: {model_file}")
        elif not model_path.is_file():
            errors.append(f"❌ NOT A FILE: {model_file}")
        else:
            # Check if readable
            try:
                with open(model_path, 'rb') as f:
                    f.read(1)  # Try to read at least 1 byte
                print(f"✅ {model_file} ({model_path.stat().st_size:,} bytes)")
            except PermissionError:
                errors.append(f"❌ PERMISSION DENIED: {model_file}")
            except Exception as e:
                errors.append(f"❌ ERROR READING {model_file}: {e}")
    
    print()
    
    if errors:
        print("=" * 70)
        print("❌ CONTAINER STARTUP VERIFICATION FAILED")
        print("=" * 70)
        for error in errors:
            print(f"  {error}")
        print()
        print("To fix:")
        print(f"  1. Ensure all model files exist in: {models_dir}")
        print(f"  2. Check file permissions (should be readable)")
        print(f"  3. Verify models are copied from models/production/ during build")
        return False
    else:
        print("=" * 70)
        print("✅ ALL MODELS VERIFIED - Container ready to start")
        print("=" * 70)
        print(f"   Total models: {len(REQUIRED_MODELS)}")
        print(f"   Location: {models_dir}")
        return True


if __name__ == "__main__":
    # Default to container path, allow override
    models_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/app/data/processed/models")
    
    success = verify_models(models_dir)
    sys.exit(0 if success else 1)
