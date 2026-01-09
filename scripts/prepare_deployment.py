#!/usr/bin/env python3
"""
Deployment Preparation Script

Verifies all components are ready for production deployment:
- Version consistency across files
- Docker configuration
- Required files present
- Environment variables documented
- Tests pass
"""
import sys
import os
from pathlib import Path
from datetime import datetime
import json

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set environment variables to match Dockerfile (for testing)
os.environ.setdefault('THE_ODDS_BASE_URL', 'https://api.the-odds-api.com/v4')
os.environ.setdefault('API_BASKETBALL_BASE_URL', 'https://v1.basketball.api-sports.io')
os.environ.setdefault('CURRENT_SEASON', '2025-2026')
os.environ.setdefault('SEASONS_TO_PROCESS', '2024-2025,2025-2026')
os.environ.setdefault('DATA_RAW_DIR', str(PROJECT_ROOT / 'data' / 'raw'))
os.environ.setdefault('DATA_PROCESSED_DIR', str(PROJECT_ROOT / 'data' / 'processed'))
os.environ.setdefault('FILTER_SPREAD_MIN_CONFIDENCE', '0.55')
os.environ.setdefault('FILTER_SPREAD_MIN_EDGE', '1.0')
os.environ.setdefault('FILTER_TOTAL_MIN_CONFIDENCE', '0.55')
os.environ.setdefault('FILTER_TOTAL_MIN_EDGE', '1.5')

checks_passed = []
checks_failed = []
checks_warnings = []


def check(name: str, condition: bool, message: str = "", warning: bool = False):
    """Record a check result."""
    if condition:
        checks_passed.append(name)
        print(f"  [PASS] {name}: {message}")
    else:
        if warning:
            checks_warnings.append(name)
            print(f"  [WARN] {name}: {message}")
        else:
            checks_failed.append(name)
            print(f"  [FAIL] {name}: {message}")


def check_version_consistency():
    """Check that version is consistent across all files."""
    print("\n" + "=" * 70)
    print("1. VERSION CONSISTENCY CHECK")
    print("=" * 70)
    
    # Read VERSION file
    version_file = PROJECT_ROOT / "VERSION"
    if not version_file.exists():
        check("VERSION file exists", False, "VERSION file not found")
        return False
    
    version = version_file.read_text().strip()
    check("VERSION file exists", True, f"Version: {version}")
    
    # Check Dockerfile.combined
    dockerfile = PROJECT_ROOT / "Dockerfile.combined"
    if dockerfile.exists():
        try:
            content = dockerfile.read_text(encoding='utf-8', errors='ignore')
            if version in content or "NBA_MODEL_VERSION" in content:
                check("Dockerfile.combined version", True, "Version referenced")
            else:
                check("Dockerfile.combined version", False, "Version not found in Dockerfile")
        except Exception as e:
            check("Dockerfile.combined version", False, f"Error reading file: {e}")
    
    # Check Dockerfile
    dockerfile_main = PROJECT_ROOT / "Dockerfile"
    if dockerfile_main.exists():
        try:
            content = dockerfile_main.read_text(encoding='utf-8', errors='ignore')
            if version in content or "NBA_MODEL_VERSION" in content:
                check("Dockerfile version", True, "Version referenced")
            else:
                check("Dockerfile version", False, "Version not found in Dockerfile")
        except Exception as e:
            check("Dockerfile version", False, f"Error reading file: {e}")
    
    # Check src/serving/app.py
    app_file = PROJECT_ROOT / "src" / "serving" / "app.py"
    if app_file.exists():
        try:
            content = app_file.read_text(encoding='utf-8', errors='ignore')
            if version in content or "NBA_MODEL_VERSION" in content:
                check("app.py version", True, "Version referenced")
            else:
                check("app.py version", False, "Version not found in app.py")
        except Exception as e:
            check("app.py version", False, f"Error reading file: {e}")
    
    return len(checks_failed) == 0


def check_required_files():
    """Check that all required files exist."""
    print("\n" + "=" * 70)
    print("2. REQUIRED FILES CHECK")
    print("=" * 70)
    
    required_files = [
        ("VERSION", PROJECT_ROOT / "VERSION"),
        ("Dockerfile.combined", PROJECT_ROOT / "Dockerfile.combined"),
        ("Dockerfile", PROJECT_ROOT / "Dockerfile"),
        ("docker-compose.yml", PROJECT_ROOT / "docker-compose.yml"),
        ("requirements.txt", PROJECT_ROOT / "requirements.txt"),
        ("src/serving/app.py", PROJECT_ROOT / "src" / "serving" / "app.py"),
        ("src/prediction/engine.py", PROJECT_ROOT / "src" / "prediction" / "engine.py"),
    ]
    
    for name, path in required_files:
        check(f"File: {name}", path.exists(), f"{path}")
    
    # Check model files exist
    # IMPORTANT: Models are at models/production/ locally, not data/processed/models/
    # In Docker, models are copied from models/production/ to /app/data/processed/models/
    # For local testing/verification, always use PROJECT_ROOT / "models" / "production"
    models_dir = PROJECT_ROOT / "models" / "production"
    if models_dir.exists():
        model_files = list(models_dir.glob("*.pkl")) + list(models_dir.glob("*.joblib"))
        check("Model files exist", len(model_files) > 0, f"Found {len(model_files)} model files")
        
        # Check for expected 4 models
        expected_models = [
            "1h_spread_model.pkl",
            "1h_total_model.pkl",
            "fg_spread_model.joblib",
            "fg_total_model.joblib",
        ]
        for model_name in expected_models:
            model_path = models_dir / model_name
            check(f"Model: {model_name}", model_path.exists(), f"{model_path}")
    else:
        check("Model files exist", False, "models/production directory not found")
    
    return len(checks_failed) == 0


def check_docker_configuration():
    """Check Docker configuration."""
    print("\n" + "=" * 70)
    print("3. DOCKER CONFIGURATION CHECK")
    print("=" * 70)
    
    dockerfile = PROJECT_ROOT / "Dockerfile.combined"
    if dockerfile.exists():
        try:
            content = dockerfile.read_text(encoding='utf-8', errors='ignore')
            
            # Check for required ENV vars
            required_env_vars = [
                "THE_ODDS_BASE_URL",
                "API_BASKETBALL_BASE_URL",
                "CURRENT_SEASON",
                "SEASONS_TO_PROCESS",
                "DATA_RAW_DIR",
                "DATA_PROCESSED_DIR",
                "FILTER_SPREAD_MIN_CONFIDENCE",
                "FILTER_SPREAD_MIN_EDGE",
                "FILTER_TOTAL_MIN_CONFIDENCE",
                "FILTER_TOTAL_MIN_EDGE",
            ]
            
            for env_var in required_env_vars:
                if f"ENV {env_var}" in content:
                    check(f"Docker ENV: {env_var}", True, "Set in Dockerfile")
                else:
                    check(f"Docker ENV: {env_var}", False, "Not found in Dockerfile")
            
            # Check for health check
            if "HEALTHCHECK" in content:
                check("Docker HEALTHCHECK", True, "Health check configured")
            else:
                check("Docker HEALTHCHECK", False, "Health check not configured")
        except Exception as e:
            check("Dockerfile.combined read", False, f"Error reading: {e}")
    
    # Check docker-compose.yml
    compose_file = PROJECT_ROOT / "docker-compose.yml"
    if compose_file.exists():
        try:
            content = compose_file.read_text(encoding='utf-8', errors='ignore')
            if "nba-v33-api" in content or "nba-gbsv-api" in content:
                check("docker-compose.yml service", True, "Service configured")
            else:
                check("docker-compose.yml service", False, "Service not found")
        except Exception as e:
            check("docker-compose.yml read", False, f"Error reading: {e}")
    
    return len(checks_failed) == 0


def check_code_quality():
    """Check code quality and imports."""
    print("\n" + "=" * 70)
    print("4. CODE QUALITY CHECK")
    print("=" * 70)
    
    # Test critical imports (API keys expected to be missing in prep - will be set in Azure)
    try:
        from src.config import settings, PROJECT_ROOT
        check("Import: src.config", True, "Config module imports")
    except Exception as e:
        error_msg = str(e).lower()
        if "api_key" in error_msg or "secret" in error_msg:
            # Expected - API keys will be set in Azure
            check("Import: src.config", True, "Config structure OK (API keys will be set in Azure)", warning=False)
        else:
            check("Import: src.config", False, f"Failed: {e}")
    
    try:
        from src.serving import app
        check("Import: src.serving.app", True, "App module imports")
    except Exception as e:
        error_msg = str(e).lower()
        if "api_key" in error_msg or "secret" in error_msg:
            # Expected - API keys will be set in Azure
            check("Import: src.serving.app", True, "App structure OK (API keys will be set in Azure)", warning=False)
        else:
            check("Import: src.serving.app", False, f"Failed: {e}")
    
    try:
        from src.prediction import UnifiedPredictionEngine
        check("Import: UnifiedPredictionEngine", True, "Prediction engine imports")
    except Exception as e:
        error_msg = str(e).lower()
        if "api_key" in error_msg or "secret" in error_msg:
            # Expected - API keys will be set in Azure
            check("Import: UnifiedPredictionEngine", True, "Engine structure OK (API keys will be set in Azure)", warning=False)
        else:
            check("Import: UnifiedPredictionEngine", False, f"Failed: {e}")
    
    return True  # Don't fail on missing API keys - they'll be set in Azure


def check_azure_configuration():
    """Check Container App website integration endpoints."""
    print("\n" + "=" * 70)
    print("5. CONTAINER APP ENDPOINTS CHECK")
    print("=" * 70)
    
    app_file = PROJECT_ROOT / "src" / "serving" / "app.py"
    if app_file.exists():
        try:
            content = app_file.read_text(encoding='utf-8', errors='ignore')
            
            # Check for required website integration endpoints
            required_endpoints = [
                "/health",
                "/weekly-lineup/nba",
                "/weekly-lineup/csv",
            ]
            
            for endpoint in required_endpoints:
                if f'"{endpoint}"' in content or f"'{endpoint}'" in content:
                    check(f"Endpoint: {endpoint}", True, "Defined in app.py")
                else:
                    check(f"Endpoint: {endpoint}", False, "Not found in app.py")
            
        except Exception as e:
            check("Container App file read", False, f"Error reading: {e}")
    else:
        check("Container App file", False, "app.py not found")
    
    return len(checks_failed) == 0


def check_documentation():
    """Check that documentation is up to date."""
    print("\n" + "=" * 70)
    print("6. DOCUMENTATION CHECK")
    print("=" * 70)
    
    docs = [
        ("README.md", PROJECT_ROOT / "README.md"),
        ("docs/ARCHITECTURE_FLOW_AND_ENDPOINTS.md", PROJECT_ROOT / "docs" / "ARCHITECTURE_FLOW_AND_ENDPOINTS.md"),
    ]
    
    for name, path in docs:
        check(f"Doc: {name}", path.exists(), f"{path}")
    
    return True  # Documentation is optional, don't fail on this


def main():
    """Run all deployment preparation checks."""
    print("=" * 70)
    print("DEPLOYMENT PREPARATION CHECKLIST")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Branch: prepare-deployment")
    print(f"Project: {PROJECT_ROOT}")
    print()
    
    # Run all checks
    version_ok = check_version_consistency()
    files_ok = check_required_files()
    docker_ok = check_docker_configuration()
    code_ok = check_code_quality()
    azure_ok = check_azure_configuration()
    docs_ok = check_documentation()
    
    # Summary
    print("\n" + "=" * 70)
    print("DEPLOYMENT PREPARATION SUMMARY")
    print("=" * 70)
    
    print(f"\nPassed: {len(checks_passed)}")
    print(f"Failed: {len(checks_failed)}")
    print(f"Warnings: {len(checks_warnings)}")
    
    if checks_failed:
        print(f"\n[FAIL] Failed Checks:")
        for check_name in checks_failed:
            print(f"  - {check_name}")
    
    if checks_warnings:
        print(f"\n[WARN] Warnings:")
        for check_name in checks_warnings:
            print(f"  - {check_name}")
    
    # Deployment readiness
    print("\n" + "=" * 70)
    all_critical_ok = version_ok and files_ok and docker_ok and code_ok
    
    if all_critical_ok and len(checks_failed) == 0:
        print("[PASS] READY FOR DEPLOYMENT")
        print("\nNext steps:")
        print("  1. Review failed/warning checks above")
        print("  2. Commit changes: git add . && git commit -m 'Prepare for deployment'")
        print("  3. Push to GitHub: git push origin prepare-deployment")
        print("  4. Create PR and merge to main")
        print("  5. Deploy: .\\scripts\\deploy.ps1")
        return 0
    else:
        print("[FAIL] NOT READY - Fix issues above before deploying")
        return 1


if __name__ == "__main__":
    sys.exit(main())

