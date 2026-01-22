#!/usr/bin/env python3
"""
🔐 SINGLE SOURCE OF TRUTH: Environment & Secrets Validator

This script validates that ALL required secrets/configs are present
BEFORE anything tries to run and fails cryptically.

Run this FIRST in any environment:
  - Local development: python scripts/validate_environment.py
  - Docker build:      Added to Dockerfile as health check
  - GitHub Actions:    First step in every workflow
  - Codespace:         Run on terminal open

EXIT CODES:
  0 = All good
  1 = Missing required secrets/config
  2 = Invalid configuration
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# ============================================================================
# SINGLE SOURCE OF TRUTH: Required Configuration
# ============================================================================

REQUIRED_SECRETS = {
    "THE_ODDS_API_KEY": {
        "description": "The Odds API key for live odds data",
        "env_var": "THE_ODDS_API_KEY",
        "secret_file": "secrets/THE_ODDS_API_KEY",
        "keyvault_name": "THE-ODDS-API-KEY",
        "github_secret": "THE_ODDS_API_KEY",
    },
    "API_BASKETBALL_KEY": {
        "description": "API-Basketball key for stats data",
        "env_var": "API_BASKETBALL_KEY",
        "secret_file": "secrets/API_BASKETBALL_KEY",
        "keyvault_name": "API-BASKETBALL-KEY",
        "github_secret": "API_BASKETBALL_KEY",
    },
}

REQUIRED_AZURE_OIDC = {
    "AZURE_CLIENT_ID": "Azure AD App Registration Client ID",
    "AZURE_TENANT_ID": "Azure AD Tenant ID",
    "AZURE_SUBSCRIPTION_ID": "Azure Subscription ID",
}

OPTIONAL_SECRETS = {
    "ACTION_NETWORK_USERNAME": "Action Network username",
    "ACTION_NETWORK_PASSWORD": "Action Network password",
    "SERVICE_API_KEY": "API authentication key (for protected endpoints)",
}

# ============================================================================
# Detection: What environment are we in?
# ============================================================================

def detect_environment() -> str:
    """Detect the current runtime environment."""
    if os.environ.get("GITHUB_ACTIONS") == "true":
        return "github_actions"
    if os.environ.get("CODESPACES") == "true":
        return "codespace"
    if os.environ.get("DOCKER_CONTAINER") or Path("/.dockerenv").exists():
        return "docker"
    if os.environ.get("AZURE_FUNCTIONS_ENVIRONMENT"):
        return "azure_function"
    return "local"


def get_project_root() -> Path:
    """Get project root directory."""
    # Try to find it relative to this script
    script_dir = Path(__file__).resolve().parent
    if (script_dir.parent / "VERSION").exists():
        return script_dir.parent
    # Fallback to cwd
    return Path.cwd()


# ============================================================================
# Validation Functions
# ============================================================================

def check_secret(name: str, config: dict) -> tuple[bool, str, str]:
    """
    Check if a secret is available from any source.
    Returns: (found, source, value_preview)
    """
    # 1. Check environment variable
    env_val = os.environ.get(config["env_var"])
    if env_val:
        preview = f"{env_val[:4]}...{env_val[-4:]}" if len(env_val) > 8 else "****"
        return True, "env_var", preview
    
    # 2. Check Docker secrets file
    project_root = get_project_root()
    secret_file = project_root / config["secret_file"]
    if secret_file.exists():
        content = secret_file.read_text().strip()
        if content and not content.startswith("your_"):
            preview = f"{content[:4]}...{content[-4:]}" if len(content) > 8 else "****"
            return True, "secret_file", preview
    
    # 3. Check /run/secrets (Docker Swarm style)
    docker_secret = Path(f"/run/secrets/{name}")
    if docker_secret.exists():
        return True, "docker_swarm", "****"
    
    return False, "not_found", ""


def check_azure_oidc() -> dict[str, bool]:
    """Check Azure OIDC credentials (for GitHub Actions)."""
    results = {}
    for name in REQUIRED_AZURE_OIDC:
        # In GitHub Actions, these are passed as secrets
        # We can't directly check them, but we can note if they're referenced
        results[name] = bool(os.environ.get(name))
    return results


def validate_version_consistency() -> tuple[bool, list[str]]:
    """Check that VERSION file matches other version references."""
    project_root = get_project_root()
    issues = []
    
    version_file = project_root / "VERSION"
    if not version_file.exists():
        return False, ["VERSION file not found"]
    
    version = version_file.read_text().strip()
    
    # Check model_pack.json
    model_pack = project_root / "models/production/model_pack.json"
    if model_pack.exists():
        try:
            data = json.loads(model_pack.read_text())
            if data.get("version") != version:
                issues.append(f"model_pack.json version ({data.get('version')}) != VERSION ({version})")
        except json.JSONDecodeError:
            issues.append("model_pack.json is invalid JSON")
    
    # Check feature_importance.json
    feature_imp = project_root / "models/production/feature_importance.json"
    if feature_imp.exists():
        try:
            data = json.loads(feature_imp.read_text())
            if data.get("version") != version:
                issues.append(f"feature_importance.json version ({data.get('version')}) != VERSION ({version})")
        except json.JSONDecodeError:
            issues.append("feature_importance.json is invalid JSON")
    
    return len(issues) == 0, issues


# ============================================================================
# Main Validation
# ============================================================================

def validate_all(verbose: bool = True) -> bool:
    """Run all validations. Returns True if all passed."""
    env = detect_environment()
    project_root = get_project_root()
    
    if verbose:
        print("=" * 60)
        print("🔐 NBA Model Environment Validator")
        print("=" * 60)
        print(f"Environment: {env}")
        print(f"Project root: {project_root}")
        print()
    
    all_passed = True
    
    # 1. Required Secrets
    if verbose:
        print("📋 REQUIRED SECRETS:")
    
    for name, config in REQUIRED_SECRETS.items():
        found, source, preview = check_secret(name, config)
        if found:
            if verbose:
                print(f"  ✅ {name}: {source} ({preview})")
        else:
            all_passed = False
            if verbose:
                print(f"  ❌ {name}: NOT FOUND")
                print(f"     → Set env var: export {config['env_var']}=your_key")
                print(f"     → Or create file: {config['secret_file']}")
    
    if verbose:
        print()
    
    # 2. Azure OIDC (only relevant in GitHub Actions)
    if env == "github_actions":
        if verbose:
            print("📋 AZURE OIDC (GitHub Actions):")
        
        oidc = check_azure_oidc()
        for name, found in oidc.items():
            if found:
                if verbose:
                    print(f"  ✅ {name}")
            else:
                # Don't fail - these are injected by the workflow
                if verbose:
                    print(f"  ⚠️  {name}: Not in env (should be in secrets)")
        
        if verbose:
            print()
    
    # 3. Version Consistency
    if verbose:
        print("📋 VERSION CONSISTENCY:")
    
    version_ok, version_issues = validate_version_consistency()
    if version_ok:
        if verbose:
            version = (project_root / "VERSION").read_text().strip()
            print(f"  ✅ All files at version: {version}")
    else:
        all_passed = False
        for issue in version_issues:
            if verbose:
                print(f"  ❌ {issue}")
        if verbose:
            print("     → Run: python scripts/bump_version.py <VERSION>")
    
    if verbose:
        print()
    
    # 4. Optional secrets (informational)
    if verbose:
        print("📋 OPTIONAL SECRETS:")
        for name, desc in OPTIONAL_SECRETS.items():
            val = os.environ.get(name)
            secret_file = project_root / "secrets" / name
            if val or secret_file.exists():
                print(f"  ✅ {name}")
            else:
                print(f"  ⚪ {name}: not set (optional)")
        print()
    
    # Summary
    if verbose:
        print("=" * 60)
        if all_passed:
            print("✅ ALL VALIDATIONS PASSED")
        else:
            print("❌ VALIDATION FAILED - Fix issues above")
        print("=" * 60)
    
    return all_passed


def print_setup_instructions():
    """Print setup instructions for the current environment."""
    env = detect_environment()
    
    print("\n" + "=" * 60)
    print("🔧 SETUP INSTRUCTIONS")
    print("=" * 60)
    
    if env == "local":
        print("""
LOCAL DEVELOPMENT:
1. Copy .env.example to .env:
   cp .env.example .env

2. Edit .env and add your API keys:
   THE_ODDS_API_KEY=your_actual_key
   API_BASKETBALL_KEY=your_actual_key

3. Create Docker secrets (for container builds):
   python scripts/manage_secrets.py from-env
""")
    
    elif env == "codespace":
        print("""
CODESPACE:
1. Add secrets to Codespace settings:
   - Go to github.com/settings/codespaces
   - Add THE_ODDS_API_KEY and API_BASKETBALL_KEY
   - Restart the Codespace

2. Or create a .env file (temporary, lost on rebuild):
   cp .env.example .env
   # Edit with your keys
""")
    
    elif env == "docker":
        print("""
DOCKER:
1. Create secrets/ directory with key files:
   mkdir -p secrets
   echo "your_key" > secrets/THE_ODDS_API_KEY
   echo "your_key" > secrets/API_BASKETBALL_KEY

2. Or pass as environment variables:
   docker run -e THE_ODDS_API_KEY=xxx -e API_BASKETBALL_KEY=xxx ...
""")
    
    elif env == "github_actions":
        print("""
GITHUB ACTIONS:
1. Go to repo Settings → Secrets → Actions
2. Add these secrets:
   - THE_ODDS_API_KEY
   - API_BASKETBALL_KEY
   - AZURE_CLIENT_ID (for OIDC)
   - AZURE_TENANT_ID (for OIDC)
   - AZURE_SUBSCRIPTION_ID (for OIDC)
""")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate NBA Model environment")
    parser.add_argument("--quiet", "-q", action="store_true", help="Only output on failure")
    parser.add_argument("--help-setup", action="store_true", help="Show setup instructions")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()
    
    if args.help_setup:
        print_setup_instructions()
        sys.exit(0)
    
    if args.json:
        # JSON mode for programmatic use
        results = {
            "environment": detect_environment(),
            "secrets": {},
            "version_consistent": False,
        }
        for name, config in REQUIRED_SECRETS.items():
            found, source, _ = check_secret(name, config)
            results["secrets"][name] = {"found": found, "source": source}
        
        version_ok, issues = validate_version_consistency()
        results["version_consistent"] = version_ok
        results["version_issues"] = issues
        results["all_passed"] = all(s["found"] for s in results["secrets"].values()) and version_ok
        
        print(json.dumps(results, indent=2))
        sys.exit(0 if results["all_passed"] else 1)
    
    passed = validate_all(verbose=not args.quiet)
    
    if not passed:
        print_setup_instructions()
        sys.exit(1)
    
    sys.exit(0)
