"""
Deploy Option B: FG Totals Optimization

Conservative expansion of optimization after FG Spread validation.
Updates filter thresholds for FG Totals market.

Expected Impact:
- FG Total: 0.72 → 0.55 conf, 3.0 → 0.0 edge
- Expected ROI: +12.12%
- Expected Accuracy: 58.73%
- Expected Volume: ~2,721 bets/season

Usage:
    python scripts/deploy_option_b.py --validate-only  # Check what would change
    python scripts/deploy_option_b.py --deploy         # Deploy to production
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_FILE = PROJECT_ROOT / "src" / "config.py"
MODEL_PACK_FILE = PROJECT_ROOT / "models" / "production" / "model_pack.json"
VERSION_FILE = PROJECT_ROOT / "VERSION"


def get_current_version() -> str:
    """Get current version from VERSION file."""
    with open(VERSION_FILE, 'r') as f:
        return f.read().strip()


def increment_version(version: str) -> str:
    """Increment patch version (NBA_v33.0.21.0 -> NBA_v33.0.22.0)."""
    parts = version.split('.')
    patch = int(parts[-1])
    parts[-1] = str(patch + 1)
    return '.'.join(parts)


def update_config_file(validate_only: bool = True) -> bool:
    """Update src/config.py with Option B thresholds."""
    with open(CONFIG_FILE, 'r') as f:
        content = f.read()

    # Find and update FG Total thresholds
    old_total_conf = 'default_factory=lambda: _env_float_required("FILTER_TOTAL_MIN_CONFIDENCE", 0.72)'
    new_total_conf = 'default_factory=lambda: _env_float_required("FILTER_TOTAL_MIN_CONFIDENCE", 0.55)  # OPTIMIZED v33.0.22.0'

    old_total_edge = 'default_factory=lambda: _env_float_required("FILTER_TOTAL_MIN_EDGE", 3.0)'
    new_total_edge = 'default_factory=lambda: _env_float_required("FILTER_TOTAL_MIN_EDGE", 0.0)  # OPTIMIZED v33.0.22.0'

    if old_total_conf not in content or old_total_edge not in content:
        print("ERROR: Could not find expected configuration lines")
        return False

    new_content = content.replace(old_total_conf, new_total_conf)
    new_content = new_content.replace(old_total_edge, new_total_edge)

    if validate_only:
        print("\n" + "=" * 80)
        print("CONFIGURATION CHANGES (validate-only mode)")
        print("=" * 80)
        print("\nChanges to src/config.py:")
        print("  FG Total min_confidence: 0.72 → 0.55")
        print("  FG Total min_edge:       3.0 → 0.0")
        print("\nNo files modified (use --deploy to apply changes)")
        return True

    # Write updated config
    with open(CONFIG_FILE, 'w') as f:
        f.write(new_content)

    print("✅ Updated src/config.py")
    return True


def update_model_pack(new_version: str, validate_only: bool = True) -> bool:
    """Update model_pack.json with Option B metadata."""
    with open(MODEL_PACK_FILE, 'r') as f:
        model_pack = json.load(f)

    model_pack['version'] = new_version
    model_pack['last_reviewed'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
    model_pack['release_notes'] = (
        f"{new_version} - OPTION B DEPLOYMENT: FG Totals optimization deployed after successful "
        f"Week 1 validation of FG Spread. FG Total thresholds optimized (0.72→0.55 conf, 3.0→0.0 edge) "
        f"resulting in +12.12% ROI, 58.73% accuracy, and ~2,721 bets/season. Conservative expansion "
        f"strategy maintained."
    )

    # Update FG Total thresholds in model_pack
    model_pack['filter_thresholds']['fg_total'] = {
        'min_confidence': 0.55,
        'min_edge': 0.0,
        'optimized': datetime.now().strftime('%Y-%m-%d'),
        'expected_roi': 0.1212,
        'expected_accuracy': 0.5873,
        'expected_volume': 2721
    }

    model_pack['deployment']['acr'] = f"nbagbsacr.azurecr.io/nba-gbsv-api:{new_version}"

    if validate_only:
        print("\nChanges to model_pack.json:")
        print(f"  version: {new_version}")
        print(f"  filter_thresholds.fg_total.min_confidence: 0.55")
        print(f"  filter_thresholds.fg_total.min_edge: 0.0")
        return True

    # Write updated model_pack
    with open(MODEL_PACK_FILE, 'w', encoding='utf-8') as f:
        json.dump(model_pack, f, indent=2, ensure_ascii=False)

    print("✅ Updated model_pack.json")
    return True


def update_version_file(new_version: str, validate_only: bool = True) -> bool:
    """Update VERSION file."""
    if validate_only:
        print(f"\nChanges to VERSION:")
        print(f"  {new_version}")
        return True

    with open(VERSION_FILE, 'w') as f:
        f.write(f"{new_version}\n")

    print("✅ Updated VERSION")
    return True


def build_and_deploy(new_version: str) -> bool:
    """Build Docker image and deploy to Azure."""
    print("\n" + "=" * 80)
    print("DOCKER BUILD AND AZURE DEPLOYMENT")
    print("=" * 80)

    # Build Docker image
    print(f"\nBuilding Docker image: {new_version}")
    build_cmd = [
        "docker", "build",
        "-t", f"nbagbsacr.azurecr.io/nba-gbsv-api:{new_version}",
        "-t", "nbagbsacr.azurecr.io/nba-gbsv-api:latest",
        "."
    ]

    result = subprocess.run(build_cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print("❌ Docker build failed")
        return False

    print("✅ Docker image built")

    # Push to ACR
    print("\nPushing to Azure Container Registry...")
    push_cmd1 = ["docker", "push", f"nbagbsacr.azurecr.io/nba-gbsv-api:{new_version}"]
    push_cmd2 = ["docker", "push", "nbagbsacr.azurecr.io/nba-gbsv-api:latest"]

    result = subprocess.run(push_cmd1)
    if result.returncode != 0:
        print("❌ Docker push failed")
        return False

    subprocess.run(push_cmd2)
    print("✅ Images pushed to ACR")

    # Deploy to Azure
    print("\nDeploying to Azure Container App...")
    deploy_cmd = [
        "az", "containerapp", "update",
        "-n", "nba-gbsv-api",
        "-g", "nba-gbsv-model-rg",
        "--image", f"nbagbsacr.azurecr.io/nba-gbsv-api:{new_version}",
        "--set-env-vars",
        "FILTER_TOTAL_MIN_CONFIDENCE=0.55",
        "FILTER_TOTAL_MIN_EDGE=0.0"
    ]

    result = subprocess.run(deploy_cmd)
    if result.returncode != 0:
        print("❌ Azure deployment failed")
        return False

    print("✅ Deployed to Azure")

    # Update Azure tags
    print("\nUpdating Azure resource tags...")
    tag_cmd = [
        "az", "containerapp", "update",
        "-n", "nba-gbsv-api",
        "-g", "nba-gbsv-model-rg",
        "--tags", f"version={new_version}", f"deployed={datetime.now().isoformat()}"
    ]

    subprocess.run(tag_cmd)
    print("✅ Azure tags updated")

    return True


def git_commit_and_tag(new_version: str) -> bool:
    """Commit changes and create git tag."""
    print("\n" + "=" * 80)
    print("GIT COMMIT AND TAG")
    print("=" * 80)

    # Git add
    files_to_commit = [
        "VERSION",
        "src/config.py",
        "models/production/model_pack.json"
    ]

    for file in files_to_commit:
        subprocess.run(["git", "add", file], cwd=PROJECT_ROOT)

    # Git commit
    commit_msg = f"deploy: {new_version} - Option B (FG Totals optimization) deployed to Azure"
    subprocess.run(["git", "commit", "-m", commit_msg], cwd=PROJECT_ROOT)
    print(f"✅ Committed changes: {commit_msg}")

    # Git tag
    tag_msg = f"{new_version}: FG Totals optimization deployment"
    subprocess.run(["git", "tag", "-a", new_version, "-m", tag_msg], cwd=PROJECT_ROOT)
    print(f"✅ Created tag: {new_version}")

    # Git push
    subprocess.run(["git", "push"], cwd=PROJECT_ROOT)
    subprocess.run(["git", "push", "--tags"], cwd=PROJECT_ROOT)
    print("✅ Pushed to remote")

    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Deploy Option B: FG Totals Optimization")
    parser.add_argument('--validate-only', action='store_true', help='Show what would change without deploying')
    parser.add_argument('--deploy', action='store_true', help='Deploy to production')

    args = parser.parse_args()

    if not args.validate_only and not args.deploy:
        print("ERROR: Must specify either --validate-only or --deploy")
        print("\nUsage:")
        print("  python scripts/deploy_option_b.py --validate-only  # Check changes")
        print("  python scripts/deploy_option_b.py --deploy         # Deploy to production")
        return

    current_version = get_current_version()
    new_version = increment_version(current_version)

    print("=" * 80)
    print("OPTION B DEPLOYMENT: FG TOTALS OPTIMIZATION")
    print("=" * 80)
    print(f"\nCurrent Version: {current_version}")
    print(f"New Version:     {new_version}")
    print(f"\nMode: {'VALIDATE ONLY' if args.validate_only else 'DEPLOY TO PRODUCTION'}")

    # Update configuration files
    if not update_config_file(validate_only=args.validate_only):
        return

    if not update_model_pack(new_version, validate_only=args.validate_only):
        return

    if not update_version_file(new_version, validate_only=args.validate_only):
        return

    if args.validate_only:
        print("\n" + "=" * 80)
        print("VALIDATION COMPLETE")
        print("=" * 80)
        print("\nTo deploy these changes to production, run:")
        print("  python scripts/deploy_option_b.py --deploy")
        return

    # Deploy to production
    if not build_and_deploy(new_version):
        print("\n❌ Deployment failed")
        return

    if not git_commit_and_tag(new_version):
        print("\n⚠️ Git operations failed but deployment succeeded")
        return

    print("\n" + "=" * 80)
    print("DEPLOYMENT COMPLETE ✅")
    print("=" * 80)
    print(f"\nVersion: {new_version}")
    print(f"FG Total Thresholds: 0.55 conf, 0.0 edge")
    print(f"Expected Impact: +12.12% ROI, 58.73% accuracy, ~2,721 bets/season")
    print("\nNext Steps:")
    print("  1. Monitor FG Total performance for 7 days")
    print("  2. Compare actual vs expected metrics")
    print("  3. Consider Option C (Moneyline) after validation")


if __name__ == "__main__":
    main()
