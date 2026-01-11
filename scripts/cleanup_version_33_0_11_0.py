#!/usr/bin/env python3
"""
Cleanup script to decommission NBA_v33.0.11.0 references.

This script:
1. Updates hardcoded version strings to use resolve_version() where appropriate
2. Updates metadata files (model_pack.json, feature_importance.json) to current version
3. Updates display strings in scripts to use current version
4. Optionally updates documentation files

Usage:
    python scripts/cleanup_version_33_0_11_0.py --dry-run    # Preview changes
    python scripts/cleanup_version_33_0_11_0.py              # Apply changes
    python scripts/cleanup_version_33_0_11_0.py --docs       # Also update docs
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Tuple

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.utils.version import resolve_version


def get_current_version() -> str:
    """Get current version from VERSION file."""
    return resolve_version()


def find_version_references() -> List[Tuple[Path, int, str]]:
    """Find all files containing 33.0.11.0 references."""
    patterns = [
        r'33\.0\.11\.0',
        r'v33\.0\.11\.0',
        r'NBA_v33\.0\.11\.0',
    ]
    
    references = []
    
    # Files to search
    search_paths = [
        project_root / 'scripts',
        project_root / 'src',
        project_root / 'models',
        project_root / 'tests',
    ]
    
    for search_path in search_paths:
        if not search_path.exists():
            continue
            
        for file_path in search_path.rglob('*.py'):
            if '__pycache__' in str(file_path):
                continue
                
            try:
                content = file_path.read_text(encoding='utf-8')
                for pattern in patterns:
                    for match in re.finditer(pattern, content, re.IGNORECASE):
                        references.append((file_path, match.start(), match.group()))
            except Exception:
                pass
    
    # Also check JSON files
    for json_file in [
        project_root / 'models' / 'production' / 'model_pack.json',
        project_root / 'models' / 'production' / 'feature_importance.json',
        project_root / 'pyproject.toml',
    ]:
        if json_file.exists():
            try:
                content = json_file.read_text(encoding='utf-8')
                for pattern in patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        references.append((json_file, 0, 'found'))
            except Exception:
                pass
    
    return references


def update_run_slate_py(file_path: Path, current_version: str, dry_run: bool) -> bool:
    """Update run_slate.py to use resolve_version() instead of hardcoded version."""
    content = file_path.read_text(encoding='utf-8')
    original = content
    
    # Check if already imports resolve_version
    needs_import = 'from src.utils.version import resolve_version' not in content
    
    # Replace hardcoded versions in f-strings (these are inside f-strings, so need {resolve_version()...})
    # Line 318: HTML subtitle - in f-string
    content = re.sub(
        r'\| v33\.0\.11\.0',
        r'| {resolve_version().replace("NBA_v", "v")}',
        content
    )
    
    # Line 462: HTML footer - in f-string
    content = re.sub(
        r'Model v33\.0\.11\.0',
        r'Model {resolve_version().replace("NBA_v", "v")}',
        content
    )
    
    # Line 832: Print statement - in f-string
    content = re.sub(
        r'NBA PREDICTION SYSTEM v33\.0\.11\.0',
        r'NBA PREDICTION SYSTEM {resolve_version().replace("NBA_v", "v")}',
        content
    )
    
    # Add import if needed and we're making changes
    if needs_import and content != original:
        # Find a good place to add import (after zoneinfo import)
        content = re.sub(
            r'(from zoneinfo import ZoneInfo\n)',
            r'\1from src.utils.version import resolve_version\n',
            content
        )
    
    if content != original:
        if not dry_run:
            file_path.write_text(content, encoding='utf-8')
        return True
    return False


def update_train_models_py(file_path: Path, current_version: str, dry_run: bool) -> bool:
    """Update train_models.py MODEL_VERSION constant."""
    content = file_path.read_text(encoding='utf-8')
    original = content
    
    # Replace MODEL_VERSION = "33.0.11.0" with current version
    content = re.sub(
        r'MODEL_VERSION\s*=\s*["\']33\.0\.11\.0["\']',
        f'MODEL_VERSION = "{current_version.replace("NBA_v", "")}"',
        content,
        flags=re.IGNORECASE
    )
    
    if content != original:
        if not dry_run:
            file_path.write_text(content, encoding='utf-8')
        return True
    return False


def update_model_pack_json(file_path: Path, current_version: str, dry_run: bool) -> bool:
    """Update model_pack.json version fields."""
    try:
        data = json.loads(file_path.read_text(encoding='utf-8'))
        original = json.dumps(data, indent=2)
        
        # Update version fields
        if 'version' in data and '33.0.11.0' in str(data.get('version', '')):
            data['version'] = current_version
        
        if 'git_tag' in data and '33.0.11.0' in str(data.get('git_tag', '')):
            data['git_tag'] = current_version
        
        if 'release_notes' in data and '33.0.11.0' in str(data.get('release_notes', '')):
            data['release_notes'] = data['release_notes'].replace('NBA_v33.0.11.0', current_version)
        
        if 'deployment' in data and isinstance(data['deployment'], dict):
            if 'acr' in data['deployment'] and '33.0.11.0' in str(data['deployment'].get('acr', '')):
                data['deployment']['acr'] = data['deployment']['acr'].replace('NBA_v33.0.11.0', current_version)
        
        updated = json.dumps(data, indent=2)
        
        if updated != original:
            if not dry_run:
                file_path.write_text(updated, encoding='utf-8')
            return True
    except Exception as e:
        print(f"  ⚠️  Error updating {file_path}: {e}")
    
    return False


def update_feature_importance_json(file_path: Path, current_version: str, dry_run: bool) -> bool:
    """Update feature_importance.json version field."""
    try:
        data = json.loads(file_path.read_text(encoding='utf-8'))
        original = json.dumps(data, indent=2)
        
        # Update version in each market entry
        updated = False
        for market_key in data:
            if isinstance(data[market_key], dict) and 'version' in data[market_key]:
                if '33.0.11.0' in str(data[market_key].get('version', '')):
                    data[market_key]['version'] = current_version
                    updated = True
        
        if updated:
            new_content = json.dumps(data, indent=2)
            if new_content != original:
                if not dry_run:
                    file_path.write_text(new_content, encoding='utf-8')
                return True
    except Exception as e:
        print(f"  ⚠️  Error updating {file_path}: {e}")
    
    return False


def update_pyproject_toml(file_path: Path, current_version: str, dry_run: bool) -> bool:
    """Update pyproject.toml version field."""
    content = file_path.read_text(encoding='utf-8')
    original = content
    
    # Replace version = "33.0.11.0"
    content = re.sub(
        r'version\s*=\s*["\']33\.0\.11\.0["\']',
        f'version = "{current_version.replace("NBA_v", "")}"',
        content,
        flags=re.IGNORECASE
    )
    
    if content != original:
        if not dry_run:
            file_path.write_text(content, encoding='utf-8')
        return True
    return False


def update_production_comparison_py(file_path: Path, current_version: str, dry_run: bool) -> bool:
    """Update production_comparison.py version string."""
    content = file_path.read_text(encoding='utf-8')
    original = content
    
    content = re.sub(
        r'NBA v33\.0\.11\.0',
        f'NBA {current_version}',
        content,
        flags=re.IGNORECASE
    )
    
    if content != original:
        if not dry_run:
            file_path.write_text(content, encoding='utf-8')
        return True
    return False


def update_serving_app_py(file_path: Path, current_version: str, dry_run: bool) -> bool:
    """Update serving/app.py docstring example."""
    content = file_path.read_text(encoding='utf-8')
    original = content
    
    # Update docstring example
    content = re.sub(
        r'"version":\s*"v33\.0\.11\.0"',
        f'"version": "v{current_version.replace("NBA_v", "")}"',
        content,
        flags=re.IGNORECASE
    )
    
    if content != original:
        if not dry_run:
            file_path.write_text(content, encoding='utf-8')
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description='Cleanup NBA_v33.0.11.0 references')
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without applying them'
    )
    parser.add_argument(
        '--docs',
        action='store_true',
        help='Also update documentation files (README, CHANGELOG, etc.)'
    )
    args = parser.parse_args()
    
    current_version = get_current_version()
    print(f"Current version: {current_version}")
    print(f"Mode: {'DRY RUN (preview only)' if args.dry_run else 'APPLY CHANGES'}")
    print("=" * 80)
    
    updates_made = []
    
    # 1. Update run_slate.py (critical - affects user-facing output)
    run_slate_path = project_root / 'scripts' / 'run_slate.py'
    if run_slate_path.exists():
        print(f"\n[1] Updating {run_slate_path.relative_to(project_root)}...")
        if update_run_slate_py(run_slate_path, current_version, args.dry_run):
            updates_made.append(str(run_slate_path))
            print("  [OK] Updated hardcoded version strings")
        else:
            print("  [SKIP] No changes needed")
    
    # 2. Update train_models.py
    train_models_path = project_root / 'scripts' / 'train_models.py'
    if train_models_path.exists():
        print(f"\n[2] Updating {train_models_path.relative_to(project_root)}...")
        if update_train_models_py(train_models_path, current_version, args.dry_run):
            updates_made.append(str(train_models_path))
            print("  [OK] Updated MODEL_VERSION constant")
        else:
            print("  [SKIP] No changes needed")
    
    # 3. Update model_pack.json
    model_pack_path = project_root / 'models' / 'production' / 'model_pack.json'
    if model_pack_path.exists():
        print(f"\n[3] Updating {model_pack_path.relative_to(project_root)}...")
        if update_model_pack_json(model_pack_path, current_version, args.dry_run):
            updates_made.append(str(model_pack_path))
            print("  [OK] Updated version metadata")
        else:
            print("  [SKIP] No changes needed")
    
    # 4. Update feature_importance.json
    feature_importance_path = project_root / 'models' / 'production' / 'feature_importance.json'
    if feature_importance_path.exists():
        print(f"\n[4] Updating {feature_importance_path.relative_to(project_root)}...")
        if update_feature_importance_json(feature_importance_path, current_version, args.dry_run):
            updates_made.append(str(feature_importance_path))
            print("  [OK] Updated version metadata")
        else:
            print("  [SKIP] No changes needed")
    
    # 5. Update pyproject.toml
    pyproject_path = project_root / 'pyproject.toml'
    if pyproject_path.exists():
        print(f"\n[5] Updating {pyproject_path.relative_to(project_root)}...")
        if update_pyproject_toml(pyproject_path, current_version, args.dry_run):
            updates_made.append(str(pyproject_path))
            print("  [OK] Updated package version")
        else:
            print("  [SKIP] No changes needed")
    
    # 6. Update production_comparison.py
    prod_comp_path = project_root / 'scripts' / 'production_comparison.py'
    if prod_comp_path.exists():
        print(f"\n[6] Updating {prod_comp_path.relative_to(project_root)}...")
        if update_production_comparison_py(prod_comp_path, current_version, args.dry_run):
            updates_made.append(str(prod_comp_path))
            print("  [OK] Updated version string")
        else:
            print("  [SKIP] No changes needed")
    
    # 7. Update serving/app.py docstring
    serving_app_path = project_root / 'src' / 'serving' / 'app.py'
    if serving_app_path.exists():
        print(f"\n[7] Updating {serving_app_path.relative_to(project_root)}...")
        if update_serving_app_py(serving_app_path, current_version, args.dry_run):
            updates_made.append(str(serving_app_path))
            print("  [OK] Updated docstring example")
        else:
            print("  [SKIP] No changes needed")
    
    # Summary
    print("\n" + "=" * 80)
    if updates_made:
        print(f"[SUCCESS] Updated {len(updates_made)} file(s):")
        for file_path in updates_made:
            print(f"   - {Path(file_path).relative_to(project_root)}")
    else:
        print("[INFO] No files needed updates")
    
    if args.dry_run:
        print("\n[NOTE] Run without --dry-run to apply changes")
    else:
        print("\n[SUCCESS] Cleanup complete!")
        print("\n[NOTE] Documentation files (README.md, CHANGELOG.md, etc.) were not updated.")
        print("       These contain historical references and can be updated manually if needed.")


if __name__ == '__main__':
    main()
