#!/usr/bin/env python3
"""
Bump version across all files in the NBA prediction repository.

Usage:
    python scripts/bump_version.py NBA_v33.0.9.0
    python scripts/bump_version.py NBA_v33.1.0.0 --dry-run
"""
import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Files that need version updates
VERSION_FILES = [
    ("VERSION", r"NBA_v\d+\.\d+\.\d+\.\d+", "{version}"),
    ("src/serving/app.py", r'RELEASE_VERSION = os\.getenv\("NBA_MODEL_VERSION", "NBA_v\d+\.\d+\.\d+\.\d+"\)',
     'RELEASE_VERSION = os.getenv("NBA_MODEL_VERSION", "{version}")'),
    ("src/prediction/engine.py", r'MODEL_VERSION = os\.getenv\("NBA_MODEL_VERSION", "NBA_v\d+\.\d+\.\d+\.\d+"\)',
     'MODEL_VERSION = os.getenv("NBA_MODEL_VERSION", "{version}")'),
    ("src/monitoring/prediction_logger.py", r'_MODEL_VERSION = os\.getenv\("NBA_MODEL_VERSION", "NBA_v\d+\.\d+\.\d+\.\d+"\)',
     '_MODEL_VERSION = os.getenv("NBA_MODEL_VERSION", "{version}")'),
    ("models/production/model_pack.json", r'"version":\s*"NBA_v\d+\.\d+\.\d+\.\d+"',
     '  "version": "{version}"'),
    ("models/production/model_pack.json", r'"git_tag":\s*"NBA_v\d+\.\d+\.\d+\.\d+"',
     '  "git_tag": "{version}"'),
    ("models/production/model_pack.json", r'"acr":\s*"nbagbsacr\.azurecr\.io/nba-gbsv-api:NBA_v\d+\.\d+\.\d+\.\d+"',
     '    "acr": "nbagbsacr.azurecr.io/nba-gbsv-api:{version}"'),
    ("models/production/feature_importance.json", r'"version":\s*"NBA_v\d+\.\d+\.\d+\.\d+"',
     '  "version": "{version}"'),
    ("tests/test_serving.py", r'"version":\s*"NBA_v\d+\.\d+\.\d+\.\d+"',
     '        "version": "{version}"'),
    (".github/copilot-instructions.md", r"nba-gbsv-api:NBA_v\d+\.\d+\.\d+\.\d+",
     "nba-gbsv-api:{version}"),
    ("README.md", r"current:\s*`NBA_v\d+\.\d+\.\d+\.\d+`",
     "current: `{version}`"),
    ("infra/nba/main.json", r'"defaultValue":\s*"NBA_v\d+\.\d+\.\d+\.\d+"',
     '      "defaultValue": "{version}"'),
]


def validate_version(version: str) -> bool:
    """Validate version format."""
    pattern = r"^NBA_v\d+\.\d+\.\d+\.\d+$"
    return bool(re.match(pattern, version))


def find_and_replace(file_path: Path, pattern: str, replacement: str, new_version: str, dry_run: bool = False) -> Tuple[bool, int]:
    """Find and replace version in a file."""
    try:
        content = file_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"  [WARN] File not found: {file_path}")
        return False, 0
    
    # Count matches before replacement
    matches = len(re.findall(pattern, content))
    if matches == 0:
        print(f"  [WARN] No matches found in {file_path}")
        return False, 0
    
    # Perform replacement
    new_content = re.sub(pattern, replacement.format(version=new_version), content)
    
    if dry_run:
        print(f"  [DRY RUN] Would update {file_path} ({matches} occurrence(s))")
        return True, matches
    else:
        file_path.write_text(new_content, encoding="utf-8")
        print(f"  [OK] Updated {file_path} ({matches} occurrence(s))")
        return True, matches


def main():
    parser = argparse.ArgumentParser(description="Bump NBA model version across all files")
    parser.add_argument("version", help="New version (e.g., NBA_v33.0.9.0)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed without modifying files")
    args = parser.parse_args()
    
    # Validate version format
    if not validate_version(args.version):
        print(f"[ERROR] Invalid version format: {args.version}")
        print("   Expected format: NBA_v<MAJOR>.<MINOR>.<PATCH>.<BUILD>")
        print("   Example: NBA_v33.0.9.0")
        sys.exit(1)
    
    print("=" * 80)
    print(f"NBA Model Version Bump")
    print("=" * 80)
    print(f"  New Version: {args.version}")
    print(f"  Mode:        {'DRY RUN' if args.dry_run else 'LIVE'}")
    print("=" * 80)
    print()
    
    # Track results
    updated_files: List[str] = []
    failed_files: List[str] = []
    total_changes = 0
    
    # Update each file
    for file_rel, pattern, replacement in VERSION_FILES:
        file_path = PROJECT_ROOT / file_rel
        print(f"Processing: {file_rel}")
        
        success, count = find_and_replace(file_path, pattern, replacement, args.version, args.dry_run)
        
        if success:
            updated_files.append(file_rel)
            total_changes += count
        else:
            failed_files.append(file_rel)
        print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  [OK] Updated: {len(updated_files)} file(s)")
    print(f"  [ERR] Failed:  {len(failed_files)} file(s)")
    print(f"  Total:   {total_changes} change(s)")
    print()
    
    if updated_files:
        print("Updated files:")
        for f in updated_files:
            print(f"  - {f}")
        print()
    
    if failed_files:
        print("Failed files:")
        for f in failed_files:
            print(f"  - {f}")
        print()
    
    if args.dry_run:
        print("[DRY RUN] complete - no files were modified")
        print(f"   Run without --dry-run to apply changes")
    else:
        print("[OK] Version bump complete!")
        print()
        print("Next steps:")
        print(f"  1. Review changes: git diff")
        print(f"  2. Test locally:   pytest tests -v")
        print(f"  3. Commit changes: git add . && git commit -m 'chore: bump version to {args.version}'")
        print(f"  4. Tag release:    git tag {args.version}")
        print(f"  5. Push:           git push origin main --tags")
        print(f"  6. Deploy:         ./scripts/deploy.ps1")
    
    print("=" * 80)
    
    # Exit with error if any failures
    if failed_files:
        sys.exit(1)


if __name__ == "__main__":
    main()
