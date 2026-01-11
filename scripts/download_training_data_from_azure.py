#!/usr/bin/env python3
"""
Download Training Data from Azure Blob Storage (Single Source of Truth)

This script downloads the quality-checked training data and manifest from Azure.
Use this to ensure you're working with the canonical, validated dataset.

Usage:
    python scripts/download_training_data_from_azure.py              # Download latest
    python scripts/download_training_data_from_azure.py --version v2026.01.11  # Specific version
    python scripts/download_training_data_from_azure.py --list       # List available versions
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
STORAGE_ACCOUNT = "nbagbsvstrg"
CONTAINER = "nbahistoricaldata"
BLOB_PREFIX = "training_data"

# Local paths
DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUTPUT = DATA_DIR / "training_data_from_azure.csv"
MANIFEST_OUTPUT = DATA_DIR / "training_data_manifest.json"

# Status symbols
OK = "[✓]"
FAIL = "[✗]"
INFO = "[i]"


def calculate_checksum(file_path: Path) -> str:
    """Calculate SHA256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def list_versions() -> list[str]:
    """List available versions in blob storage."""
    cmd = (
        f'az storage blob list --account-name {STORAGE_ACCOUNT} '
        f'--container-name {CONTAINER} --prefix "{BLOB_PREFIX}/" '
        f'--auth-mode key --query "[].name" -o tsv'
    )
    result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    if result.returncode != 0:
        print(f"{FAIL} Failed to list blobs: {result.stderr}")
        return []

    versions = set()
    for line in result.stdout.strip().split("\n"):
        if line.startswith(f"{BLOB_PREFIX}/"):
            parts = line.replace(f"{BLOB_PREFIX}/", "").split("/")
            if parts and parts[0] not in ("latest", ""):
                versions.add(parts[0])

    return sorted(versions, reverse=True)


def download_blob(blob_name: str, output_path: Path) -> bool:
    """Download a blob to local file."""
    cmd = (
        f'az storage blob download --account-name {STORAGE_ACCOUNT} '
        f'--container-name {CONTAINER} --name "{blob_name}" '
        f'--file "{output_path}" --auth-mode key'
    )
    result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Download training data from Azure Blob Storage")
    parser.add_argument("--version", default="latest", help="Version to download (default: latest)")
    parser.add_argument("--list", action="store_true", help="List available versions")
    parser.add_argument("--output", type=Path, default=None, help="Output file path")
    parser.add_argument("--verify", action="store_true", help="Verify checksum after download")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print(" DOWNLOAD TRAINING DATA FROM AZURE (SINGLE SOURCE OF TRUTH)")
    print("=" * 60)

    if args.list:
        print(f"\n{INFO} Listing available versions...")
        versions = list_versions()
        if versions:
            print(f"\n  Available versions:")
            for v in versions:
                print(f"    - {v}")
            print(f"\n  Use: python {Path(__file__).name} --version <version>")
        else:
            print(f"  {FAIL} No versions found")
        return

    version = args.version
    output_path = args.output or DEFAULT_OUTPUT

    # Determine blob paths
    if version == "latest":
        data_blob = f"{BLOB_PREFIX}/latest/training_data_complete_2023_with_injuries.csv"
        manifest_blob = f"{BLOB_PREFIX}/latest/manifest.json"
    else:
        data_blob = f"{BLOB_PREFIX}/{version}/training_data_complete_2023_with_injuries.csv"
        manifest_blob = f"{BLOB_PREFIX}/{version}/manifest.json"

    print(f"\n{INFO} Version: {version}")
    print(f"{INFO} Data blob: {data_blob}")
    print(f"{INFO} Output: {output_path}")

    # Download manifest first
    print(f"\n{INFO} Downloading manifest...")
    if download_blob(manifest_blob, MANIFEST_OUTPUT):
        print(f"  {OK} Manifest downloaded: {MANIFEST_OUTPUT}")
        manifest = json.loads(MANIFEST_OUTPUT.read_text())
        print(f"  {INFO} Version: {manifest.get('version', 'unknown')}")
        print(f"  {INFO} Generated: {manifest.get('generated_at', 'unknown')}")
        print(f"  {INFO} Games: {manifest.get('quality_metrics', {}).get('total_games', 'unknown')}")
    else:
        print(f"  {FAIL} Failed to download manifest")
        manifest = None

    # Download data
    print(f"\n{INFO} Downloading training data...")
    if download_blob(data_blob, output_path):
        print(f"  {OK} Data downloaded: {output_path}")
        print(f"  {INFO} Size: {output_path.stat().st_size / (1024*1024):.1f} MB")
    else:
        print(f"  {FAIL} Failed to download data")
        sys.exit(1)

    # Verify checksum
    if args.verify and manifest:
        print(f"\n{INFO} Verifying checksum...")
        expected = manifest.get("sha256", "")
        actual = calculate_checksum(output_path)
        if actual == expected:
            print(f"  {OK} Checksum verified: {actual[:16]}...")
        else:
            print(f"  {FAIL} Checksum mismatch!")
            print(f"      Expected: {expected[:16]}...")
            print(f"      Actual:   {actual[:16]}...")
            sys.exit(1)

    print("\n" + "=" * 60)
    print(f"{OK} DOWNLOAD COMPLETE")
    print(f"\nSingle Source of Truth Data:")
    print(f"  {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
