#!/usr/bin/env python3
"""
Fetch external datasets from Kaggle.
Requires 'kaggle' CLI to be installed and configured.

Datasets:
- isaacjackson06/nba-rest-and-performance-24-25 (Rest & Performance)
- eoinamoore/historical-nba-data-and-player-box-scores (Historical Box Scores)
- chrismunch/nba-game-team-statistics (Team Stats)
- rickmcintire/mighunba2024 (2024 Data)
- rickmcintire/mighuretro2024 (Retro Data)
- isaienkov/nba-2k20-data-analysis-visualization (2K20 Ratings)
- zachht/wnba-odds-history (WNBA Odds - Optional)
"""
import subprocess
import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "kaggle"

DATASETS = [
    "isaacjackson06/nba-rest-and-performance-24-25",
    "eoinamoore/historical-nba-data-and-player-box-scores",
    "chrismunch/nba-game-team-statistics",
    "rickmcintire/mighunba2024",
    "rickmcintire/mighuretro2024",
    "isaienkov/nba-2k20-data-analysis-visualization",
    # "zachht/wnba-odds-history"  # Commented out by default as it's WNBA
]

def check_kaggle_auth():
    """Check if Kaggle API is working."""
    try:
        subprocess.run(["kaggle", "datasets", "list"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def download_dataset(dataset_slug: str, output_dir: Path):
    """Download and unzip a Kaggle dataset."""
    print(f"Downloading {dataset_slug}...")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_slug, "-p", str(output_dir), "--unzip"],
            check=True
        )
        print(f"[OK] Downloaded {dataset_slug}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to download {dataset_slug}: {e}")

def main():
    if not check_kaggle_auth():
        print("Error: Kaggle CLI not authenticated or not installed.")
        print("Please ensure you have 'kaggle' installed and your 'kaggle.json' in the correct location.")
        return 1

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading datasets to {DATA_DIR}...")

    for dataset in DATASETS:
        download_dataset(dataset, DATA_DIR)

    print("\nDownload complete.")
    print(f"Files in {DATA_DIR}:")
    for f in DATA_DIR.glob("*"):
        print(f"  - {f.name}")
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
