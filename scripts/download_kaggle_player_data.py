#!/usr/bin/env python3
"""
Download eoinamoore's Kaggle NBA dataset for player box scores.

Dataset: https://www.kaggle.com/datasets/eoinamoore/historical-nba-data-and-player-box-scores
Updated daily, covers 1947-present (including 2024-2026 seasons we need!)

Files needed:
- PlayerStatistics.csv - Box scores for every player, every game
- Players.csv - Player biographical info (height, weight, position, draft info)
- Games.csv - All games with dates, teams, scores

This fills our injury data gap:
- If player is on roster but NOT in PlayerStatistics.csv for a game = INACTIVE
"""

import os
import sys
import subprocess
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Paths
DATA_DIR = Path(__file__).parent.parent / "data" / "external" / "kaggle_nba"
DATASET_NAME = "eoinamoore/historical-nba-data-and-player-box-scores"


def check_kaggle_cli():
    """Check if Kaggle CLI is installed and configured."""
    try:
        result = subprocess.run(
            ["kaggle", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print(f"✓ Kaggle CLI found: {result.stdout.strip()}")
            return True
        else:
            print("✗ Kaggle CLI not working properly")
            return False
    except FileNotFoundError:
        print("✗ Kaggle CLI not installed")
        print("  Install with: pip install kaggle")
        print("  Then configure: https://www.kaggle.com/docs/api")
        return False
    except subprocess.TimeoutExpired:
        print("✗ Kaggle CLI timed out")
        return False


def check_kaggle_auth():
    """Check if Kaggle API credentials are configured."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        print(f"✓ Kaggle credentials found: {kaggle_json}")
        return True
    else:
        print(f"✗ Kaggle credentials not found at {kaggle_json}")
        print("  1. Go to https://www.kaggle.com/settings")
        print("  2. Click 'Create New Token'")
        print("  3. Save kaggle.json to ~/.kaggle/")
        return False


def download_dataset():
    """Download the Kaggle dataset."""
    print(f"\nDownloading dataset: {DATASET_NAME}")
    print(f"Target directory: {DATA_DIR}")
    
    # Create directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download
    result = subprocess.run(
        [
            "kaggle", "datasets", "download",
            "-d", DATASET_NAME,
            "-p", str(DATA_DIR),
            "--unzip"
        ],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✓ Download complete!")
        return True
    else:
        print(f"✗ Download failed: {result.stderr}")
        return False


def verify_files():
    """Verify required files were downloaded."""
    required = ["PlayerStatistics.csv", "Players.csv", "Games.csv"]
    missing = []
    
    print("\nVerifying downloaded files:")
    for fname in required:
        fpath = DATA_DIR / fname
        if fpath.exists():
            size_mb = fpath.stat().st_size / (1024 * 1024)
            print(f"  ✓ {fname} ({size_mb:.1f} MB)")
        else:
            print(f"  ✗ {fname} - MISSING")
            missing.append(fname)
    
    return len(missing) == 0


def show_data_preview():
    """Show a preview of the downloaded data."""
    import pandas as pd
    
    print("\n" + "="*60)
    print("DATA PREVIEW")
    print("="*60)
    
    # Players.csv
    players_file = DATA_DIR / "Players.csv"
    if players_file.exists():
        df = pd.read_csv(players_file, nrows=5)
        print(f"\nPlayers.csv columns: {list(df.columns)}")
        print(f"Sample rows:\n{df.head(2)}")
    
    # PlayerStatistics.csv
    stats_file = DATA_DIR / "PlayerStatistics.csv"
    if stats_file.exists():
        df = pd.read_csv(stats_file, nrows=5)
        print(f"\nPlayerStatistics.csv columns: {list(df.columns)}")
        
        # Count rows efficiently
        print("Counting rows (this may take a moment)...")
        row_count = sum(1 for _ in open(stats_file, 'r', encoding='utf-8')) - 1
        print(f"Total player-game records: {row_count:,}")
    
    # Games.csv
    games_file = DATA_DIR / "Games.csv"
    if games_file.exists():
        df = pd.read_csv(games_file, nrows=5)
        print(f"\nGames.csv columns: {list(df.columns)}")
        
        # Get date range
        df_full = pd.read_csv(games_file, usecols=['gameDate'] if 'gameDate' in df.columns else [df.columns[0]])
        date_col = 'gameDate' if 'gameDate' in df_full.columns else df_full.columns[0]
        if date_col in df_full.columns:
            dates = pd.to_datetime(df_full[date_col], errors='coerce')
            print(f"Date range: {dates.min()} to {dates.max()}")
            print(f"Total games: {len(df_full):,}")


def main():
    print("="*60)
    print("KAGGLE NBA DATASET DOWNLOADER")
    print("="*60)
    print(f"\nDataset: {DATASET_NAME}")
    print("Purpose: Fill injury data gap for 2024-2026 seasons")
    print("="*60)
    
    # Check prerequisites
    if not check_kaggle_cli():
        return 1
    
    if not check_kaggle_auth():
        return 1
    
    # Download
    if not download_dataset():
        return 1
    
    # Verify
    if not verify_files():
        print("\n⚠️ Some required files are missing!")
        return 1
    
    # Preview
    try:
        show_data_preview()
    except Exception as e:
        print(f"\n⚠️ Could not preview data: {e}")
    
    print("\n" + "="*60)
    print("✓ DOWNLOAD COMPLETE!")
    print("="*60)
    print(f"\nFiles saved to: {DATA_DIR}")
    print("\nNext step: Run scripts/infer_inactive_from_kaggle.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
