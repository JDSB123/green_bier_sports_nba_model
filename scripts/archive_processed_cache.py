#!/usr/bin/env python3
"""
Archive processed cache directories and stale artifacts.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings


def archive_directory(source: Path, archive_root: Path) -> None:
    if not source.exists() or not any(source.iterdir()):
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = archive_root / timestamp / source.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(dest))
    source.mkdir(parents=True, exist_ok=True)
    print(f"Archived {source} -> {dest}")


def archive_old_files(
    processed_dir: Path,
    patterns: Iterable[str],
    archive_root: Path,
    cutoff: datetime,
) -> None:
    for pattern in patterns:
        for file_path in processed_dir.glob(pattern):
            mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            if mtime >= cutoff:
                continue
            dest = archive_root / "slate_analysis" / file_path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(file_path), str(dest))
            print(f"Archived stale file {file_path} -> {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive processed cache and stale analysis files.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(settings.data_processed_dir) / "cache",
        help="Cache directory to rotate.",
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=Path(settings.data_processed_dir) / "archive",
        help="Root directory for archived artifacts.",
    )
    parser.add_argument(
        "--slate-days",
        type=int,
        default=3,
        help="Archive slate_analysis/pick_review files older than N days (<=0 disables).",
    )
    args = parser.parse_args()

    archive_dir = args.archive_dir
    archive_dir.mkdir(parents=True, exist_ok=True)

    if args.cache_dir:
        archive_directory(args.cache_dir, archive_dir)

    if args.slate_days > 0:
        cutoff = datetime.now() - timedelta(days=args.slate_days)
        processed_dir = Path(settings.data_processed_dir)
        patterns = [
            "slate_analysis_*.json",
            "slate_analysis_*.txt",
            "pick_review_*.json",
            "pick_review_*.md",
        ]
        archive_old_files(processed_dir, patterns, archive_dir, cutoff)


if __name__ == "__main__":
    main()

