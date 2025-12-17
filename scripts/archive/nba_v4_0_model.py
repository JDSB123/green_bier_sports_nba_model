#!/usr/bin/env python3
"""
Single source of truth entrypoint for NBA v4.0 model slate analysis.
Wraps `analyze_todays_slate.py` so downstream users have one canonical CLI.
"""
from pathlib import Path
import sys

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_todays_slate import main as analyze_main  # noqa: E402


if __name__ == "__main__":
    analyze_main()

