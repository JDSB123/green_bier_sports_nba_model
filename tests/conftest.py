"""Shared pytest fixtures and configuration hooks."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root (which contains the `src` package) is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT_STR = str(PROJECT_ROOT)
if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)
