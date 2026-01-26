#!/usr/bin/env python3
"""
Run the production runtime isolation test without triggering coverage thresholds.

This avoids pytest.ini addopts that enforce coverage on a single-file test.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_PATH = PROJECT_ROOT / "tests" / "test_production_no_historical_runtime_imports.py"


def main() -> int:
    if not TEST_PATH.exists():
        print(f"[FAIL] Missing test file: {TEST_PATH}")
        return 1

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cov",
        "-o",
        "addopts=",
        str(TEST_PATH),
    ]
    print(f"[RUN] {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
