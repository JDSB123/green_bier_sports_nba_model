#!/usr/bin/env python3
"""
Preflight freshness check:
- Fail fast if required secrets are missing (no silent fallbacks).
- Fetch fresh odds (same script used in production flow).
- Run invariant tests to ensure 4-market integrity (no placeholders, no conflicts).

Usage:
    python scripts/preflight_freshness.py

Assumptions:
- Run from repo root.
- THE_ODDS_API_KEY and API_BASKETBALL_KEY are set as environment variables.
- Network access is available to fetch odds.
"""

from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REQUIRED_SECRETS = ["THE_ODDS_API_KEY", "API_BASKETBALL_KEY"]


class PreflightError(RuntimeError):
    pass


def require_env_secrets() -> None:
    missing = [name for name in REQUIRED_SECRETS if not os.environ.get(name)]
    if missing:
        raise PreflightError(
            f"Missing required secrets in environment: {', '.join(missing)}. "
            "Set them in your dev environment (no fallbacks allowed)."
        )


def run_cmd(cmd: list[str], desc: str) -> None:
    print(f"\n=== {desc} ===")
    proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True)
    if proc.returncode != 0:
        raise PreflightError(f"Command failed: {desc}")


def main() -> int:
    try:
        # 1) Require secrets present
        require_env_secrets()
        print("Secrets present: " + ", ".join(REQUIRED_SECRETS))

        # 2) Fetch fresh odds (no cache/placeholder)
        run_cmd([sys.executable, "scripts/ingest_all.py", "--refresh"], "Fetch fresh odds")

        # 3) Run invariant tests (4 markets, no silent conflicts)
        run_cmd([sys.executable, "-m", "pytest", "tests/test_prediction_invariants.py", "-v"], "Run invariant tests")

        print("\nPreflight success: fresh data fetched and invariants hold.")
        return 0
    except PreflightError as e:
        print(f"\nPreflight failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
