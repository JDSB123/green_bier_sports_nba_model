#!/usr/bin/env python3
"""
CI sanity checks to catch config/env drift early.

Steps:
1) Load .env.example into environment (without overwriting existing vars).
2) Import src.config to ensure required env vars are present.
3) Run a lightweight pytest selection (default: all).
4) Validate Bicep syntax for infra/nba/main.bicep if az is available.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_EXAMPLE = PROJECT_ROOT / ".env.example"
BICEP_FILE = PROJECT_ROOT / "infra" / "nba" / "main.bicep"


def load_env_example():
    """Load .env.example into process env if keys are missing."""
    if not ENV_EXAMPLE.exists():
        print(f"[WARN] .env.example not found at {ENV_EXAMPLE}")
        return
    with ENV_EXAMPLE.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip()
            if key and key not in os.environ:
                os.environ[key] = val
    print("[OK] Loaded defaults from .env.example for missing vars")


def import_config():
    """Import src.config to ensure required envs are present."""
    try:
        import src.config  # noqa: F401
        print("[OK] src.config import succeeded (env requirements met)")
    except Exception as e:
        print(f"[FAIL] src.config import failed: {e}")
        sys.exit(1)


def run_pytest():
    """Run pytest (can be scoped by PYTEST_ARGS env)."""
    args = os.getenv("PYTEST_ARGS", "").split()
    cmd = [sys.executable, "-m", "pytest"] + args
    print(f"[RUN] {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[FAIL] pytest exited with {result.returncode}")
        sys.exit(result.returncode)
    print("[OK] pytest passed")


def run_bicep_validate():
    """Validate Bicep if az is available."""
    if not BICEP_FILE.exists():
        print(f"[SKIP] Bicep file not found: {BICEP_FILE}")
        return
    try:
        subprocess.run(["az", "bicep", "version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        print("[SKIP] az bicep not available; skipping Bicep validation")
        return
    cmd = ["az", "bicep", "build", "--file", str(BICEP_FILE), "--outdir", str(BICEP_FILE.parent)]
    print(f"[RUN] {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[FAIL] Bicep validation failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    print("[OK] Bicep validation passed")


def main():
    load_env_example()
    import_config()
    run_pytest()
    run_bicep_validate()


if __name__ == "__main__":
    main()
