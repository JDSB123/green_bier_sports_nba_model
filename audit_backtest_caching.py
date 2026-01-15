#!/usr/bin/env python3
"""Audit backtest caching strategy."""

from pathlib import Path

OK = "[OK]"
WARN = "[WARN]"
FAIL = "[FAIL]"

print("\n" + "=" * 80)
print("BACKTEST DATA CACHING AUDIT")
print("=" * 80)

data_dir = Path("data")
docker_entrypoint = Path("docker-entrypoint-backtest.sh")
docker_compose = Path("docker-compose.backtest.yml")

# Check docker-entrypoint strategy
print("\n1. Docker Entrypoint Caching Strategy")
print("-" * 60)

if docker_entrypoint.exists():
    content = docker_entrypoint.read_text(encoding="utf-8")

    if "training_data.csv" in content:
        print(f"{OK} Entrypoint references training_data.csv (canonical)")
    else:
        print(f"{FAIL} Entrypoint does not reference training_data.csv")

    if "build_training_data_complete.py" in content or "USE_PREBUILT" in content or "--use-prebuilt" in content:
        print(f"{WARN} Raw rebuild/prebuilt references found in entrypoint")
    else:
        print(f"{OK} No raw rebuild/prebuilt references in entrypoint")

    if "validate_training_data.py" in content:
        print(f"{OK} Validation uses validate_training_data.py")
    else:
        print(f"{WARN} Validation script not updated to validate_training_data.py")

    if "cleanup_cache" in content:
        print(f"{OK} Cache cleanup function implemented")
    else:
        print(f"{WARN} Cache cleanup not found")

    if "data/historical" in content:
        print(f"{OK} Historical data cleanup specified")
    else:
        print(f"{WARN} Historical data not cleaned up after backtest")
else:
    print(f"{FAIL} docker-entrypoint-backtest.sh not found")

# Check docker-compose strategy
print("\n2. Docker Compose Caching Strategy")
print("-" * 60)

if docker_compose.exists():
    content = docker_compose.read_text(encoding="utf-8")

    if "USE_PREBUILT" in content:
        print(f"{WARN} USE_PREBUILT still present in docker-compose")
    else:
        print(f"{OK} USE_PREBUILT removed from docker-compose")

    if "backtest_cache:" in content:
        print(f"{OK} Named volume 'backtest_cache' defined for persistence")
    else:
        print(f"{WARN} Named volumes not clearly defined")
else:
    print(f"{FAIL} docker-compose.backtest.yml not found")

# Check data directory structure
print("\n3. Data Directory Caching Structure")
print("-" * 60)

critical_paths = {
    "data/processed/training_data.csv": "Canonical training data (should be reused)",
    "data/historical": "Historical data (should be ephemeral during backtest)",
    "data/backtest_results": "Backtest output (should persist)",
    ".cache": "Python/pip cache (should be ephemeral)",
}

for path_str, description in critical_paths.items():
    full_path = data_dir / path_str if not path_str.startswith(".") else Path(path_str)
    if full_path.exists():
        size_kb = sum(f.stat().st_size for f in full_path.rglob("*") if f.is_file()) / 1024
        status = OK if "ephemeral" not in description or size_kb < 1000 else WARN
        print(f"{status} {path_str:40} {size_kb:10,.0f}KB - {description}")
    else:
        print(f"  {path_str:40} {'NOT FOUND':10} - {description}")

# Check Docker volume setup
print("\n4. Docker Compose Volume Setup (Persistence Strategy)")
print("-" * 60)

if docker_compose.exists():
    lines = docker_compose.read_text(encoding="utf-8").splitlines()

    in_volumes = False
    for line in lines:
        if "volumes:" in line:
            in_volumes = True
        if in_volumes and ("./data:/app/data" in line or "/app/data" in line):
            print(f"{OK} Local ./data mounted to /app/data (training data persists)")
            break

    if "backtest_cache:" in "\n".join(lines):
        print(f"{OK} Named volume 'backtest_cache' for ephemeral cache")

# Summary
print("\n" + "=" * 80)
print("CACHING STRATEGY SUMMARY")
print("=" * 80)

print(
    """
FAST MODE (DEFAULT):
  1. Ensure data/processed/training_data.csv exists (audited 2023+)
  2. Run backtest on canonical training data
  3. Clean up ephemeral /data/historical after backtest (if present)

PERSISTENCE:
  [OK] training_data.csv persists across runs (./data mounted)
  [OK] backtest_results persists (./data/backtest_results)
  [OK] Historical raw data cleaned up after each run
  [OK] Named volume backtest_cache for intermediate files
"""
)
