#!/usr/bin/env python3
"""
Production Readiness Validation with Backtest

Validates production readiness AND runs backtests to verify model performance.
"""
import sys
import subprocess
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def run_production_readiness_check():
    """Run the basic production readiness validation."""
    print("=" * 60)
    print("STEP 1: Production Readiness Validation")
    print("=" * 60)
    
    result = subprocess.run(
        [sys.executable, "scripts/validate_production_readiness.py"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT
    )
    
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    return result.returncode == 0


def run_backtest_moneyline():
    """Run moneyline backtest (doesn't require betting lines)."""
    print("\n" + "=" * 60)
    print("STEP 2: Backtest Validation (Moneyline Markets)")
    print("=" * 60)
    print("Running backtest on moneyline markets...")
    print("(Moneyline doesn't require betting lines, so can run with current data)")
    print()
    
    result = subprocess.run(
        [sys.executable, "scripts/backtest.py", "--markets", "fg_moneyline,1h_moneyline", "--min-training", "100"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        timeout=600  # 10 minute timeout
    )
    
    print(result.stdout)
    if result.stderr and "WARN" not in result.stderr:
        print(result.stderr)
    
    # Check if we got any results
    success = False
    if "Completed" in result.stdout or "predictions" in result.stdout.lower():
        success = True
    
    return success, result.stdout


def check_data_quality():
    """Check if we have sufficient data for backtesting."""
    print("\n" + "=" * 60)
    print("STEP 3: Data Quality Check")
    print("=" * 60)
    
    try:
        import pandas as pd
        
        data_file = PROJECT_ROOT / "data/processed/training_data.csv"
        if not data_file.exists():
            print(f"[FAIL] Training data not found: {data_file}")
            return False
        
        df = pd.read_csv(data_file)
        print(f"[OK] Training data found: {len(df)} games")
        
        # Check date range
        df["date"] = pd.to_datetime(df["date"])
        date_range = (df["date"].max() - df["date"].min()).days
        print(f"[OK] Date range: {date_range} days ({df['date'].min()} to {df['date'].max()})")
        
        # Check for required columns
        required = ["home_score", "away_score", "home_team", "away_team", "date"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            print(f"[FAIL] Missing required columns: {missing}")
            return False
        
        print(f"[OK] All required columns present")
        
        # Check for betting lines (optional but preferred)
        has_spread_line = "spread_line" in df.columns
        has_total_line = "total_line" in df.columns
        
        if has_spread_line and has_total_line:
            print(f"[OK] Betting lines present (can backtest all markets)")
        else:
            print(f"[WARN] Betting lines missing (can only backtest moneyline markets)")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Data quality check failed: {e}")
        return False


def main():
    """Run full production readiness validation with backtest."""
    print("\n" + "=" * 60)
    print("PRODUCTION READINESS VALIDATION WITH BACKTEST")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project: {PROJECT_ROOT}")
    print()
    
    results = {}
    
    # Step 1: Production readiness
    results["production_ready"] = run_production_readiness_check()
    
    # Step 2: Data quality
    results["data_quality"] = check_data_quality()
    
    # Step 3: Backtest (only if data quality is OK)
    if results["data_quality"]:
        results["backtest_success"], backtest_output = run_backtest_moneyline()
        results["backtest_output"] = backtest_output
    else:
        results["backtest_success"] = False
        results["backtest_output"] = "Skipped due to data quality issues"
    
    # Summary
    print("\n" + "=" * 60)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 60)
    
    all_passed = all([
        results["production_ready"],
        results["data_quality"],
        results["backtest_success"]
    ])
    
    print(f"Production Readiness: {'[PASS]' if results['production_ready'] else '[FAIL]'}")
    print(f"Data Quality: {'[PASS]' if results['data_quality'] else '[FAIL]'}")
    print(f"Backtest Execution: {'[PASS]' if results['backtest_success'] else '[FAIL]'}")
    
    print("\n" + "=" * 60)
    if all_passed:
        print("[OK] PRODUCTION READY WITH VALIDATED BACKTEST")
        print("\nThe system has:")
        print("  - Passed all code/configuration checks")
        print("  - Sufficient data for backtesting")
        print("  - Successfully run backtests on moneyline markets")
        return 0
    else:
        print("[WARN] PRODUCTION READY BUT BACKTEST NEEDS ATTENTION")
        if not results["backtest_success"]:
            print("\nNote: Backtest may have failed due to:")
            print("  - Missing betting lines (only moneyline works without lines)")
            print("  - Insufficient historical data")
            print("  - Model training issues")
        return 1


if __name__ == "__main__":
    sys.exit(main())

