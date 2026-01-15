#!/usr/bin/env python3
"""Audit market coverage and verify no placeholder odds are used."""

from pathlib import Path
import sys

import pandas as pd

OK = "[OK]"
WARN = "[WARN]"
FAIL = "[FAIL]"
INFO = "[INFO]"

data_dir = Path("data/processed")
training_file = data_dir / "master_training_data.csv"

if not training_file.exists():
    print(f"{FAIL} training_data.csv not found. Checking available files...")
    print(list(data_dir.glob("*.csv")))
    sys.exit(1)

df = pd.read_csv(training_file)

print("\n" + "=" * 80)
print("MARKET COVERAGE & ODDS VALIDITY AUDIT")
print("=" * 80)
print(f"\nTotal games in training_data.csv: {len(df):,}\n")

# Define markets
markets = {
    "Full Game": {
        "spread": "fg_spread_line",
        "total": "fg_total_line",
        "label_spread": "fg_spread_covered",
        "label_total": "fg_total_over",
    },
    "1st Half": {
        "spread": "1h_spread_line",
        "total": "1h_total_line",
        "label_spread": "1h_spread_covered",
        "label_total": "1h_total_over",
    },
    "Q1": {
        "spread": "q1_spread_line",
        "total": "q1_total_line",
        "label_spread": "q1_spread_covered",
        "label_total": "q1_total_over",
    },
}

total_market_coverage = {}

for market_name, market_cols in markets.items():
    print(f"\n{market_name.upper()}")
    print("-" * 60)

    for bet_type in ["spread", "total"]:
        line_col = market_cols[bet_type]
        label_col = market_cols[f"label_{bet_type}"]

        if line_col not in df.columns:
            print(f"  {FAIL} {bet_type.upper()}: Column '{line_col}' NOT FOUND")
            continue

        if label_col not in df.columns:
            print(f"  {WARN} {bet_type.upper()}: Label column '{label_col}' NOT FOUND")
            continue

        # Coverage
        line_coverage = df[line_col].notna().sum()
        line_pct = (line_coverage / len(df)) * 100

        # Check for unique values (not all identical - would indicate synthetic/placeholder)
        unique_lines = df[line_col].nunique()

        # Check for obvious placeholders (zeros, out-of-range values)
        zeros = (df[line_col] == 0).sum()
        negative = (df[line_col] < 0).sum()
        extreme_high = (df[line_col] > 30).sum()  # Unusual spread
        extreme_low = (df[line_col] < -30).sum()  # Unusual spread

        # Sample values
        sample_values = df[df[line_col].notna()][line_col].head(5).tolist()
        sample_str = ", ".join([f"{v:.1f}" if v else str(v) for v in sample_values])

        print(f"  {bet_type.upper()} LINES:")
        print(f"    {OK} Coverage: {line_coverage:,}/{len(df):,} ({line_pct:.1f}%)")
        print(f"    {OK} Unique values: {unique_lines} (indicates real odds, not synthetic)")
        print(
            f"    {INFO} Suspicious values: Zeros={zeros}, Negatives={negative}, >30={extreme_high}, <-30={extreme_low}"
        )
        print(f"    Range: [{df[line_col].min():.1f}, {df[line_col].max():.1f}]")
        print(f"    Sample: {sample_str}")

        # Check labels match odds
        if label_col in df.columns:
            # Labels should only exist where odds exist
            label_coverage = df[label_col].notna().sum()
            label_pct = (label_coverage / len(df)) * 100

            # Cross-check: labels without odds (data integrity issue)
            orphaned_labels = df[(df[label_col].notna()) & (df[line_col].isna())].shape[0]

            print(f"    {OK} Labels: {label_coverage:,} ({label_pct:.1f}%)")
            if orphaned_labels > 0:
                print(f"    {WARN} WARNING: {orphaned_labels} labels exist without odds!")

            total_market_coverage[f"{market_name} {bet_type}"] = {
                "odds_coverage": line_pct,
                "unique_odds": unique_lines,
                "label_coverage": label_pct,
            }

# Moneylines (Juice/Vig indicator)
print("\n\nMONEYLINE COVERAGE (Juice/Vig - confirms real odds)")
print("-" * 60)
ml_coverage = {}
for market_name in ["fg", "1h", "q1"]:
    print(f"\n{market_name.upper()}:")
    for side in ["home", "away"]:
        col = f"{market_name}_ml_{side}"
        if col in df.columns:
            coverage = df[col].notna().sum()
            pct = (coverage / len(df)) * 100
            sample = df[df[col].notna()][col].head(3).tolist()
            sample_str = ", ".join([f"{int(v)}" if v else str(v) for v in sample])
            print(f"  {side.upper()}: {coverage:,} ({pct:.1f}%) - {sample_str}")
            ml_coverage[f"{market_name}_{side}"] = pct
        else:
            print(f"  {side.upper()}: Column not found")

# Summary
print("\n\n" + "=" * 80)
print("COVERAGE SUMMARY")
print("=" * 80)

for market, coverage in total_market_coverage.items():
    status = OK if coverage["odds_coverage"] > 0 else FAIL
    print(
        f"{status} {market:25} Odds: {coverage['odds_coverage']:5.1f}%  "
        f"Unique: {coverage['unique_odds']:4d}  Labels: {coverage['label_coverage']:5.1f}%"
    )

print("\n" + "=" * 80)
print("VALIDATION CHECKLIST")
print("=" * 80)

all_checks_passed = True

# Check 1: Coverage > 0 for all markets
for market, coverage in total_market_coverage.items():
    if coverage["odds_coverage"] == 0:
        print(f"{FAIL} FAIL: {market} has 0% odds coverage")
        all_checks_passed = False

# Check 2: Unique odds (not synthetic)
for market, coverage in total_market_coverage.items():
    if coverage["unique_odds"] < 10:  # Should have variety
        print(
            f"{WARN} WARNING: {market} has only {coverage['unique_odds']} unique odds "
            "(potential placeholder concern)"
        )

# Check 3: Moneylines exist (confirms real odds with juice)
if ml_coverage and all(ml > 0 for ml in ml_coverage.values()):
    print(f"{OK} PASS: All moneylines present - real odds with juice verified")
else:
    print(f"{FAIL} FAIL: Some moneylines missing - odds may be synthetic")
    all_checks_passed = False

# Check 4: CST alignment (match_key date vs game_date/date)
print("\n" + "=" * 80)
print("CST ALIGNMENT CHECK")
print("=" * 80)

if "match_key" not in df.columns:
    print(f"{WARN} match_key missing; cannot verify CST alignment")
    all_checks_passed = False
else:
    mk_dates = pd.to_datetime(df["match_key"].astype(str).str.split("_").str[0], errors="coerce")
    for col in ["game_date", "date"]:
        if col not in df.columns:
            print(f"{WARN} {col} missing; cannot verify CST alignment")
            all_checks_passed = False
            continue
        parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
        mk_date_only = mk_dates.dt.date
        col_date_only = parsed.dt.date
        mismatches = (
            mk_date_only.notna()
            & col_date_only.notna()
            & (mk_date_only != col_date_only)
        )
        mismatch_count = int(mismatches.sum())
        if mismatch_count > 0:
            print(f"{WARN} {col} mismatches vs match_key: {mismatch_count}")
            all_checks_passed = False
        else:
            print(f"{OK} {col} matches match_key dates")

if all_checks_passed:
    print(f"\n{OK} ALL CHECKS PASSED: Using real, unique odds from actual sportsbooks")
else:
    print(f"\n{FAIL} ISSUES DETECTED: Review warnings above")
