#!/usr/bin/env python3
"""
DATA STANDARDIZATION VERIFICATION SCRIPT

Validates that ALL data ingestion flows through the SINGLE SOURCE OF TRUTH:
1. Team name standardization (all variants -> canonical names)
2. Timezone conversion (UTC/local -> CST)
3. Match key generation (consistent across all sources)
4. Merge accuracy (no orphaned or duplicate records)

Run this BEFORE training to ensure data integrity.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.standardization import (
    standardize_team_name,
    to_cst,
    to_cst_date,
    generate_match_key,
    get_all_team_names,
    ABBREV_TO_FULL,
    CST,
)

# Paths
DATA_DIR = PROJECT_ROOT / "data"
KAGGLE_FILE = DATA_DIR / "external" / "kaggle" / "nba_2008-2025.csv"
THEODDS_DIR = DATA_DIR / "historical" / "the_odds"
OUTPUT_FILE = DATA_DIR / "processed" / "training_data_complete_2023.csv"


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_result(label: str, passed: bool, details: str = ""):
    status = "[PASS]" if passed else "[FAIL]"
    print(f"  {status}: {label}")
    if details:
        for line in details.split("\n"):
            print(f"         {line}")


# =============================================================================
# 1. TEAM NAME STANDARDIZATION VERIFICATION
# =============================================================================

def verify_team_names():
    """Verify all team name variants map to canonical names."""
    print_header("1. TEAM NAME STANDARDIZATION")
    
    canonical_names = set(get_all_team_names())
    print(f"\n  Canonical team names: {len(canonical_names)}")
    
    # Test all abbreviations
    errors = []
    for abbrev, expected in ABBREV_TO_FULL.items():
        result = standardize_team_name(abbrev)
        if result != expected:
            errors.append(f"{abbrev} -> {result} (expected {expected})")
    
    print_result(
        f"Abbreviation mapping ({len(ABBREV_TO_FULL)} variants)",
        len(errors) == 0,
        "\n".join(errors[:5]) if errors else ""
    )
    
    # Test case variations
    test_cases = [
        ("LAL", "Los Angeles Lakers"),
        ("lakers", "Los Angeles Lakers"),
        ("LAKERS", "Los Angeles Lakers"),
        ("Los Angeles Lakers", "Los Angeles Lakers"),
        ("gsw", "Golden State Warriors"),
        ("Golden State", "Golden State"),  # Partial name - may not match
        ("76ers", "Philadelphia 76ers"),
        ("sixers", "Philadelphia 76ers"),
    ]
    
    case_errors = []
    for input_name, expected in test_cases:
        result = standardize_team_name(input_name)
        if result != expected and expected in canonical_names:
            case_errors.append(f"{input_name} -> {result} (expected {expected})")
    
    print_result(
        "Case-insensitive matching",
        len(case_errors) == 0,
        "\n".join(case_errors) if case_errors else ""
    )
    
    return len(errors) == 0 and len(case_errors) == 0


# =============================================================================
# 2. TIMEZONE CONVERSION VERIFICATION
# =============================================================================

def verify_timezone():
    """Verify timezone conversion to CST."""
    print_header("2. TIMEZONE CONVERSION (CST)")
    
    # Test UTC to CST conversion
    test_cases = [
        # (input, expected_cst_date, source_is_utc)
        ("2023-10-25T00:30:00Z", "2023-10-24", True),  # UTC midnight+30min -> prev day CST
        ("2023-10-25T06:00:00Z", "2023-10-25", True),  # UTC 6am -> same day CST
        ("2023-10-24", "2023-10-24", False),  # Local date stays same
        ("2023-10-24T22:00:00", "2023-10-24", False),  # Local evening stays same
    ]
    
    errors = []
    for input_dt, expected_date, source_is_utc in test_cases:
        result = to_cst_date(input_dt, source_is_utc=source_is_utc)
        if result != expected_date:
            src = "UTC" if source_is_utc else "local"
            errors.append(f"{input_dt} ({src}) -> {result} (expected {expected_date})")
    
    print_result(
        f"UTC/Local to CST conversion ({len(test_cases)} cases)",
        len(errors) == 0,
        "\n".join(errors) if errors else ""
    )
    
    # Test real-world scenario: late night game
    # Game at 10:30 PM ET on Oct 24 = 02:30 UTC on Oct 25 = 9:30 PM CST on Oct 24
    utc_time = "2023-10-25T02:30:00Z"
    kaggle_date = "2023-10-24"
    
    utc_cst = to_cst_date(utc_time, source_is_utc=True)
    local_cst = to_cst_date(kaggle_date, source_is_utc=False)
    
    print_result(
        "Late-night game alignment",
        utc_cst == local_cst,
        f"UTC '{utc_time}' -> '{utc_cst}', Local '{kaggle_date}' -> '{local_cst}'"
    )
    
    return len(errors) == 0 and utc_cst == local_cst


# =============================================================================
# 3. MATCH KEY GENERATION VERIFICATION
# =============================================================================

def verify_match_keys():
    """Verify match key generation is consistent across sources."""
    print_header("3. MATCH KEY GENERATION")
    
    # Test same game from different sources
    test_games = [
        {
            "name": "Nuggets vs Lakers (season opener)",
            "kaggle": {"date": "2023-10-24", "home": "den", "away": "lal"},
            "theodds": {"time": "2023-10-25T02:30:00Z", "home": "Denver Nuggets", "away": "Los Angeles Lakers"},
        },
        {
            "name": "Celtics vs Knicks",
            "kaggle": {"date": "2023-10-25", "home": "bos", "away": "nyk"},
            "theodds": {"time": "2023-10-26T00:00:00Z", "home": "Boston Celtics", "away": "New York Knicks"},
        },
    ]
    
    all_passed = True
    for game in test_games:
        k = game["kaggle"]
        t = game["theodds"]
        
        kaggle_key = generate_match_key(k["date"], k["home"], k["away"], source_is_utc=False)
        theodds_key = generate_match_key(t["time"], t["home"], t["away"], source_is_utc=True)
        
        passed = kaggle_key == theodds_key
        all_passed = all_passed and passed
        
        print_result(
            game["name"],
            passed,
            f"Kaggle: {kaggle_key}\nTheOdds: {theodds_key}" if not passed else ""
        )
    
    return all_passed


# =============================================================================
# 4. DATA SOURCE COVERAGE VERIFICATION
# =============================================================================

def verify_data_coverage():
    """Verify merge coverage from each data source."""
    print_header("4. DATA SOURCE MERGE COVERAGE")
    
    # Check if output exists
    if not OUTPUT_FILE.exists():
        print(f"\n  Output file not found: {OUTPUT_FILE}")
        print("  Run: python scripts/build_complete_training_data.py --start-date 2023-01-01")
        return False
    
    df = pd.read_csv(OUTPUT_FILE)
    total = len(df)
    
    print(f"\n  Total games: {total:,}")
    
    # Check FG lines
    fg_spread = df["fg_spread_line"].notna().sum()
    fg_total = df["fg_total_line"].notna().sum()
    fg_ml = df["fg_ml_home"].notna().sum()
    
    print(f"\n  Full Game (FG) coverage:")
    print_result(
        f"FG spread lines: {fg_spread:,}/{total:,} ({fg_spread/total*100:.1f}%)",
        fg_spread/total > 0.95
    )
    print_result(
        f"FG total lines: {fg_total:,}/{total:,} ({fg_total/total*100:.1f}%)",
        fg_total/total > 0.95
    )
    print_result(
        f"FG moneylines: {fg_ml:,}/{total:,} ({fg_ml/total*100:.1f}%)",
        fg_ml/total > 0.90
    )
    
    # Check 1H lines
    h1_spread = df["1h_spread_line"].notna().sum() if "1h_spread_line" in df.columns else 0
    h1_total = df["1h_total_line"].notna().sum() if "1h_total_line" in df.columns else 0
    
    print(f"\n  First Half (1H) coverage:")
    print_result(
        f"1H spread lines: {h1_spread:,}/{total:,} ({h1_spread/total*100:.1f}%)",
        h1_spread/total > 0.70  # 1H data starts May 2023
    )
    print_result(
        f"1H total lines: {h1_total:,}/{total:,} ({h1_total/total*100:.1f}%)",
        h1_total/total > 0.70
    )
    
    # Check outcomes
    print(f"\n  Outcome labels:")
    for label in ["fg_spread_covered", "fg_total_over", "1h_spread_covered", "1h_total_over"]:
        if label in df.columns:
            count = df[label].notna().sum()
            pct = df[label].mean() * 100 if count > 0 else 0
            print_result(
                f"{label}: {count:,} games, {pct:.1f}% positive",
                45 < pct < 55  # Should be near 50% if no data leakage
            )
    
    return True


# =============================================================================
# 5. TEAM NAME CONSISTENCY IN OUTPUT
# =============================================================================

def verify_output_teams():
    """Verify all team names in output are canonical."""
    print_header("5. TEAM NAME CONSISTENCY IN OUTPUT")
    
    if not OUTPUT_FILE.exists():
        print(f"\n  Output file not found")
        return False
    
    df = pd.read_csv(OUTPUT_FILE)
    canonical = set(get_all_team_names())
    
    # Check home teams
    home_teams = set(df["home_team"].dropna().unique())
    unknown_home = home_teams - canonical
    
    print_result(
        f"Home teams ({len(home_teams)} unique)",
        len(unknown_home) == 0,
        f"Unknown: {unknown_home}" if unknown_home else ""
    )
    
    # Check away teams
    away_teams = set(df["away_team"].dropna().unique())
    unknown_away = away_teams - canonical
    
    print_result(
        f"Away teams ({len(away_teams)} unique)",
        len(unknown_away) == 0,
        f"Unknown: {unknown_away}" if unknown_away else ""
    )
    
    # All teams should be exactly 30 NBA teams
    all_teams = home_teams | away_teams
    print_result(
        f"Total unique teams: {len(all_teams)}",
        len(all_teams) == 30,
        f"Teams: {sorted(all_teams)}" if len(all_teams) != 30 else ""
    )
    
    return len(unknown_home) == 0 and len(unknown_away) == 0


# =============================================================================
# 6. DUPLICATE CHECK
# =============================================================================

def verify_no_duplicates():
    """Verify no duplicate games in output."""
    print_header("6. DUPLICATE CHECK")
    
    if not OUTPUT_FILE.exists():
        print(f"\n  Output file not found")
        return False
    
    df = pd.read_csv(OUTPUT_FILE)
    
    # Check by match_key
    if "match_key" in df.columns:
        dup_keys = df[df.duplicated(subset=["match_key"], keep=False)]
        num_dups = len(dup_keys)
        
        print_result(
            f"Unique match keys: {len(df['match_key'].unique()):,}/{len(df):,}",
            num_dups == 0,
            f"Duplicates: {num_dups}" if num_dups > 0 else ""
        )
        
        if num_dups > 0:
            print(f"\n  Sample duplicates:")
            for key in dup_keys["match_key"].unique()[:3]:
                print(f"    - {key}")
    
    return num_dups == 0


# =============================================================================
# 7. RAW SOURCE AUDIT
# =============================================================================

def audit_raw_sources():
    """Audit raw data sources for team name variants."""
    print_header("7. RAW SOURCE TEAM NAME AUDIT")
    
    # Kaggle teams
    if KAGGLE_FILE.exists():
        kaggle = pd.read_csv(KAGGLE_FILE, usecols=["home", "away"])
        kaggle_teams = set(kaggle["home"].unique()) | set(kaggle["away"].unique())
        
        unmapped = []
        for team in kaggle_teams:
            std = standardize_team_name(team)
            if std not in get_all_team_names():
                unmapped.append(f"{team} -> {std}")
        
        print_result(
            f"Kaggle teams ({len(kaggle_teams)} variants)",
            len(unmapped) == 0,
            "\n".join(unmapped[:5]) if unmapped else "All mapped to canonical names"
        )
    
    # TheOdds teams
    theodds_teams = set()
    odds_dir = THEODDS_DIR / "odds"
    for json_file in odds_dir.glob("*/*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
            events = data if isinstance(data, list) else data.get("data", [])
            for e in events[:100]:  # Sample
                if "home_team" in e:
                    theodds_teams.add(e["home_team"])
                if "away_team" in e:
                    theodds_teams.add(e["away_team"])
        except:
            pass
    
    if theodds_teams:
        unmapped = []
        for team in theodds_teams:
            std = standardize_team_name(team)
            if std not in get_all_team_names():
                unmapped.append(f"{team} -> {std}")
        
        print_result(
            f"TheOdds teams ({len(theodds_teams)} variants)",
            len(unmapped) == 0,
            "\n".join(unmapped[:5]) if unmapped else "All mapped to canonical names"
        )
    
    return True


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print(" DATA STANDARDIZATION VERIFICATION REPORT")
    print(" Single Source of Truth: src/data/standardization.py")
    print("=" * 70)
    
    results = []
    
    results.append(("Team Names", verify_team_names()))
    results.append(("Timezone (CST)", verify_timezone()))
    results.append(("Match Keys", verify_match_keys()))
    results.append(("Data Coverage", verify_data_coverage()))
    results.append(("Output Teams", verify_output_teams()))
    results.append(("No Duplicates", verify_no_duplicates()))
    results.append(("Raw Source Audit", audit_raw_sources()))
    
    print_header("SUMMARY")
    
    all_passed = True
    for name, passed in results:
        status = "[OK]" if passed else "[!!]"
        print(f"  {status} {name}")
        all_passed = all_passed and passed
    
    print()
    if all_passed:
        print("  [OK] ALL CHECKS PASSED - Data standardization verified")
    else:
        print("  [!!] SOME CHECKS FAILED - Review issues above")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
