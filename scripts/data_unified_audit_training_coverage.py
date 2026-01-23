#!/usr/bin/env python3
"""
Audit Training Data Coverage

Validates canonical training_data.csv for:
- Feature coverage (injuries, travel, ELO, markets)
- Point-in-time price integrity
- Date ranges by market type
- Source attribution for key columns

Usage:
    python scripts/data_unified_audit_training_coverage.py
    python scripts/data_unified_audit_training_coverage.py --verbose
    python scripts/data_unified_audit_training_coverage.py --output audit_report.json
"""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# COLUMN DEFINITIONS BY CATEGORY
# ============================================================================

# Core market line columns (6 markets: FG/1H × spread/total/moneyline)
MARKET_COLUMNS = {
    "fg_spread": {
        "line": ["fg_spread_line", "spread_line"],
        "price_home": ["fg_spread_price_home", "spread_price_home"],
        "price_away": ["fg_spread_price_away", "spread_price_away"],
        "label": ["fg_spread_covered"],
        "min_date": "2023-05-01",  # Spreads only available from May 2023
    },
    "fg_total": {
        "line": ["fg_total_line", "total_line"],
        "price_over": ["fg_total_price_over", "total_price_over"],
        "price_under": ["fg_total_price_under", "total_price_under"],
        "label": ["fg_total_over"],
        "min_date": "2023-01-01",
    },
    "fg_moneyline": {
        "line_home": ["fg_ml_home", "moneyline_home", "to_fg_ml_home"],
        "line_away": ["fg_ml_away", "moneyline_away", "to_fg_ml_away"],
        "label": [],  # Derived from home_score > away_score
        "min_date": "2023-01-01",
    },
    "1h_spread": {
        "line": ["1h_spread_line", "fh_spread_line"],
        "price_home": ["1h_spread_price_home"],
        "price_away": ["1h_spread_price_away"],
        "label": ["1h_spread_covered"],
        "min_date": "2023-05-01",
    },
    "1h_total": {
        "line": ["1h_total_line", "fh_total_line"],
        "price_over": ["1h_total_price_over"],
        "price_under": ["1h_total_price_under"],
        "label": ["1h_total_over"],
        "min_date": "2023-01-01",
    },
    "1h_moneyline": {
        "line_home": ["1h_ml_home", "to_1h_ml_home", "exp_1h_ml_home"],
        "line_away": ["1h_ml_away", "to_1h_ml_away", "exp_1h_ml_away"],
        "label": [],  # Derived from 1H scores
        "min_date": "2023-01-01",
    },
}

# Injury feature columns
INJURY_COLUMNS = [
    "home_injury_spread_impact",
    "away_injury_spread_impact",
    "injury_spread_diff",
    "home_star_out",
    "away_star_out",
    "home_injury_total_impact",
    "away_injury_total_impact",
]

# Travel feature columns
TRAVEL_COLUMNS = [
    "away_travel_distance",
    "away_timezone_change",
    "away_travel_fatigue",
    "is_away_long_trip",
    "is_away_cross_country",
    "away_b2b_travel_penalty",
    "home_court_advantage",
    "dynamic_hca",
]

# ELO feature columns
ELO_COLUMNS = [
    "home_elo",
    "away_elo",
    "elo_diff",
    "elo_prob",
]

# Core required columns
REQUIRED_COLUMNS = [
    "game_date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
]

# Actual value columns for backtesting
ACTUAL_VALUE_COLUMNS = [
    "home_1h",
    "away_1h",
    "home_q1",
    "home_q2",
    "away_q1",
    "away_q2",
]


def check_column_coverage(
    df: pd.DataFrame,
    columns: List[str],
    category: str,
    min_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Check coverage for a list of columns."""
    results = {}

    # Filter by date if specified
    if min_date and "game_date" in df.columns:
        df_filtered = df[df["game_date"] >= min_date].copy()
        date_filter_applied = True
    else:
        df_filtered = df
        date_filter_applied = False

    total_rows = len(df_filtered)

    for col in columns:
        if col in df_filtered.columns:
            non_null = df_filtered[col].notna().sum()
            coverage_pct = (non_null / total_rows *
                            100) if total_rows > 0 else 0

            # Get date range where column has values
            col_dates = df_filtered[df_filtered[col].notna()]["game_date"]
            if len(col_dates) > 0:
                date_range = {
                    "min": str(col_dates.min())[:10],
                    "max": str(col_dates.max())[:10],
                }
            else:
                date_range = None

            results[col] = {
                "present": True,
                "non_null": int(non_null),
                "total": total_rows,
                "coverage_pct": round(coverage_pct, 1),
                "date_range": date_range,
                "date_filter": min_date if date_filter_applied else None,
            }
        else:
            results[col] = {
                "present": False,
                "non_null": 0,
                "total": total_rows,
                "coverage_pct": 0.0,
                "date_range": None,
                "date_filter": min_date if date_filter_applied else None,
            }

    return results


def check_market_coverage(df: pd.DataFrame) -> Dict[str, Any]:
    """Check coverage for all 6 markets with appropriate date filters."""
    market_results = {}

    for market_key, config in MARKET_COLUMNS.items():
        min_date = config.get("min_date")

        # Filter data by minimum date
        if min_date and "game_date" in df.columns:
            df_filtered = df[df["game_date"] >= min_date].copy()
        else:
            df_filtered = df.copy()

        total_games = len(df_filtered)

        # Check line column (use first available)
        line_cols = config.get("line", config.get("line_home", []))
        if isinstance(line_cols, list) and line_cols:
            line_col = None
            line_coverage = 0
            for col in line_cols:
                if col in df_filtered.columns:
                    cov = df_filtered[col].notna().sum()
                    if cov > line_coverage:
                        line_col = col
                        line_coverage = cov
        else:
            line_col = None
            line_coverage = 0

        # Check label column
        label_cols = config.get("label", [])
        label_col = None
        label_coverage = 0
        for col in label_cols:
            if col in df_filtered.columns:
                cov = df_filtered[col].notna().sum()
                if cov > label_coverage:
                    label_col = col
                    label_coverage = cov

        market_results[market_key] = {
            "min_date": min_date,
            "total_games": total_games,
            "line_column": line_col,
            "line_coverage": int(line_coverage),
            "line_coverage_pct": round(line_coverage / total_games * 100, 1) if total_games > 0 else 0,
            "label_column": label_col,
            "label_coverage": int(label_coverage),
            "label_coverage_pct": round(label_coverage / total_games * 100, 1) if total_games > 0 else 0,
            "backtest_ready": line_coverage > 0 and (label_coverage > 0 or not label_cols),
        }

    return market_results


def check_point_in_time_integrity(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Check that price columns represent opening (point-in-time) values.

    This validates that prices use opening snapshots, not closing lines.
    """
    results = {
        "checks_performed": [],
        "warnings": [],
        "status": "OK",
    }

    # Check 1: Opening vs closing spread lines (if both exist)
    if "spread_opening_line" in df.columns and "spread_closing_line" in df.columns:
        both_exist = df[["spread_opening_line",
                         "spread_closing_line"]].notna().all(axis=1)
        if both_exist.sum() > 100:
            diff = (df.loc[both_exist, "spread_closing_line"] -
                    df.loc[both_exist, "spread_opening_line"]).abs()
            avg_movement = diff.mean()
            results["checks_performed"].append({
                "check": "opening_vs_closing_spread",
                "samples": int(both_exist.sum()),
                "avg_line_movement": round(avg_movement, 2),
                "interpretation": "Lines move on average, confirming different snapshots"
            })

    # Check 2: Verify spread_line matches opening (not closing)
    if "fg_spread_line" in df.columns and "spread_opening_line" in df.columns:
        both_exist = df[["fg_spread_line",
                         "spread_opening_line"]].notna().all(axis=1)
        if both_exist.sum() > 100:
            matches_opening = (df.loc[both_exist, "fg_spread_line"] ==
                               df.loc[both_exist, "spread_opening_line"]).mean()
            results["checks_performed"].append({
                "check": "fg_spread_uses_opening",
                "samples": int(both_exist.sum()),
                "match_rate": round(matches_opening * 100, 1),
                "interpretation": "High match = using opening lines (good for backtesting)"
            })
            if matches_opening < 0.5:
                results["warnings"].append(
                    "fg_spread_line may not be using opening lines - verify data pipeline"
                )
                results["status"] = "WARNING"

    # Check 3: Price columns exist and have reasonable values
    price_cols = [
        "fg_spread_price_home", "fg_spread_price_away",
        "fg_total_price_over", "fg_total_price_under",
    ]
    for col in price_cols:
        if col in df.columns:
            valid = df[col].notna()
            if valid.sum() > 0:
                values = df.loc[valid, col]
                # American odds typically range from -200 to +200 for spreads
                reasonable = ((values >= -300) & (values <= 300)).mean()
                results["checks_performed"].append({
                    "check": f"{col}_reasonable_range",
                    "samples": int(valid.sum()),
                    "in_range_pct": round(reasonable * 100, 1),
                })

    return results


def generate_audit_report(df: pd.DataFrame, verbose: bool = False) -> Dict[str, Any]:
    """Generate comprehensive audit report."""

    # Parse dates
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

    report = {
        "generated_at": datetime.now().isoformat(),
        "file_stats": {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "date_range": {
                "min": str(df["game_date"].min())[:10] if "game_date" in df.columns else None,
                "max": str(df["game_date"].max())[:10] if "game_date" in df.columns else None,
            },
        },
        "required_columns": check_column_coverage(df, REQUIRED_COLUMNS, "required"),
        "injury_features": check_column_coverage(df, INJURY_COLUMNS, "injury"),
        "travel_features": check_column_coverage(df, TRAVEL_COLUMNS, "travel"),
        "elo_features": check_column_coverage(df, ELO_COLUMNS, "elo"),
        "actual_values": check_column_coverage(df, ACTUAL_VALUE_COLUMNS, "actuals"),
        "market_coverage": check_market_coverage(df),
        "point_in_time_integrity": check_point_in_time_integrity(df),
    }

    # Summary statistics
    summary = {
        "injury_avg_coverage": 0,
        "travel_avg_coverage": 0,
        "elo_avg_coverage": 0,
        "markets_ready": [],
        "markets_not_ready": [],
        "overall_status": "OK",
        "issues": [],
    }

    # Calculate averages
    for category, features in [
        ("injury", report["injury_features"]),
        ("travel", report["travel_features"]),
        ("elo", report["elo_features"]),
    ]:
        coverages = [v["coverage_pct"]
                     for v in features.values() if v["present"]]
        if coverages:
            summary[f"{category}_avg_coverage"] = round(
                sum(coverages) / len(coverages), 1)

    # Check market readiness
    for market, data in report["market_coverage"].items():
        if data["backtest_ready"]:
            summary["markets_ready"].append(market)
        else:
            summary["markets_not_ready"].append(market)
            summary["issues"].append(f"{market}: missing line or label data")

    # Check for major issues
    if summary["injury_avg_coverage"] < 50:
        summary["issues"].append(
            f"Low injury coverage: {summary['injury_avg_coverage']}%")
        summary["overall_status"] = "WARNING"

    if summary["travel_avg_coverage"] < 50:
        summary["issues"].append(
            f"Low travel coverage: {summary['travel_avg_coverage']}%")
        summary["overall_status"] = "WARNING"

    if report["point_in_time_integrity"]["status"] != "OK":
        summary["issues"].extend(report["point_in_time_integrity"]["warnings"])
        summary["overall_status"] = "WARNING"

    report["summary"] = summary

    return report


def print_report(report: Dict[str, Any], verbose: bool = False):
    """Print formatted audit report."""

    print("=" * 80)
    print("TRAINING DATA COVERAGE AUDIT")
    print("=" * 80)
    print(f"Generated: {report['generated_at']}")
    print()

    # File stats
    stats = report["file_stats"]
    print(
        f"File: {stats['total_rows']:,} rows × {stats['total_columns']} columns")
    print(
        f"Date range: {stats['date_range']['min']} to {stats['date_range']['max']}")
    print()

    # Market coverage table
    print("-" * 80)
    print("MARKET COVERAGE (6 Markets)")
    print("-" * 80)
    print(f"{'Market':<15} {'Min Date':<12} {'Games':<8} {'Line Col':<25} {'Line %':<8} {'Label %':<8} {'Ready'}")
    print("-" * 80)

    for market, data in report["market_coverage"].items():
        ready_str = "✓" if data["backtest_ready"] else "✗"
        line_col = data["line_column"] or "N/A"
        if len(line_col) > 23:
            line_col = line_col[:20] + "..."
        print(
            f"{market:<15} {data['min_date'] or 'N/A':<12} "
            f"{data['total_games']:<8} {line_col:<25} "
            f"{data['line_coverage_pct']:>5.1f}%  {data['label_coverage_pct']:>5.1f}%   {ready_str}"
        )
    print()

    # Feature coverage summary
    print("-" * 80)
    print("FEATURE COVERAGE BY CATEGORY")
    print("-" * 80)

    for category, label in [
        ("injury_features", "INJURY"),
        ("travel_features", "TRAVEL"),
        ("elo_features", "ELO"),
        ("actual_values", "ACTUALS"),
    ]:
        features = report[category]
        present = sum(1 for v in features.values() if v["present"])
        total = len(features)
        coverages = [v["coverage_pct"]
                     for v in features.values() if v["present"]]
        avg_cov = sum(coverages) / len(coverages) if coverages else 0

        print(
            f"\n{label}: {present}/{total} columns present, avg coverage: {avg_cov:.1f}%")

        if verbose:
            for col, data in features.items():
                status = "✓" if data["present"] else "✗"
                cov = f"{data['coverage_pct']:>5.1f}%" if data["present"] else "N/A"
                date_info = f"({data['date_range']['min']} to {data['date_range']['max']})" if data.get(
                    "date_range") else ""
                print(f"  {status} {col:<35} {cov} {date_info}")

    # Point-in-time integrity
    print()
    print("-" * 80)
    print("POINT-IN-TIME PRICE INTEGRITY")
    print("-" * 80)

    pit = report["point_in_time_integrity"]
    print(f"Status: {pit['status']}")

    if pit["checks_performed"]:
        for check in pit["checks_performed"]:
            print(
                f"  • {check['check']}: {check.get('match_rate', check.get('avg_line_movement', 'N/A'))}")

    if pit["warnings"]:
        print("Warnings:")
        for warn in pit["warnings"]:
            print(f"  ⚠ {warn}")

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    summary = report["summary"]
    print(f"Overall Status: {summary['overall_status']}")
    print(
        f"Markets Ready for Backtest: {', '.join(summary['markets_ready']) or 'None'}")
    print(
        f"Markets NOT Ready: {', '.join(summary['markets_not_ready']) or 'None'}")
    print(f"Injury Coverage: {summary['injury_avg_coverage']}%")
    print(f"Travel Coverage: {summary['travel_avg_coverage']}%")
    print(f"ELO Coverage: {summary['elo_avg_coverage']}%")

    if summary["issues"]:
        print("\nIssues Found:")
        for issue in summary["issues"]:
            print(f"  ⚠ {issue}")
    else:
        print("\n✓ No major issues found")


def main():
    parser = argparse.ArgumentParser(
        description="Audit training data coverage for NBA prediction system"
    )
    parser.add_argument(
        "--data",
        default="data/processed/training_data.csv",
        help="Path to training data CSV",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed column-by-column breakdown",
    )
    parser.add_argument(
        "--output", "-o",
        help="Save report to JSON file",
    )
    args = parser.parse_args()

    data_path = PROJECT_ROOT / args.data

    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        sys.exit(1)

    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path, low_memory=False)

    report = generate_audit_report(df, verbose=args.verbose)
    print_report(report, verbose=args.verbose)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nReport saved to: {output_path}")

    # Exit with appropriate code
    if report["summary"]["overall_status"] != "OK":
        sys.exit(1)


if __name__ == "__main__":
    main()
