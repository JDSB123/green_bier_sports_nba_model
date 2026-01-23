#!/usr/bin/env python3
"""Audit historical training/odds data for NaNs and conflicting values.

Goal
- Find NaNs and coverage gaps in *canonical* columns used for backtests.
- Detect duplicate game keys and conflicting line/odds values across sources.
- Produce a human-readable Markdown + machine-readable JSON report.

Notes
- This script does NOT impute/fill values.
- Conflicts are reported; resolution should happen upstream in the builder/merger.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from src.utils.historical_guard import resolve_historical_output_root, require_historical_mode, ensure_historical_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_DATA = _DEFAULT_PROCESSED_DIR / "training_data.csv"
OUTPUT_DIR = resolve_historical_output_root("diagnostics")


@dataclass(frozen=True)
class ColumnConflict:
    canonical: str
    other: str
    threshold: float


def _as_datetime(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    # Normalize to date (no time) for stable grouping
    return parsed.dt.normalize()


def _pick_first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _compute_season(date: pd.Timestamp) -> str:
    """NBA season label aligned to real season boundaries.

    - Oct-Dec: season starts this calendar year
    - Jan-Jun: season starts previous year
    - Jul-Sep (offseason): treat as upcoming season
    """
    if pd.isna(date):
        return "unknown"
    y = int(date.year)
    m = int(date.month)
    if m >= 10:
        start = y
    elif m <= 6:
        start = y - 1
    else:
        start = y
    return f"{start}-{start+1}"


def _safe_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _normalize_season_value(value: Any) -> str:
    """Normalize legacy season formats to 'YYYY-YYYY'.

    Examples:
    - 2023 -> '2023-2024'
    - '2025-26' -> '2025-2026'
    - '2025-2026' -> '2025-2026'
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "unknown"

    s = str(value).strip()
    if not s:
        return "unknown"

    # pure year
    if s.isdigit() and len(s) == 4:
        y = int(s)
        return f"{y}-{y+1}"

    # 'YYYY-YY'
    if "-" in s:
        left, right = s.split("-", 1)
        left = left.strip()
        right = right.strip()
        if left.isdigit() and len(left) == 4:
            y = int(left)
            if right.isdigit() and len(right) == 2:
                return f"{y}-{y+1}"
            if right.isdigit() and len(right) == 4:
                return f"{y}-{int(right)}"

    return s


def _summarize_nulls(df: pd.DataFrame, cols: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for c in cols:
        if c not in df.columns:
            out[c] = {"present": False}
            continue
        nulls = int(df[c].isna().sum())
        out[c] = {
            "present": True,
            "nulls": nulls,
            "null_pct": float(nulls / max(1, len(df))),
        }
    return out


def _find_duplicates(df: pd.DataFrame, key_col: str) -> Dict[str, Any]:
    vc = df[key_col].value_counts(dropna=False)
    dup_keys = vc[vc > 1]
    sample_keys = dup_keys.head(25).index.tolist()
    return {
        "duplicate_key_count": int((vc > 1).sum()),
        "duplicate_row_count": int(dup_keys.sum()) if len(dup_keys) else 0,
        "sample_keys": sample_keys,
    }


def _conflict_report(
    df: pd.DataFrame,
    key_cols: List[str],
    conflicts: List[ColumnConflict],
) -> Dict[str, Any]:
    reports: Dict[str, Any] = {}

    for conflict in conflicts:
        canonical, other, threshold = conflict.canonical, conflict.other, conflict.threshold
        if canonical not in df.columns or other not in df.columns:
            reports[f"{canonical}__vs__{other}"] = {"present": False}
            continue

        a = _safe_float_series(df[canonical])
        b = _safe_float_series(df[other])
        both = a.notna() & b.notna()
        if both.sum() == 0:
            reports[f"{canonical}__vs__{other}"] = {
                "present": True,
                "rows_with_both": 0,
                "conflict_rows": 0,
            }
            continue

        delta = (a[both] - b[both]).abs()
        bad = delta >= threshold

        # Collect a small sample of the worst conflicts
        sample = (
            df.loc[both].assign(_delta=delta)
            .sort_values("_delta", ascending=False)
            .head(25)
        )
        sample_records = sample[key_cols + [canonical,
                                            other, "_delta"]].to_dict(orient="records")

        reports[f"{canonical}__vs__{other}"] = {
            "present": True,
            "rows_with_both": int(both.sum()),
            "conflict_rows": int(bad.sum()),
            "conflict_pct_of_both": float(bad.mean()),
            "threshold": threshold,
            "sample": sample_records,
        }

    return reports


def _write_markdown(report: Dict[str, Any], path: Path) -> None:
    def pct(x: float) -> str:
        return f"{x*100:.2f}%"

    lines: List[str] = []
    lines.append("# Historical Data Integrity Audit")
    lines.append("")
    lines.append(f"- Generated: {report['generated_at']}")
    lines.append(f"- Data: `{report['data_path']}`")
    lines.append(f"- Rows: {report['row_count']:,}")
    lines.append("")

    lines.append("## Key Columns")
    for k, v in report["key_columns"].items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")

    lines.append("## Null Coverage (Canonical)")
    lines.append("| Column | Nulls | Null % |")
    lines.append("|---|---:|---:|")
    for c, info in report["canonical_nulls"].items():
        if not info.get("present"):
            lines.append(f"| {c} | MISSING | MISSING |")
        else:
            lines.append(
                f"| {c} | {info['nulls']:,} | {pct(info['null_pct'])} |")
    lines.append("")

    dup = report["duplicates"]
    lines.append("## Duplicate Game Keys")
    lines.append(f"- Keys with duplicates: {dup['duplicate_key_count']:,}")
    lines.append(f"- Duplicate rows total: {dup['duplicate_row_count']:,}")
    if dup["sample_keys"]:
        lines.append("- Sample keys:")
        for k in dup["sample_keys"][:10]:
            lines.append(f"  - `{k}`")
    lines.append("")

    lines.append("## Conflicts (Across Sources)")
    lines.append(
        "Conflicts are rows where both values exist and differ by at least the threshold.")
    lines.append("")
    lines.append(
        "| Pair | Rows w/ both | Conflicts | Conflict % | Threshold |")
    lines.append("|---|---:|---:|---:|---:|")

    for pair, info in report["conflicts"].items():
        if not info.get("present"):
            lines.append(f"| {pair} | MISSING | MISSING | MISSING | MISSING |")
        else:
            lines.append(
                f"| {pair} | {info['rows_with_both']:,} | {info['conflict_rows']:,} | {pct(info.get('conflict_pct_of_both', 0.0))} | {info.get('threshold', 'N/A')} |"
            )

    lines.append("")

    lines.append("## Coverage by Season")
    lines.append(
        "| Season | Games | FG spread line | FG total line | 1H spread line | 1H total line |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in report["coverage_by_season"]:
        lines.append(
            "| {season} | {games:,} | {fg_spread:.2%} | {fg_total:.2%} | {h1_spread:.2%} | {h1_total:.2%} |".format(
                **row)
        )

    if report.get("season_alignment"):
        sa = report["season_alignment"]
        lines.append("")
        lines.append("## Season Alignment")
        lines.append(
            f"- Rows with a season mismatch: {sa['mismatch_rows']:,} / {sa['total_rows']:,} ({pct(sa['mismatch_pct'])})")
        if sa.get("sample"):
            lines.append("- Sample mismatches:")
            for r in sa["sample"][:10]:
                lines.append(
                    "  - date={date}, season_col={season_col}, computed={computed} ({home} vs {away})".format(
                        **r)
                )

    path.write_text("\n".join(lines), encoding="utf-8")


def audit(data_path: Path) -> Dict[str, Any]:
    df = pd.read_csv(data_path, low_memory=False)

    date_col = _pick_first_existing(df, ["date", "game_date"]) or "date"
    if date_col not in df.columns:
        raise ValueError("No date/game_date column found")

    df["_audit_date"] = _as_datetime(df[date_col])

    home_col = _pick_first_existing(df, ["home_team", "home"]) or "home_team"
    away_col = _pick_first_existing(df, ["away_team", "away"]) or "away_team"

    game_id_col = _pick_first_existing(
        df, ["game_id", "match_key"])  # prefer game_id
    if game_id_col is None:
        # last resort (not ideal): date+teams
        df["_game_key"] = df["_audit_date"].astype(
            str) + "|" + df[home_col].astype(str) + "|" + df[away_col].astype(str)
        game_id_col = "_game_key"
    else:
        df["_game_key"] = df[game_id_col].astype(str)

    key_cols = ["_game_key", "_audit_date", home_col, away_col]

    canonical_cols = [
        "spread_line",
        "total_line",
        "1h_spread_line",
        "1h_total_line",
        "fg_spread_line",
        "fg_total_line",
        "fg_total_actual",
        "1h_total_actual",
    ]

    conflict_pairs: List[ColumnConflict] = [
        # FG lines
        ColumnConflict("spread_line", "to_fg_spread", 0.25),
        ColumnConflict("spread_line", "kaggle_fg_spread", 0.25),
        ColumnConflict("total_line", "to_fg_total", 0.5),
        ColumnConflict("total_line", "kaggle_fg_total", 0.5),
        ColumnConflict("spread_line", "open_spread", 0.25),
        ColumnConflict("spread_line", "close_spread", 0.25),
        ColumnConflict("total_line", "open_total", 0.5),
        ColumnConflict("total_line", "close_total", 0.5),
        # 1H lines
        ColumnConflict("1h_spread_line", "to_1h_spread", 0.25),
        ColumnConflict("1h_spread_line", "exp_1h_spread", 0.25),
        ColumnConflict("1h_total_line", "to_1h_total", 0.5),
        ColumnConflict("1h_total_line", "exp_1h_total", 0.5),
        # 1H moneyline prices (if present)
        ColumnConflict("1h_ml_home", "to_1h_ml_home", 1.0),
        ColumnConflict("1h_ml_home", "exp_1h_ml_home", 1.0),
        ColumnConflict("1h_ml_away", "to_1h_ml_away", 1.0),
        ColumnConflict("1h_ml_away", "exp_1h_ml_away", 1.0),
    ]

    # Coverage by season for the 4 primary markets
    def coverage(col: str) -> float:
        if col not in df.columns:
            return 0.0
        return float(df[col].notna().mean())

    df["_season"] = df["_audit_date"].apply(_compute_season)
    season_rows = []
    for season, sdf in df.groupby("_season", dropna=False):
        season_rows.append(
            {
                "season": str(season),
                "games": int(len(sdf)),
                "fg_spread": float(sdf.get("spread_line").notna().mean()) if "spread_line" in sdf.columns else 0.0,
                "fg_total": float(sdf.get("total_line").notna().mean()) if "total_line" in sdf.columns else 0.0,
                "h1_spread": float(sdf.get("1h_spread_line").notna().mean()) if "1h_spread_line" in sdf.columns else 0.0,
                "h1_total": float(sdf.get("1h_total_line").notna().mean()) if "1h_total_line" in sdf.columns else 0.0,
            }
        )
    season_rows = sorted(season_rows, key=lambda r: r["season"])

    season_alignment: Dict[str, Any] = {}
    if "season" in df.columns:
        season_col = df["season"].apply(_normalize_season_value).astype(str)
        computed = df["_season"].astype(str)
        mismatch = season_col.notna() & (season_col != computed) & (computed != "unknown")
        sample = (
            df.loc[mismatch]
            .head(25)
            .assign(
                date=df["_audit_date"].astype(str),
                season_col=season_col,
                computed=computed,
                home=df[home_col].astype(str),
                away=df[away_col].astype(str),
            )[["date", "season_col", "computed", "home", "away"]]
            .to_dict(orient="records")
        )
        season_alignment = {
            "total_rows": int(len(df)),
            "mismatch_rows": int(mismatch.sum()),
            "mismatch_pct": float(mismatch.mean()) if len(df) else 0.0,
            "sample": sample,
        }

    report: Dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "data_path": str(data_path),
        "row_count": int(len(df)),
        "key_columns": {
            "date": date_col,
            "home": home_col,
            "away": away_col,
            "game_id": game_id_col,
        },
        "canonical_nulls": _summarize_nulls(df, canonical_cols),
        "duplicates": _find_duplicates(df, "_game_key"),
        "conflicts": _conflict_report(df, key_cols=key_cols, conflicts=conflict_pairs),
        "coverage_by_season": season_rows,
        "season_alignment": season_alignment,
    }

    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit historical training data for NaNs/coverage/conflicts")
    require_historical_mode()
    parser.add_argument("--data", default=str(DEFAULT_DATA),
                        help="Path to training data CSV")
    parser.add_argument("--out-dir", default=str(OUTPUT_DIR),
                        help="Directory for audit outputs")
    args = parser.parse_args()
    ensure_historical_path(Path(args.out_dir), "out-dir")

    data_path = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = audit(data_path)

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"historical_audit_{stamp}.json"
    md_path = out_dir / f"historical_audit_{stamp}.md"

    json_path.write_text(json.dumps(
        report, indent=2, default=str), encoding="utf-8")
    _write_markdown(report, md_path)

    print(f"Audit report written: {md_path}")
    print(f"JSON report written: {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
