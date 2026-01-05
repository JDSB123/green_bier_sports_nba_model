#!/usr/bin/env python3
"""
Leakage-safe backtest using the *actual production model artifacts*.

This backtest differs from the existing walk-forward scripts:
- It DOES NOT retrain models.
- It loads the frozen production artifacts from `models/production/`.
- It computes features for each game using ONLY games strictly BEFORE that game
  (default cutoff: previous calendar days, excluding same-day games).

Intended inputs:
- A feature-rich dataset with outcomes + quarter scores + (historical) lines:
  - `data/processed/training_data_theodds.csv` (recommended; real historical
    FG+1H lines)
  - `data/processed/training_data.csv` (often missing real 1H lines; 1H
    markets may be skipped)

Outputs:
- Timestamped predictions CSV
- Timestamped summary JSON
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.features import FeatureEngineer  # noqa: E402
from src.prediction.engine import UnifiedPredictionEngine  # noqa: E402


@dataclass(frozen=True)
class MarketSpec:
    market_key: str  # fg_spread, fg_total, 1h_spread, 1h_total
    kind: str  # spread|total
    period: str  # fg|1h
    label_col: str
    # Line columns, in preference order (first non-null wins)
    line_cols: List[str]
    # Pregame-only columns that must be present (non-null)
    # to avoid placeholders
    required_pregame_cols: List[str]


MARKETS: Dict[str, MarketSpec] = {
    "fg_spread": MarketSpec(
        market_key="fg_spread",
        kind="spread",
        period="fg",
        label_col="spread_covered",
        line_cols=["spread_line", "fg_spread_line"],
        required_pregame_cols=[
            # Injury
            "home_injury_spread_impact",
            "away_injury_spread_impact",
            "injury_spread_diff",
            "home_star_out",
            "away_star_out",
            # Splits / RLM
            "is_rlm_spread",
            "sharp_side_spread",
            "spread_public_home_pct",
            "spread_ticket_money_diff",
            "spread_movement",
        ],
    ),
    "fg_total": MarketSpec(
        market_key="fg_total",
        kind="total",
        period="fg",
        label_col="total_over",
        line_cols=["total_line", "fg_total_line"],
        required_pregame_cols=[
            "is_rlm_total",
            "sharp_side_total",
        ],
    ),
    "1h_spread": MarketSpec(
        market_key="1h_spread",
        kind="spread",
        period="1h",
        label_col="1h_spread_covered",
        line_cols=["fh_spread_line", "1h_spread_line"],
        required_pregame_cols=[],
    ),
    "1h_total": MarketSpec(
        market_key="1h_total",
        kind="total",
        period="1h",
        label_col="1h_total_over",
        line_cols=["fh_total_line", "1h_total_line"],
        required_pregame_cols=[],
    ),
}

PREGAME_PASS_THROUGH_COLS: List[str] = sorted(
    {
        # Injury
        "has_injury_data",
        "home_injury_spread_impact",
        "away_injury_spread_impact",
        "injury_spread_diff",
        "home_star_out",
        "away_star_out",
        # Splits / RLM
        "has_real_splits",
        "is_rlm_spread",
        "sharp_side_spread",
        "spread_public_home_pct",
        "spread_ticket_money_diff",
        "spread_movement",
        "is_rlm_total",
        "sharp_side_total",
        # Totals split fields (harmless if unused)
        "over_public_pct",
        "under_public_pct",
        "over_money_pct",
        "under_money_pct",
        "total_ticket_money_diff",
        "total_movement",
    }
)


def _profit_flat_110(pred: int, actual: int) -> float:
    return (100.0 / 110.0) if int(pred) == int(actual) else -1.0


def _as_datetime(s: str | None) -> Optional[pd.Timestamp]:
    if not s:
        return None
    ts = pd.to_datetime(s, errors="coerce", utc=False)
    return ts if not pd.isna(ts) else None


def _select_first_nonnull(
    row: pd.Series,
    cols: Iterable[str],
) -> Optional[float]:
    for c in cols:
        if c in row.index and pd.notna(row[c]):
            try:
                return float(row[c])
            except Exception:
                return None
    return None


def _history_slice(
    df_all: pd.DataFrame,
    game_ts: pd.Timestamp,
    cutoff_granularity: str,
) -> pd.DataFrame:
    if cutoff_granularity == "day":
        d = game_ts.date()
        return df_all[df_all["date"].dt.date < d]
    if cutoff_granularity == "datetime":
        return df_all[df_all["date"] < game_ts]
    raise ValueError(f"Unknown cutoff_granularity: {cutoff_granularity}")


def _build_game_input_row(
    row: pd.Series,
    fg_spread_line: Optional[float],
    fg_total_line: Optional[float],
    fh_spread_line: Optional[float],
    fh_total_line: Optional[float],
) -> pd.Series:
    """
    Build the minimal pregame payload used by FeatureEngineer.

    IMPORTANT: This intentionally does NOT pass through postgame fields
    (scores/labels) to avoid any accidental leakage through new/unknown
    feature logic.
    """
    payload: Dict[str, Any] = {
        "date": row["date"],
        "home_team": row.get("home_team"),
        "away_team": row.get("away_team"),
        "spread_line": fg_spread_line,
        "total_line": fg_total_line,
        # FeatureEngineer uses fh_* for 1H line features
        "fh_spread_line": fh_spread_line,
        "fh_total_line": fh_total_line,
    }

    # Pass through real pregame-only inputs (splits/injuries) so we don't
    # accidentally default them inside FeatureEngineer.
    for c in PREGAME_PASS_THROUGH_COLS:
        if c in row.index and pd.notna(row[c]):
            payload[c] = row[c]

    return pd.Series(payload)


def _build_1h_features_only(
    game_input: pd.Series,
    history: pd.DataFrame,
    fe: FeatureEngineer,
    fh_spread_line: Optional[float],
    fh_total_line: Optional[float],
) -> Dict[str, float]:
    """
    Build ONLY the 1H features the production 1H models expect.

    This preserves market independence and avoids computing unrelated FG-only
    feature blocks (H2H/SOS/etc) when FG markets are not backtestable due to
    missing real pregame feeds.
    """
    home_team = game_input.get("home_team")
    away_team = game_input.get("away_team")
    game_date = pd.to_datetime(game_input.get("date"), errors="coerce")

    if not home_team or not away_team or pd.isna(game_date):
        return {}

    try:
        home_rest = fe.compute_rest_days(history, home_team, game_date)
        away_rest = fe.compute_rest_days(history, away_team, game_date)
    except Exception:
        return {}

    home_b2b = 1.0 if home_rest == 0 else 0.0
    away_b2b = 1.0 if away_rest == 0 else 0.0

    away_travel = fe.compute_travel_features(
        history,
        team=away_team,
        opponent=home_team,
        game_date=game_date,
        is_home=False,
    )
    hca = fe.compute_dynamic_hca(
        history,
        home_team=home_team,
        away_team=away_team,
        game_date=game_date,
        home_rest=home_rest,
    )

    home_1h = fe.compute_period_rolling_stats(history, home_team, game_date, "1h")
    away_1h = fe.compute_period_rolling_stats(history, away_team, game_date, "1h")
    if not home_1h or not away_1h:
        return {}

    home_1h_ppg = float(home_1h["ppg_1h"])
    home_1h_papg = float(home_1h["papg_1h"])
    home_1h_margin = float(home_1h["margin_1h"])
    away_1h_ppg = float(away_1h["ppg_1h"])
    away_1h_papg = float(away_1h["papg_1h"])
    away_1h_margin = float(away_1h["margin_1h"])

    hca_1h = float(hca) * 0.5
    travel_fatigue_1h = float(away_travel.get("travel_fatigue", 0.0)) * 0.5

    predicted_margin_1h = ((home_1h_margin - away_1h_margin) / 2.0) + hca_1h - (
        travel_fatigue_1h
    )

    home_1h_expected = (home_1h_ppg + away_1h_papg) / 2.0
    away_1h_expected = (away_1h_ppg + home_1h_papg) / 2.0
    predicted_total_1h = home_1h_expected + away_1h_expected

    feats: Dict[str, float] = {
        "home_ppg_1h": home_1h_ppg,
        "home_papg_1h": home_1h_papg,
        "home_margin_1h": home_1h_margin,
        "away_ppg_1h": away_1h_ppg,
        "away_papg_1h": away_1h_papg,
        "away_margin_1h": away_1h_margin,
        "predicted_margin_1h": float(predicted_margin_1h),
        "predicted_total_1h": float(predicted_total_1h),
        "home_b2b": float(home_b2b),
        "away_b2b": float(away_b2b),
    }

    if fh_spread_line is not None and not pd.isna(fh_spread_line):
        feats["1h_spread_line"] = float(fh_spread_line)
    if fh_total_line is not None and not pd.isna(fh_total_line):
        feats["1h_total_line"] = float(fh_total_line)

    return feats


@dataclass
class BacktestSummary:
    generated_at: str
    data: str
    models_dir: str
    cutoff_granularity: str
    start_date: Optional[str]
    end_date: Optional[str]
    markets: List[str]
    require_real_1h_lines: bool
    lookback: int
    min_history_games: int
    n_games_considered: int
    n_predictions: int
    per_market: Dict[str, Dict[str, Any]]
    skipped: Dict[str, int]
    model_info: Dict[str, Any]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Backtest frozen production models (leakage-safe)"
    )
    p.add_argument(
        "--data",
        type=str,
        default="data/processed/training_data_theodds.csv",
        help="CSV with outcomes + quarter scores + lines",
    )
    p.add_argument(
        "--models-dir",
        type=str,
        default="models/production",
        help="Directory containing production .joblib models",
    )
    p.add_argument(
        "--markets",
        type=str,
        default="all",
        help=(
            "Comma-separated market keys or 'all' "
            "(fg_spread,fg_total,1h_spread,1h_total)"
        ),
    )
    p.add_argument(
        "--start-date", type=str, default=None, help="YYYY-MM-DD (inclusive)"
    )
    p.add_argument(
        "--end-date", type=str, default=None, help="YYYY-MM-DD (inclusive)"
    )
    p.add_argument(
        "--cutoff-granularity",
        type=str,
        default="day",
        choices=["day", "datetime"],
        help=(
            "History cutoff: 'day' excludes same-day games (safest); "
            "'datetime' uses timestamps"
        ),
    )
    p.add_argument(
        "--allow-derived-1h-lines",
        action="store_true",
        help=(
            "Allow using derived 1h_* lines when fh_* lines are missing. "
            "Default is leakage-safe (fh_* only)."
        ),
    )
    p.add_argument(
        "--lookback",
        type=int,
        default=10,
        help="Lookback window for feature engineering",
    )
    p.add_argument(
        "--min-history-games",
        type=int,
        default=10,
        help="Minimum total historical games before attempting features",
    )
    p.add_argument(
        "--high-conf",
        type=float,
        default=0.60,
        help="High confidence threshold",
    )
    p.add_argument("--output-dir", type=str, default="data/backtest_results")
    p.add_argument("--tag", type=str, default="production")
    return p.parse_args()


def _has_required_pregame_inputs(row: pd.Series, spec: MarketSpec) -> bool:
    # Hard rule: do not backtest with placeholders.
    # If the dataset explicitly indicates "no real splits/injury data", we must
    # skip markets that rely on those features instead of using zeros.
    if spec.market_key in {"fg_spread", "fg_total"}:
        if "has_real_splits" not in row.index or pd.isna(
            row["has_real_splits"]
        ):
            return False
        if float(row["has_real_splits"]) != 1.0:
            return False

    if spec.market_key == "fg_spread":
        if "has_injury_data" not in row.index or pd.isna(
            row["has_injury_data"]
        ):
            return False
        if float(row["has_injury_data"]) != 1.0:
            return False

    for c in spec.required_pregame_cols:
        if c not in row.index:
            return False
        if pd.isna(row[c]):
            return False
    return True


def _print_production_model_identity(models_dir: Path) -> None:
    """
    Print the exact artifacts we will load as "production".

    This is intentionally printed BEFORE any predictions run.
    """
    pack_path = models_dir / "model_pack.json"
    if pack_path.exists():
        try:
            pack = json.loads(pack_path.read_text(encoding="utf-8"))
            version = pack.get("version")
            created_at = pack.get("created_at")
            print(
                f"[PROD] model_pack.json: version={version}, "
                f"created_at={created_at}"
            )
        except Exception as exc:
            print(f"[PROD] model_pack.json present but unreadable: {exc}")
    else:
        print("[PROD] model_pack.json: MISSING (using raw joblib artifacts)")

    artifacts = [
        "fg_spread_model.joblib",
        "fg_total_model.joblib",
        "1h_spread_model.joblib",
        "1h_total_model.joblib",
    ]
    for fname in artifacts:
        p = models_dir / fname
        if not p.exists():
            print(f"[PROD] MISSING artifact: {p}")
            continue

        payload = joblib.load(p)
        model = payload.get("pipeline") or payload.get("model")
        feats = payload.get("feature_columns") or []
        if hasattr(model, "feature_names_in_"):
            feats = list(model.feature_names_in_)

        meta = payload.get("meta") or {}
        print(
            f"[PROD] {fname}: model_type={meta.get('model_type')}, "
            f"n_features={len(feats)}"
        )


def main() -> int:
    args = parse_args()

    # Reduce noisy warnings (e.g., offseason "long rest" messages) that can
    # overwhelm output during large historical runs.
    logging.getLogger().setLevel(logging.ERROR)
    logging.getLogger("src.modeling.features").setLevel(logging.ERROR)

    data_path = (PROJECT_ROOT / args.data).resolve()
    models_dir = (PROJECT_ROOT / args.models_dir).resolve()
    out_dir = (PROJECT_ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Missing data: {data_path}")
    if not models_dir.exists():
        raise FileNotFoundError(f"Missing models dir: {models_dir}")

    _print_production_model_identity(models_dir)

    if args.markets.strip().lower() == "all":
        markets = list(MARKETS.keys())
    else:
        markets = [m.strip() for m in args.markets.split(",") if m.strip()]
        unknown = [m for m in markets if m not in MARKETS]
        if unknown:
            raise ValueError(
                f"Unknown markets: {unknown} (valid: {list(MARKETS.keys())})"
            )

    start_ts = _as_datetime(args.start_date)
    end_ts = _as_datetime(args.end_date)
    require_real_1h_lines = not bool(args.allow_derived_1h_lines)

    df_all = pd.read_csv(data_path)
    if "date" not in df_all.columns:
        raise ValueError("Input data missing required column: date")
    if not {"home_team", "away_team"}.issubset(df_all.columns):
        raise ValueError(
            "Input data missing required columns: home_team, away_team"
        )

    df_all["date"] = pd.to_datetime(df_all["date"], errors="coerce")
    df_all = df_all.dropna(subset=["date"]).sort_values("date").reset_index(
        drop=True
    )

    # Precompute date-only for leakage-safe "day" cutoff.
    df_all["date_only"] = df_all["date"].dt.date

    # Map each date_only to its first row index for fast history slicing.
    # For cutoff=day, history for games on D is df_all.iloc[:first_idx(D)].
    day_first_idx = (
        df_all.reset_index()
        .drop_duplicates(subset=["date_only"], keep="first")[["date_only", "index"]]
        .set_index("date_only")["index"]
        .to_dict()
    )

    mask = pd.Series([True] * len(df_all))
    if start_ts is not None:
        mask &= df_all["date"] >= start_ts
    if end_ts is not None:
        mask &= df_all["date"] <= end_ts

    # Keep df_all indices so we can slice history by iloc[:idx].
    df_eval = df_all.loc[mask].copy()

    engine = UnifiedPredictionEngine(models_dir=models_dir, require_all=True)
    fe = FeatureEngineer(lookback=int(args.lookback))

    # If we have no real splits/injury snapshots, FG markets are not
    # backtestable under the "no placeholders" rule. Detect this once so we
    # can avoid computing unused feature blocks.
    has_any_real_splits = (
        "has_real_splits" in df_eval.columns
        and (df_eval["has_real_splits"] == 1).any()
    )
    has_any_injury = (
        "has_injury_data" in df_eval.columns
        and (df_eval["has_injury_data"] == 1).any()
    )
    allow_fg_total = bool(has_any_real_splits)
    allow_fg_spread = bool(has_any_real_splits and has_any_injury)

    skipped: Dict[str, int] = {
        "missing_label": 0,
        "missing_line": 0,
        "missing_pregame_inputs": 0,
        "insufficient_history": 0,
        "feature_error": 0,
        "prediction_error": 0,
    }

    preds_rows: List[Dict[str, Any]] = []

    for i, (idx, row) in enumerate(df_eval.iterrows(), start=1):
        game_ts: pd.Timestamp = row["date"]

        if args.cutoff_granularity == "day":
            history_end = day_first_idx.get(row["date_only"], int(idx))
        else:
            history_end = int(idx)

        history = df_all.iloc[:history_end]
        if len(history) < int(args.min_history_games):
            skipped["insufficient_history"] += 1
            continue

        # Lines (used both for feature payload and for evaluation)
        if allow_fg_spread:
            fg_spread_line = _select_first_nonnull(
                row, MARKETS["fg_spread"].line_cols
            )
        else:
            fg_spread_line = None

        if allow_fg_total:
            fg_total_line = _select_first_nonnull(
                row, MARKETS["fg_total"].line_cols
            )
        else:
            fg_total_line = None

        # 1H: prefer fh_* always; optionally forbid derived 1h_*.
        if require_real_1h_lines:
            fh_spread_line = _select_first_nonnull(row, ["fh_spread_line"])
            fh_total_line = _select_first_nonnull(row, ["fh_total_line"])
        else:
            fh_spread_line = _select_first_nonnull(
                row, MARKETS["1h_spread"].line_cols
            )
            fh_total_line = _select_first_nonnull(
                row, MARKETS["1h_total"].line_cols
            )

        game_input = _build_game_input_row(
            row,
            fg_spread_line=fg_spread_line,
            fg_total_line=fg_total_line,
            fh_spread_line=fh_spread_line,
            fh_total_line=fh_total_line,
        )

        # Fast path: if FG markets can't be evaluated (no real pregame feeds),
        # compute ONLY the 1H feature pipeline.
        if not (allow_fg_spread or allow_fg_total):
            features = _build_1h_features_only(
                game_input=game_input,
                history=history,
                fe=fe,
                fh_spread_line=fh_spread_line,
                fh_total_line=fh_total_line,
            )
            if not features:
                skipped["feature_error"] += 1
                continue
        else:
            try:
                features = fe.build_game_features(game_input, history)
            except Exception:
                skipped["feature_error"] += 1
                continue

            if not features:
                skipped["feature_error"] += 1
                continue

        try:
            pred_all = engine.predict_all_markets(
                features=features,
                fg_spread_line=fg_spread_line,
                fg_total_line=fg_total_line,
                fh_spread_line=fh_spread_line,
                fh_total_line=fh_total_line,
            )
        except Exception:
            skipped["prediction_error"] += 1
            continue

        # Evaluate requested markets
        for m in markets:
            spec = MARKETS[m]

            # Enforce: no placeholders/no silent defaulting for pregame-only
            # inputs.
            # If we don't have the required pregame fields in the dataset row,
            # we skip this market for this game rather than default to zeros.
            if not _has_required_pregame_inputs(row, spec):
                skipped["missing_pregame_inputs"] += 1
                continue

            actual_raw = row.get(spec.label_col)
            if actual_raw is None or pd.isna(actual_raw):
                skipped["missing_label"] += 1
                continue

            # Select line for this market (may be None even if other markets
            # have lines)
            if spec.period == "fg":
                if spec.kind == "spread":
                    line_val = fg_spread_line
                else:
                    line_val = fg_total_line
            else:
                if spec.kind == "spread":
                    line_val = fh_spread_line
                else:
                    line_val = fh_total_line

            if line_val is None or pd.isna(line_val):
                skipped["missing_line"] += 1
                continue

            # Pull the right period bucket
            if spec.period == "fg":
                period_bucket = "full_game"
            else:
                period_bucket = "first_half"
            kind_bucket = "spread" if spec.kind == "spread" else "total"
            pred = (pred_all.get(period_bucket) or {}).get(kind_bucket) or {}
            if not pred:
                skipped["prediction_error"] += 1
                continue

            # Convert to our label encoding (1 = home cover / over)
            bet_side = pred.get("bet_side")
            if spec.kind == "spread":
                pred_label = 1 if bet_side == "home" else 0
                proba = float(pred.get("home_cover_prob", np.nan))
            else:
                pred_label = 1 if bet_side == "over" else 0
                proba = float(pred.get("over_prob", np.nan))

            actual = int(actual_raw)
            profit = _profit_flat_110(pred_label, actual)

            preds_rows.append(
                {
                    "date": game_ts,
                    "home_team": row.get("home_team"),
                    "away_team": row.get("away_team"),
                    "market": m,
                    "line": float(line_val),
                    "bet_side": bet_side,
                    "pred": int(pred_label),
                    "actual": int(actual),
                    "proba": proba,
                    "confidence": float(pred.get("confidence", np.nan)),
                    "edge": float(pred.get("edge", np.nan)),
                    "raw_edge": float(pred.get("raw_edge", np.nan)),
                    "passes_filter": bool(pred.get("passes_filter", False)),
                    "filter_reason": pred.get("filter_reason"),
                    "profit": float(profit),
                    "correct": int(pred_label == actual),
                }
            )

        if i % 200 == 0:
            print(
                f"[PROGRESS] games={i}/{len(df_eval)} "
                f"preds={len(preds_rows)} skipped={skipped}"
            )

    preds_df = pd.DataFrame(preds_rows)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"{args.tag}_" if args.tag else ""
    preds_path = out_dir / f"backtest_prod_preds_{tag}{run_ts}.csv"
    preds_df.to_csv(preds_path, index=False)

    per_market: Dict[str, Dict[str, Any]] = {}
    if preds_df.empty:
        for m in markets:
            per_market[m] = {"n": 0}
    else:
        for m in markets:
            md = preds_df[preds_df["market"] == m].copy()
            if md.empty:
                per_market[m] = {"n": 0}
                continue
            acc_all = float(md["correct"].mean())
            prof_all = float(md["profit"].sum())
            roi_all = float(prof_all / len(md))

            md_f = md[md["passes_filter"]].copy()
            acc_f = float(md_f["correct"].mean()) if len(md_f) else 0.0
            prof_f = float(md_f["profit"].sum()) if len(md_f) else 0.0
            roi_f = float(prof_f / len(md_f)) if len(md_f) else 0.0

            md_hc = md[md["confidence"] >= float(args.high_conf)].copy()
            acc_hc = float(md_hc["correct"].mean()) if len(md_hc) else 0.0
            prof_hc = float(md_hc["profit"].sum()) if len(md_hc) else 0.0
            roi_hc = float(prof_hc / len(md_hc)) if len(md_hc) else 0.0

            per_market[m] = {
                "n": int(len(md)),
                "accuracy_all": acc_all,
                "roi_all": roi_all,
                "profit_all": prof_all,
                "n_filtered": int(len(md_f)),
                "accuracy_filtered": acc_f,
                "roi_filtered": roi_f,
                "profit_filtered": prof_f,
                "n_high_conf": int(len(md_hc)),
                "accuracy_high_conf": acc_hc,
                "roi_high_conf": roi_hc,
                "profit_high_conf": prof_hc,
            }

    summary = BacktestSummary(
        generated_at=run_ts,
        data=str(data_path),
        models_dir=str(models_dir),
        cutoff_granularity=args.cutoff_granularity,
        start_date=args.start_date,
        end_date=args.end_date,
        markets=markets,
        require_real_1h_lines=bool(require_real_1h_lines),
        lookback=int(args.lookback),
        min_history_games=int(args.min_history_games),
        n_games_considered=int(len(df_eval)),
        n_predictions=int(len(preds_df)),
        per_market=per_market,
        skipped=skipped,
        model_info=engine.get_model_info(),
    )

    summary_path = out_dir / f"backtest_prod_summary_{tag}{run_ts}.json"
    summary_path.write_text(
        json.dumps(asdict(summary), indent=2),
        encoding="utf-8",
    )

    print(f"[OK] Wrote predictions: {preds_path}")
    print(f"[OK] Wrote summary: {summary_path}")
    print(f"[OK] Predictions: {len(preds_df)} across markets={markets}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
