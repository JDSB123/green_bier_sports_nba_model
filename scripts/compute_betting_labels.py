"""
Compute betting outcome labels for merged training data.

Adds/updates columns for:
- fg_spread_covered: 1 if home team covers full game spread
- fg_total_over: 1 if full game total goes over the line
- 1h_spread_covered: 1 if home team covers 1st half spread
- 1h_total_over: 1 if 1st half total goes over the line
- fg_home_win: 1 if home team wins outright

Also ensures essential columns exist with consistent naming.
"""

import argparse
import logging
from pathlib import Path
import json
import time

import pandas as pd
import numpy as np

# region agent log
_DEBUG_LOG_PATH = Path(__file__).resolve().parents[1] / ".cursor" / "debug.log"


def _agent_log(run_id: str, hypothesis_id: str, location: str, message: str, data: dict) -> None:
    try:
        payload = {
            "sessionId": "debug-session",
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        _DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass

# endregion

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_merged_data(path: Path) -> pd.DataFrame:
    """Load the merged training data file."""
    logger.info(f"Loading merged data from: {path}")
    df = pd.read_csv(path, low_memory=False)
    logger.info(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure consistent column naming."""
    df = df.copy()
    
    # Map alternative column names to standard names
    column_mapping = {
        'date': 'game_date',
        'spread': 'fg_spread_line',
        'total': 'fg_total_line',
        'h2_spread': '1h_spread_line',
        'h2_total': '1h_total_line',
    }
    
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns and new_name not in df.columns:
            df[new_name] = df[old_name]
            logger.info(f"  Created {new_name} from {old_name}")
    
    # Ensure game_date is datetime
    if 'game_date' in df.columns:
        df['game_date'] = pd.to_datetime(df['game_date'], format='mixed', errors='coerce')
    
    return df


def compute_first_half_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 1H scores from quarter data if not already present."""
    df = df.copy()
    
    # region agent log
    run_id = "run1"
    cols = df.columns
    _agent_log(
        run_id=run_id,
        hypothesis_id="H1",
        location="scripts/compute_betting_labels.py:compute_first_half_scores:entry",
        message="Compute 1H scores: input column coverage",
        data={
            "rows": int(len(df)),
            "has_home_1h": bool("home_1h" in cols),
            "has_away_1h": bool("away_1h" in cols),
            "home_1h_non_null": int(df["home_1h"].notna().sum()) if "home_1h" in cols else None,
            "away_1h_non_null": int(df["away_1h"].notna().sum()) if "away_1h" in cols else None,
            "has_quarters": all(c in cols for c in ["home_q1", "home_q2", "away_q1", "away_q2"]),
            "home_q1_non_null": int(df["home_q1"].notna().sum()) if "home_q1" in cols else None,
            "home_q2_non_null": int(df["home_q2"].notna().sum()) if "home_q2" in cols else None,
            "away_q1_non_null": int(df["away_q1"].notna().sum()) if "away_q1" in cols else None,
            "away_q2_non_null": int(df["away_q2"].notna().sum()) if "away_q2" in cols else None,
        },
    )
    # endregion

    # Check if 1H columns exist
    if 'home_1h' not in df.columns or df['home_1h'].isna().sum() > len(df) * 0.5:
        # Try to compute from quarters
        if all(col in df.columns for col in ['home_q1', 'home_q2', 'away_q1', 'away_q2']):
            df['home_1h'] = df['home_q1'].fillna(0) + df['home_q2'].fillna(0)
            df['away_1h'] = df['away_q1'].fillna(0) + df['away_q2'].fillna(0)
            
            # Only keep where we have actual quarter data
            q_mask = df['home_q1'].notna() & df['home_q2'].notna()
            df.loc[~q_mask, 'home_1h'] = np.nan
            df.loc[~q_mask, 'away_1h'] = np.nan
            
            computed = q_mask.sum()
            logger.info(f"  Computed 1H scores for {computed} games from quarter data")

            # region agent log
            _agent_log(
                run_id=run_id,
                hypothesis_id="H1",
                location="scripts/compute_betting_labels.py:compute_first_half_scores:computed",
                message="Computed 1H from Q1+Q2",
                data={
                    "computed_rows": int(computed),
                    "home_1h_non_null_after": int(df["home_1h"].notna().sum()),
                    "away_1h_non_null_after": int(df["away_1h"].notna().sum()),
                },
            )
            # endregion
    
    return df


def compute_betting_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all betting outcome labels."""
    df = df.copy()
    
    # region agent log
    run_id = "run1"
    df_dates = pd.to_datetime(df.get("game_date"), format="mixed", errors="coerce")
    is_2526 = df_dates >= pd.to_datetime("2025-10-01")
    pre_counts = {}
    for c in ["1h_spread_covered", "1h_total_over", "home_1h", "away_1h", "home_q1", "home_q2", "away_q1", "away_q2"]:
        if c in df.columns:
            pre_counts[c] = {
                "overall_non_null": int(df[c].notna().sum()),
                "s2526_non_null": int(df.loc[is_2526, c].notna().sum()),
            }
    _agent_log(
        run_id=run_id,
        hypothesis_id="H2",
        location="scripts/compute_betting_labels.py:compute_betting_labels:pre_wipe",
        message="Pre-wipe 1H label/score coverage (overall vs 2025-26 subset)",
        data=pre_counts,
    )
    # endregion

    # Always recompute labels from source-of-truth score/line fields.
    # This prevents stale/placeholder label values from surviving merges.
    label_cols = [
        "fg_spread_covered",
        "fg_total_over",
        "fg_home_win",
        "1h_spread_covered",
        "1h_total_over",
        "1h_home_win",
    ]
    for col in label_cols:
        if col in df.columns:
            df[col] = np.nan

    # ==================
    # FULL GAME LABELS
    # ==================
    
    # Full game margin (home perspective)
    df["fg_margin"] = df["home_score"] - df["away_score"]
    logger.info(f"  Computed fg_margin for {df['fg_margin'].notna().sum()} games")
    
    # Full game total
    df["fg_total_actual"] = df["home_score"] + df["away_score"]
    logger.info(f"  Computed fg_total_actual for {df['fg_total_actual'].notna().sum()} games")
    
    # FG Spread Covered (home team covers spread)
    # If spread is -3.5, home must win by 4+ to cover
    # If spread is +5.5, home must not lose by 6+ to cover
    if 'fg_spread_line' in df.columns:
        spread_mask = df['fg_spread_line'].notna() & df['fg_margin'].notna()
        # Handle both conventions: spread relative to home team
        # Negative spread means home is favored, must win by more than abs(spread)
        df.loc[spread_mask, 'fg_spread_covered'] = (
            df.loc[spread_mask, 'fg_margin'] + df.loc[spread_mask, 'fg_spread_line'] > 0
        ).astype(int)
        
        # Handle pushes (set to NaN)
        push_mask = (df['fg_margin'] + df['fg_spread_line']).abs() < 0.01
        df.loc[spread_mask & push_mask, 'fg_spread_covered'] = np.nan
        
        computed = df['fg_spread_covered'].notna().sum()
        logger.info(f"  Computed fg_spread_covered for {computed} games")
    else:
        logger.warning("  Missing fg_spread_line - cannot compute fg_spread_covered")
    
    # FG Total Over
    if 'fg_total_line' in df.columns:
        total_mask = df['fg_total_line'].notna() & df['fg_total_actual'].notna()
        df.loc[total_mask, 'fg_total_over'] = (
            df.loc[total_mask, 'fg_total_actual'] > df.loc[total_mask, 'fg_total_line']
        ).astype(int)
        
        # Handle pushes
        push_mask = (df['fg_total_actual'] - df['fg_total_line']).abs() < 0.01
        df.loc[total_mask & push_mask, 'fg_total_over'] = np.nan
        
        computed = df['fg_total_over'].notna().sum()
        logger.info(f"  Computed fg_total_over for {computed} games")
    else:
        logger.warning("  Missing fg_total_line - cannot compute fg_total_over")
    
    # FG Home Win (moneyline label)
    df["fg_home_win"] = (df["fg_margin"] > 0).astype(int)
    df.loc[df["fg_margin"].isna(), "fg_home_win"] = np.nan
    computed = df["fg_home_win"].notna().sum()
    logger.info(f"  Computed fg_home_win for {computed} games")
    
    # ==================
    # FIRST HALF LABELS
    # ==================
    
    # 1H margin
    if "home_1h" in df.columns and "away_1h" in df.columns:
        df["1h_margin"] = df["home_1h"] - df["away_1h"]
        df["1h_total_actual"] = df["home_1h"] + df["away_1h"]

        # Only where we have 1H scores (real data)
        h1_mask = df["home_1h"].notna() & df["away_1h"].notna()
        computed_1h = int(h1_mask.sum())
        logger.info(f"  Computed 1h_margin/1h_total_actual for {computed_1h} games")
    else:
        df["1h_margin"] = np.nan
        df["1h_total_actual"] = np.nan
    
    # 1H Spread Covered
    if "1h_spread_line" in df.columns and "1h_margin" in df.columns:
        spread_1h_mask = df["1h_spread_line"].notna() & df["1h_margin"].notna()
        df.loc[spread_1h_mask, '1h_spread_covered'] = (
            df.loc[spread_1h_mask, '1h_margin'] + df.loc[spread_1h_mask, '1h_spread_line'] > 0
        ).astype(int)
        
        # Handle pushes
        push_mask = (df['1h_margin'] + df['1h_spread_line']).abs() < 0.01
        df.loc[spread_1h_mask & push_mask, '1h_spread_covered'] = np.nan
        
        computed = df['1h_spread_covered'].notna().sum()
        logger.info(f"  Computed 1h_spread_covered for {computed} games")
    else:
        logger.warning("  Missing 1h_spread_line or 1h_margin - cannot compute 1h_spread_covered")
    
    # 1H Total Over
    if "1h_total_line" in df.columns and "1h_total_actual" in df.columns:
        total_1h_mask = df["1h_total_line"].notna() & df["1h_total_actual"].notna()
        df.loc[total_1h_mask, '1h_total_over'] = (
            df.loc[total_1h_mask, '1h_total_actual'] > df.loc[total_1h_mask, '1h_total_line']
        ).astype(int)
        
        # Handle pushes
        push_mask = (df['1h_total_actual'] - df['1h_total_line']).abs() < 0.01
        df.loc[total_1h_mask & push_mask, '1h_total_over'] = np.nan
        
        computed = df['1h_total_over'].notna().sum()
        logger.info(f"  Computed 1h_total_over for {computed} games")
    else:
        logger.warning("  Missing 1h_total_line or 1h_total_actual - cannot compute 1h_total_over")
    
    # 1H Home Win
    if "1h_margin" in df.columns:
        df["1h_home_win"] = (df["1h_margin"] > 0).astype(int)
        df.loc[df["1h_margin"].isna(), "1h_home_win"] = np.nan
        computed = df["1h_home_win"].notna().sum()
        logger.info(f"  Computed 1h_home_win for {computed} games")
    
    # region agent log
    post_counts = {}
    for c in ["1h_spread_covered", "1h_total_over", "1h_home_win"]:
        if c in df.columns:
            post_counts[c] = {
                "overall_non_null": int(df[c].notna().sum()),
                "s2526_non_null": int(df.loc[is_2526, c].notna().sum()),
                "s2526_value_counts": df.loc[is_2526, c].value_counts(dropna=False).to_dict(),
            }
    _agent_log(
        run_id=run_id,
        hypothesis_id="H1",
        location="scripts/compute_betting_labels.py:compute_betting_labels:post_compute",
        message="Post-compute 1H labels coverage (overall vs 2025-26 subset)",
        data=post_counts,
    )
    # endregion

    return df


def validate_labels(df: pd.DataFrame) -> None:
    """Validate betting labels are computed correctly."""
    logger.info("Validating betting labels...")
    
    labels = [
        ('fg_spread_covered', 'fg_spread_line'),
        ('fg_total_over', 'fg_total_line'),
        ('fg_home_win', 'home_score'),
        ('1h_spread_covered', '1h_spread_line'),
        ('1h_total_over', '1h_total_line'),
        ('1h_home_win', 'home_1h'),
    ]
    
    for label, prereq in labels:
        if label in df.columns:
            valid = df[label].notna().sum()
            total = len(df)
            pct = valid / total * 100
            logger.info(f"  {label}: {valid}/{total} ({pct:.1f}%)")
        else:
            logger.warning(f"  {label}: MISSING")


def main():
    parser = argparse.ArgumentParser(description='Compute betting labels for merged training data')
    parser.add_argument(
        '--input',
        type=str,
        default='data/processed/training_data_all_seasons.csv',
        help='Input merged data file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file (defaults to overwriting input)'
    )
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path
    
    # Verify input exists
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1
    
    # Load data
    df = load_merged_data(input_path)
    
    # Normalize column names
    df = normalize_column_names(df)

    # region agent log
    run_id = "run1"
    dates = pd.to_datetime(df.get("game_date"), format="mixed", errors="coerce")
    has_dates = dates.notna()
    s2526 = dates >= pd.to_datetime("2025-10-01")
    alt_cols = [c for c in df.columns if c.lower() in {"home_1h_score", "away_1h_score", "home_first_half", "away_first_half"}]
    _agent_log(
        run_id=run_id,
        hypothesis_id="H1",
        location="scripts/compute_betting_labels.py:main:after_load",
        message="Dataset audit for 1H/quarter score availability",
        data={
            "rows": int(len(df)),
            "cols": int(len(df.columns)),
            "valid_game_date": int(has_dates.sum()),
            "date_min": str(dates[has_dates].min()) if has_dates.any() else None,
            "date_max": str(dates[has_dates].max()) if has_dates.any() else None,
            "rows_2526": int(s2526.sum()),
            "alt_halftime_cols_present": alt_cols,
            "home_1h_2526_non_null": int(df.loc[s2526, "home_1h"].notna().sum()) if "home_1h" in df.columns else None,
            "home_q1_2526_non_null": int(df.loc[s2526, "home_q1"].notna().sum()) if "home_q1" in df.columns else None,
        },
    )
    # endregion
    
    # Compute 1H scores from quarters if needed
    df = compute_first_half_scores(df)
    
    # Compute betting labels
    df = compute_betting_labels(df)
    
    # Validate
    validate_labels(df)
    
    # Write output
    logger.info(f"Writing updated data to: {output_path}")
    df.to_csv(output_path, index=False)
    logger.info(f"Successfully wrote {len(df)} games with computed labels")
    
    return 0


if __name__ == '__main__':
    exit(main())
