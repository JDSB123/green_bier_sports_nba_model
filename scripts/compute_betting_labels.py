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

import pandas as pd
import numpy as np

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
    
    return df


def compute_betting_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all betting outcome labels."""
    df = df.copy()
    
    # ==================
    # FULL GAME LABELS
    # ==================
    
    # Full game margin (home perspective)
    if 'fg_margin' not in df.columns:
        df['fg_margin'] = df['home_score'] - df['away_score']
        logger.info(f"  Computed fg_margin for {df['fg_margin'].notna().sum()} games")
    
    # Full game total
    if 'fg_total_actual' not in df.columns:
        df['fg_total_actual'] = df['home_score'] + df['away_score']
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
    
    # FG Home Win (moneyline)
    if 'fg_home_win' not in df.columns:
        df['fg_home_win'] = (df['fg_margin'] > 0).astype(int)
        df.loc[df['fg_margin'].isna(), 'fg_home_win'] = np.nan
        computed = df['fg_home_win'].notna().sum()
        logger.info(f"  Computed fg_home_win for {computed} games")
    
    # ==================
    # FIRST HALF LABELS
    # ==================
    
    # 1H margin
    if 'home_1h' in df.columns and 'away_1h' in df.columns:
        df['1h_margin'] = df['home_1h'] - df['away_1h']
        df['1h_total_actual'] = df['home_1h'] + df['away_1h']
        
        # Only where we have 1H data
        h1_mask = df['home_1h'].notna() & df['away_1h'].notna()
        computed_1h = h1_mask.sum()
        logger.info(f"  Computed 1h_margin/1h_total_actual for {computed_1h} games")
    
    # 1H Spread Covered
    if '1h_spread_line' in df.columns and '1h_margin' in df.columns:
        spread_1h_mask = df['1h_spread_line'].notna() & df['1h_margin'].notna()
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
    if '1h_total_line' in df.columns and '1h_total_actual' in df.columns:
        total_1h_mask = df['1h_total_line'].notna() & df['1h_total_actual'].notna()
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
    if '1h_margin' in df.columns:
        df['1h_home_win'] = (df['1h_margin'] > 0).astype(int)
        df.loc[df['1h_margin'].isna(), '1h_home_win'] = np.nan
        computed = df['1h_home_win'].notna().sum()
        logger.info(f"  Computed 1h_home_win for {computed} games")
    
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
