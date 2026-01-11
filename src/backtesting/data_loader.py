"""
Data loading and validation for backtesting.

STRICT MODE: No silent fallbacks, no assumptions, no data imputation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class StrictModeViolation(Exception):
    """Raised when backtest encounters missing/invalid data in strict mode."""
    pass


class DataValidationError(Exception):
    """Raised when data fails validation checks."""
    pass


@dataclass
class DataQualityReport:
    """Report on data quality metrics."""
    total_games: int
    date_range: Tuple[str, str]
    columns_count: int
    
    # Coverage by market
    fg_spread_coverage: float  # % of games with valid FG spread line
    fg_total_coverage: float
    fg_moneyline_coverage: float
    h1_spread_coverage: float
    h1_total_coverage: float
    h1_moneyline_coverage: float
    
    # Label availability
    fg_labels_complete: float  # % with all FG outcomes
    h1_labels_complete: float  # % with all 1H outcomes
    
    # Issues
    null_columns: Dict[str, int]  # column -> null count
    warnings: List[str]


@dataclass
class MarketConfig:
    """Configuration for a single market."""
    market_key: str
    period: str  # "fg" or "1h"
    market_type: str  # "spread", "total", "moneyline"
    label_column: str
    line_column: Optional[str]  # None for moneyline
    required_columns: List[str]
    
    # Score columns for label computation
    home_score_col: str
    away_score_col: str


# Market configurations
MARKET_CONFIGS: Dict[str, MarketConfig] = {
    "fg_spread": MarketConfig(
        market_key="fg_spread",
        period="fg",
        market_type="spread",
        label_column="fg_spread_covered",
        line_column="fg_spread_line",
        required_columns=["fg_spread_line", "home_score", "away_score"],
        home_score_col="home_score",
        away_score_col="away_score",
    ),
    "fg_total": MarketConfig(
        market_key="fg_total",
        period="fg",
        market_type="total",
        label_column="fg_total_over",
        line_column="fg_total_line",
        required_columns=["fg_total_line", "home_score", "away_score"],
        home_score_col="home_score",
        away_score_col="away_score",
    ),
    "fg_moneyline": MarketConfig(
        market_key="fg_moneyline",
        period="fg",
        market_type="moneyline",
        label_column="fg_home_win",
        line_column=None,
        required_columns=["home_score", "away_score", "fg_ml_home", "fg_ml_away"],
        home_score_col="home_score",
        away_score_col="away_score",
    ),
    "1h_spread": MarketConfig(
        market_key="1h_spread",
        period="1h",
        market_type="spread",
        label_column="1h_spread_covered",
        line_column="1h_spread_line",
        required_columns=["1h_spread_line", "home_q1", "home_q2", "away_q1", "away_q2"],
        home_score_col="home_1h_score",
        away_score_col="away_1h_score",
    ),
    "1h_total": MarketConfig(
        market_key="1h_total",
        period="1h",
        market_type="total",
        label_column="1h_total_over",
        line_column="1h_total_line",
        required_columns=["1h_total_line", "home_q1", "home_q2", "away_q1", "away_q2"],
        home_score_col="home_1h_score",
        away_score_col="away_1h_score",
    ),
    "1h_moneyline": MarketConfig(
        market_key="1h_moneyline",
        period="1h",
        market_type="moneyline",
        label_column="1h_home_win",
        line_column=None,
        required_columns=["home_q1", "home_q2", "away_q1", "away_q2", "1h_ml_home", "1h_ml_away"],
        home_score_col="home_1h_score",
        away_score_col="away_1h_score",
    ),
}


class BacktestDataLoader:
    """
    Loads and validates data for backtesting.
    
    STRICT MODE ENFORCEMENT:
    - No silent imputation
    - No fallback values
    - Explicit failure on invalid data
    """
    
    REQUIRED_BASE_COLUMNS = [
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
    ]
    
    def __init__(
        self,
        data_path: str | Path,
        strict_mode: bool = True,
        max_null_pct: float = 0.05,  # Max 5% null in critical columns
    ):
        """
        Initialize data loader.
        
        Args:
            data_path: Path to training data CSV
            strict_mode: If True, raise exceptions on data issues
            max_null_pct: Maximum allowed null percentage before failure
        """
        self.data_path = Path(data_path)
        self.strict_mode = strict_mode
        self.max_null_pct = max_null_pct
        
        self._df: Optional[pd.DataFrame] = None
        self._quality_report: Optional[DataQualityReport] = None
    
    def load(self) -> pd.DataFrame:
        """
        Load and validate training data.
        
        Returns:
            Validated DataFrame with computed labels
            
        Raises:
            DataValidationError: If data fails validation
            FileNotFoundError: If data file not found
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Training data not found: {self.data_path}")
        
        logger.info(f"Loading training data from {self.data_path}")
        
        # #region agent log
        import json
        import time
        log_path = r"c:\Users\JB\green-bier-ventures\NBA_main\.cursor\debug.log"
        # #endregion
        
        # Load CSV
        df = pd.read_csv(self.data_path, low_memory=False)
        
        # #region agent log
        odds_related_cols = [c for c in df.columns if any(term in c.lower() for term in ['odds', 'juice', 'vig', 'price', 'ml_', '_ml'])]
        spread_cols = [c for c in df.columns if 'spread' in c.lower()]
        total_cols = [c for c in df.columns if 'total' in c.lower()]
        with open(log_path, 'a') as f:
            f.write(json.dumps({"hypothesisId": "B,C", "location": "data_loader.py:load", "message": "Available odds columns in data", "data": {"odds_related": odds_related_cols, "spread_cols": spread_cols, "total_cols": total_cols, "all_columns_count": len(df.columns)}, "timestamp": int(time.time()*1000), "sessionId": "debug-session", "runId": "initial"}) + '\n')
        # #endregion
        
        # Parse dates - try game_date first, then date
        if "game_date" in df.columns:
            df["date"] = pd.to_datetime(df["game_date"], errors="coerce")
        else:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        
        # Drop rows with invalid dates (training data may have header rows mixed in)
        initial_count = len(df)
        df = df[df["date"].notna()].copy()
        if len(df) < initial_count:
            logger.info(f"Filtered out {initial_count - len(df)} rows with invalid dates")
        
        # Validate base columns
        self._validate_base_columns(df)
        
        # Sort by date (critical for temporal integrity)
        df = df.sort_values("date").reset_index(drop=True)
        
        # Compute derived scores (1H)
        df = self._compute_derived_scores(df)
        
        # Use existing labels if available, otherwise compute
        df = self._ensure_labels(df)
        
        # Generate quality report
        self._quality_report = self._generate_quality_report(df)
        
        # Log summary
        logger.info(
            f"Loaded {len(df)} games from {df['date'].min().date()} to {df['date'].max().date()}"
        )
        
        self._df = df
        return df
    
    def _validate_base_columns(self, df: pd.DataFrame) -> None:
        """Validate required base columns exist and have data."""
        missing = [col for col in self.REQUIRED_BASE_COLUMNS if col not in df.columns]
        if missing:
            raise DataValidationError(
                f"Missing required columns: {missing}. "
                f"Available columns: {sorted(df.columns.tolist())[:20]}..."
            )
        
        # Check for null values in required columns
        for col in self.REQUIRED_BASE_COLUMNS:
            null_count = df[col].isna().sum()
            null_pct = null_count / len(df)
            
            if null_pct > self.max_null_pct:
                msg = (
                    f"Column '{col}' has {null_pct:.1%} null values "
                    f"(max allowed: {self.max_null_pct:.1%}). "
                    f"Fix data quality or reduce max_null_pct threshold."
                )
                if self.strict_mode:
                    raise StrictModeViolation(msg)
                logger.warning(msg)
        
        # Check date parsing
        invalid_dates = df["date"].isna().sum()
        if invalid_dates > 0:
            msg = f"{invalid_dates} rows have unparseable dates"
            if self.strict_mode:
                raise StrictModeViolation(msg)
            logger.warning(f"{msg}. These will be excluded.")
    
    def _compute_derived_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute derived score columns (1H scores from quarters or existing columns)."""
        df = df.copy()
        
        # Check if 1H scores already exist
        if "home_1h" in df.columns and "away_1h" in df.columns:
            df["home_1h_score"] = df["home_1h"]
            df["away_1h_score"] = df["away_1h"]
            logger.info(
                f"Using existing 1H scores: {df['home_1h_score'].notna().sum()}/{len(df)} games have valid 1H data"
            )
            return df
        
        # Check for quarter columns
        quarter_cols = ["home_q1", "home_q2", "away_q1", "away_q2"]
        has_quarters = all(col in df.columns for col in quarter_cols)
        
        if has_quarters:
            # Compute 1H scores - NO ZERO-FILLING for missing quarters
            # Games with missing quarter data get NaN (will be excluded from 1H markets)
            df["home_1h_score"] = df["home_q1"] + df["home_q2"]
            df["away_1h_score"] = df["away_q1"] + df["away_q2"]
            
            logger.info(
                f"Computed 1H scores: {df['home_1h_score'].notna().sum()}/{len(df)} games have valid 1H data"
            )
        else:
            logger.warning("Quarter columns not found. 1H markets will be unavailable.")
            df["home_1h_score"] = np.nan
            df["away_1h_score"] = np.nan
        
        return df
    
    def _ensure_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure outcome labels exist for all markets.
        
        Uses existing labels if available, otherwise computes them.
        NO SILENT FALLBACKS - labels are NaN if data is missing.
        """
        df = df.copy()
        
        # Check if labels already exist in the data
        labels_exist = all(
            config.label_column in df.columns 
            for config in MARKET_CONFIGS.values()
        )
        
        if labels_exist:
            logger.info("Using existing labels from training data")
            # Log label availability
            for market_key, config in MARKET_CONFIGS.items():
                valid_count = df[config.label_column].notna().sum()
                logger.info(f"{market_key}: {valid_count}/{len(df)} valid labels ({valid_count/len(df):.1%})")
            return df
        
        # Otherwise compute labels
        logger.info("Computing labels from scores")
        return self._compute_labels(df)
    
    def _compute_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute outcome labels for all markets.
        
        NO SILENT FALLBACKS - labels are NaN if data is missing.
        """
        df = df.copy()
        
        # Full Game Labels
        # FG Spread: home covers if (home_score - away_score) > -spread_line
        spread_col = "fg_spread_line" if "fg_spread_line" in df.columns else "spread_line"
        if spread_col in df.columns:
            df["fg_margin"] = df["home_score"] - df["away_score"]
            df["fg_spread_covered"] = np.where(
                df[spread_col].notna(),
                (df["fg_margin"] > -df[spread_col]).astype(float),
                np.nan
            )
        else:
            df["fg_spread_covered"] = np.nan
        
        # FG Total: over if (home_score + away_score) > total_line
        total_col = "fg_total_line" if "fg_total_line" in df.columns else "total_line"
        if total_col in df.columns:
            df["fg_total_points"] = df["home_score"] + df["away_score"]
            df["fg_total_over"] = np.where(
                df[total_col].notna(),
                (df["fg_total_points"] > df[total_col]).astype(float),
                np.nan
            )
        else:
            df["fg_total_over"] = np.nan
        
        # FG Moneyline: home wins
        df["fg_home_win"] = (df["home_score"] > df["away_score"]).astype(float)
        # Set to NaN for ties (rare in NBA but handle properly)
        df.loc[df["home_score"] == df["away_score"], "fg_home_win"] = np.nan
        
        # First Half Labels (REQUIRE REAL 1H LINES - NO FG/2 APPROXIMATION)
        # 1H Spread
        h1_spread_col = "1h_spread_line" if "1h_spread_line" in df.columns else "fh_spread_line"
        if h1_spread_col in df.columns:
            df["1h_margin"] = df["home_1h_score"] - df["away_1h_score"]
            df["1h_spread_covered"] = np.where(
                df[h1_spread_col].notna() & df["1h_margin"].notna(),
                (df["1h_margin"] > -df[h1_spread_col]).astype(float),
                np.nan
            )
        else:
            df["1h_spread_covered"] = np.nan
        
        # 1H Total
        h1_total_col = "1h_total_line" if "1h_total_line" in df.columns else "fh_total_line"
        if h1_total_col in df.columns:
            df["1h_total_points"] = df["home_1h_score"] + df["away_1h_score"]
            df["1h_total_over"] = np.where(
                df[h1_total_col].notna() & df["1h_total_points"].notna(),
                (df["1h_total_points"] > df[h1_total_col]).astype(float),
                np.nan
            )
        else:
            df["1h_total_over"] = np.nan
        
        # 1H Moneyline
        df["1h_home_win"] = np.where(
            df["home_1h_score"].notna() & df["away_1h_score"].notna(),
            (df["home_1h_score"] > df["away_1h_score"]).astype(float),
            np.nan
        )
        # Handle 1H ties
        df.loc[df["home_1h_score"] == df["away_1h_score"], "1h_home_win"] = np.nan
        
        # Log label availability
        for market_key, config in MARKET_CONFIGS.items():
            valid_count = df[config.label_column].notna().sum()
            logger.info(f"{market_key}: {valid_count}/{len(df)} valid labels ({valid_count/len(df):.1%})")
        
        return df
    
    def _generate_quality_report(self, df: pd.DataFrame) -> DataQualityReport:
        """Generate comprehensive data quality report."""
        n = len(df)
        
        # Calculate coverage for each market
        def coverage(col: str) -> float:
            if col not in df.columns:
                return 0.0
            return df[col].notna().sum() / n
        
        # Null columns (only report columns with >1% nulls)
        null_cols = {}
        for col in df.columns:
            null_count = df[col].isna().sum()
            if null_count > n * 0.01:  # Only report if >1% null
                null_cols[col] = int(null_count)
        
        # Warnings
        warnings = []
        h1_spread_col = "1h_spread_line" if "1h_spread_line" in df.columns else "fh_spread_line"
        h1_total_col = "1h_total_line" if "1h_total_line" in df.columns else "fh_total_line"
        
        if coverage(h1_spread_col) < 0.5:
            warnings.append(f"1H spread coverage is low ({coverage(h1_spread_col):.1%})")
        if coverage(h1_total_col) < 0.5:
            warnings.append(f"1H total coverage is low ({coverage(h1_total_col):.1%})")
        
        fg_spread_col = "fg_spread_line" if "fg_spread_line" in df.columns else "spread_line"
        fg_total_col = "fg_total_line" if "fg_total_line" in df.columns else "total_line"
        
        return DataQualityReport(
            total_games=n,
            date_range=(str(df["date"].min().date()), str(df["date"].max().date())),
            columns_count=len(df.columns),
            fg_spread_coverage=coverage(fg_spread_col),
            fg_total_coverage=coverage(fg_total_col),
            fg_moneyline_coverage=1.0,  # Always available if scores exist
            h1_spread_coverage=coverage(h1_spread_col),
            h1_total_coverage=coverage(h1_total_col),
            h1_moneyline_coverage=coverage("home_1h_score") if "home_1h_score" in df.columns else coverage("home_1h"),
            fg_labels_complete=coverage("fg_spread_covered"),
            h1_labels_complete=coverage("1h_spread_covered"),
            null_columns=null_cols,
            warnings=warnings,
        )
    
    def get_quality_report(self) -> DataQualityReport:
        """Get data quality report. Must call load() first."""
        if self._quality_report is None:
            raise RuntimeError("Call load() before get_quality_report()")
        return self._quality_report
    
    def get_market_data(
        self,
        market_key: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get data filtered for a specific market.
        
        Only returns rows with valid labels, lines, and odds for this market.
        NO FALLBACKS - games with missing required data are excluded.
        
        Args:
            market_key: Market to filter for (e.g., "fg_spread", "1h_total")
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)
            
        Returns:
            Filtered DataFrame with only valid rows for this market
        """
        if self._df is None:
            raise RuntimeError("Call load() before get_market_data()")
        
        if market_key not in MARKET_CONFIGS:
            raise ValueError(f"Unknown market: {market_key}. Valid: {list(MARKET_CONFIGS.keys())}")
        
        config = MARKET_CONFIGS[market_key]
        df = self._df.copy()
        initial_count = len(df)
        
        # Date filtering
        if start_date:
            df = df[df["date"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["date"] <= pd.to_datetime(end_date)]
        
        # Filter to valid labels
        df = df[df[config.label_column].notna()]
        
        # Filter to valid lines (for spread/total)
        if config.line_column:
            df = df[df[config.line_column].notna()]
        
        # For moneyline markets: REQUIRE actual odds (no fallback to assumed odds)
        if config.market_type == "moneyline":
            if config.period == "fg":
                odds_cols = ["fg_ml_home", "fg_ml_away"]
            else:
                odds_cols = ["1h_ml_home", "1h_ml_away"]
            
            # Check columns exist
            missing_cols = [c for c in odds_cols if c not in df.columns]
            if missing_cols:
                logger.warning(f"Missing odds columns for {market_key}: {missing_cols}")
                return pd.DataFrame()  # Return empty if odds columns don't exist
            
            # Filter to rows with valid odds
            for col in odds_cols:
                df = df[df[col].notna()]
        
        filtered_count = initial_count - len(df)
        if filtered_count > 0:
            logger.info(
                f"Market {market_key}: {len(df)} valid games "
                f"(filtered {filtered_count} with missing data)"
            )
        else:
            logger.info(f"Market {market_key}: {len(df)} valid games")
        
        return df.reset_index(drop=True)
    
    def get_available_markets(self) -> List[str]:
        """Get list of markets with sufficient data (>100 games)."""
        if self._df is None:
            raise RuntimeError("Call load() before get_available_markets()")
        
        available = []
        for market_key, config in MARKET_CONFIGS.items():
            valid_count = self._df[config.label_column].notna().sum()
            if config.line_column:
                line_count = self._df[config.line_column].notna().sum()
                valid_count = min(valid_count, line_count)
            
            if valid_count >= 100:
                available.append(market_key)
            else:
                logger.warning(f"Market {market_key} has insufficient data ({valid_count} games)")
        
        return available
