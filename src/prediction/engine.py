"""
NBA v6.0 - Unified Prediction Engine

PRODUCTION: 9 INDEPENDENT Markets (Q1 + 1H + FG)

First Quarter:
- Q1 Spread
- Q1 Total
- Q1 Moneyline

First Half:
- 1H Spread
- 1H Total
- 1H Moneyline

Full Game:
- FG Spread
- FG Total
- FG Moneyline

ARCHITECTURE: Each period has INDEPENDENT models trained on period-specific
features. No cross-period dependencies.

STRICT MODE: No fallbacks, no silent failures. All models must exist.
"""
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import logging
import joblib

from src.prediction.spreads import SpreadPredictor
from src.prediction.totals import TotalPredictor
from src.prediction.moneyline import MoneylinePredictor
from src.modeling.period_features import MODEL_CONFIGS
from src.config import filter_thresholds

logger = logging.getLogger(__name__)


class ModelNotFoundError(Exception):
    """Raised when a required model file is missing."""
    pass


class PeriodPredictor:
    """
    Predictor for a single period (Q1, 1H, or FG).

    Each period predictor has its own independent models for
    spread, total, and moneyline.
    """

    def __init__(
        self,
        period: str,
        spread_model: Any,
        spread_features: List[str],
        total_model: Any,
        total_features: List[str],
        moneyline_model: Any,
        moneyline_features: List[str],
    ):
        self.period = period
        self.spread_model = spread_model
        self.spread_features = spread_features
        self.total_model = total_model
        self.total_features = total_features
        self.moneyline_model = moneyline_model
        self.moneyline_features = moneyline_features

    def predict_spread(
        self,
        features: Dict[str, float],
        spread_line: float,
    ) -> Dict[str, Any]:
        """Predict spread outcome for this period."""
        import pandas as pd
        from src.prediction.confidence import calculate_confidence_from_probabilities

        # Prepare features
        feature_df = pd.DataFrame([features])

        # STRICT MODE: Fail on missing features - NO SILENT ZERO-FILL
        missing = set(self.spread_features) - set(feature_df.columns)
        if missing:
            raise ValueError(
                f"[{self.period}_spread] MISSING {len(missing)} REQUIRED FEATURES: {sorted(missing)[:10]}. "
                f"Feature pipeline is broken - fix data ingestion, do not zero-fill."
            )

        X = feature_df[self.spread_features]

        # Get prediction
        spread_proba = self.spread_model.predict_proba(X)[0]
        home_cover_prob = float(spread_proba[1])
        away_cover_prob = float(spread_proba[0])
        confidence = calculate_confidence_from_probabilities(home_cover_prob, away_cover_prob)
        bet_side = "home" if home_cover_prob > 0.5 else "away"

        # Get predicted margin from features
        margin_key = f"predicted_margin_{self.period}" if self.period != "fg" else "predicted_margin"
        predicted_margin = features.get(margin_key, 0)
        edge = predicted_margin - spread_line

        # Filter logic using configurable thresholds
        min_conf = filter_thresholds.spread_min_confidence
        min_edge = filter_thresholds.spread_min_edge
        passes_filter = confidence >= min_conf and abs(edge) >= min_edge
        filter_reason = None
        if not passes_filter:
            if confidence < min_conf:
                filter_reason = f"Low confidence: {confidence:.1%}"
            else:
                filter_reason = f"Low edge: {edge:+.1f}"

        return {
            "home_cover_prob": home_cover_prob,
            "away_cover_prob": away_cover_prob,
            "predicted_margin": predicted_margin,
            "confidence": confidence,
            "bet_side": bet_side,
            "edge": edge,
            "model_edge_pct": abs(confidence - 0.5),
            "passes_filter": passes_filter,
            "filter_reason": filter_reason,
        }

    def predict_total(
        self,
        features: Dict[str, float],
        total_line: float,
    ) -> Dict[str, Any]:
        """Predict total outcome for this period."""
        import pandas as pd
        from src.prediction.confidence import calculate_confidence_from_probabilities

        # Prepare features
        feature_df = pd.DataFrame([features])

        # STRICT MODE: Fail on missing features - NO SILENT ZERO-FILL
        missing = set(self.total_features) - set(feature_df.columns)
        if missing:
            raise ValueError(
                f"[{self.period}_total] MISSING {len(missing)} REQUIRED FEATURES: {sorted(missing)[:10]}. "
                f"Feature pipeline is broken - fix data ingestion, do not zero-fill."
            )

        X = feature_df[self.total_features]

        # Get prediction
        total_proba = self.total_model.predict_proba(X)[0]
        over_prob = float(total_proba[1])
        under_prob = float(total_proba[0])
        confidence = calculate_confidence_from_probabilities(over_prob, under_prob)
        bet_side = "over" if over_prob > 0.5 else "under"

        # Get predicted total from features
        total_key = f"predicted_total_{self.period}" if self.period != "fg" else "predicted_total"
        predicted_total = features.get(total_key, total_line)
        edge = predicted_total - total_line if bet_side == "over" else total_line - predicted_total

        # Filter logic using configurable thresholds
        min_conf = filter_thresholds.total_min_confidence
        min_edge = filter_thresholds.total_min_edge
        passes_filter = confidence >= min_conf and abs(edge) >= min_edge
        filter_reason = None
        if not passes_filter:
            if confidence < min_conf:
                filter_reason = f"Low confidence: {confidence:.1%}"
            else:
                filter_reason = f"Low edge: {edge:+.1f}"

        return {
            "over_prob": over_prob,
            "under_prob": under_prob,
            "predicted_total": predicted_total,
            "confidence": confidence,
            "bet_side": bet_side,
            "edge": edge,
            "model_edge_pct": abs(confidence - 0.5),
            "passes_filter": passes_filter,
            "filter_reason": filter_reason,
        }

    def predict_moneyline(
        self,
        features: Dict[str, float],
        home_ml_odds: int,
        away_ml_odds: int,
    ) -> Dict[str, Any]:
        """Predict moneyline outcome for this period."""
        import pandas as pd
        from src.prediction.confidence import calculate_confidence_from_probabilities

        # Prepare features
        feature_df = pd.DataFrame([features])

        # STRICT MODE: Fail on missing features - NO SILENT ZERO-FILL
        missing = set(self.moneyline_features) - set(feature_df.columns)
        if missing:
            raise ValueError(
                f"[{self.period}_moneyline] MISSING {len(missing)} REQUIRED FEATURES: {sorted(missing)[:10]}. "
                f"Feature pipeline is broken - fix data ingestion, do not zero-fill."
            )

        X = feature_df[self.moneyline_features]

        # Get prediction
        ml_proba = self.moneyline_model.predict_proba(X)[0]
        home_win_prob = float(ml_proba[1])
        away_win_prob = float(ml_proba[0])
        confidence = calculate_confidence_from_probabilities(home_win_prob, away_win_prob)

        # Calculate implied probabilities from odds
        def american_to_implied(odds: int) -> float:
            if odds > 0:
                return 100 / (odds + 100)
            else:
                return abs(odds) / (abs(odds) + 100)

        home_implied = american_to_implied(home_ml_odds)
        away_implied = american_to_implied(away_ml_odds)

        # Calculate edge
        home_edge = home_win_prob - home_implied
        away_edge = away_win_prob - away_implied

        # Determine recommended bet using configurable thresholds
        min_edge_pct = filter_thresholds.moneyline_min_edge_pct
        min_conf = filter_thresholds.moneyline_min_confidence
        if home_edge > away_edge and home_edge > min_edge_pct:
            recommended_bet = "home"
            edge = home_edge
        elif away_edge > min_edge_pct:
            recommended_bet = "away"
            edge = away_edge
        else:
            recommended_bet = None
            edge = max(home_edge, away_edge)

        passes_filter = recommended_bet is not None and confidence >= min_conf
        filter_reason = None
        if not passes_filter:
            if recommended_bet is None:
                filter_reason = f"No edge: home={home_edge:+.1%}, away={away_edge:+.1%}"
            else:
                filter_reason = f"Low confidence: {confidence:.1%}"

        return {
            "home_win_prob": home_win_prob,
            "away_win_prob": away_win_prob,
            "home_implied_prob": home_implied,
            "away_implied_prob": away_implied,
            "home_edge": home_edge,
            "away_edge": away_edge,
            "confidence": confidence,
            "recommended_bet": recommended_bet,
            "passes_filter": passes_filter,
            "filter_reason": filter_reason,
        }

    def predict_all(
        self,
        features: Dict[str, float],
        spread_line: Optional[float] = None,
        total_line: Optional[float] = None,
        home_ml_odds: Optional[int] = None,
        away_ml_odds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Predict all markets for this period."""
        result = {}

        if spread_line is not None:
            result["spread"] = self.predict_spread(features, spread_line)

        if total_line is not None:
            result["total"] = self.predict_total(features, total_line)

        if home_ml_odds is not None and away_ml_odds is not None:
            result["moneyline"] = self.predict_moneyline(features, home_ml_odds, away_ml_odds)

        return result


class UnifiedPredictionEngine:
    """
    NBA v6.0 - Production Prediction Engine

    9 INDEPENDENT Markets:

    First Quarter (Q1):
    - Q1 Spread
    - Q1 Total
    - Q1 Moneyline

    First Half (1H):
    - 1H Spread
    - 1H Total
    - 1H Moneyline

    Full Game (FG):
    - FG Spread
    - FG Total
    - FG Moneyline

    ARCHITECTURE:
    - Each period has independent models trained on period-specific features
    - No cross-period dependencies
    - All 9 models required for full functionality
    """

    def __init__(self, models_dir: Path, require_all: bool = True):
        """
        Initialize unified prediction engine.

        STRICT MODE (v6.0): All 9 models REQUIRED. No fallbacks.

        Args:
            models_dir: Path to models directory
            require_all: DEPRECATED - always True. All 9 models required.
        """
        if not require_all:
            raise ValueError(
                "STRICT MODE ENFORCED: require_all=False is no longer supported. "
                "All 9 models must be present. No silent fallbacks."
            )
        self.models_dir = Path(models_dir)
        self.loaded_models: Dict[str, bool] = {}

        if not self.models_dir.exists():
            raise ModelNotFoundError(
                f"Models directory does not exist: {self.models_dir}\n"
                f"Run: python scripts/train_all_models.py"
            )

        # STRICT MODE: All 9 models MUST be loaded - NO FALLBACKS
        # Initialize period predictors
        self.q1_predictor: Optional[PeriodPredictor] = None
        self.h1_predictor: Optional[PeriodPredictor] = None
        self.fg_predictor: Optional[PeriodPredictor] = None

        # Load Q1 models - WILL RAISE if any missing
        logger.info("Loading Q1 models (spread, total, moneyline)...")
        q1_models = self._load_period_models("q1")
        self.q1_predictor = PeriodPredictor("q1", *q1_models)
        logger.info("Q1 predictor initialized (3/3 models)")

        # Load 1H models - WILL RAISE if any missing
        logger.info("Loading 1H models (spread, total, moneyline)...")
        h1_models = self._load_period_models("1h")
        self.h1_predictor = PeriodPredictor("1h", *h1_models)
        logger.info("1H predictor initialized (3/3 models)")

        # Load FG models - WILL RAISE if any missing
        logger.info("Loading FG models (spread, total, moneyline)...")
        fg_models = self._load_period_models("fg")
        self.fg_predictor = PeriodPredictor("fg", *fg_models)
        logger.info("FG predictor initialized (3/3 models)")

        # Legacy predictors for backwards compatibility
        self._init_legacy_predictors()

        # Verify all 9 models loaded
        loaded_count = sum(1 for v in self.loaded_models.values() if v)
        if loaded_count < 9:
            missing = [k for k, v in self.loaded_models.items() if not v]
            raise ModelNotFoundError(
                f"FATAL: Only {loaded_count}/9 models loaded. Missing: {missing}\n"
                f"Run: python scripts/train_all_models.py"
            )
        logger.info(f"SUCCESS: All 9/9 models loaded - STRICT MODE ACTIVE")

    def _load_period_models(
        self,
        period: str,
    ) -> Optional[Tuple[Any, List[str], Any, List[str], Any, List[str]]]:
        """Load all models for a period."""
        spread_key = f"{period}_spread"
        total_key = f"{period}_total"
        ml_key = f"{period}_moneyline"

        try:
            spread_model, spread_features = self._load_model(spread_key)
            self.loaded_models[spread_key] = True
        except Exception as e:
            logger.warning(f"Could not load {spread_key}: {e}")
            self.loaded_models[spread_key] = False
            spread_model, spread_features = None, []

        try:
            total_model, total_features = self._load_model(total_key)
            self.loaded_models[total_key] = True
        except Exception as e:
            logger.warning(f"Could not load {total_key}: {e}")
            self.loaded_models[total_key] = False
            total_model, total_features = None, []

        try:
            ml_model, ml_features = self._load_model(ml_key)
            self.loaded_models[ml_key] = True
        except Exception as e:
            logger.warning(f"Could not load {ml_key}: {e}")
            self.loaded_models[ml_key] = False
            ml_model, ml_features = None, []

        # STRICT MODE: ALL 3 models required per period - NO FALLBACKS
        if spread_model is None:
            raise ModelNotFoundError(
                f"[{period}] MISSING spread model. Train with: python scripts/train_all_models.py --markets {period}_spread"
            )
        if total_model is None:
            raise ModelNotFoundError(
                f"[{period}] MISSING total model. Train with: python scripts/train_all_models.py --markets {period}_total"
            )
        if ml_model is None:
            raise ModelNotFoundError(
                f"[{period}] MISSING moneyline model. Train with: python scripts/train_all_models.py --markets {period}_moneyline"
            )

        return (
            spread_model, spread_features,
            total_model, total_features,
            ml_model, ml_features,
        )

    def _load_model(self, model_key: str) -> Tuple[Any, List[str]]:
        """Load a single model.

        Supports both .joblib (combined model+features) and .pkl (separate files) formats.
        """
        import pickle

        config = MODEL_CONFIGS.get(model_key)
        if not config:
            raise ValueError(f"Unknown model key: {model_key}")

        model_path = self.models_dir / config["model_file"]

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Handle .pkl files (1H models use separate model and features files)
        if model_path.suffix == ".pkl":
            with open(model_path, "rb") as f:
                model = pickle.load(f)

            # Load features from separate file if specified
            features = []
            if "features_file" in config:
                features_path = self.models_dir / config["features_file"]
                if features_path.exists():
                    with open(features_path, "rb") as f:
                        features = pickle.load(f)
                else:
                    # Fall back to config features
                    features = config.get("features", [])
            else:
                features = config.get("features", [])

            logger.info(f"Loaded {model_key} (pkl) with {len(features)} features")
            return model, features

        # Handle .joblib files (combined format)
        data = joblib.load(model_path)

        model = data.get("pipeline") or data.get("model")
        features = data.get("feature_columns", [])

        if model is None:
            raise ValueError(f"Invalid model file: {model_path}")

        logger.info(f"Loaded {model_key} with {len(features)} features")
        return model, features

    def _init_legacy_predictors(self):
        """Initialize legacy predictors for backwards compatibility."""
        # These are used by old code paths
        if self.fg_predictor:
            # Create legacy SpreadPredictor
            self.spread_predictor = SpreadPredictor(
                fg_model=self.fg_predictor.spread_model,
                fg_feature_columns=self.fg_predictor.spread_features,
                fh_model=self.h1_predictor.spread_model if self.h1_predictor else self.fg_predictor.spread_model,
                fh_feature_columns=self.h1_predictor.spread_features if self.h1_predictor else self.fg_predictor.spread_features,
            )

            # Create legacy TotalPredictor
            self.total_predictor = TotalPredictor(
                fg_model=self.fg_predictor.total_model,
                fg_feature_columns=self.fg_predictor.total_features,
                fh_model=self.h1_predictor.total_model if self.h1_predictor else self.fg_predictor.total_model,
                fh_feature_columns=self.h1_predictor.total_features if self.h1_predictor else self.fg_predictor.total_features,
            )

            # Create legacy MoneylinePredictor
            self.moneyline_predictor = MoneylinePredictor(
                model=self.fg_predictor.moneyline_model,
                feature_columns=self.fg_predictor.moneyline_features,
                fh_model=self.h1_predictor.spread_model if self.h1_predictor else self.fg_predictor.spread_model,
                fh_feature_columns=self.h1_predictor.spread_features if self.h1_predictor else self.fg_predictor.spread_features,
            )

    def predict_quarter(
        self,
        features: Dict[str, float],
        spread_line: Optional[float] = None,
        total_line: Optional[float] = None,
        home_ml_odds: Optional[int] = None,
        away_ml_odds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate predictions for Q1 markets.

        Args:
            features: Feature dictionary with Q1-specific features
            spread_line: Q1 spread line
            total_line: Q1 total line
            home_ml_odds: Q1 home moneyline odds
            away_ml_odds: Q1 away moneyline odds

        Returns:
            Predictions for Q1 Spread, Total, and Moneyline
        """
        if self.q1_predictor is None:
            raise ModelNotFoundError("Q1 models not loaded")

        return self.q1_predictor.predict_all(
            features, spread_line, total_line, home_ml_odds, away_ml_odds
        )

    def predict_first_half(
        self,
        features: Dict[str, float],
        spread_line: Optional[float] = None,
        total_line: Optional[float] = None,
        home_ml_odds: Optional[int] = None,
        away_ml_odds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate predictions for 1H markets.

        Args:
            features: Feature dictionary with 1H-specific features
            spread_line: 1H spread line
            total_line: 1H total line
            home_ml_odds: 1H home moneyline odds
            away_ml_odds: 1H away moneyline odds

        Returns:
            Predictions for 1H Spread, Total, and Moneyline
        """
        if self.h1_predictor is None:
            raise ModelNotFoundError("1H models not loaded")

        return self.h1_predictor.predict_all(
            features, spread_line, total_line, home_ml_odds, away_ml_odds
        )

    def predict_full_game(
        self,
        features: Dict[str, float],
        spread_line: Optional[float] = None,
        total_line: Optional[float] = None,
        home_ml_odds: Optional[int] = None,
        away_ml_odds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate predictions for FG markets.

        Args:
            features: Feature dictionary with FG-specific features
            spread_line: FG spread line
            total_line: FG total line
            home_ml_odds: FG home moneyline odds
            away_ml_odds: FG away moneyline odds

        Returns:
            Predictions for FG Spread, Total, and Moneyline
        """
        if self.fg_predictor is None:
            raise ModelNotFoundError("FG models not loaded")

        return self.fg_predictor.predict_all(
            features, spread_line, total_line, home_ml_odds, away_ml_odds
        )

    def predict_all_markets(
        self,
        features: Dict[str, float],
        # Full game lines
        fg_spread_line: Optional[float] = None,
        fg_total_line: Optional[float] = None,
        # First half lines
        fh_spread_line: Optional[float] = None,
        fh_total_line: Optional[float] = None,
        # First quarter lines
        q1_spread_line: Optional[float] = None,
        q1_total_line: Optional[float] = None,
        # Moneyline odds
        home_ml_odds: Optional[int] = None,
        away_ml_odds: Optional[int] = None,
        fh_home_ml_odds: Optional[int] = None,
        fh_away_ml_odds: Optional[int] = None,
        q1_home_ml_odds: Optional[int] = None,
        q1_away_ml_odds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate predictions for ALL 9 markets.

        Args:
            features: Feature dictionary with all period features
            fg_spread_line: FG spread line
            fg_total_line: FG total line
            fh_spread_line: 1H spread line
            fh_total_line: 1H total line
            q1_spread_line: Q1 spread line
            q1_total_line: Q1 total line
            home_ml_odds: FG home moneyline odds
            away_ml_odds: FG away moneyline odds
            fh_home_ml_odds: 1H home moneyline odds
            fh_away_ml_odds: 1H away moneyline odds
            q1_home_ml_odds: Q1 home moneyline odds
            q1_away_ml_odds: Q1 away moneyline odds

        Returns:
            Predictions for all 9 markets grouped by period
        """
        result = {
            "first_quarter": {},
            "first_half": {},
            "full_game": {},
        }

        # Q1 predictions
        if self.q1_predictor and (q1_spread_line is not None or q1_total_line is not None):
            try:
                result["first_quarter"] = self.predict_quarter(
                    features,
                    spread_line=q1_spread_line,
                    total_line=q1_total_line,
                    home_ml_odds=q1_home_ml_odds,
                    away_ml_odds=q1_away_ml_odds,
                )
            except Exception as e:
                logger.warning(f"Q1 prediction failed: {e}")

        # 1H predictions
        if self.h1_predictor and (fh_spread_line is not None or fh_total_line is not None):
            try:
                result["first_half"] = self.predict_first_half(
                    features,
                    spread_line=fh_spread_line,
                    total_line=fh_total_line,
                    home_ml_odds=fh_home_ml_odds,
                    away_ml_odds=fh_away_ml_odds,
                )
            except Exception as e:
                logger.warning(f"1H prediction failed: {e}")

        # FG predictions
        if self.fg_predictor and (fg_spread_line is not None or fg_total_line is not None):
            try:
                result["full_game"] = self.predict_full_game(
                    features,
                    spread_line=fg_spread_line,
                    total_line=fg_total_line,
                    home_ml_odds=home_ml_odds,
                    away_ml_odds=away_ml_odds,
                )
            except Exception as e:
                logger.warning(f"FG prediction failed: {e}")

        return result

    def get_model_info(self) -> Dict[str, Any]:
        """Return info about loaded models."""
        return {
            "version": "6.0",
            "architecture": "9-model independent",
            "markets": sum(1 for v in self.loaded_models.values() if v),
            "markets_list": [k for k, v in self.loaded_models.items() if v],
            "periods": ["first_quarter", "first_half", "full_game"],
            "models_dir": str(self.models_dir),
            "loaded_models": self.loaded_models,
            "predictors": {
                "q1": self.q1_predictor is not None,
                "1h": self.h1_predictor is not None,
                "fg": self.fg_predictor is not None,
            },
        }
