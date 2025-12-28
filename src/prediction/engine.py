"""
NBA v33.0.8.0 - Unified Prediction Engine

PRODUCTION: 6 INDEPENDENT Markets (1H + FG)

First Half:
- 1H Spread
- 1H Total
- 1H Moneyline

Full Game:
- FG Spread
- FG Total
- FG Moneyline

ARCHITECTURE: Each period has INDEPENDENT models trained on period-specific
features. No cross-period dependencies. Uses matchup-based formulas for
predicted totals (not scaled from FG).

v33.0.8.0 FIXES:
    - bet_side now based on EDGE calculation, not classifier
    - Added classifier sanity check to detect data drift (extreme probabilities)
    - Simple filter: confidence + edge thresholds (NO dual-signal requirement)

FEATURE VALIDATION:
    Controlled by PREDICTION_FEATURE_MODE environment variable:
    - "strict" (default): Raise error on missing features
    - "warn": Log warning, zero-fill missing features
    - "silent": Zero-fill without logging

    See src/prediction/feature_validation.py for details.
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
# Single source of truth for version - read from environment variable
MODEL_VERSION = os.getenv("NBA_MODEL_VERSION", "NBA_v33.0.8.0")

import logging
import joblib

from src.prediction.spreads import SpreadPredictor
from src.prediction.totals import TotalPredictor
from src.prediction.moneyline import MoneylinePredictor
from src.prediction.feature_validation import (
    validate_and_prepare_features,
    MissingFeaturesError,
    get_feature_mode,
)
from src.modeling.period_features import MODEL_CONFIGS
from src.config import filter_thresholds

# Import monitoring components (lazy loaded to avoid circular imports)
_signal_tracker = None
_feature_tracker = None

def _get_signal_tracker():
    """Lazy load signal tracker to avoid circular imports."""
    global _signal_tracker
    if _signal_tracker is None:
        try:
            from src.monitoring.signal_tracker import get_signal_tracker
            _signal_tracker = get_signal_tracker()
        except ImportError:
            pass
    return _signal_tracker

def _get_feature_tracker():
    """Lazy load feature tracker to avoid circular imports."""
    global _feature_tracker
    if _feature_tracker is None:
        try:
            from src.monitoring.feature_completeness import get_feature_tracker
            _feature_tracker = get_feature_tracker()
        except ImportError:
            pass
    return _feature_tracker

logger = logging.getLogger(__name__)


class ModelNotFoundError(Exception):
    """Raised when a required model file is missing."""
    pass


class PeriodPredictor:
    """
    Predictor for a single period (1H or FG).

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
        if self.spread_model is None:
            raise ModelNotFoundError(f"Spread model for {self.period} not loaded")

        import pandas as pd
        from src.prediction.confidence import calculate_confidence_from_probabilities

        # Prepare features using unified validation
        feature_df = pd.DataFrame([features])
        X, missing = validate_and_prepare_features(
            feature_df,
            self.spread_features,
            market=f"{self.period}_spread",
        )

        # =====================================================================
        # SIGNAL 1: CLASSIFIER (ML model trained on historical patterns)
        # =====================================================================
        spread_proba = self.spread_model.predict_proba(X)[0]
        home_cover_prob = float(spread_proba[1])
        away_cover_prob = float(spread_proba[0])
        confidence = calculate_confidence_from_probabilities(home_cover_prob, away_cover_prob)
        classifier_side = "home" if home_cover_prob > 0.5 else "away"

        # =====================================================================
        # SIGNAL 2: POINT PREDICTION (model's predicted margin vs market line)
        # =====================================================================
        margin_key = f"predicted_margin_{self.period}" if self.period != "fg" else "predicted_margin"
        predicted_margin = features.get(margin_key)

        # CRITICAL: Log and warn when predicted margin is missing
        if predicted_margin is None:
            logger.warning(
                f"[{self.period}_spread] MISSING predicted_margin feature (key: {margin_key}). "
                f"Defaulting to 0. This may affect edge calculation accuracy. "
                f"Available features with 'margin': {[k for k in features.keys() if 'margin' in k.lower()]}"
            )
            predicted_margin = 0
        elif predicted_margin == 0:
            # Log when margin is exactly 0 (might be intentional or might indicate issue)
            logger.debug(f"[{self.period}_spread] predicted_margin is exactly 0 (may be intentional)")

        # EDGE CALCULATION:
        # spread_line is the HOME spread from sportsbooks (negative = home favored)
        # predicted_margin is positive when home wins by X points
        #
        # Example: home -3.5 (spread_line = -3.5), model predicts home +5
        # - Market implies home wins by ~3.5 points
        # - Model says home wins by 5 points
        # - Home beats the spread by: 5 - 3.5 = 1.5 points
        # Formula: edge = predicted_margin + spread_line
        edge = predicted_margin + spread_line
        prediction_side = "home" if edge > 0 else "away"

        # v33.0.8.0 FIX: Classifier sanity check - detect broken/drifted models
        # If classifier outputs extreme probability (>99% or <1%), it's unreliable
        classifier_extreme = (home_cover_prob > 0.99 or home_cover_prob < 0.01)
        if classifier_extreme:
            logger.warning(
                f"[{self.period}_spread] EXTREME classifier probability: {home_cover_prob:.4f}. "
                f"Model may have data drift."
            )

        # v33.0.8.0 FIX: Use EDGE-BASED prediction as authoritative bet_side
        # The edge calculation is more robust than a potentially drifted classifier
        bet_side = prediction_side

        # For filtering and display, use absolute edge value
        edge_abs = abs(edge)

        # Filter logic: confidence AND edge threshold (NO dual-signal requirement)
        min_conf = filter_thresholds.spread_min_confidence
        min_edge = filter_thresholds.spread_min_edge

        # Simple filter: just need confidence and edge to pass
        passes_filter = confidence >= min_conf and edge_abs >= min_edge

        filter_reason = None
        if not passes_filter:
            if confidence < min_conf:
                filter_reason = f"Low confidence: {confidence:.1%}"
            else:
                filter_reason = f"Low edge: {edge_abs:.1f} pts (min: {min_edge})"

        return {
            "home_cover_prob": home_cover_prob,
            "away_cover_prob": away_cover_prob,
            "predicted_margin": predicted_margin,
            "spread_line": spread_line,
            "confidence": confidence,
            "side": bet_side,  # Alias for downstream code expecting generic side
            "bet_side": bet_side,
            "edge": edge_abs,  # Always positive for the recommended side
            "raw_edge": edge,  # Signed edge for diagnostics
            "classifier_side": classifier_side,  # What the ML classifier says
            "prediction_side": prediction_side,  # What the point prediction says
            "classifier_extreme": classifier_extreme,  # True if classifier is unreliable
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
        if self.total_model is None:
            raise ModelNotFoundError(f"Total model for {self.period} not loaded")

        import pandas as pd
        from src.prediction.confidence import calculate_confidence_from_probabilities

        # Prepare features using unified validation
        feature_df = pd.DataFrame([features])
        X, missing = validate_and_prepare_features(
            feature_df,
            self.total_features,
            market=f"{self.period}_total",
        )

        # =====================================================================
        # SIGNAL 1: CLASSIFIER (ML model trained on historical patterns)
        # =====================================================================
        total_proba = self.total_model.predict_proba(X)[0]
        over_prob = float(total_proba[1])
        under_prob = float(total_proba[0])
        confidence = calculate_confidence_from_probabilities(over_prob, under_prob)
        classifier_side = "over" if over_prob > 0.5 else "under"

        # =====================================================================
        # SIGNAL 2: POINT PREDICTION (model's predicted total vs market line)
        # =====================================================================
        total_key = f"predicted_total_{self.period}" if self.period != "fg" else "predicted_total"
        predicted_total = features.get(total_key)

        # CRITICAL: Log and warn when predicted total is missing
        if predicted_total is None:
            logger.warning(
                f"[{self.period}_total] MISSING predicted_total feature (key: {total_key}). "
                f"Defaulting to market line ({total_line}). This may affect edge calculation. "
                f"Available features with 'total': {[k for k in features.keys() if 'total' in k.lower()]}"
            )
            predicted_total = total_line
        elif predicted_total == total_line:
            # Log when predicted exactly matches line (edge will be 0)
            logger.debug(f"[{self.period}_total] predicted_total equals market line ({total_line})")

        # EDGE CALCULATION:
        # Compare model's predicted total to market line
        # Positive = model predicts OVER the line
        # Negative = model predicts UNDER the line
        #
        # Example: predicted_total = 230, total_line = 225
        # - Model says 230 points, market says 225
        # - Edge = 230 - 225 = +5 (OVER by 5 points)
        edge = predicted_total - total_line
        prediction_side = "over" if edge > 0 else "under"

        # v33.0.8.0 FIX: Classifier sanity check - detect broken/drifted models
        # If classifier outputs extreme probability (>99% or <1%), it's unreliable
        # This catches the FG Total model data drift issue (always outputs 100% over)
        classifier_extreme = (over_prob > 0.99 or over_prob < 0.01)
        if classifier_extreme:
            logger.warning(
                f"[{self.period}_total] EXTREME classifier probability: over={over_prob:.4f}. "
                f"Model may have data drift."
            )

        # v33.0.8.0 FIX: Use EDGE-BASED prediction as authoritative bet_side
        # The edge calculation is more robust than a potentially drifted classifier
        bet_side = prediction_side

        # For filtering and display, use absolute edge value
        edge_abs = abs(edge)

        # Filter logic: confidence AND edge threshold (NO dual-signal requirement)
        min_conf = filter_thresholds.total_min_confidence
        min_edge = filter_thresholds.total_min_edge

        # Simple filter: just need confidence and edge to pass
        passes_filter = confidence >= min_conf and edge_abs >= min_edge

        filter_reason = None
        if not passes_filter:
            if confidence < min_conf:
                filter_reason = f"Low confidence: {confidence:.1%}"
            else:
                filter_reason = f"Low edge: {edge_abs:.1f} pts (min: {min_edge})"

        return {
            "over_prob": over_prob,
            "under_prob": under_prob,
            "predicted_total": predicted_total,
            "total_line": total_line,
            "confidence": confidence,
            "side": bet_side,  # Alias for downstream code expecting generic side
            "bet_side": bet_side,
            "edge": edge_abs,  # Always positive for the recommended side
            "raw_edge": edge,  # Signed edge for diagnostics
            "classifier_side": classifier_side,  # What the ML classifier says
            "prediction_side": prediction_side,  # What the point prediction says
            "classifier_extreme": classifier_extreme,  # True if classifier is unreliable
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
        if self.moneyline_model is None:
            raise ModelNotFoundError(f"Moneyline model for {self.period} not loaded")

        import pandas as pd
        from src.prediction.confidence import calculate_confidence_from_probabilities

        # Validate odds to prevent division by zero
        if home_ml_odds == 0:
            raise ValueError("home_ml_odds cannot be zero - invalid American odds")
        if away_ml_odds == 0:
            raise ValueError("away_ml_odds cannot be zero - invalid American odds")

        # Prepare features using unified validation
        feature_df = pd.DataFrame([features])
        X, missing = validate_and_prepare_features(
            feature_df,
            self.moneyline_features,
            market=f"{self.period}_moneyline",
        )

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

        # v33.0.6.0 FIX: Moneyline DISABLED per user request
        # if home_ml_odds is not None and away_ml_odds is not None:
        #     result["moneyline"] = self.predict_moneyline(features, home_ml_odds, away_ml_odds)

        return result


class UnifiedPredictionEngine:
    """
    NBA v33.0.7.0 - Production Prediction Engine

    4 ACTIVE Markets (1H + FG spreads/totals; moneyline disabled):

    First Half (1H):
    - 1H Spread
    - 1H Total

    Full Game (FG):
    - FG Spread
    - FG Total

    ARCHITECTURE:
    - Each period has independent models trained on period-specific features
    - No cross-period dependencies
    - Uses matchup-based formulas for predicted totals
    """

    def __init__(self, models_dir: Path, require_all: bool = True):
        """
        Initialize unified prediction engine.

        Args:
            models_dir: Path to models directory
            require_all: DEPRECATED - always True for 1H/FG models.
        """
        if not require_all:
            raise ValueError(
                "STRICT MODE ENFORCED: require_all=False is no longer supported. "
                "All 6 models (1H + FG) must be present. No silent fallbacks."
            )
        self.models_dir = Path(models_dir)
        self.loaded_models: Dict[str, bool] = {}

        if not self.models_dir.exists():
            raise ModelNotFoundError(
                f"Models directory does not exist: {self.models_dir}\n"
                f"Run: python scripts/train_all_models.py"
            )

        # Initialize period predictors (1H + FG only)
        self.h1_predictor: Optional[PeriodPredictor] = None
        self.fg_predictor: Optional[PeriodPredictor] = None

        logger.info("[v33.0.7.0] Loading 1H + FG models...")

        # Load 1H models - WILL RAISE if any missing
        logger.info("Loading 1H models (spread, total; moneyline disabled)...")
        h1_models = self._load_period_models("1h")
        self.h1_predictor = PeriodPredictor("1h", *h1_models)
        logger.info("1H predictor initialized (2/2 models - Moneyline disabled)")

        # Load FG models - WILL RAISE if any missing
        logger.info("Loading FG models (spread, total; moneyline disabled)...")
        fg_models = self._load_period_models("fg")
        self.fg_predictor = PeriodPredictor("fg", *fg_models)
        logger.info("FG predictor initialized (2/2 models - Moneyline disabled)")

        # Legacy predictors for backwards compatibility
        self._init_legacy_predictors()

        # Verify loaded models (v33.0.6.0 expects 4 total: 1H + FG Spreads/Totals)
        # Moneyline is disabled, so we filter those out from the count check
        loaded_count = sum(1 for k, v in self.loaded_models.items() if v and (k.startswith("1h_") or k.startswith("fg_")) and "moneyline" not in k)
        if loaded_count < 4:
            missing = [k for k, v in self.loaded_models.items() if (k.startswith("1h_") or k.startswith("fg_")) and not v and "moneyline" not in k]
            logger.warning(
                f"PARTIAL LOAD: Only {loaded_count}/4 models loaded (1H+FG Spreads/Totals). Missing: {missing}\n"
                f"Some predictions may be skipped."
            )
        else:
            logger.info("SUCCESS: All 4/4 models loaded (1H + FG Spreads/Totals)")

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

        # v33.0.6.0 FIX: Moneyline DISABLED per user request
        # try:
        #     ml_model, ml_features = self._load_model(ml_key)
        #     self.loaded_models[ml_key] = True
        # except Exception as e:
        #     logger.warning(f"Could not load {ml_key}: {e}")
        #     self.loaded_models[ml_key] = False
        ml_model, ml_features = None, []
        self.loaded_models[ml_key] = False # Mark as not loaded (or ignored)

        # v33.0.6.0 STRICT MODE: Spread and Total required (Moneyline DISABLED)
        missing_models = []
        if spread_model is None:
            missing_models.append(f"{period}_spread")
        if total_model is None:
            missing_models.append(f"{period}_total")
        # v33.0.6.0: Moneyline intentionally disabled - do NOT check for it

        if missing_models:
            raise ModelNotFoundError(
                f"STRICT MODE: Missing models for {period}: {missing_models}. "
                f"Spread and Total models are required (Moneyline is disabled in v33.0.6.0)."
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
            model = None
            try:
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
            except Exception as e:
                logger.warning(f"pickle.load failed for {model_path}: {e}")

            # Check if the loaded object is a valid model (has predict methods)
            # If not, it might be a joblib file saved with .pkl extension (common in this repo)
            if not (hasattr(model, "predict") or hasattr(model, "predict_proba")):
                logger.info(f"Object loaded from {model_path} via pickle does not look like a model. Trying joblib...")
                try:
                    model = joblib.load(model_path)
                    logger.info(f"Successfully loaded {model_path} using joblib fallback.")
                except Exception as e:
                    logger.warning(f"joblib fallback failed for {model_path}: {e}")

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

            # AUTO-CORRECTION: Use model's internal feature names if available
            if hasattr(model, "feature_names_in_"):
                model_features = model.feature_names_in_.tolist()
                if set(model_features) != set(features):
                    logger.warning(f"Feature mismatch for {model_key}. Using model's internal features ({len(model_features)}) instead of config/file ({len(features)}).")
                features = model_features

            logger.info(f"Loaded {model_key} (pkl) with {len(features)} features")
            return model, features

        # Handle .joblib files (combined format)
        data = joblib.load(model_path)

        model = data.get("pipeline") or data.get("model")
        features = data.get("feature_columns", [])

        if model is None:
            raise ValueError(f"Invalid model file: {model_path}")

        # AUTO-CORRECTION: Use model's internal feature names if available
        if hasattr(model, "feature_names_in_"):
            model_features = model.feature_names_in_.tolist()
            if set(model_features) != set(features):
                logger.warning(f"Feature mismatch for {model_key}. Using model's internal features ({len(model_features)}) instead of config/file ({len(features)}).")
            features = model_features

        logger.info(f"Loaded {model_key} with {len(features)} features")

        # v6.5 FIX: Verify model class indices match our assumptions
        # Expected: class 0 = away/under, class 1 = home/over
        if hasattr(model, "classes_"):
            classes = model.classes_.tolist() if hasattr(model.classes_, "tolist") else list(model.classes_)
            if len(classes) == 2:
                logger.info(f"[{model_key}] Model classes: {classes} (expected [0, 1] or [False, True])")
                # Check if classes are in expected order
                if classes not in [[0, 1], [False, True], [0.0, 1.0]]:
                    logger.warning(
                        f"[{model_key}] UNEXPECTED CLASS ORDER: {classes}. "
                        f"Predictions may be inverted! Expected [0, 1] where 0=away/under, 1=home/over."
                    )
            else:
                logger.warning(f"[{model_key}] Model has {len(classes)} classes (expected 2): {classes}")

        return model, features

    def _init_legacy_predictors(self):
        """Initialize legacy predictors for backwards compatibility."""
        # These are used by old code paths
        if self.fg_predictor:
            # Create legacy SpreadPredictor
            if self.fg_predictor.spread_model:
                try:
                    self.spread_predictor = SpreadPredictor(
                        fg_model=self.fg_predictor.spread_model,
                        fg_feature_columns=self.fg_predictor.spread_features,
                        fh_model=self.h1_predictor.spread_model if (self.h1_predictor and self.h1_predictor.spread_model) else self.fg_predictor.spread_model,
                        fh_feature_columns=self.h1_predictor.spread_features if (self.h1_predictor and self.h1_predictor.spread_model) else self.fg_predictor.spread_features,
                    )
                except Exception as e:
                    logger.warning(f"Failed to init legacy SpreadPredictor: {e}")

            # Create legacy TotalPredictor
            if self.fg_predictor.total_model:
                try:
                    self.total_predictor = TotalPredictor(
                        fg_model=self.fg_predictor.total_model,
                        fg_feature_columns=self.fg_predictor.total_features,
                        fh_model=self.h1_predictor.total_model if (self.h1_predictor and self.h1_predictor.total_model) else self.fg_predictor.total_model,
                        fh_feature_columns=self.h1_predictor.total_features if (self.h1_predictor and self.h1_predictor.total_model) else self.fg_predictor.total_features,
                    )
                except Exception as e:
                    logger.warning(f"Failed to init legacy TotalPredictor: {e}")

            # Create legacy MoneylinePredictor
            if self.fg_predictor.moneyline_model:
                try:
                    self.moneyline_predictor = MoneylinePredictor(
                        model=self.fg_predictor.moneyline_model,
                        feature_columns=self.fg_predictor.moneyline_features,
                        fh_model=self.h1_predictor.spread_model if (self.h1_predictor and self.h1_predictor.spread_model) else self.fg_predictor.spread_model,
                        fh_feature_columns=self.h1_predictor.spread_features if (self.h1_predictor and self.h1_predictor.spread_model) else self.fg_predictor.spread_features,
                    )
                except Exception as e:
                    logger.warning(f"Failed to init legacy MoneylinePredictor: {e}")

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
        # Moneyline odds
        home_ml_odds: Optional[int] = None,
        away_ml_odds: Optional[int] = None,
        fh_home_ml_odds: Optional[int] = None,
        fh_away_ml_odds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate predictions for all 6 markets (1H + FG).

        Args:
            features: Feature dictionary with all period features
            fg_spread_line: FG spread line
            fg_total_line: FG total line
            fh_spread_line: 1H spread line
            fh_total_line: 1H total line
            home_ml_odds: FG home moneyline odds
            away_ml_odds: FG away moneyline odds
            fh_home_ml_odds: 1H home moneyline odds
            fh_away_ml_odds: 1H away moneyline odds

        Returns:
            Predictions for all 6 markets grouped by period
        """
        result = {
            "first_half": {},
            "full_game": {},
        }

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
            "version": MODEL_VERSION,
            "architecture": "4-model independent (1H + FG spreads/totals; moneyline disabled)",
            "markets": sum(1 for v in self.loaded_models.values() if v),
            "markets_list": [k for k, v in self.loaded_models.items() if v],
            "periods": ["first_half", "full_game"],
            "models_dir": str(self.models_dir),
            "loaded_models": self.loaded_models,
            "predictors": {
                "1h": self.h1_predictor is not None,
                "fg": self.fg_predictor is not None,
            },
        }
