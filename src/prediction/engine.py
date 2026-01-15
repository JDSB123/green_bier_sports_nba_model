"""
NBA v33.0.11.0 - Unified Prediction Engine

PRODUCTION: 4 INDEPENDENT Markets (1H + FG spreads/totals)

First Half:
- 1H Spread
- 1H Total

Full Game:
- FG Spread
- FG Total

ARCHITECTURE: Each period has INDEPENDENT models trained on period-specific
features. No cross-period dependencies. Uses matchup-based formulas for
predicted totals (not scaled from FG).

v33.0.11.0 FIXES:
    - bet_side now based on EDGE calculation, not classifier
    - Added classifier sanity check to detect data drift (extreme probabilities)
    - FIXED: 1H models now use 1H-specific features (not FG features)
    - Simple filter: confidence + edge thresholds (NO dual-signal requirement)

FEATURE VALIDATION:
    Controlled by PREDICTION_FEATURE_MODE environment variable:
    - "strict" (default): Raise error on missing features
    - "warn": Log warning, zero-fill missing features
    - "silent": Zero-fill without logging

    See src/prediction/feature_validation.py for details.
"""
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from src.utils.version import resolve_version

# Single source of truth for version - env overrides VERSION file
MODEL_VERSION = resolve_version()

import logging
import joblib

from src.prediction.spreads import SpreadPredictor
from src.prediction.totals import TotalPredictor
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


def map_1h_features_to_fg_names(features: Dict[str, float]) -> Dict[str, float]:
    """
    Map 1H-specific feature names to FG feature names for model compatibility.

    The 1H models were trained on FG feature names but should use 1H data.
    This function copies 1H features to the FG feature names the model expects.

    Args:
        features: Dict with both FG and 1H features

    Returns:
        Dict with 1H features mapped to FG names (for 1H model predictions)
    """
    mapped_features = dict(features)  # Copy all features

    # Core statistical features mapping
    feature_mappings = {
        # PPG (points per game)
        "home_ppg_1h": "home_ppg",
        "away_ppg_1h": "away_ppg",
        "home_papg_1h": "home_papg",
        "away_papg_1h": "away_papg",

        # Margins
        "home_margin_1h": "home_avg_margin",
        "away_margin_1h": "away_avg_margin",

        # Differentials
        "ppg_diff_1h": "ppg_diff",
        "margin_diff_1h": "win_pct_diff",  # Using win_pct_diff as closest equivalent

        # Win rates
        "home_1h_win_pct": "home_win_pct",
        "away_1h_win_pct": "away_win_pct",

        # Pace
        "home_pace_1h": "home_pace_factor",
        "away_pace_1h": "away_pace_factor",

        # Recent form (last 5)
        "home_l5_margin_1h": "home_l5_margin",
        "away_l5_margin_1h": "away_l5_margin",

        # Recent form (last 10)
        "home_l10_margin_1h": "home_l10_margin",
        "away_l10_margin_1h": "away_l10_margin",

        # Consistency (standard deviation)
        "home_margin_std_1h": "home_form_adj",  # Using form_adj as closest equivalent
        "away_margin_std_1h": "away_form_adj",

        # Efficiency ratings
        "home_ortg_1h": "home_ortg",
        "away_ortg_1h": "away_ortg",
        "home_drtg_1h": "home_drtg",
        "away_drtg_1h": "away_drtg",
        "home_net_rtg_1h": "home_net_rtg",
        "away_net_rtg_1h": "away_net_rtg",

        # Position (standings)
        "home_position_1h": "home_position",
        "away_position_1h": "away_position",

        # H2H
        "h2h_margin_1h": "h2h_win_rate",  # Using win_rate as closest equivalent

        # Rest (same for both periods)
        "home_rest_days_1h": "home_rest_days",
        "away_rest_days_1h": "away_rest_days",
        "home_rest_adj_1h": "home_rest_adj",
        "away_rest_adj_1h": "away_rest_adj",
        "rest_margin_adj_1h": "rest_margin_adj",

        # Travel (same for both periods)
        "away_travel_distance_1h": "away_travel_distance",
        "away_timezone_change_1h": "away_timezone_change",
        "away_travel_fatigue_1h": "away_travel_fatigue",
        "is_away_long_trip_1h": "is_away_long_trip",
        "is_away_cross_country_1h": "is_away_cross_country",
        "away_b2b_travel_penalty_1h": "away_b2b_travel_penalty",
        "travel_advantage_1h": "travel_advantage",

        # Home court advantage
        "dynamic_hca_1h": "home_court_advantage",

        # Injuries (same for both periods)
        "home_injury_impact_ppg_1h": "home_injury_impact_ppg",
        "away_injury_impact_ppg_1h": "away_injury_impact_ppg",
        "injury_margin_adj_1h": "injury_margin_adj",

        # Elo (same for both periods)
        "home_elo_1h": "home_elo",
        "away_elo_1h": "away_elo",
        "elo_diff_1h": "elo_diff",
        "elo_prob_home_1h": "elo_prob_home",
    }

    # Apply mappings - copy 1H features to FG names if they exist
    for h1_feature, fg_feature in feature_mappings.items():
        if h1_feature in features:
            mapped_features[fg_feature] = features[h1_feature]
            logger.debug(f"Mapped {h1_feature} -> {fg_feature} = {features[h1_feature]}")

    return mapped_features


class PeriodPredictor:
    """
    Predictor for a single period (1H or FG).

    Each period predictor has its own independent models for
    spread and total.
    """

    def __init__(
        self,
        period: str,
        spread_model: Any,
        spread_features: List[str],
        total_model: Any,
        total_features: List[str],
    ):
        self.period = period
        self.spread_model = spread_model
        self.spread_features = spread_features
        self.total_model = total_model
        self.total_features = total_features

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

        # FIX: For 1H models, map 1H features to FG feature names that the model expects
        if self.period == "1h":
            features = map_1h_features_to_fg_names(features)
            logger.debug(f"[{self.period}_spread] Mapped 1H features to FG names for model compatibility")

        # Prepare features using unified validation (add line context for compatibility)
        margin_key = f"predicted_margin_{self.period}" if self.period != "fg" else "predicted_margin"
        feature_payload = dict(features)
        feature_payload["spread_line"] = spread_line
        feature_payload["fg_spread_line"] = spread_line
        if self.period != "fg":
            feature_payload["1h_spread_line"] = spread_line
            feature_payload["fh_spread_line"] = spread_line

        predicted_margin_value = feature_payload.get(margin_key)
        if predicted_margin_value is not None:
            if self.period == "fg":
                feature_payload["spread_vs_predicted"] = predicted_margin_value - (-spread_line)
            else:
                feature_payload["spread_vs_predicted_1h"] = predicted_margin_value - (-spread_line)
                feature_payload["fh_spread_vs_predicted"] = predicted_margin_value - (-spread_line)

        feature_df = pd.DataFrame([feature_payload])
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
        predicted_margin = feature_payload.get(margin_key)

        # CRITICAL: predicted_margin is REQUIRED - do not allow defaults
        if predicted_margin is None:
            available_margin_keys = [k for k in features.keys() if 'margin' in k.lower()]
            raise ValueError(
                f"[{self.period}_spread] MISSING REQUIRED predicted_margin feature (key: {margin_key}). "
                f"Cannot proceed without margin calculation. "
                f"Available features with 'margin': {available_margin_keys}. "
                f"Fix feature pipeline to ensure {margin_key} is always computed."
            )
        if predicted_margin == 0:
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

        # v33.0.11.0 FIX: Classifier sanity check - detect broken/drifted models
        # If classifier outputs extreme probability (>99% or <1%), it's unreliable
        classifier_extreme = (home_cover_prob > 0.99 or home_cover_prob < 0.01)
        if classifier_extreme:
            logger.warning(
                f"[{self.period}_spread] EXTREME classifier probability: {home_cover_prob:.4f}. "
                f"Model may have data drift."
            )

        # v33.0.11.0 FIX: Use EDGE-BASED prediction as authoritative bet_side
        # The edge calculation is more robust than a potentially drifted classifier
        bet_side = prediction_side

        # For filtering and display, use absolute edge value
        edge_abs = abs(edge)

        # Filter logic: confidence AND edge threshold (NO dual-signal requirement)
        # Use calibrated probability from src/prediction/confidence.py for consistency
        if self.period == "1h":
            min_conf = filter_thresholds.fh_spread_min_confidence
            min_edge = filter_thresholds.fh_spread_min_edge
        else:
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

        # FIX: For 1H models, map 1H features to FG feature names that the model expects
        if self.period == "1h":
            features = map_1h_features_to_fg_names(features)
            logger.debug(f"[{self.period}_total] Mapped 1H features to FG names for model compatibility")

        # Prepare features using unified validation (add line context for compatibility)
        total_key = f"predicted_total_{self.period}" if self.period != "fg" else "predicted_total"
        feature_payload = dict(features)
        feature_payload["total_line"] = total_line
        feature_payload["fg_total_line"] = total_line
        if self.period != "fg":
            feature_payload["1h_total_line"] = total_line
            feature_payload["fh_total_line"] = total_line

        predicted_total_value = feature_payload.get(total_key)
        if predicted_total_value is not None:
            if self.period == "fg":
                feature_payload["total_vs_predicted"] = predicted_total_value - total_line
            else:
                feature_payload["total_vs_predicted_1h"] = predicted_total_value - total_line
                feature_payload["fh_total_vs_predicted"] = predicted_total_value - total_line

        feature_df = pd.DataFrame([feature_payload])
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
        predicted_total = feature_payload.get(total_key)

        # CRITICAL: predicted_total is REQUIRED - do not allow defaults
        if predicted_total is None:
            available_total_keys = [k for k in features.keys() if 'total' in k.lower()]
            raise ValueError(
                f"[{self.period}_total] MISSING REQUIRED predicted_total feature (key: {total_key}). "
                f"Cannot proceed without total calculation. "
                f"Available features with 'total': {available_total_keys}. "
                f"Fix feature pipeline to ensure {total_key} is always computed."
            )
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

        # v33.0.11.0 FIX: Classifier sanity check - detect broken/drifted models
        # If classifier outputs extreme probability (>99% or <1%), it's unreliable
        # This catches the FG Total model data drift issue (always outputs 100% over)
        classifier_extreme = (over_prob > 0.99 or over_prob < 0.01)
        if classifier_extreme:
            logger.warning(
                f"[{self.period}_total] EXTREME classifier probability: over={over_prob:.4f}. "
                f"Model may have data drift."
            )

        # v33.0.11.0 FIX: Use EDGE-BASED prediction as authoritative bet_side
        # The edge calculation is more robust than a potentially drifted classifier
        bet_side = prediction_side

        # For filtering and display, use absolute edge value
        edge_abs = abs(edge)

        # Filter logic: confidence AND edge threshold (NO dual-signal requirement)
        if self.period == "1h":
            min_conf = filter_thresholds.fh_total_min_confidence
            min_edge = filter_thresholds.fh_total_min_edge
        else:
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

    def predict_all(
        self,
        features: Dict[str, float],
        spread_line: Optional[float] = None,
        total_line: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Predict all markets for this period."""
        result = {}

        if spread_line is not None:
            result["spread"] = self.predict_spread(features, spread_line)

        if total_line is not None:
            result["total"] = self.predict_total(features, total_line)

        return result


class UnifiedPredictionEngine:
    """
    NBA v34.0 - Production Prediction Engine

    4 ACTIVE Markets (1H + FG spreads/totals):

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
                "All 4 models (1H + FG spreads/totals) must be present. No silent fallbacks."
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

        logger.info("[v33.0.11.0] Loading 1H + FG models (spread/total only)...")

        # Load 1H models - spread/total required
        logger.info("Loading 1H models (spread, total)...")
        h1_models = self._load_period_models("1h")
        self.h1_predictor = PeriodPredictor("1h", *h1_models)
        logger.info("1H predictor initialized (2/2 models)")

        # Load FG models - spread/total required
        logger.info("Loading FG models (spread, total)...")
        fg_models = self._load_period_models("fg")
        self.fg_predictor = PeriodPredictor("fg", *fg_models)
        logger.info("FG predictor initialized (2/2 models)")

        # Legacy predictors for backwards compatibility
        self._init_legacy_predictors()

        # Verify loaded models (v33.0.11.0 expects spreads/totals)
        loaded_count = sum(
            1
            for k, v in self.loaded_models.items()
            if v and (k.startswith("1h_") or k.startswith("fg_"))
        )
        if loaded_count < 4:
            missing = [k for k, v in self.loaded_models.items() if (k.startswith("1h_") or k.startswith("fg_")) and not v]
            logger.warning(
                f"PARTIAL LOAD: Only {loaded_count}/4 models loaded (1H+FG Spreads/Totals). Missing: {missing}\n"
                f"Some predictions may be skipped."
            )
        else:
            logger.info("SUCCESS: All required models loaded (1H + FG spreads/totals)")

    def _load_period_models(
        self,
        period: str,
    ) -> Optional[Tuple[Any, List[str], Any, List[str], Any, List[str]]]:
        """Load all models for a period."""
        spread_key = f"{period}_spread"
        total_key = f"{period}_total"

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

        # Spread and Total required
        missing_models = []
        if spread_model is None:
            missing_models.append(f"{period}_spread")
        if total_model is None:
            missing_models.append(f"{period}_total")

        if missing_models:
            raise ModelNotFoundError(
                f"STRICT MODE: Missing models for {period}: {missing_models}. "
                f"Spread and Total models are required."
            )

        return (
            spread_model, spread_features,
            total_model, total_features,
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

        # Verify model class indices match our assumptions
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
        """
        Initialize legacy predictors for backwards compatibility.

        DEPRECATED (v33.0.11.0): These predictors are NOT used by any production code path.
        All predictions go through PeriodPredictor.predict_spread/predict_total.
        Keeping for API stability but may be removed in v34.0.
        """
        # DEPRECATED: These are NOT used by predict_all_markets() or any serving code
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

    def predict_first_half(
        self,
        features: Dict[str, float],
        spread_line: Optional[float] = None,
        total_line: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Generate predictions for 1H markets.

        Args:
            features: Feature dictionary with 1H-specific features
            spread_line: 1H spread line
            total_line: 1H total line
        Returns:
            Predictions for 1H Spread and Total
        """
        if self.h1_predictor is None:
            raise ModelNotFoundError("1H models not loaded")

        return self.h1_predictor.predict_all(features, spread_line, total_line)

    def predict_full_game(
        self,
        features: Dict[str, float],
        spread_line: Optional[float] = None,
        total_line: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Generate predictions for FG markets.

        Args:
            features: Feature dictionary with FG-specific features
            spread_line: FG spread line
            total_line: FG total line
        Returns:
            Predictions for FG Spread and Total
        """
        if self.fg_predictor is None:
            raise ModelNotFoundError("FG models not loaded")

        return self.fg_predictor.predict_all(features, spread_line, total_line)

    def predict_all_markets(
        self,
        features: Dict[str, float],
        # Full game lines
        fg_spread_line: Optional[float] = None,
        fg_total_line: Optional[float] = None,
        # First half lines
        fh_spread_line: Optional[float] = None,
        fh_total_line: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Generate predictions for all 4 markets (1H + FG spreads/totals).

        Args:
            features: Feature dictionary with all period features
            fg_spread_line: FG spread line
            fg_total_line: FG total line
            fh_spread_line: 1H spread line
            fh_total_line: 1H total line
        Returns:
            Predictions for all 4 markets grouped by period
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
                )
            except Exception as e:
                logger.warning(f"FG prediction failed: {e}")

        return result

    def get_model_info(self) -> Dict[str, Any]:
        """Return info about loaded models."""
        return {
            "version": MODEL_VERSION,
            "architecture": "1H + FG spreads/totals only",
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

