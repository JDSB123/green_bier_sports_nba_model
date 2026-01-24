"""
Sharp Money Weighted Combination System (v34.2.0).

ARCHITECTURE:
This module combines ML model predictions with sharp money signals to determine:
1. WHICH SIDE to bet (not just confidence!)
2. Whether to play at all (signals cancel = no play)
3. Final confidence based on signal agreement

FORMULA:
    combined_edge = (model_edge × MODEL_WEIGHT) + (sharp_edge × SHARP_WEIGHT)
    
    Where:
        MODEL_WEIGHT = 0.60 (we built the model, we trust it somewhat)
        SHARP_WEIGHT = 0.40 (Pinnacle knows things we don't)
    
    If combined_edge > threshold → Bet that side
    If combined_edge < -threshold → Bet opposite side
    If |combined_edge| < threshold → NO PLAY (signals cancel)

KEY INSIGHT:
Old system: Model says HOME, sharps say AWAY → Still bet HOME with lower confidence ❌
New system: Model says HOME, sharps say AWAY → Combined math decides side (or NO PLAY) ✓
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class WeightedConfig:
    """Configuration for weighted combination system."""
    
    # Weight allocation (must sum to 1.0)
    model_weight: float = 0.60
    sharp_weight: float = 0.40
    
    # Minimum edge threshold for a play (in points)
    # If |combined_edge| < this, NO PLAY
    min_edge_spread: float = 0.5   # 0.5 pts minimum edge for spread bets
    min_edge_total: float = 1.0    # 1.0 pts minimum edge for totals
    
    # Confidence scaling (how combined_edge maps to confidence)
    # confidence = base + (combined_edge * scale_factor), capped at max
    base_confidence: float = 0.50
    confidence_scale_spread: float = 0.10  # Each point of edge = +10% confidence
    confidence_scale_total: float = 0.05   # Each point of edge = +5% confidence
    max_confidence: float = 0.85
    min_confidence: float = 0.52  # Minimum confidence if we're playing


# Global config
WEIGHTED_CONFIG = WeightedConfig()


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SharpSignal:
    """Sharp money signal for a single market."""
    
    # Pinnacle vs square divergence (primary signal)
    pinnacle_line: Optional[float] = None
    square_line: Optional[float] = None
    
    # Action Network betting splits (secondary signal)
    ticket_pct: Optional[float] = None  # Ticket % on one side
    money_pct: Optional[float] = None   # Money % on one side (RLM when differs from ticket)
    
    # Line movement
    opening_line: Optional[float] = None
    current_line: Optional[float] = None
    
    @property
    def has_pinnacle(self) -> bool:
        return self.pinnacle_line is not None
    
    @property
    def divergence(self) -> Optional[float]:
        """Divergence between Pinnacle and square books."""
        if self.pinnacle_line is None or self.square_line is None:
            return None
        return self.pinnacle_line - self.square_line
    
    @property
    def line_move(self) -> Optional[float]:
        """Line movement from open."""
        if self.opening_line is None or self.current_line is None:
            return None
        return self.current_line - self.opening_line
    
    @property
    def has_rlm(self) -> bool:
        """True if reverse line movement detected."""
        if self.ticket_pct is None or self.money_pct is None:
            return False
        # RLM = big difference between ticket and money %
        return abs(self.ticket_pct - self.money_pct) > 15


@dataclass 
class WeightedResult:
    """Result of weighted combination calculation."""
    
    # Core outputs
    final_side: str           # "home", "away", "over", "under", or "NO_PLAY"
    final_confidence: float   # Combined confidence (0.0-1.0)
    combined_edge: float      # Combined edge in points
    is_play: bool             # False if signals cancel out
    
    # Components for transparency
    model_side: str           # What model originally said
    model_edge: float         # Model's edge in points
    sharp_side: str           # What sharps say (from divergence)
    sharp_edge: float         # Sharp edge in points (from divergence magnitude)
    
    # Signal alignment
    signals_agree: bool       # True if model and sharps on same side
    side_flipped: bool        # True if final side differs from model
    
    # Rationale
    rationale: List[str]      # Explanation bullets


# ============================================================================
# SPREAD COMBINATION
# ============================================================================

def combine_spread_signals(
    model_predicted_margin: float,
    market_spread: float,
    sharp_signal: SharpSignal,
    config: WeightedConfig = WEIGHTED_CONFIG,
) -> WeightedResult:
    """
    Combine ML model spread prediction with sharp money signals.
    
    Args:
        model_predicted_margin: Model's predicted home margin (positive = home wins by X)
        market_spread: Current market spread line (negative = home favored)
        sharp_signal: Sharp money signal data
        config: Weighting configuration
        
    Returns:
        WeightedResult with final side and confidence
        
    Example:
        Model predicts home margin +3.0, market spread -1.0
        Model edge = 3.0 - (-(-1.0)) = 3.0 - 1.0 = +2.0 pts (home covers)
        
        Pinnacle at -3.0, Square at -1.0 → divergence = -2.0 (sharps like home MORE)
        Sharp edge toward HOME = +2.0 pts
        
        Combined = (2.0 * 0.6) + (2.0 * 0.4) = 1.2 + 0.8 = +2.0 HOME
    """
    rationale = []
    
    # =========================================================================
    # STEP 1: Calculate model edge
    # =========================================================================
    # Model edge = predicted_margin - (-spread)
    # If model predicts home by 5 and spread is -3, edge = 5 - 3 = +2 (home covers)
    # Positive edge = home covers, negative = away covers
    model_edge = model_predicted_margin - (-market_spread)
    model_side = "home" if model_edge > 0 else "away"
    
    rationale.append(f"MODEL: {model_side.upper()} by {abs(model_edge):.1f} pts")
    
    # =========================================================================
    # STEP 2: Calculate sharp edge from Pinnacle divergence
    # =========================================================================
    sharp_edge = 0.0
    sharp_side = "neutral"
    
    if sharp_signal.has_pinnacle and sharp_signal.divergence is not None:
        # Divergence = Pinnacle - Square
        # Negative divergence = Pinnacle more negative = sharps like HOME
        # Positive divergence = Pinnacle less negative = sharps like AWAY
        divergence = sharp_signal.divergence
        
        if divergence < -0.25:
            # Sharps favor HOME (Pinnacle line is more negative)
            sharp_side = "home"
            sharp_edge = abs(divergence)  # Magnitude of their conviction
            rationale.append(f"SHARP: HOME by {sharp_edge:.1f} pts (Pinnacle {divergence:+.1f} from square)")
        elif divergence > 0.25:
            # Sharps favor AWAY (Pinnacle line is less negative)
            sharp_side = "away"
            sharp_edge = -abs(divergence)  # Negative = toward away
            rationale.append(f"SHARP: AWAY by {abs(divergence):.1f} pts (Pinnacle {divergence:+.1f} from square)")
        else:
            rationale.append(f"SHARP: Neutral (divergence {divergence:+.1f})")
    else:
        rationale.append("SHARP: No Pinnacle data - using model only")
        # Without sharp data, use 100% model weight
        sharp_edge = 0.0
    
    # =========================================================================
    # STEP 3: Combine signals
    # =========================================================================
    if sharp_signal.has_pinnacle:
        combined_edge = (model_edge * config.model_weight) + (sharp_edge * config.sharp_weight)
    else:
        # No sharp data - use model alone but reduce confidence
        combined_edge = model_edge * 0.8  # 20% haircut without sharp confirmation
    
    rationale.append(f"COMBINED: {combined_edge:+.2f} pts = ({model_edge:.1f} × {config.model_weight}) + ({sharp_edge:.1f} × {config.sharp_weight})")
    
    # =========================================================================
    # STEP 4: Determine final side and if it's a play
    # =========================================================================
    signals_agree = (model_edge > 0 and sharp_edge >= 0) or (model_edge < 0 and sharp_edge <= 0)
    
    if abs(combined_edge) < config.min_edge_spread:
        # Signals cancel out - NO PLAY
        final_side = "NO_PLAY"
        is_play = False
        final_confidence = 0.0
        rationale.append(f"❌ NO PLAY: Combined edge {combined_edge:+.2f} < min {config.min_edge_spread}")
    else:
        final_side = "home" if combined_edge > 0 else "away"
        is_play = True
        
        # Calculate confidence based on combined edge magnitude
        raw_confidence = config.base_confidence + (abs(combined_edge) * config.confidence_scale_spread)
        final_confidence = min(config.max_confidence, max(config.min_confidence, raw_confidence))
        
        if signals_agree:
            rationale.append(f"✓ SIGNALS AGREE: {final_side.upper()} ({final_confidence:.1%} confidence)")
        else:
            rationale.append(f"⚠️ SIGNALS CONFLICT: Combined favors {final_side.upper()} ({final_confidence:.1%})")
    
    side_flipped = (model_side != final_side) and (final_side != "NO_PLAY")
    if side_flipped:
        rationale.append(f"🔄 SIDE FLIPPED: Model said {model_side.upper()}, now {final_side.upper()}")
    
    return WeightedResult(
        final_side=final_side,
        final_confidence=final_confidence,
        combined_edge=combined_edge,
        is_play=is_play,
        model_side=model_side,
        model_edge=model_edge,
        sharp_side=sharp_side,
        sharp_edge=sharp_edge,
        signals_agree=signals_agree,
        side_flipped=side_flipped,
        rationale=rationale,
    )


# ============================================================================
# TOTAL COMBINATION
# ============================================================================

def combine_total_signals(
    model_predicted_total: float,
    market_total: float,
    sharp_signal: SharpSignal,
    config: WeightedConfig = WEIGHTED_CONFIG,
) -> WeightedResult:
    """
    Combine ML model total prediction with sharp money signals.
    
    Args:
        model_predicted_total: Model's predicted total points
        market_total: Current market total line
        sharp_signal: Sharp money signal data
        config: Weighting configuration
        
    Returns:
        WeightedResult with final side and confidence
        
    Example:
        Model predicts 228, market total 224
        Model edge = 228 - 224 = +4.0 pts (model likes OVER)
        
        Pinnacle at 226, Square at 224 → divergence = +2.0 (sharps like OVER too)
        Sharp edge = +2.0 pts
        
        Combined = (4.0 * 0.6) + (2.0 * 0.4) = 2.4 + 0.8 = +3.2 OVER
    """
    rationale = []
    
    # =========================================================================
    # STEP 1: Calculate model edge
    # =========================================================================
    # Positive = over, negative = under
    model_edge = model_predicted_total - market_total
    model_side = "over" if model_edge > 0 else "under"
    
    rationale.append(f"MODEL: {model_side.upper()} by {abs(model_edge):.1f} pts")
    
    # =========================================================================
    # STEP 2: Calculate sharp edge from Pinnacle divergence
    # =========================================================================
    sharp_edge = 0.0
    sharp_side = "neutral"
    
    if sharp_signal.has_pinnacle and sharp_signal.divergence is not None:
        # Divergence = Pinnacle - Square
        # Positive divergence = Pinnacle higher = sharps like OVER
        # Negative divergence = Pinnacle lower = sharps like UNDER
        divergence = sharp_signal.divergence
        
        if divergence > 0.5:
            sharp_side = "over"
            sharp_edge = abs(divergence)
            rationale.append(f"SHARP: OVER by {sharp_edge:.1f} pts (Pinnacle {divergence:+.1f} from square)")
        elif divergence < -0.5:
            sharp_side = "under"
            sharp_edge = -abs(divergence)  # Negative = toward under
            rationale.append(f"SHARP: UNDER by {abs(divergence):.1f} pts (Pinnacle {divergence:+.1f} from square)")
        else:
            rationale.append(f"SHARP: Neutral (divergence {divergence:+.1f})")
    else:
        rationale.append("SHARP: No Pinnacle data - using model only")
        sharp_edge = 0.0
    
    # =========================================================================
    # STEP 3: Combine signals
    # =========================================================================
    if sharp_signal.has_pinnacle:
        combined_edge = (model_edge * config.model_weight) + (sharp_edge * config.sharp_weight)
    else:
        combined_edge = model_edge * 0.8
    
    rationale.append(f"COMBINED: {combined_edge:+.2f} pts = ({model_edge:.1f} × {config.model_weight}) + ({sharp_edge:.1f} × {config.sharp_weight})")
    
    # =========================================================================
    # STEP 4: Determine final side and if it's a play
    # =========================================================================
    signals_agree = (model_edge > 0 and sharp_edge >= 0) or (model_edge < 0 and sharp_edge <= 0)
    
    if abs(combined_edge) < config.min_edge_total:
        final_side = "NO_PLAY"
        is_play = False
        final_confidence = 0.0
        rationale.append(f"❌ NO PLAY: Combined edge {combined_edge:+.2f} < min {config.min_edge_total}")
    else:
        final_side = "over" if combined_edge > 0 else "under"
        is_play = True
        
        raw_confidence = config.base_confidence + (abs(combined_edge) * config.confidence_scale_total)
        final_confidence = min(config.max_confidence, max(config.min_confidence, raw_confidence))
        
        if signals_agree:
            rationale.append(f"✓ SIGNALS AGREE: {final_side.upper()} ({final_confidence:.1%} confidence)")
        else:
            rationale.append(f"⚠️ SIGNALS CONFLICT: Combined favors {final_side.upper()} ({final_confidence:.1%})")
    
    side_flipped = (model_side != final_side) and (final_side != "NO_PLAY")
    if side_flipped:
        rationale.append(f"🔄 SIDE FLIPPED: Model said {model_side.upper()}, now {final_side.upper()}")
    
    return WeightedResult(
        final_side=final_side,
        final_confidence=final_confidence,
        combined_edge=combined_edge,
        is_play=is_play,
        model_side=model_side,
        model_edge=model_edge,
        sharp_side=sharp_side,
        sharp_edge=sharp_edge,
        signals_agree=signals_agree,
        side_flipped=side_flipped,
        rationale=rationale,
    )


# ============================================================================
# HELPER: Build SharpSignal from features dict
# ============================================================================

def build_sharp_signal_spread(
    features: Dict[str, Any],
    market_spread: Optional[float] = None,
) -> SharpSignal:
    """Build SharpSignal for spread from features dictionary."""
    return SharpSignal(
        pinnacle_line=features.get("pinnacle_spread"),
        square_line=features.get("square_avg_spread") or market_spread,
        ticket_pct=features.get("spread_ticket_home_pct"),
        money_pct=features.get("spread_money_home_pct"),
        opening_line=features.get("spread_open"),
        current_line=market_spread,
    )


def build_sharp_signal_total(
    features: Dict[str, Any],
    market_total: Optional[float] = None,
) -> SharpSignal:
    """Build SharpSignal for total from features dictionary."""
    return SharpSignal(
        pinnacle_line=features.get("pinnacle_total"),
        square_line=features.get("square_avg_total") or market_total,
        ticket_pct=features.get("total_ticket_over_pct"),
        money_pct=features.get("total_money_over_pct"),
        opening_line=features.get("total_open"),
        current_line=market_total,
    )


# ============================================================================
# INTEGRATION FUNCTIONS (for predict_unified_full_game.py)
# ============================================================================

def apply_weighted_combination_spread(
    prediction: Dict[str, Any],
    features: Dict[str, Any],
    market_spread: Optional[float] = None,
    config: WeightedConfig = WEIGHTED_CONFIG,
) -> Tuple[Dict[str, Any], WeightedResult]:
    """
    Apply weighted combination to spread prediction.
    
    This REPLACES the old sharp_adjustments approach:
    - Old: Model says HOME → still bet HOME, just lower confidence
    - New: Model + Sharps combined → may change side or NO PLAY
    
    Args:
        prediction: Original prediction from ML engine
        features: Features dict (contains Pinnacle data)
        market_spread: Current market spread
        config: Weighting configuration
        
    Returns:
        (updated_prediction, weighted_result)
    """
    model_margin = prediction.get("predicted_margin", 0)
    spread = market_spread or prediction.get("spread_line", 0)
    
    sharp_signal = build_sharp_signal_spread(features, spread)
    result = combine_spread_signals(model_margin, spread, sharp_signal, config)
    
    # Update prediction with combined result
    updated = prediction.copy()
    
    if result.is_play:
        updated["bet_side"] = result.final_side
        updated["confidence"] = result.final_confidence
        updated["combined_edge"] = result.combined_edge
        updated["passes_filter"] = True
        updated["filter_reason"] = ""
    else:
        updated["bet_side"] = result.model_side  # Show what model said
        updated["confidence"] = 0.0
        updated["combined_edge"] = result.combined_edge
        updated["passes_filter"] = False
        updated["filter_reason"] = "Signals cancel - NO PLAY"
    
    # Add tracking fields
    updated["model_side"] = result.model_side
    updated["model_edge"] = result.model_edge
    updated["sharp_side"] = result.sharp_side
    updated["sharp_edge"] = result.sharp_edge
    updated["signals_agree"] = result.signals_agree
    updated["side_flipped"] = result.side_flipped
    updated["sharp_rationale"] = result.rationale
    
    return updated, result


def apply_weighted_combination_total(
    prediction: Dict[str, Any],
    features: Dict[str, Any],
    market_total: Optional[float] = None,
    config: WeightedConfig = WEIGHTED_CONFIG,
) -> Tuple[Dict[str, Any], WeightedResult]:
    """
    Apply weighted combination to total prediction.
    
    Args:
        prediction: Original prediction from ML engine
        features: Features dict (contains Pinnacle data)
        market_total: Current market total
        config: Weighting configuration
        
    Returns:
        (updated_prediction, weighted_result)
    """
    model_total = prediction.get("predicted_total", 0)
    total = market_total or prediction.get("total_line", 0)
    
    sharp_signal = build_sharp_signal_total(features, total)
    result = combine_total_signals(model_total, total, sharp_signal, config)
    
    # Update prediction with combined result
    updated = prediction.copy()
    
    if result.is_play:
        updated["bet_side"] = result.final_side
        updated["confidence"] = result.final_confidence
        updated["combined_edge"] = result.combined_edge
        updated["passes_filter"] = True
        updated["filter_reason"] = ""
    else:
        updated["bet_side"] = result.model_side
        updated["confidence"] = 0.0
        updated["combined_edge"] = result.combined_edge
        updated["passes_filter"] = False
        updated["filter_reason"] = "Signals cancel - NO PLAY"
    
    updated["model_side"] = result.model_side
    updated["model_edge"] = result.model_edge
    updated["sharp_side"] = result.sharp_side
    updated["sharp_edge"] = result.sharp_edge
    updated["signals_agree"] = result.signals_agree
    updated["side_flipped"] = result.side_flipped
    updated["sharp_rationale"] = result.rationale
    
    return updated, result
