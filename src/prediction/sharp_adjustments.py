"""
Sharp Money Confidence Adjustments for NBA Predictions.

ARCHITECTURE (following NCAAM best practices):
- Sharp signals do NOT modify ML predictions
- Sharp signals modify CONFIDENCE which affects:
  1. EV calculation (lower confidence = lower EV)
  2. Recommendation gating (confidence below threshold = no rec)
  3. Display/rationale (show sharp alignment status)

SIGNAL HIERARCHY (applied in order):
1. Sharp alignment check (-15%) - Betting against Pinnacle direction
2. Sharp-square divergence (±5%) - Pinnacle differs from DraftKings/FanDuel
3. RLM boost (+5%) - Line moving against public when aligned
4. Steam moves (±8%) - Large, fast line moves

DATA SOURCES:
- Pinnacle lines from The-Odds-API (sharp book reference)
- DraftKings/FanDuel averages (square book reference)
- Action Network ticket/money splits (RLM detection)
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION - Thresholds following NCAAM best practices
# ============================================================================

@dataclass
class SharpConfig:
    """Configuration for sharp money adjustments."""
    
    # Sharp alignment penalty when betting against Pinnacle direction
    against_sharp_penalty: float = 0.15  # -15%
    
    # Sharp-square divergence adjustments
    sharp_square_boost: float = 0.05     # +5% when aligned with sharps
    sharp_square_penalty: float = 0.05   # -5% when against sharps
    
    # Divergence thresholds
    spread_divergence_threshold: float = 0.5   # 0.5 pt = meaningful for spreads
    total_divergence_threshold: float = 1.0    # 1.0 pt = meaningful for totals
    
    # RLM (Reverse Line Movement) adjustments
    rlm_boost: float = 0.05              # +5% for RLM aligned
    
    # Steam move adjustments (large, fast line moves)
    steam_boost: float = 0.05            # +5% when aligned with steam
    steam_penalty: float = 0.08          # -8% when against steam
    steam_threshold_spread: float = 1.5  # 1.5 pt move = steam
    steam_threshold_total: float = 2.0   # 2.0 pt move = steam
    
    # Line movement adjustments (normal moves)
    move_boost: float = 0.03             # +3% when aligned
    move_penalty: float = 0.05           # -5% when against
    move_threshold_spread: float = 0.5   # 0.5 pt = significant
    move_threshold_total: float = 1.0    # 1.0 pt = significant


# Global config instance
SHARP_CONFIG = SharpConfig()


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SharpContext:
    """Sharp money context for a single game."""
    
    # Pinnacle (sharp book) lines
    pinnacle_spread: Optional[float] = None
    pinnacle_total: Optional[float] = None
    
    # Square book average (DraftKings, FanDuel, BetMGM)
    square_spread: Optional[float] = None
    square_total: Optional[float] = None
    
    # Opening lines (for line movement)
    spread_open: Optional[float] = None
    total_open: Optional[float] = None
    
    # Action Network betting splits (for RLM)
    spread_ticket_home_pct: Optional[float] = None
    spread_money_home_pct: Optional[float] = None
    total_ticket_over_pct: Optional[float] = None
    total_money_over_pct: Optional[float] = None
    
    # Current consensus lines
    consensus_spread: Optional[float] = None
    consensus_total: Optional[float] = None
    
    @property
    def has_pinnacle_spread(self) -> bool:
        return self.pinnacle_spread is not None
    
    @property
    def has_pinnacle_total(self) -> bool:
        return self.pinnacle_total is not None
    
    @property
    def spread_divergence(self) -> Optional[float]:
        """Spread difference between sharp and square books."""
        if self.pinnacle_spread is None:
            return None
        square = self.square_spread or self.consensus_spread
        if square is None:
            return None
        return self.pinnacle_spread - square
    
    @property
    def total_divergence(self) -> Optional[float]:
        """Total difference between sharp and square books."""
        if self.pinnacle_total is None:
            return None
        square = self.square_total or self.consensus_total
        if square is None:
            return None
        return self.pinnacle_total - square
    
    @property
    def spread_move(self) -> Optional[float]:
        """Line movement from open (positive = moved toward away)."""
        if self.spread_open is None or self.consensus_spread is None:
            return None
        return self.consensus_spread - self.spread_open
    
    @property
    def total_move(self) -> Optional[float]:
        """Line movement from open (positive = moved up)."""
        if self.total_open is None or self.consensus_total is None:
            return None
        return self.consensus_total - self.total_open


@dataclass
class SharpAdjustmentResult:
    """Result of sharp money adjustment."""
    original_confidence: float
    adjusted_confidence: float
    adjustments_applied: list
    sharp_aligned: bool
    rationale: list


# ============================================================================
# SHARP ALIGNMENT CHECKS
# ============================================================================

def _check_spread_sharp_alignment(
    bet_side: str,
    model_predicted_margin: float,
    consensus_spread: float,
    pinnacle_spread: Optional[float],
) -> Tuple[bool, str]:
    """
    Check if our spread bet aligns with Pinnacle direction.
    
    Args:
        bet_side: "home" or "away"
        model_predicted_margin: Model's predicted home margin
        consensus_spread: Consensus spread line (negative = home favored)
        pinnacle_spread: Pinnacle spread line
        
    Returns:
        (is_aligned, reason_str)
    """
    if pinnacle_spread is None:
        return True, "No Pinnacle spread available"
    
    # Model's view: Does model think home beats the spread?
    # predicted_margin > -spread means home covers
    model_home_covers = model_predicted_margin > (-consensus_spread)
    model_side = "home" if model_home_covers else "away"
    
    # Sharp's view: Does Pinnacle differ from consensus in our direction?
    # If Pinnacle has home at -4 and consensus at -3, Pinnacle likes home MORE
    sharp_diff = pinnacle_spread - consensus_spread
    # sharp_diff < 0 means Pinnacle moved home spread more negative = sharps like home
    sharp_favors_home = sharp_diff < -0.25  # Give some tolerance
    sharp_favors_away = sharp_diff > 0.25
    
    if bet_side == "home":
        if sharp_favors_away:
            return False, f"Betting HOME but Pinnacle favors AWAY (diff: {sharp_diff:+.1f})"
        elif sharp_favors_home:
            return True, f"Aligned with sharps: Pinnacle favors HOME (diff: {sharp_diff:+.1f})"
        else:
            return True, "Neutral - Pinnacle near consensus"
    else:  # away
        if sharp_favors_home:
            return False, f"Betting AWAY but Pinnacle favors HOME (diff: {sharp_diff:+.1f})"
        elif sharp_favors_away:
            return True, f"Aligned with sharps: Pinnacle favors AWAY (diff: {sharp_diff:+.1f})"
        else:
            return True, "Neutral - Pinnacle near consensus"


def _check_total_sharp_alignment(
    bet_side: str,  # "over" or "under"
    model_predicted_total: float,
    consensus_total: float,
    pinnacle_total: Optional[float],
) -> Tuple[bool, str]:
    """
    Check if our total bet aligns with Pinnacle direction.
    
    Args:
        bet_side: "over" or "under"
        model_predicted_total: Model's predicted total
        consensus_total: Consensus total line
        pinnacle_total: Pinnacle total line
        
    Returns:
        (is_aligned, reason_str)
    """
    if pinnacle_total is None:
        return True, "No Pinnacle total available"
    
    # Sharp's view: Higher Pinnacle total = sharps like OVER
    sharp_diff = pinnacle_total - consensus_total
    sharp_favors_over = sharp_diff > 0.5
    sharp_favors_under = sharp_diff < -0.5
    
    if bet_side == "over":
        if sharp_favors_under:
            return False, f"Betting OVER but Pinnacle favors UNDER (diff: {sharp_diff:+.1f})"
        elif sharp_favors_over:
            return True, f"Aligned with sharps: Pinnacle favors OVER (diff: {sharp_diff:+.1f})"
        else:
            return True, "Neutral - Pinnacle near consensus"
    else:  # under
        if sharp_favors_over:
            return False, f"Betting UNDER but Pinnacle favors OVER (diff: {sharp_diff:+.1f})"
        elif sharp_favors_under:
            return True, f"Aligned with sharps: Pinnacle favors UNDER (diff: {sharp_diff:+.1f})"
        else:
            return True, "Neutral - Pinnacle near consensus"


# ============================================================================
# SHARP-SQUARE DIVERGENCE DETECTION
# ============================================================================

def _detect_sharp_square_divergence(
    market_type: str,  # "spread" or "total"
    sharp_context: SharpContext,
    config: SharpConfig = SHARP_CONFIG,
) -> Tuple[bool, float, str]:
    """
    Detect when sharp books (Pinnacle) have moved but square books haven't.
    
    This is an alternative to public betting percentages - we infer sharp action
    from the difference between sharp and square lines.
    
    Returns:
        (divergence_detected, divergence_amount, direction_str)
    """
    if market_type == "spread":
        divergence = sharp_context.spread_divergence
        threshold = config.spread_divergence_threshold
        if divergence is None:
            return False, 0.0, "No data"
        
        detected = abs(divergence) >= threshold
        if divergence < 0:
            direction = "Sharps favor HOME"
        elif divergence > 0:
            direction = "Sharps favor AWAY"
        else:
            direction = "Neutral"
        return detected, divergence, direction
    
    else:  # total
        divergence = sharp_context.total_divergence
        threshold = config.total_divergence_threshold
        if divergence is None:
            return False, 0.0, "No data"
        
        detected = abs(divergence) >= threshold
        if divergence > 0:
            direction = "Sharps favor OVER"
        elif divergence < 0:
            direction = "Sharps favor UNDER"
        else:
            direction = "Neutral"
        return detected, divergence, direction


# ============================================================================
# RLM (REVERSE LINE MOVEMENT) DETECTION
# ============================================================================

def _detect_rlm_spread(
    sharp_context: SharpContext,
    public_threshold: float = 0.65,
) -> Tuple[bool, str]:
    """
    Detect reverse line movement on spread.
    
    RLM occurs when:
    - Public heavily on one side (>65% tickets)
    - But line moves TOWARD the public side (sharps betting other way)
    
    Returns:
        (rlm_detected, sharp_side_str)
    """
    pct_home = sharp_context.spread_ticket_home_pct
    move = sharp_context.spread_move
    
    if pct_home is None or move is None:
        return False, ""
    
    public_home = pct_home >= public_threshold * 100
    public_away = pct_home <= (1 - public_threshold) * 100
    
    if not (public_home or public_away):
        return False, ""
    
    # RLM: Public on home but line moving TOWARD home (more negative)
    # This means sharps are on AWAY, forcing books to move line
    if public_home and move < -0.5:
        return True, "away"
    
    # RLM: Public on away but line moving TOWARD away (less negative/more positive)
    if public_away and move > 0.5:
        return True, "home"
    
    return False, ""


def _detect_rlm_total(
    sharp_context: SharpContext,
    public_threshold: float = 0.65,
) -> Tuple[bool, str]:
    """
    Detect reverse line movement on total.
    
    Returns:
        (rlm_detected, sharp_side_str)
    """
    pct_over = sharp_context.total_ticket_over_pct
    move = sharp_context.total_move
    
    if pct_over is None or move is None:
        return False, ""
    
    public_over = pct_over >= public_threshold * 100
    public_under = pct_over <= (1 - public_threshold) * 100
    
    if not (public_over or public_under):
        return False, ""
    
    # RLM: Public on over but line moving DOWN
    if public_over and move < -1.0:
        return True, "under"
    
    # RLM: Public on under but line moving UP
    if public_under and move > 1.0:
        return True, "over"
    
    return False, ""


# ============================================================================
# MAIN ADJUSTMENT FUNCTIONS
# ============================================================================

def apply_sharp_adjustments_spread(
    prediction: Dict[str, Any],
    sharp_context: SharpContext,
    config: SharpConfig = SHARP_CONFIG,
) -> SharpAdjustmentResult:
    """
    Apply sharp money adjustments to spread prediction confidence.
    
    Args:
        prediction: Prediction dict from engine (must have bet_side, confidence, predicted_margin)
        sharp_context: Sharp money context
        config: Adjustment configuration
        
    Returns:
        SharpAdjustmentResult with adjusted confidence and rationale
    """
    original_confidence = prediction.get("confidence", 0.5)
    adjusted = original_confidence
    adjustments = []
    rationale = []
    sharp_aligned = True
    
    bet_side = prediction.get("bet_side", prediction.get("side", ""))
    predicted_margin = prediction.get("predicted_margin", 0)
    spread_line = prediction.get("spread_line", 0)
    
    # 1. SHARP ALIGNMENT CHECK (biggest impact: -15%)
    if sharp_context.has_pinnacle_spread:
        is_aligned, reason = _check_spread_sharp_alignment(
            bet_side=bet_side,
            model_predicted_margin=predicted_margin,
            consensus_spread=sharp_context.consensus_spread or spread_line,
            pinnacle_spread=sharp_context.pinnacle_spread,
        )
        
        if not is_aligned:
            penalty = config.against_sharp_penalty
            adjusted *= (1 - penalty)
            adjustments.append(f"Against sharp: -{penalty*100:.0f}%")
            rationale.append(f"⚠️ AGAINST SHARPS: {reason}")
            sharp_aligned = False
        else:
            rationale.append(f"✓ Sharp aligned: {reason}")
    
    # 2. SHARP-SQUARE DIVERGENCE
    divergence_detected, divergence_amt, direction = _detect_sharp_square_divergence(
        "spread", sharp_context, config
    )
    
    if divergence_detected:
        # Are we aligned with the sharp side?
        sharp_on_home = divergence_amt < 0
        aligned_with_divergence = (
            (bet_side == "home" and sharp_on_home) or
            (bet_side == "away" and not sharp_on_home)
        )
        
        if aligned_with_divergence:
            adjusted *= (1 + config.sharp_square_boost)
            adjustments.append(f"Sharp-square aligned: +{config.sharp_square_boost*100:.0f}%")
            rationale.append(f"✓ Sharp-square divergence ({divergence_amt:+.1f}): {direction}")
        else:
            adjusted *= (1 - config.sharp_square_penalty)
            adjustments.append(f"Sharp-square against: -{config.sharp_square_penalty*100:.0f}%")
            rationale.append(f"⚠️ Against sharp-square divergence: {direction}")
            sharp_aligned = False
    
    # 3. RLM CHECK
    rlm_detected, rlm_sharp_side = _detect_rlm_spread(sharp_context)
    
    if rlm_detected:
        if bet_side == rlm_sharp_side:
            adjusted *= (1 + config.rlm_boost)
            adjustments.append(f"RLM aligned: +{config.rlm_boost*100:.0f}%")
            rationale.append(f"✓ RLM detected: Sharps on {rlm_sharp_side.upper()}")
        else:
            # Don't double-penalize (sharp alignment already caught this)
            rationale.append(f"⚠️ RLM against us: Sharps on {rlm_sharp_side.upper()}")
    
    # 4. LINE MOVEMENT / STEAM
    move = sharp_context.spread_move
    if move is not None:
        move_magnitude = abs(move)
        
        # Determine if move is aligned with our bet
        # Positive move = line moved toward away = favors away bettors
        move_favors_away = move > 0
        aligned_with_move = (
            (bet_side == "away" and move_favors_away) or
            (bet_side == "home" and not move_favors_away)
        )
        
        # Steam move (large)
        if move_magnitude >= config.steam_threshold_spread:
            if aligned_with_move:
                adjusted *= (1 + config.steam_boost)
                adjustments.append(f"Steam aligned: +{config.steam_boost*100:.0f}%")
                rationale.append(f"✓ Steam move aligned ({move:+.1f} pts)")
            else:
                adjusted *= (1 - config.steam_penalty)
                adjustments.append(f"Steam against: -{config.steam_penalty*100:.0f}%")
                rationale.append(f"⚠️ Steam move against ({move:+.1f} pts)")
                
        # Normal significant move
        elif move_magnitude >= config.move_threshold_spread:
            if aligned_with_move:
                adjusted *= (1 + config.move_boost)
                adjustments.append(f"Move aligned: +{config.move_boost*100:.0f}%")
            else:
                adjusted *= (1 - config.move_penalty)
                adjustments.append(f"Move against: -{config.move_penalty*100:.0f}%")
    
    # Clamp confidence
    adjusted = min(0.99, max(0.01, adjusted))
    
    return SharpAdjustmentResult(
        original_confidence=original_confidence,
        adjusted_confidence=adjusted,
        adjustments_applied=adjustments,
        sharp_aligned=sharp_aligned,
        rationale=rationale,
    )


def apply_sharp_adjustments_total(
    prediction: Dict[str, Any],
    sharp_context: SharpContext,
    config: SharpConfig = SHARP_CONFIG,
) -> SharpAdjustmentResult:
    """
    Apply sharp money adjustments to total prediction confidence.
    
    Args:
        prediction: Prediction dict from engine (must have bet_side, confidence, predicted_total)
        sharp_context: Sharp money context
        config: Adjustment configuration
        
    Returns:
        SharpAdjustmentResult with adjusted confidence and rationale
    """
    original_confidence = prediction.get("confidence", 0.5)
    adjusted = original_confidence
    adjustments = []
    rationale = []
    sharp_aligned = True
    
    bet_side = prediction.get("bet_side", prediction.get("side", ""))
    predicted_total = prediction.get("predicted_total", 0)
    total_line = prediction.get("total_line", 0)
    
    # 1. SHARP ALIGNMENT CHECK
    if sharp_context.has_pinnacle_total:
        is_aligned, reason = _check_total_sharp_alignment(
            bet_side=bet_side,
            model_predicted_total=predicted_total,
            consensus_total=sharp_context.consensus_total or total_line,
            pinnacle_total=sharp_context.pinnacle_total,
        )
        
        if not is_aligned:
            penalty = config.against_sharp_penalty
            adjusted *= (1 - penalty)
            adjustments.append(f"Against sharp: -{penalty*100:.0f}%")
            rationale.append(f"⚠️ AGAINST SHARPS: {reason}")
            sharp_aligned = False
        else:
            rationale.append(f"✓ Sharp aligned: {reason}")
    
    # 2. SHARP-SQUARE DIVERGENCE
    divergence_detected, divergence_amt, direction = _detect_sharp_square_divergence(
        "total", sharp_context, config
    )
    
    if divergence_detected:
        # Are we aligned with the sharp side?
        sharp_on_over = divergence_amt > 0
        aligned_with_divergence = (
            (bet_side == "over" and sharp_on_over) or
            (bet_side == "under" and not sharp_on_over)
        )
        
        if aligned_with_divergence:
            adjusted *= (1 + config.sharp_square_boost)
            adjustments.append(f"Sharp-square aligned: +{config.sharp_square_boost*100:.0f}%")
            rationale.append(f"✓ Sharp-square divergence ({divergence_amt:+.1f}): {direction}")
        else:
            adjusted *= (1 - config.sharp_square_penalty)
            adjustments.append(f"Sharp-square against: -{config.sharp_square_penalty*100:.0f}%")
            rationale.append(f"⚠️ Against sharp-square divergence: {direction}")
            sharp_aligned = False
    
    # 3. RLM CHECK
    rlm_detected, rlm_sharp_side = _detect_rlm_total(sharp_context)
    
    if rlm_detected:
        if bet_side == rlm_sharp_side:
            adjusted *= (1 + config.rlm_boost)
            adjustments.append(f"RLM aligned: +{config.rlm_boost*100:.0f}%")
            rationale.append(f"✓ RLM detected: Sharps on {rlm_sharp_side.upper()}")
        else:
            rationale.append(f"⚠️ RLM against us: Sharps on {rlm_sharp_side.upper()}")
    
    # 4. LINE MOVEMENT / STEAM
    move = sharp_context.total_move
    if move is not None:
        move_magnitude = abs(move)
        
        # Positive move = line went up = favors over bettors
        move_favors_over = move > 0
        aligned_with_move = (
            (bet_side == "over" and move_favors_over) or
            (bet_side == "under" and not move_favors_over)
        )
        
        # Steam move
        if move_magnitude >= config.steam_threshold_total:
            if aligned_with_move:
                adjusted *= (1 + config.steam_boost)
                adjustments.append(f"Steam aligned: +{config.steam_boost*100:.0f}%")
                rationale.append(f"✓ Steam move aligned ({move:+.1f} pts)")
            else:
                adjusted *= (1 - config.steam_penalty)
                adjustments.append(f"Steam against: -{config.steam_penalty*100:.0f}%")
                rationale.append(f"⚠️ Steam move against ({move:+.1f} pts)")
                
        # Normal significant move
        elif move_magnitude >= config.move_threshold_total:
            if aligned_with_move:
                adjusted *= (1 + config.move_boost)
                adjustments.append(f"Move aligned: +{config.move_boost*100:.0f}%")
            else:
                adjusted *= (1 - config.move_penalty)
                adjustments.append(f"Move against: -{config.move_penalty*100:.0f}%")
    
    # Clamp confidence
    adjusted = min(0.99, max(0.01, adjusted))
    
    return SharpAdjustmentResult(
        original_confidence=original_confidence,
        adjusted_confidence=adjusted,
        adjustments_applied=adjustments,
        sharp_aligned=sharp_aligned,
        rationale=rationale,
    )


# ============================================================================
# HELPER: Build SharpContext from features dict
# ============================================================================

def build_sharp_context_from_features(
    features: Dict[str, Any],
    spread_line: Optional[float] = None,
    total_line: Optional[float] = None,
) -> SharpContext:
    """
    Build SharpContext from prediction features dict.
    
    Expects features to contain:
    - pinnacle_spread, pinnacle_total (from fetch_sharp_square_lines)
    - square_spread_avg, square_total_avg (from fetch_sharp_square_lines)
    - spread_ticket_home_pct, spread_money_home_pct (from Action Network)
    - total_ticket_over_pct, total_money_over_pct (from Action Network)
    """
    return SharpContext(
        # Pinnacle
        pinnacle_spread=features.get("pinnacle_spread"),
        pinnacle_total=features.get("pinnacle_total"),
        
        # Square books
        square_spread=features.get("square_spread_avg"),
        square_total=features.get("square_total_avg"),
        
        # Opening lines (if available)
        spread_open=features.get("spread_open"),
        total_open=features.get("total_open"),
        
        # Action Network betting splits
        spread_ticket_home_pct=features.get("spread_home_ticket_pct"),
        spread_money_home_pct=features.get("spread_home_money_pct"),
        total_ticket_over_pct=features.get("total_over_ticket_pct"),
        total_money_over_pct=features.get("total_over_money_pct"),
        
        # Consensus (fallback to line args)
        consensus_spread=features.get("consensus_spread", spread_line),
        consensus_total=features.get("consensus_total", total_line),
    )
