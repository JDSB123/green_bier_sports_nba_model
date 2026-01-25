"""
Sharp Money Weighted Combination System (v34.3.0).

ARCHITECTURE:
This module combines ML model predictions with ALL AVAILABLE sharp money signals:
1. Pinnacle vs Square divergence (primary sharp signal)
2. Reverse Line Movement (RLM) - ticket % vs money %
3. Line Movement (steam moves from open)
4. Money vs Ticket divergence magnitude

FORMULA:
    sharp_edge = (
        pinnacle_component × 0.50 +   # Primary: Pinnacle divergence
        rlm_component × 0.25 +        # Secondary: RLM detection
        line_move_component × 0.15 + # Tertiary: Steam/line movement
        money_ticket_mag × 0.10      # Quaternary: Money/ticket divergence magnitude
    )

    combined_edge = (model_edge × MODEL_WEIGHT) + (sharp_edge × SHARP_WEIGHT)

    Where:
        MODEL_WEIGHT = 0.55 (model foundation)
        SHARP_WEIGHT = 0.45 (sharp signals are MAXIMIZED)

    If combined_edge > threshold → Bet that side
    If combined_edge < -threshold → Bet opposite side
    If |combined_edge| < threshold → NO PLAY (signals cancel)

SHARP SIGNAL BREAKDOWN:
1. PINNACLE DIVERGENCE (50% of sharp_edge):
   - Pinnacle line vs average of square books (DK, FD, MGM)
   - If Pinnacle more negative on spread → sharps like HOME
   - If Pinnacle higher on total → sharps like OVER

2. REVERSE LINE MOVEMENT (25% of sharp_edge):
   - When line moves AGAINST public ticket %
   - 70% tickets on HOME but line moves toward AWAY = SHARP AWAY
   - Calculated as: (money_pct - ticket_pct) / 100 scaled to points

3. LINE MOVEMENT / STEAM (15% of sharp_edge):
   - Significant move from opening line
   - Steam = rapid move in one direction
   - Measured as: current_line - opening_line

4. MONEY vs TICKET MAGNITUDE (10% of sharp_edge):
   - Pure magnitude of money/ticket split
   - Large divergence = strong sharp action regardless of direction
   - Boosts confidence when sharps are heavily positioned

KEY INSIGHT:
v34.2.0: Only used Pinnacle divergence (40% of potential signal strength)
v34.3.0: Uses ALL 4 signals (100% of potential signal strength) ← MAXIMUM
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
    model_weight: float = 0.55  # Reduced from 0.60 to give sharps more weight
    sharp_weight: float = 0.45  # Increased from 0.40 - MAXIMIZE sharp signals

    # Sharp signal component weights (must sum to 1.0)
    pinnacle_weight: float = 0.50    # Primary: Pinnacle divergence
    rlm_weight: float = 0.25         # Secondary: Reverse Line Movement
    line_move_weight: float = 0.15   # Tertiary: Steam/line movement
    money_ticket_weight: float = 0.10  # Quaternary: Money/ticket magnitude

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

    # RLM detection thresholds
    rlm_threshold: float = 10.0   # Min % difference for RLM (money - ticket)
    steam_threshold: float = 1.0  # Min line move for steam detection (points)


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
    # Money % on one side (RLM when differs from ticket)
    money_pct: Optional[float] = None

    # Line movement
    opening_line: Optional[float] = None
    current_line: Optional[float] = None
    # Detection thresholds
    rlm_threshold: float = WEIGHTED_CONFIG.rlm_threshold
    steam_threshold: float = WEIGHTED_CONFIG.steam_threshold

    @property
    def has_pinnacle(self) -> bool:
        return self.pinnacle_line is not None and self.square_line is not None

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
        return abs(self.ticket_pct - self.money_pct) > self.rlm_threshold

    @property
    def rlm_direction(self) -> float:
        """
        RLM direction as points-equivalent edge.
        Positive = sharp money favoring side (higher money % relative to tickets)
        Example: 30% tickets but 60% money → +0.30 edge toward that side
        """
        if self.ticket_pct is None or self.money_pct is None:
            return 0.0
        # Scale: (money - ticket) / 100, then multiply by 3 to get points-equivalent
        # 30% difference → 0.9 pts edge
        return ((self.money_pct - self.ticket_pct) / 100.0) * 3.0

    @property
    def money_ticket_magnitude(self) -> float:
        """
        Magnitude of money vs ticket split (absolute value).
        Large divergence = strong sharp positioning regardless of direction.
        Returns 0-1 scale based on divergence.
        """
        if self.ticket_pct is None or self.money_pct is None:
            return 0.0
        divergence = abs(self.money_pct - self.ticket_pct)
        # Scale: 0% = 0, 50% = 1.0 (max)
        return min(1.0, divergence / 50.0)

    @property
    def has_steam(self) -> bool:
        """True if significant line movement detected (potential steam)."""
        if self.line_move is None:
            return False
        return abs(self.line_move) >= self.steam_threshold

    @property
    def steam_edge(self) -> float:
        """
        Steam move edge in points.
        Line moving toward a side = sharps driving it that way.
        For spreads: Negative move (more negative) = sharps like HOME
        For totals: Positive move = sharps like OVER
        """
        if self.line_move is None:
            return 0.0
        # Cap at +/- 3 points to prevent outliers from dominating
        return max(-3.0, min(3.0, self.line_move))


@dataclass
class WeightedResult:
    """Result of weighted combination calculation."""

    # Core outputs
    final_side: str           # "home", "away", "over", "under", or "NO_PLAY"
    final_confidence: float   # Combined confidence (0.0-1.0)
    combined_edge: float      # Combined edge in points
    is_play: bool             # False if signals cancel out

    # Model component
    model_side: str           # What model originally said
    model_edge: float         # Model's edge in points

    # Sharp composite components (NEW in v34.3.0)
    # Total sharp edge (weighted sum of all components)
    sharp_edge: float
    pinnacle_component: float  # Pinnacle divergence contribution
    rlm_component: float      # RLM contribution
    line_move_component: float  # Steam/line move contribution
    money_ticket_component: float  # Money/ticket magnitude contribution

    # Sharp direction
    sharp_side: str           # What sharps say overall

    # Signal alignment
    signals_agree: bool       # True if model and sharps on same side
    side_flipped: bool        # True if final side differs from model

    # Tracking which signals fired
    has_pinnacle: bool
    has_rlm: bool
    has_steam: bool

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
    Combine ML model spread prediction with ALL sharp money signals.

    MAXIMIZES signal integration (v34.3.0):
    - Pinnacle divergence (50% of sharp weight)
    - Reverse Line Movement (25% of sharp weight)
    - Steam/line movement (15% of sharp weight)
    - Money vs Ticket magnitude (10% of sharp weight)

    Args:
        model_predicted_margin: Model's predicted home margin (positive = home wins by X)
        market_spread: Current market spread line (negative = home favored)
        sharp_signal: Sharp money signal data
        config: Weighting configuration

    Returns:
        WeightedResult with final side and confidence
    """
    rationale = []

    # =========================================================================
    # STEP 1: Calculate model edge
    # =========================================================================
    model_edge = model_predicted_margin - (-market_spread)
    model_side = "home" if model_edge > 0 else "away"
    rationale.append(
        f"MODEL: {model_side.upper()} by {abs(model_edge):.1f} pts")

    # =========================================================================
    # STEP 2: Calculate ALL sharp signal components
    # =========================================================================

    # Component 1: Pinnacle divergence (primary - 50%)
    pinnacle_component = 0.0
    if sharp_signal.has_pinnacle and sharp_signal.divergence is not None:
        divergence = sharp_signal.divergence
        # Negative divergence = Pinnacle more negative = sharps like HOME
        # Positive divergence = Pinnacle less negative = sharps like AWAY
        if abs(divergence) > 0.25:
            # Flip sign: neg divergence → positive (home)
            pinnacle_component = -divergence
            rationale.append(
                f"PINNACLE: {pinnacle_component:+.2f} edge (divergence {divergence:+.1f})")

    # Component 2: Reverse Line Movement (25%)
    rlm_component = 0.0
    if sharp_signal.has_rlm:
        rlm_component = sharp_signal.rlm_direction
        rlm_side = "HOME" if rlm_component > 0 else "AWAY"
        rationale.append(
            f"RLM: {rlm_component:+.2f} edge → {rlm_side} (money {sharp_signal.money_pct:.0f}% vs ticket {sharp_signal.ticket_pct:.0f}%)")

    # Component 3: Line movement / Steam (15%)
    line_move_component = 0.0
    if sharp_signal.has_steam:
        # For spreads: line moving more negative = sharps driving toward HOME
        # Flip: neg move → positive (home)
        line_move_component = -sharp_signal.steam_edge
        steam_side = "HOME" if line_move_component > 0 else "AWAY"
        rationale.append(
            f"STEAM: {line_move_component:+.2f} edge → {steam_side} (line moved {sharp_signal.line_move:+.1f})")

    # Component 4: Money vs Ticket magnitude (10%)
    # This boosts conviction when sharps are heavily positioned
    money_ticket_component = 0.0
    if sharp_signal.money_ticket_magnitude > 0.2:  # Only if significant divergence
        # Direction follows RLM direction, magnitude from split
        if rlm_component != 0:
            direction = 1.0 if rlm_component > 0 else -1.0
        elif pinnacle_component != 0:
            direction = 1.0 if pinnacle_component > 0 else -1.0
        else:
            direction = 0.0
        money_ticket_component = direction * \
            sharp_signal.money_ticket_magnitude * 2.0  # Scale to points
        if abs(money_ticket_component) > 0.1:
            rationale.append(
                f"MONEY/TICKET MAG: {money_ticket_component:+.2f} edge (split: {sharp_signal.money_ticket_magnitude:.1%})")

    # =========================================================================
    # STEP 3: Combine sharp components into weighted sharp_edge
    # =========================================================================
    sharp_edge = (
        pinnacle_component * config.pinnacle_weight +
        rlm_component * config.rlm_weight +
        line_move_component * config.line_move_weight +
        money_ticket_component * config.money_ticket_weight
    )

    sharp_side = "home" if sharp_edge > 0.1 else (
        "away" if sharp_edge < -0.1 else "neutral")

    signal_count = sum([
        1 if sharp_signal.has_pinnacle else 0,
        1 if sharp_signal.has_rlm else 0,
        1 if sharp_signal.has_steam else 0,
    ])
    rationale.append(
        f"SHARP TOTAL: {sharp_edge:+.2f} edge → {sharp_side.upper()} ({signal_count}/3 signals active)")

    # =========================================================================
    # STEP 4: Combine model + sharp into final edge
    # =========================================================================
    if signal_count > 0:
        combined_edge = (model_edge * config.model_weight) + \
            (sharp_edge * config.sharp_weight)
        rationale.append(
            f"COMBINED: {combined_edge:+.2f} = ({model_edge:.1f} × {config.model_weight}) + ({sharp_edge:.2f} × {config.sharp_weight})")
    else:
        # No sharp data - use model alone with haircut
        combined_edge = model_edge * 0.75
        rationale.append(
            f"COMBINED: {combined_edge:+.2f} (model only with 25% haircut - no sharp data)")

    # =========================================================================
    # STEP 5: Determine final side and if it's a play
    # =========================================================================
    signals_agree = (model_edge > 0 and sharp_edge >= 0) or (
        model_edge < 0 and sharp_edge <= 0)

    if abs(combined_edge) < config.min_edge_spread:
        final_side = "NO_PLAY"
        is_play = False
        final_confidence = 0.0
        rationale.append(
            f"❌ NO PLAY: Combined edge {combined_edge:+.2f} < min {config.min_edge_spread}")
    else:
        final_side = "home" if combined_edge > 0 else "away"
        is_play = True

        raw_confidence = config.base_confidence + \
            (abs(combined_edge) * config.confidence_scale_spread)
        final_confidence = min(config.max_confidence, max(
            config.min_confidence, raw_confidence))

        if signals_agree:
            rationale.append(
                f"✓ SIGNALS AGREE: {final_side.upper()} ({final_confidence:.1%} confidence)")
        else:
            rationale.append(
                f"⚠️ SIGNALS CONFLICT: Combined favors {final_side.upper()} ({final_confidence:.1%})")

    side_flipped = (model_side != final_side) and (final_side != "NO_PLAY")
    if side_flipped:
        rationale.append(
            f"🔄 SIDE FLIPPED: Model said {model_side.upper()}, now {final_side.upper()}")

    return WeightedResult(
        final_side=final_side,
        final_confidence=final_confidence,
        combined_edge=combined_edge,
        is_play=is_play,
        model_side=model_side,
        model_edge=model_edge,
        sharp_edge=sharp_edge,
        pinnacle_component=pinnacle_component,
        rlm_component=rlm_component,
        line_move_component=line_move_component,
        money_ticket_component=money_ticket_component,
        sharp_side=sharp_side,
        signals_agree=signals_agree,
        side_flipped=side_flipped,
        has_pinnacle=sharp_signal.has_pinnacle,
        has_rlm=sharp_signal.has_rlm,
        has_steam=sharp_signal.has_steam,
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
    Combine ML model total prediction with ALL sharp money signals.

    MAXIMIZES signal integration (v34.3.0):
    - Pinnacle divergence (50% of sharp weight)
    - Reverse Line Movement (25% of sharp weight)
    - Steam/line movement (15% of sharp weight)
    - Money vs Ticket magnitude (10% of sharp weight)

    Args:
        model_predicted_total: Model's predicted total points
        market_total: Current market total line
        sharp_signal: Sharp money signal data
        config: Weighting configuration

    Returns:
        WeightedResult with final side and confidence
    """
    rationale = []

    # =========================================================================
    # STEP 1: Calculate model edge
    # =========================================================================
    model_edge = model_predicted_total - market_total
    model_side = "over" if model_edge > 0 else "under"
    rationale.append(
        f"MODEL: {model_side.upper()} by {abs(model_edge):.1f} pts")

    # =========================================================================
    # STEP 2: Calculate ALL sharp signal components
    # =========================================================================

    # Component 1: Pinnacle divergence (primary - 50%)
    pinnacle_component = 0.0
    if sharp_signal.has_pinnacle and sharp_signal.divergence is not None:
        divergence = sharp_signal.divergence
        # Positive divergence = Pinnacle higher = sharps like OVER
        # Negative divergence = Pinnacle lower = sharps like UNDER
        if abs(divergence) > 0.5:
            pinnacle_component = divergence  # Positive = over
            rationale.append(
                f"PINNACLE: {pinnacle_component:+.2f} edge (divergence {divergence:+.1f})")

    # Component 2: Reverse Line Movement (25%)
    rlm_component = 0.0
    if sharp_signal.has_rlm:
        rlm_component = sharp_signal.rlm_direction
        rlm_side = "OVER" if rlm_component > 0 else "UNDER"
        rationale.append(
            f"RLM: {rlm_component:+.2f} edge → {rlm_side} (money {sharp_signal.money_pct:.0f}% vs ticket {sharp_signal.ticket_pct:.0f}%)")

    # Component 3: Line movement / Steam (15%)
    line_move_component = 0.0
    if sharp_signal.has_steam:
        # For totals: line moving higher = sharps driving toward OVER
        line_move_component = sharp_signal.steam_edge  # Positive = over
        steam_side = "OVER" if line_move_component > 0 else "UNDER"
        rationale.append(
            f"STEAM: {line_move_component:+.2f} edge → {steam_side} (line moved {sharp_signal.line_move:+.1f})")

    # Component 4: Money vs Ticket magnitude (10%)
    money_ticket_component = 0.0
    if sharp_signal.money_ticket_magnitude > 0.2:
        if rlm_component != 0:
            direction = 1.0 if rlm_component > 0 else -1.0
        elif pinnacle_component != 0:
            direction = 1.0 if pinnacle_component > 0 else -1.0
        else:
            direction = 0.0
        money_ticket_component = direction * sharp_signal.money_ticket_magnitude * 2.0
        if abs(money_ticket_component) > 0.1:
            rationale.append(
                f"MONEY/TICKET MAG: {money_ticket_component:+.2f} edge (split: {sharp_signal.money_ticket_magnitude:.1%})")

    # =========================================================================
    # STEP 3: Combine sharp components into weighted sharp_edge
    # =========================================================================
    sharp_edge = (
        pinnacle_component * config.pinnacle_weight +
        rlm_component * config.rlm_weight +
        line_move_component * config.line_move_weight +
        money_ticket_component * config.money_ticket_weight
    )

    sharp_side = "over" if sharp_edge > 0.2 else (
        "under" if sharp_edge < -0.2 else "neutral")

    signal_count = sum([
        1 if sharp_signal.has_pinnacle else 0,
        1 if sharp_signal.has_rlm else 0,
        1 if sharp_signal.has_steam else 0,
    ])
    rationale.append(
        f"SHARP TOTAL: {sharp_edge:+.2f} edge → {sharp_side.upper()} ({signal_count}/3 signals active)")

    # =========================================================================
    # STEP 4: Combine model + sharp into final edge
    # =========================================================================
    if signal_count > 0:
        combined_edge = (model_edge * config.model_weight) + \
            (sharp_edge * config.sharp_weight)
        rationale.append(
            f"COMBINED: {combined_edge:+.2f} = ({model_edge:.1f} × {config.model_weight}) + ({sharp_edge:.2f} × {config.sharp_weight})")
    else:
        combined_edge = model_edge * 0.75
        rationale.append(
            f"COMBINED: {combined_edge:+.2f} (model only with 25% haircut - no sharp data)")

    # =========================================================================
    # STEP 5: Determine final side and if it's a play
    # =========================================================================
    signals_agree = (model_edge > 0 and sharp_edge >= 0) or (
        model_edge < 0 and sharp_edge <= 0)

    if abs(combined_edge) < config.min_edge_total:
        final_side = "NO_PLAY"
        is_play = False
        final_confidence = 0.0
        rationale.append(
            f"❌ NO PLAY: Combined edge {combined_edge:+.2f} < min {config.min_edge_total}")
    else:
        final_side = "over" if combined_edge > 0 else "under"
        is_play = True

        raw_confidence = config.base_confidence + \
            (abs(combined_edge) * config.confidence_scale_total)
        final_confidence = min(config.max_confidence, max(
            config.min_confidence, raw_confidence))

        if signals_agree:
            rationale.append(
                f"✓ SIGNALS AGREE: {final_side.upper()} ({final_confidence:.1%} confidence)")
        else:
            rationale.append(
                f"⚠️ SIGNALS CONFLICT: Combined favors {final_side.upper()} ({final_confidence:.1%})")

    side_flipped = (model_side != final_side) and (final_side != "NO_PLAY")
    if side_flipped:
        rationale.append(
            f"🔄 SIDE FLIPPED: Model said {model_side.upper()}, now {final_side.upper()}")

    return WeightedResult(
        final_side=final_side,
        final_confidence=final_confidence,
        combined_edge=combined_edge,
        is_play=is_play,
        model_side=model_side,
        model_edge=model_edge,
        sharp_edge=sharp_edge,
        pinnacle_component=pinnacle_component,
        rlm_component=rlm_component,
        line_move_component=line_move_component,
        money_ticket_component=money_ticket_component,
        sharp_side=sharp_side,
        signals_agree=signals_agree,
        side_flipped=side_flipped,
        has_pinnacle=sharp_signal.has_pinnacle,
        has_rlm=sharp_signal.has_rlm,
        has_steam=sharp_signal.has_steam,
        rationale=rationale,
    )


# ============================================================================
# HELPER: Build SharpSignal from features dict
# ============================================================================

def _first_present(features: Dict[str, Any], *keys: str) -> Optional[float]:
    """Return the first non-None value for any of the provided keys."""
    for key in keys:
        if key in features and features.get(key) is not None:
            return features.get(key)
    return None


def build_sharp_signal_spread(
    features: Dict[str, Any],
    market_spread: Optional[float] = None,
    rlm_threshold: float = WEIGHTED_CONFIG.rlm_threshold,
    steam_threshold: float = WEIGHTED_CONFIG.steam_threshold,
) -> SharpSignal:
    """Build SharpSignal for spread from features dictionary."""
    square_line = _first_present(features, "square_spread_avg", "square_avg_spread")
    if square_line is None:
        square_line = market_spread
    current_line = market_spread if market_spread is not None else _first_present(
        features, "spread_current"
    )
    return SharpSignal(
        pinnacle_line=_first_present(features, "pinnacle_spread"),
        square_line=square_line,
        ticket_pct=_first_present(
            features,
            "spread_public_home_pct",
            "spread_ticket_home_pct",
            "spread_home_ticket_pct",
        ),
        money_pct=_first_present(
            features,
            "spread_money_home_pct",
            "spread_home_money_pct",
        ),
        opening_line=_first_present(features, "spread_open"),
        current_line=current_line,
        rlm_threshold=rlm_threshold,
        steam_threshold=steam_threshold,
    )


def build_sharp_signal_total(
    features: Dict[str, Any],
    market_total: Optional[float] = None,
    rlm_threshold: float = WEIGHTED_CONFIG.rlm_threshold,
    steam_threshold: float = WEIGHTED_CONFIG.steam_threshold,
) -> SharpSignal:
    """Build SharpSignal for total from features dictionary."""
    square_line = _first_present(features, "square_total_avg", "square_avg_total")
    if square_line is None:
        square_line = market_total
    current_line = market_total if market_total is not None else _first_present(
        features, "total_current"
    )
    return SharpSignal(
        pinnacle_line=_first_present(features, "pinnacle_total"),
        square_line=square_line,
        ticket_pct=_first_present(
            features,
            "over_public_pct",
            "total_ticket_over_pct",
            "over_ticket_pct",
        ),
        money_pct=_first_present(
            features,
            "over_money_pct",
            "total_money_over_pct",
        ),
        opening_line=_first_present(features, "total_open"),
        current_line=current_line,
        rlm_threshold=rlm_threshold,
        steam_threshold=steam_threshold,
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

    This MAXIMIZES sharp signal integration (v34.3.0):
    - Uses Pinnacle divergence (50%)
    - Uses RLM from ticket/money % (25%)
    - Uses Steam from line movement (15%)
    - Uses Money/Ticket magnitude (10%)

    Args:
        prediction: Original prediction from ML engine
        features: Features dict (contains Pinnacle data)
        market_spread: Current market spread
        config: Weighting configuration

    Returns:
        (updated_prediction, weighted_result)
    """
    model_margin = prediction.get("predicted_margin", 0)
    spread = market_spread if market_spread is not None else prediction.get("spread_line")
    if spread is None:
        spread = 0

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

    # NEW v34.3.0: Component breakdown
    updated["pinnacle_component"] = result.pinnacle_component
    updated["rlm_component"] = result.rlm_component
    updated["line_move_component"] = result.line_move_component
    updated["money_ticket_component"] = result.money_ticket_component
    updated["has_pinnacle"] = result.has_pinnacle
    updated["has_rlm"] = result.has_rlm
    updated["has_steam"] = result.has_steam

    return updated, result


def apply_weighted_combination_total(
    prediction: Dict[str, Any],
    features: Dict[str, Any],
    market_total: Optional[float] = None,
    config: WeightedConfig = WEIGHTED_CONFIG,
) -> Tuple[Dict[str, Any], WeightedResult]:
    """
    Apply weighted combination to total prediction.

    This MAXIMIZES sharp signal integration (v34.3.0):
    - Uses Pinnacle divergence (50%)
    - Uses RLM from ticket/money % (25%)
    - Uses Steam from line movement (15%)
    - Uses Money/Ticket magnitude (10%)

    Args:
        prediction: Original prediction from ML engine
        features: Features dict (contains Pinnacle data)
        market_total: Current market total
        config: Weighting configuration

    Returns:
        (updated_prediction, weighted_result)
    """
    model_total = prediction.get("predicted_total", 0)
    total = market_total if market_total is not None else prediction.get("total_line")
    if total is None:
        total = 0

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

    # NEW v34.3.0: Component breakdown
    updated["pinnacle_component"] = result.pinnacle_component
    updated["rlm_component"] = result.rlm_component
    updated["line_move_component"] = result.line_move_component
    updated["money_ticket_component"] = result.money_ticket_component
    updated["has_pinnacle"] = result.has_pinnacle
    updated["has_rlm"] = result.has_rlm
    updated["has_steam"] = result.has_steam

    return updated, result
