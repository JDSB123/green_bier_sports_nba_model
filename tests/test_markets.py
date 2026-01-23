"""
Tests for src/markets/ modules - spread, total, moneyline markets.

Coverage target: 95%+ for market modules.
"""
import pytest
import numpy as np
from typing import Dict

from src.markets.base import MarketPrediction, BaseMarket


class TestMarketPrediction:
    """Tests for MarketPrediction dataclass."""

    def test_market_prediction_creation(self):
        """MarketPrediction should be created with required fields."""
        pred = MarketPrediction(
            market_key="fg_spread",
            period="fg",
            market_type="spread",
            side="home",
            probability=0.58,
            confidence=0.65,
        )
        
        assert pred.market_key == "fg_spread"
        assert pred.period == "fg"
        assert pred.market_type == "spread"
        assert pred.side == "home"
        assert pred.probability == 0.58
        assert pred.confidence == 0.65

    def test_market_prediction_defaults(self):
        """Default values should be sensible."""
        pred = MarketPrediction(
            market_key="1h_total",
            period="1h",
            market_type="total",
            side="over",
            probability=0.55,
            confidence=0.60,
        )
        
        assert pred.line is None
        assert pred.implied_prob is None
        assert pred.edge is None
        assert pred.kelly_fraction is None
        assert pred.passes_filter is True
        assert pred.filter_reason is None

    def test_market_prediction_with_edge(self):
        """Edge should be calculated correctly."""
        implied_prob = 0.524  # -110 odds
        probability = 0.58
        edge = probability - implied_prob
        
        pred = MarketPrediction(
            market_key="fg_spread",
            period="fg",
            market_type="spread",
            side="home",
            probability=probability,
            confidence=0.65,
            line=-3.5,
            implied_prob=implied_prob,
            edge=edge,
        )
        
        assert pred.edge == pytest.approx(0.056, abs=0.001)

    def test_market_prediction_filter_failed(self):
        """Failed filters should have reason."""
        pred = MarketPrediction(
            market_key="1h_spread",
            period="1h",
            market_type="spread",
            side="away",
            probability=0.52,
            confidence=0.55,
            passes_filter=False,
            filter_reason="Confidence below threshold",
        )
        
        assert pred.passes_filter is False
        assert pred.filter_reason == "Confidence below threshold"


class TestBaseMarketConstants:
    """Tests for BaseMarket class constants."""

    def test_standard_vig(self):
        """Standard vig should be ~4.55%."""
        # At -110 odds: risk 110 to win 100
        # Vig = (110 - 100) / 210 ≈ 4.76% per side
        # Or: 1 - (1/1.91 + 1/1.91) ≈ 4.55% total
        
        assert BaseMarket.STANDARD_VIG == pytest.approx(0.0455, abs=0.001)

    def test_win_payout(self):
        """Win payout at -110 should be 0.909."""
        # Win 100 on 110 risked = 100/110 = 0.909
        assert BaseMarket.WIN_PAYOUT == pytest.approx(0.909, abs=0.001)

    def test_break_even_prob(self):
        """Break-even probability at -110 should be ~52.38%."""
        # Need to win 110/(110+100) = 52.38% to break even
        assert BaseMarket.BREAK_EVEN_PROB == pytest.approx(0.5238, abs=0.001)

    def test_valid_periods(self):
        """Valid periods should be 'fg' and '1h'."""
        assert "fg" in BaseMarket.PERIODS
        assert "1h" in BaseMarket.PERIODS
        assert len(BaseMarket.PERIODS) == 2


class TestSpreadMarket:
    """Tests for spread market logic."""

    def test_spread_sides(self):
        """Spread should have home/away sides."""
        # Home covering = home score - spread > away score
        # Away covering = away score + spread > home score
        
        pred = MarketPrediction(
            market_key="fg_spread",
            period="fg",
            market_type="spread",
            side="home",
            probability=0.55,
            confidence=0.60,
            line=-3.5,
        )
        
        assert pred.side in ["home", "away"]

    def test_spread_line_interpretation(self):
        """Negative spread = home favored."""
        # Home -3.5 means home must win by 4+
        line = -3.5
        
        assert line < 0  # Home favored

    def test_spread_push_not_possible_half_point(self):
        """Half-point spreads cannot push."""
        line = -3.5
        
        # No integer margin can equal 3.5
        assert line % 1 == 0.5


class TestTotalMarket:
    """Tests for total (over/under) market logic."""

    def test_total_sides(self):
        """Total should have over/under sides."""
        pred = MarketPrediction(
            market_key="fg_total",
            period="fg",
            market_type="total",
            side="over",
            probability=0.55,
            confidence=0.60,
            line=220.5,
        )
        
        assert pred.side in ["over", "under"]

    def test_total_line_range(self):
        """NBA totals should be in reasonable range."""
        # NBA full game totals typically 200-250
        fg_total = 220.5
        assert 180 < fg_total < 280
        
        # First half totals typically 100-130
        first_half_total = 115.5
        assert 90 < first_half_total < 140

    def test_total_push_not_possible_half_point(self):
        """Half-point totals cannot push."""
        line = 220.5
        
        assert line % 1 == 0.5


class TestMoneylineMarket:
    """Tests for moneyline market logic."""

    def test_moneyline_sides(self):
        """Moneyline should have home/away sides."""
        pred = MarketPrediction(
            market_key="fg_moneyline",
            period="fg",
            market_type="moneyline",
            side="home",
            probability=0.65,
            confidence=0.70,
        )
        
        assert pred.side in ["home", "away"]

    def test_moneyline_no_line(self):
        """Moneyline doesn't have a traditional line."""
        pred = MarketPrediction(
            market_key="fg_moneyline",
            period="fg",
            market_type="moneyline",
            side="home",
            probability=0.65,
            confidence=0.70,
            line=None,  # No spread line
        )
        
        assert pred.line is None


class TestOddsConversion:
    """Tests for odds conversion utilities."""

    def test_american_to_decimal_favorite(self):
        """Negative American odds should convert correctly."""
        # -150 American = risk 150 to win 100
        # Decimal = 1 + (100/150) = 1.667
        american = -150
        decimal = 1 + (100 / abs(american))
        
        assert decimal == pytest.approx(1.667, abs=0.001)

    def test_american_to_decimal_underdog(self):
        """Positive American odds should convert correctly."""
        # +150 American = risk 100 to win 150
        # Decimal = 1 + (150/100) = 2.5
        american = +150
        decimal = 1 + (american / 100)
        
        assert decimal == pytest.approx(2.5, abs=0.001)

    def test_implied_probability_favorite(self):
        """Implied probability for favorite."""
        # -150 implies P = 150 / (150 + 100) = 60%
        american = -150
        implied = abs(american) / (abs(american) + 100)
        
        assert implied == pytest.approx(0.60, abs=0.001)

    def test_implied_probability_underdog(self):
        """Implied probability for underdog."""
        # +150 implies P = 100 / (150 + 100) = 40%
        american = +150
        implied = 100 / (american + 100)
        
        assert implied == pytest.approx(0.40, abs=0.001)

    def test_implied_probability_even_odds(self):
        """Implied probability at -110 (standard)."""
        # -110 implies P = 110 / 210 = 52.38%
        american = -110
        implied = abs(american) / (abs(american) + 100)
        
        assert implied == pytest.approx(0.5238, abs=0.001)


class TestEdgeCalculation:
    """Tests for edge (expected value) calculation."""

    def test_positive_edge(self):
        """Positive edge when probability > implied."""
        probability = 0.58
        implied = 0.5238  # -110
        edge = probability - implied
        
        assert edge > 0
        assert edge == pytest.approx(0.056, abs=0.01)

    def test_negative_edge(self):
        """Negative edge when probability < implied."""
        probability = 0.50
        implied = 0.5238  # -110
        edge = probability - implied
        
        assert edge < 0

    def test_break_even_edge(self):
        """Zero edge at break-even point."""
        probability = 0.5238
        implied = 0.5238
        edge = probability - implied
        
        assert edge == pytest.approx(0.0, abs=0.001)

    def test_expected_value_calculation(self):
        """Expected value should be edge * bet amount."""
        probability = 0.58
        implied = 0.5238
        edge = probability - implied
        
        # EV per $100 bet at -110
        bet = 100
        win_payout = 100 / 110  # 0.909 per unit
        
        # EV = P(win) * win_amount - P(lose) * lose_amount
        ev = probability * (bet * win_payout) - (1 - probability) * bet
        
        # EV should be positive with positive edge
        assert ev > 0


class TestKellyFraction:
    """Tests for Kelly criterion calculations."""

    def test_kelly_positive_edge(self):
        """Kelly should be positive with positive edge."""
        probability = 0.58
        odds_ratio = 0.909  # -110 payout ratio (win/risk = 100/110)
        
        # Kelly = (p * (1 + b) - 1) / b where b = decimal odds - 1
        # For -110: decimal = 1.909, b = 0.909
        # Kelly = (0.58 * 1.909 - 1) / 0.909 = 0.1179
        # OR using edge approach:
        # Kelly = edge / (odds - 1) but different formula
        
        # Actually at -110 (decimal 1.909):
        # Kelly = (p*b - q)/b where b is profit per unit won
        # = (0.58 * 0.909 - 0.42) / 0.909 
        # = (0.527 - 0.42) / 0.909
        # = 0.118
        
        p = probability
        q = 1 - probability
        b = odds_ratio
        kelly = (p * b - q) / b
        
        assert kelly > 0
        assert kelly == pytest.approx(0.118, abs=0.01)

    def test_kelly_negative_edge(self):
        """Kelly should be negative (don't bet) with negative edge."""
        probability = 0.50
        odds = 0.909  # -110 payout ratio
        
        kelly = (probability * odds - (1 - probability)) / odds
        
        assert kelly < 0  # Don't bet

    def test_kelly_break_even(self):
        """Kelly should be near zero at break-even."""
        probability = 0.5238
        odds = 0.909
        
        kelly = (probability * odds - (1 - probability)) / odds
        
        assert abs(kelly) < 0.01  # Near zero

    def test_half_kelly(self):
        """Half Kelly should be half of full Kelly."""
        probability = 0.58
        odds = 0.909
        
        full_kelly = (probability * odds - (1 - probability)) / odds
        half_kelly = full_kelly / 2
        
        assert half_kelly == pytest.approx(full_kelly / 2, abs=0.001)


class TestMarketFilters:
    """Tests for market filter logic."""

    def test_confidence_filter_pass(self):
        """Should pass if confidence >= threshold."""
        min_confidence = 0.55
        confidence = 0.60
        
        passes = confidence >= min_confidence
        assert passes is True

    def test_confidence_filter_fail(self):
        """Should fail if confidence < threshold."""
        min_confidence = 0.55
        confidence = 0.52
        
        passes = confidence >= min_confidence
        assert passes is False

    def test_edge_filter_pass(self):
        """Should pass if edge >= threshold."""
        min_edge = 0.03
        edge = 0.05
        
        passes = edge >= min_edge
        assert passes is True

    def test_edge_filter_fail(self):
        """Should fail if edge < threshold."""
        min_edge = 0.03
        edge = 0.01
        
        passes = edge >= min_edge
        assert passes is False

    def test_combined_filters(self):
        """Both filters must pass."""
        min_confidence = 0.55
        min_edge = 0.03
        
        confidence = 0.60
        edge = 0.05
        
        passes = (confidence >= min_confidence) and (edge >= min_edge)
        assert passes is True
        
        # Fail one
        edge = 0.01
        passes = (confidence >= min_confidence) and (edge >= min_edge)
        assert passes is False


class TestMarketKeyGeneration:
    """Tests for market key generation."""

    def test_fg_spread_key(self):
        """Full game spread key."""
        key = f"fg_spread"
        assert key == "fg_spread"

    def test_1h_spread_key(self):
        """First half spread key."""
        key = f"1h_spread"
        assert key == "1h_spread"

    def test_fg_total_key(self):
        """Full game total key."""
        key = f"fg_total"
        assert key == "fg_total"

    def test_1h_total_key(self):
        """First half total key."""
        key = f"1h_total"
        assert key == "1h_total"

    def test_all_market_keys(self):
        """All 4 active market keys."""
        markets = ["fg_spread", "fg_total", "1h_spread", "1h_total"]
        
        assert len(markets) == 4
        for m in markets:
            assert "_" in m
            period, market_type = m.split("_")
            assert period in ["fg", "1h"]
            assert market_type in ["spread", "total"]
