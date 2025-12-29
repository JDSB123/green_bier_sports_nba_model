from src.utils.odds import (
    american_to_implied_prob,
    devig_two_way,
    expected_value,
    kelly_fraction,
)


def test_american_to_implied_prob():
    assert round(american_to_implied_prob(-110), 4) == 0.5238
    assert round(american_to_implied_prob(150), 4) == 0.4


def test_devig_two_way_even_prices():
    p_a, p_b = devig_two_way(-110, -110)
    assert round(p_a, 4) == 0.5
    assert round(p_b, 4) == 0.5


def test_devig_two_way_asymmetric_prices():
    p_a, p_b = devig_two_way(-200, 180)
    assert round(p_a + p_b, 6) == 1.0
    assert p_a > p_b


def test_expected_value_even_odds():
    ev = expected_value(0.5, -110, stake=1.0)
    assert round(ev, 4) == -0.0455


def test_kelly_fraction_positive():
    kelly = kelly_fraction(0.6, -110, fraction=0.5)
    assert round(kelly, 2) == 0.08


def test_kelly_fraction_zero():
    kelly = kelly_fraction(0.5, -110, fraction=0.5)
    assert kelly == 0.0
