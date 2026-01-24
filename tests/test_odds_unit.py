from src.utils.odds import (
    american_to_implied_prob,
    break_even_common_juices,
    break_even_percent,
    devig_three_way,
    devig_two_way,
    expected_value,
    kelly_fraction,
)


def test_american_to_implied_prob_handles_invalids():
    assert american_to_implied_prob(None) is None
    assert american_to_implied_prob("nope") is None
    assert american_to_implied_prob(0) is None


def test_american_to_implied_prob_positive_and_negative():
    assert round(american_to_implied_prob(100) or 0.0, 6) == 0.5
    assert round(american_to_implied_prob(-100) or 0.0, 6) == 0.5


def test_devig_two_way_normalizes_to_one():
    a, b = devig_two_way(-110, -110)
    assert a is not None and b is not None
    assert round(a + b, 9) == 1.0


def test_devig_three_way_normalizes_to_one():
    a, b, c = devig_three_way(200, 200, 200)
    assert a is not None and b is not None and c is not None
    assert round(a + b + c, 9) == 1.0


def test_expected_value_and_kelly_fraction_edge_cases():
    assert expected_value(None, -110) is None
    assert expected_value(0.55, None) is None
    assert expected_value(0.55, -110, stake=0) is None
    assert expected_value(0.55, "bad") is None

    assert kelly_fraction(None, -110) is None
    assert kelly_fraction(0.55, None) is None
    assert kelly_fraction(0.55, "bad") is None


def test_break_even_helpers():
    assert break_even_percent(-110) is not None
    juices = break_even_common_juices()
    assert -110 in juices
