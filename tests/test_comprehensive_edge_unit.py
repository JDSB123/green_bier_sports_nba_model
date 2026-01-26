from __future__ import annotations

from datetime import date

import pytest


def _sample_inputs():
    features = {
        "predicted_margin": 10.0,
        "predicted_total": 240.0,
        "expected_pace_factor": 1.0,
        "home_margin_std": 10.0,
        "away_margin_std": 10.0,
        "home_score_std": 12.0,
        "away_score_std": 12.0,
    }

    fh_features = {
        "expected_pace_factor": 1.0,
        "home_margin_std_1h": 6.0,
        "away_margin_std_1h": 6.0,
        "home_score_std": 8.0,
        "away_score_std": 8.0,
    }

    odds = {
        "home_spread": -5.0,
        "home_spread_price": -110,
        "away_spread_price": -110,
        "total": 230.0,
        "total_over_price": -110,
        "total_under_price": -110,
        "total_price": -110,
        "fh_home_spread": -2.5,
        "fh_home_spread_price": -110,
        "fh_away_spread_price": -110,
        "fh_total": 115.0,
        "fh_total_over_price": -110,
        "fh_total_under_price": -110,
        "fh_total_price": -110,
        "as_of_utc": "2026-01-01T00:00:00Z",
    }

    game = {"home_team": "Los Angeles Lakers", "away_team": "Boston Celtics"}

    engine_predictions = {
        "full_game": {
            "spread": {
                "bet_side": "home",
                "home_cover_prob": 0.60,
                "away_cover_prob": 0.40,
                "confidence": 0.80,
                "passes_filter": True,
                "filter_reason": None,
                "signals_agree": True,
                "classifier_side": "home",
                "prediction_side": "home",
                "classifier_extreme": False,
            },
            "total": {
                "bet_side": "over",
                "over_prob": 0.60,
                "under_prob": 0.40,
                "confidence": 0.80,
                "passes_filter": True,
                "filter_reason": None,
                "signals_agree": True,
                "classifier_side": "over",
                "prediction_side": "over",
                "classifier_extreme": False,
            },
        },
        "first_half": {
            "spread": {
                "predicted_margin": 6.0,
                "bet_side": "home",
                "home_cover_prob": 0.58,
                "confidence": 0.78,
                "passes_filter": True,
                "filter_reason": None,
                "signals_agree": True,
                "classifier_side": "home",
                "prediction_side": "home",
                "classifier_extreme": False,
            },
            "total": {
                "predicted_total": 120.0,
                "bet_side": "over",
                "over_prob": 0.57,
                "confidence": 0.76,
                "passes_filter": True,
                "filter_reason": None,
                "signals_agree": True,
                "classifier_side": "over",
                "prediction_side": "over",
                "classifier_extreme": False,
            },
        },
    }

    return features, fh_features, odds, game, engine_predictions


def test_calculate_comprehensive_edge_requires_engine_predictions():
    from src.utils.comprehensive_edge import calculate_comprehensive_edge

    features, fh_features, odds, game, _ = _sample_inputs()

    with pytest.raises(ValueError):
        calculate_comprehensive_edge(features, fh_features, odds, game, engine_predictions=None)


def test_calculate_comprehensive_edge_happy_path_generates_picks():
    from src.utils.comprehensive_edge import calculate_comprehensive_edge

    features, fh_features, odds, game, engine_predictions = _sample_inputs()

    result = calculate_comprehensive_edge(
        features,
        fh_features,
        odds,
        game,
        engine_predictions=engine_predictions,
        edge_thresholds={"spread": 2.0, "total": 3.0, "1h_spread": 1.5, "1h_total": 2.0},
    )

    assert result["full_game"]["spread"]["pick"] == "Los Angeles Lakers"
    assert result["full_game"]["total"]["pick"] == "OVER"
    assert result["first_half"]["spread"]["pick"] == "Los Angeles Lakers"
    assert result["first_half"]["total"]["pick"] == "OVER"

    assert result["top_plays"]
    assert all(p["probability_source"] == "engine_ml" for p in result["top_plays"])


def test_generate_comprehensive_text_report_empty_and_non_empty():
    from src.utils.comprehensive_edge import generate_comprehensive_text_report

    empty = generate_comprehensive_text_report([], date(2026, 1, 1))
    assert "No games scheduled" in empty

    report = generate_comprehensive_text_report(
        [
            {
                "home_team": "Los Angeles Lakers",
                "away_team": "Boston Celtics",
                "time_cst": "7:00 PM",
                "comprehensive_edge": {
                    "full_game": {
                        "spread": {"pick": "Los Angeles Lakers", "edge": 3.0, "confidence": 0.8}
                    },
                    "first_half": {},
                },
            }
        ],
        date(2026, 1, 1),
    )
    assert "GAME 1" in report
    assert "Los Angeles Lakers" in report
