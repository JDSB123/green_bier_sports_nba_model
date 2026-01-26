from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Set

import joblib
import pytest


def _load_feature_names(joblib_path: Path) -> Set[str]:
    payload: dict[str, Any] = joblib.load(joblib_path)
    model = payload.get("pipeline") or payload.get("model")
    feats = payload.get("feature_columns") or payload.get("model_columns") or []

    # Prefer the model's internal feature list when present
    if hasattr(model, "feature_names_in_"):
        feats = list(model.feature_names_in_)
    return set(map(str, feats))


def _assert_no_forbidden_features(features: Iterable[str]) -> None:
    # Postgame outcomes / targets must never be used as features.
    forbidden = {
        # Final scores
        "home_score",
        "away_score",
        "actual_total",
        "actual_margin",
        # Quarter / 1H outcomes
        "home_q1",
        "home_q2",
        "home_q3",
        "home_q4",
        "away_q1",
        "away_q2",
        "away_q3",
        "away_q4",
        "home_1h_score",
        "away_1h_score",
        "actual_1h_total",
        "actual_1h_margin",
        # Labels / targets
        "home_win",
        "spread_covered",
        "total_over",
        "1h_spread_covered",
        "1h_total_over",
        "q1_spread_covered",
        "q1_total_over",
    }

    feats = set(features)
    overlap = sorted(feats & forbidden)
    assert not overlap, f"Forbidden (leaky) features present: {overlap}"


@pytest.mark.parametrize(
    "fname",
    [
        "fg_spread_model.joblib",
        "fg_total_model.joblib",
        "1h_spread_model.joblib",
        "1h_total_model.joblib",
    ],
)
def test_production_models_do_not_use_postgame_outcomes_as_features(fname: str) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    model_path = repo_root / "models" / "production" / fname
    assert model_path.exists(), f"Missing production model artifact: {model_path}"

    feature_names = _load_feature_names(model_path)
    _assert_no_forbidden_features(feature_names)
