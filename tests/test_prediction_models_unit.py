from __future__ import annotations

from pathlib import Path

import joblib
import pytest


def test_load_fg_models_fallback_paths(tmp_path, monkeypatch):
    from src.prediction.models import load_spread_model, load_total_model

    # Make sure the tracker doesn't redirect us away from the standard names.
    class _FakeTracker:
        def get_active_version(self, *_args, **_kwargs):
            return None

        def get_version_info(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr("src.prediction.models.ModelTracker", _FakeTracker)

    spreads_path = tmp_path / "fg_spread_model.joblib"
    totals_path = tmp_path / "fg_total_model.joblib"

    joblib.dump({"model": "spread_model", "feature_columns": ["a", "b"]}, spreads_path)
    joblib.dump({"pipeline": "total_model", "model_columns": ["x"]}, totals_path)

    model, cols = load_spread_model(tmp_path)
    assert model == "spread_model"
    assert cols == ["a", "b"]

    model, cols = load_total_model(tmp_path)
    assert model == "total_model"
    assert cols == ["x"]


def test_load_fg_models_tracker_redirect(tmp_path, monkeypatch):
    from src.prediction.models import load_spread_model

    redirected = tmp_path / "alt.joblib"
    joblib.dump({"model": "m", "feature_columns": []}, redirected)

    class _FakeTracker:
        def get_active_version(self, market: str):
            return "v1" if market == "spreads" else None

        def get_version_info(self, _version: str):
            return {"file_path": "alt.joblib"}

    monkeypatch.setattr("src.prediction.models.ModelTracker", _FakeTracker)

    model, cols = load_spread_model(tmp_path)
    assert model == "m"
    assert cols == []


def test_load_first_half_models_joblib_and_legacy(tmp_path: Path):
    from src.prediction.models import load_first_half_spread_model, load_first_half_total_model

    # New combined joblib format
    joblib.dump({"model": "fh_spread", "feature_columns": ["f1"]}, tmp_path / "1h_spread_model.joblib")
    joblib.dump({"pipeline": "fh_total", "model_columns": ["t1", "t2"]}, tmp_path / "1h_total_model.joblib")

    m, cols = load_first_half_spread_model(tmp_path)
    assert m == "fh_spread"
    assert cols == ["f1"]

    m, cols = load_first_half_total_model(tmp_path)
    assert m == "fh_total"
    assert cols == ["t1", "t2"]

    # Legacy .pkl format fallback
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    joblib.dump("legacy_spread", legacy_dir / "1h_spread_model.pkl")
    joblib.dump(["l1"], legacy_dir / "1h_spread_features.pkl")

    joblib.dump("legacy_total", legacy_dir / "1h_total_model.pkl")
    joblib.dump(["z"], legacy_dir / "1h_total_features.pkl")

    m, cols = load_first_half_spread_model(legacy_dir)
    assert m == "legacy_spread"
    assert cols == ["l1"]

    m, cols = load_first_half_total_model(legacy_dir)
    assert m == "legacy_total"
    assert cols == ["z"]


def test_load_models_missing_raises(tmp_path):
    from src.prediction.models import (
        load_spread_model,
        load_total_model,
        load_first_half_spread_model,
        load_first_half_total_model,
    )

    with pytest.raises(FileNotFoundError):
        load_spread_model(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_total_model(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_first_half_spread_model(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_first_half_total_model(tmp_path)
