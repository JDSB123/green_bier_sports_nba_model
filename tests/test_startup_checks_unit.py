from dataclasses import dataclass
from pathlib import Path

import pytest

import src.utils.startup_checks as sc


@dataclass(frozen=True)
class _FakeCatalog:
    source: str
    markets: list[str]


@dataclass
class _FakeValidationResult:
    is_valid: bool
    errors: list[str]
    warnings: list[str]


def test_validate_api_auth_config_requires_service_key(monkeypatch):
    monkeypatch.setenv("REQUIRE_API_AUTH", "true")
    monkeypatch.delenv("SERVICE_API_KEY", raising=False)
    assert sc._validate_api_auth_config() is not None

    monkeypatch.setenv("SERVICE_API_KEY", "abc")
    assert sc._validate_api_auth_config() is None


def test_validate_filter_thresholds_warns_when_defaults_used(monkeypatch):
    for var in [
        "FILTER_SPREAD_MIN_CONFIDENCE",
        "FILTER_SPREAD_MIN_EDGE",
        "FILTER_TOTAL_MIN_CONFIDENCE",
        "FILTER_TOTAL_MIN_EDGE",
        "FILTER_1H_SPREAD_MIN_CONFIDENCE",
        "FILTER_1H_SPREAD_MIN_EDGE",
        "FILTER_1H_TOTAL_MIN_CONFIDENCE",
        "FILTER_1H_TOTAL_MIN_EDGE",
    ]:
        monkeypatch.delenv(var, raising=False)

    warnings = sc._validate_filter_thresholds()
    assert len(warnings) == 8

    monkeypatch.setenv("FILTER_SPREAD_MIN_CONFIDENCE", "0.7")
    warnings2 = sc._validate_filter_thresholds()
    assert len(warnings2) == 7


def test_validate_feature_alignment_reports_missing_features(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(sc, "_collect_builder_feature_keys", lambda: {"a", "b"})
    monkeypatch.setattr(
        sc, "_load_model_features", lambda models_dir, market_key: (["a", "b", "c"], None)
    )

    report = sc._validate_feature_alignment(tmp_path, ["fg_spread"])
    assert report.errors
    assert "missing" in report.errors[0]


def test_run_startup_integrity_checks_raises_on_market_mismatch(monkeypatch, tmp_path: Path):
    project_root = tmp_path / "proj"
    models_dir = tmp_path / "models"
    project_root.mkdir()
    models_dir.mkdir()

    monkeypatch.setattr(
        sc, "validate_required_api_keys", lambda: _FakeValidationResult(True, [], [])
    )
    monkeypatch.setattr(sc, "_validate_api_auth_config", lambda: None)
    monkeypatch.setattr(sc, "_validate_filter_thresholds", lambda: [])
    monkeypatch.setattr(sc, "get_expected_markets", lambda: ["fg_spread", "fg_total"])
    monkeypatch.setattr(
        sc,
        "get_market_catalog",
        lambda *args, **kwargs: _FakeCatalog(source="model_pack", markets=["fg_spread"]),
    )
    monkeypatch.setattr(
        sc,
        "_validate_feature_alignment",
        lambda *args, **kwargs: sc.StartupIntegrityReport(errors=[], warnings=[]),
    )

    with pytest.raises(sc.StartupIntegrityError) as exc:
        sc.run_startup_integrity_checks(project_root=project_root, models_dir=models_dir)

    assert "model_pack missing markets" in str(exc.value)


def test_run_startup_integrity_checks_succeeds(monkeypatch, tmp_path: Path):
    project_root = tmp_path / "proj"
    models_dir = tmp_path / "models"
    project_root.mkdir()
    models_dir.mkdir()

    monkeypatch.setattr(
        sc, "validate_required_api_keys", lambda: _FakeValidationResult(True, [], [])
    )
    monkeypatch.setattr(sc, "_validate_api_auth_config", lambda: None)
    monkeypatch.setattr(sc, "_validate_filter_thresholds", lambda: [])
    monkeypatch.setattr(sc, "get_expected_markets", lambda: ["fg_spread"])
    monkeypatch.setattr(
        sc,
        "get_market_catalog",
        lambda *args, **kwargs: _FakeCatalog(source="model_pack", markets=["fg_spread"]),
    )
    monkeypatch.setattr(
        sc,
        "_validate_feature_alignment",
        lambda *args, **kwargs: sc.StartupIntegrityReport(errors=[], warnings=[]),
    )

    sc.run_startup_integrity_checks(project_root=project_root, models_dir=models_dir)
