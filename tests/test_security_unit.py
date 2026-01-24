import os

import pytest

from src.utils.security import (
    SecurityError,
    fail_fast_on_missing_keys,
    mask_api_key,
    sanitize_for_logging,
    validate_database_config,
)


def test_mask_api_key_basic():
    assert mask_api_key("") == "****"
    assert mask_api_key("abc") == "****"
    assert mask_api_key("abcd", visible_chars=4) == "abcd"
    assert mask_api_key("abcdef", visible_chars=4) == "**cdef"


def test_sanitize_for_logging_masks_sensitive_fields():
    payload = {
        "apiKey": "abcdef123456",
        "password": "supersecret",
        "nested": {"ok": True},
        "safe": 123,
    }
    sanitized = sanitize_for_logging(payload)
    assert sanitized["safe"] == 123
    assert sanitized["nested"] == {"ok": True}
    assert sanitized["apiKey"].endswith("3456")
    assert sanitized["password"].startswith("*")


def test_validate_database_config_optional_when_unset(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("DB_PASSWORD", raising=False)
    result = validate_database_config()
    assert result.is_valid
    assert result.errors == []


def test_validate_database_config_requires_password_when_url_set(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgres://example")
    monkeypatch.delenv("DB_PASSWORD", raising=False)
    result = validate_database_config()
    assert not result.is_valid
    assert any("DB_PASSWORD is required" in e for e in result.errors)


def test_fail_fast_on_missing_keys_raises(monkeypatch):
    # Force settings keys to empty by patching env vars and reloading settings-dependent values.
    # The security module reads from settings (pydantic settings), but validate_required_api_keys
    # uses settings.<key>, so we patch those attributes directly.
    from src import config

    monkeypatch.setattr(config.settings, "the_odds_api_key", "")
    monkeypatch.setattr(config.settings, "api_basketball_key", "")

    with pytest.raises(SecurityError):
        fail_fast_on_missing_keys()
