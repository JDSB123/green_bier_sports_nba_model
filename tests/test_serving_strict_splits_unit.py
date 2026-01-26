from __future__ import annotations

import pytest
from fastapi import HTTPException


def _deps_module():
    import importlib

    return importlib.import_module("src.serving.dependencies")


@pytest.mark.asyncio
async def test_fetch_required_splits_non_strict_returns_empty_on_error(monkeypatch):
    from src.config import settings

    deps_module = _deps_module()
    fetch_required_splits = deps_module.fetch_required_splits

    monkeypatch.setattr(settings, "require_action_network_splits", False, raising=False)
    monkeypatch.setattr(settings, "require_real_splits", False, raising=False)

    async def boom(*_args, **_kwargs):
        raise RuntimeError("network")

    monkeypatch.setattr(deps_module, "fetch_public_betting_splits", boom)

    games = [{"home_team": "Los Angeles Lakers", "away_team": "Boston Celtics"}]
    splits = await fetch_required_splits(games)
    assert splits == {}


@pytest.mark.asyncio
async def test_fetch_required_splits_strict_raises_on_fetch_failure(monkeypatch):
    from src.config import settings

    deps_module = _deps_module()
    fetch_required_splits = deps_module.fetch_required_splits

    monkeypatch.setattr(settings, "require_action_network_splits", True, raising=False)
    monkeypatch.setattr(settings, "require_real_splits", False, raising=False)

    async def boom(*_args, **_kwargs):
        raise RuntimeError("no action network")

    monkeypatch.setattr(deps_module, "fetch_public_betting_splits", boom)

    games = [{"home_team": "Los Angeles Lakers", "away_team": "Boston Celtics"}]
    with pytest.raises(HTTPException) as exc:
        await fetch_required_splits(games)
    assert exc.value.status_code == 502


@pytest.mark.asyncio
async def test_fetch_required_splits_strict_requires_coverage(monkeypatch):
    from src.config import settings

    deps_module = _deps_module()
    fetch_required_splits = deps_module.fetch_required_splits

    monkeypatch.setattr(settings, "require_action_network_splits", True, raising=False)
    monkeypatch.setattr(settings, "require_real_splits", True, raising=False)

    async def ok(_games, **_kwargs):
        # Missing Celtics@Lakers key on purpose.
        return {"Other@Game": object()}

    monkeypatch.setattr(deps_module, "fetch_public_betting_splits", ok)

    games = [{"home_team": "Los Angeles Lakers", "away_team": "Boston Celtics"}]
    with pytest.raises(HTTPException) as exc:
        await fetch_required_splits(games)
    assert exc.value.status_code == 502
