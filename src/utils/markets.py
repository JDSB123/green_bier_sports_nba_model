from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.modeling.unified_features import get_all_market_keys, get_period_markets


@dataclass(frozen=True)
class MarketCatalog:
    source: str
    markets: List[str]
    periods: Dict[str, List[str]]
    market_types: List[str]
    model_pack_version: Optional[str] = None
    model_pack_path: Optional[str] = None


def _candidate_model_pack_paths(project_root: Path, data_processed_dir: str) -> List[Path]:
    return [
        project_root / "models" / "production" / "model_pack.json",
    ]


def load_model_pack(
    project_root: Path, data_processed_dir: str
) -> Tuple[Optional[Dict], Optional[Path]]:
    for path in _candidate_model_pack_paths(project_root, data_processed_dir):
        if path.exists():
            try:
                return json.loads(path.read_text()), path
            except Exception:
                return None, path
    return None, None


def _extract_markets_from_pack(model_pack: Dict) -> List[str]:
    if not model_pack:
        return []

    for key in ("market_status", "backtest_results"):
        value = model_pack.get(key)
        if isinstance(value, dict) and value:
            return sorted(value.keys())

    markets_list = model_pack.get("markets_list")
    if isinstance(markets_list, list) and markets_list:
        return sorted(markets_list)

    return []


def _build_periods(markets: List[str]) -> Dict[str, List[str]]:
    periods: Dict[str, List[str]] = {}
    for market in markets:
        if "_" not in market:
            continue
        period, market_type = market.split("_", 1)
        periods.setdefault(period, []).append(market_type)

    for key in periods:
        periods[key] = sorted(periods[key])
    return periods


def get_market_catalog(project_root: Path, data_processed_dir: str) -> MarketCatalog:
    model_pack, model_pack_path = load_model_pack(project_root, data_processed_dir)
    markets = _extract_markets_from_pack(model_pack or {})

    source = "model_pack"
    if not markets:
        markets = sorted(get_all_market_keys())
        source = "model_configs"

    periods = _build_periods(markets)
    market_types = sorted({market.split("_", 1)[1] for market in markets if "_" in market})

    return MarketCatalog(
        source=source,
        markets=markets,
        periods=periods,
        market_types=market_types,
        model_pack_version=(model_pack or {}).get("version"),
        model_pack_path=str(model_pack_path) if model_pack_path else None,
    )


def get_enabled_markets(project_root: Path, data_processed_dir: str) -> List[str]:
    return get_market_catalog(project_root, data_processed_dir).markets


def get_expected_markets() -> List[str]:
    return sorted(get_all_market_keys())


def get_expected_periods() -> Dict[str, List[str]]:
    return {
        "1h": sorted([m.split("_", 1)[1] for m in get_period_markets("1h")]),
        "fg": sorted([m.split("_", 1)[1] for m in get_period_markets("fg")]),
    }
