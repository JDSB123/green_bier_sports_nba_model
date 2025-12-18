"""
Adapters that let the Dockerized FastAPI service reuse the proven v4
prediction stack (models + filters) without duplicating code.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)


def _find_repo_root() -> Path:
    """
    Walk parent directories until we locate the repository root (contains src/).
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "src").exists():
            return parent
    raise RuntimeError("Unable to locate repository root (expected a src/ directory)")


def _ensure_repo_on_path(repo_root: Path) -> None:
    """
    Ensure both <repo_root> and <repo_root>/src are importable.
    """
    for candidate in (repo_root, repo_root / "src"):
        candidate_str = str(candidate)
        if candidate.exists() and candidate_str not in sys.path:
            sys.path.append(candidate_str)


@dataclass(frozen=True)
class V4Predictors:
    spread_total_engine: "PredictionEngine"
    moneyline_predictor: "MoneylinePredictor"


def _sanitize_models_dir(models_dir: str | Path) -> Path:
    """
    Normalize and validate the models directory path.
    """
    path = Path(models_dir).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Models directory not found: {path}")
    return path


def _bootstrap_v4_context(models_dir: str | Path) -> Tuple[Path, Path]:
    """
    Resolve repo + models paths and make sure imports work.
    """
    repo_root = _find_repo_root()
    _ensure_repo_on_path(repo_root)
    models_path = _sanitize_models_dir(models_dir)
    return repo_root, models_path


@lru_cache(maxsize=1)
def load_v4_predictors(models_dir: str | Path) -> V4Predictors:
    """
    Load the canonical v4 prediction engine + moneyline predictor.
    """
    repo_root, models_path = _bootstrap_v4_context(models_dir)

    # Imports happen after repo_path has been added to sys.path.
    from src.prediction.predictor import PredictionEngine as V4PredictionEngine
    from src.prediction.moneyline import MoneylinePredictor

    logger.info("Loading v4 predictors from %s", models_path)
    engine = V4PredictionEngine(models_path)
    moneyline = MoneylinePredictor(engine.spread_model, engine.spread_features)

    return V4Predictors(spread_total_engine=engine, moneyline_predictor=moneyline)
