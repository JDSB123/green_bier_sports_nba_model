"""
Adapters that let the Dockerized FastAPI service reuse the v5 STRICT MODE
prediction stack (models + filters) without duplicating code.

STRICT MODE: All 4 models must exist. No fallbacks. No silent failures.

Supports 6 BACKTESTED markets:
- Full Game: Spread, Total, Moneyline
- First Half: Spread, Total, Moneyline
"""

from __future__ import annotations

import logging
import sys
from functools import lru_cache
from pathlib import Path

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


def _sanitize_models_dir(models_dir: str | Path) -> Path:
    """
    Normalize and validate the models directory path.
    
    Raises:
        FileNotFoundError: If models directory does not exist
    """
    path = Path(models_dir).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Models directory not found: {path}")
    return path


def _bootstrap_context(models_dir: str | Path) -> Path:
    """
    Resolve repo + models paths and make sure imports work.
    
    Returns:
        Path to models directory
        
    Raises:
        RuntimeError: If repo root cannot be found
        FileNotFoundError: If models directory does not exist
    """
    repo_root = _find_repo_root()
    _ensure_repo_on_path(repo_root)
    return _sanitize_models_dir(models_dir)


@lru_cache(maxsize=1)
def load_prediction_engine(models_dir: str | Path) -> "UnifiedPredictionEngine":
    """
    Load the STRICT MODE unified prediction engine.
    
    ALL 4 models must exist:
    - spreads_model.joblib (FG Spread)
    - totals_model.joblib (FG Total)
    - first_half_spread_model.pkl (1H Spread)
    - first_half_total_model.pkl (1H Total)
    
    Args:
        models_dir: Path to directory containing all model files
        
    Returns:
        UnifiedPredictionEngine instance with all predictors loaded
        
    Raises:
        ModelNotFoundError: If ANY required model is missing
        FileNotFoundError: If models directory does not exist
    """
    models_path = _bootstrap_context(models_dir)

    # Import after repo_path has been added to sys.path
    from src.prediction.engine import UnifiedPredictionEngine

    logger.info("Loading STRICT MODE prediction engine from %s", models_path)
    
    # This will FAIL LOUDLY if any model is missing - no silent failures
    engine = UnifiedPredictionEngine(models_path)
    
    logger.info("Successfully loaded all 4 required models for 6 markets")
    return engine
