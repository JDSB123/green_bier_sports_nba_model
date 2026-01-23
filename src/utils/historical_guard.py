"""
Historical mode guardrails.

Enforces Azure-only storage for historical outputs and blocks accidental
execution in prediction/production contexts.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _is_truthy(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def require_historical_mode() -> None:
    """Raise if historical mode is not explicitly enabled."""
    if not _is_truthy(os.getenv("HISTORICAL_MODE")):
        raise RuntimeError(
            "Historical scripts are disabled by default. "
            "Set HISTORICAL_MODE=true to proceed."
        )


def resolve_historical_output_root(subdir: str | None = None) -> Path:
    """Resolve Azure-only output root for historical artifacts.

    Requires HISTORICAL_MODE=true and HISTORICAL_OUTPUT_ROOT to be set.
    Local output is blocked unless ALLOW_LOCAL_HISTORICAL=true.
    """
    require_historical_mode()

    output_root = os.getenv("HISTORICAL_OUTPUT_ROOT")
    allow_local = _is_truthy(os.getenv("ALLOW_LOCAL_HISTORICAL"))

    if not output_root:
        if allow_local:
            root = PROJECT_ROOT / "data" / "historical"
        else:
            raise RuntimeError(
                "HISTORICAL_OUTPUT_ROOT is required for Azure-only storage. "
                "Set HISTORICAL_OUTPUT_ROOT to an Azure-mounted path, or set "
                "ALLOW_LOCAL_HISTORICAL=true for local runs."
            )
    else:
        root = Path(output_root).expanduser().resolve()
        if PROJECT_ROOT in root.parents or root == PROJECT_ROOT:
            if not allow_local:
                raise RuntimeError(
                    "Local historical outputs are blocked. "
                    "Set HISTORICAL_OUTPUT_ROOT to Azure-mounted storage or "
                    "ALLOW_LOCAL_HISTORICAL=true to override."
                )

    return root / subdir if subdir else root


def ensure_historical_path(path: Path, description: str = "path") -> None:
    """Validate a path is under HISTORICAL_OUTPUT_ROOT unless local is allowed."""
    require_historical_mode()
    allow_local = _is_truthy(os.getenv("ALLOW_LOCAL_HISTORICAL"))
    output_root = os.getenv("HISTORICAL_OUTPUT_ROOT")
    if allow_local or not output_root:
        return

    root = Path(output_root).expanduser().resolve()
    candidate = path.expanduser().resolve()
    if root not in candidate.parents and candidate != root:
        raise RuntimeError(
            f"{description} must be under HISTORICAL_OUTPUT_ROOT ({root}). "
            "Local historical paths are blocked."
        )
