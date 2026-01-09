from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_VERSION_ENV = "NBA_MODEL_VERSION"
_VERSION_FILE = "VERSION"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_version_file(path: Optional[Path] = None) -> Optional[str]:
    target = path or (_project_root() / _VERSION_FILE)
    try:
        value = target.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    return value or None


def resolve_version() -> str:
    env_value = os.getenv(_VERSION_ENV, "").strip()
    if env_value and env_value.lower() not in {"unknown", "unset", "none"}:
        return env_value

    file_value = read_version_file()
    if file_value:
        return file_value

    return "unknown"
