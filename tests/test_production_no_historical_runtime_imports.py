import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# We explicitly allow some modeling modules at runtime because they define
# schema/config/constants used by the production engine.
ALLOWED_MODELING_IMPORTS = {
    "src.modeling.unified_features",
    "src.modeling.edge_thresholds",
}

# These modules are training/backtest oriented and/or read processed historical files.
BANNED_MODELING_IMPORT_PREFIXES = (
    "src.modeling.dataset",
    "src.modeling.features",
    "src.modeling.calibration",
    "src.modeling.models",
    "src.modeling.team_factors",
)

# Production runtime surfaces we care about.
PRODUCTION_DIRS = (
    PROJECT_ROOT / "src" / "serving",
    PROJECT_ROOT / "src" / "prediction",
    PROJECT_ROOT / "src" / "features",
    PROJECT_ROOT / "src" / "ingestion",
)


def _iter_python_files():
    for base in PRODUCTION_DIRS:
        for path in base.rglob("*.py"):
            # Ignore package init files
            yield path


def _collect_imports(py_path: Path) -> set[str]:
    tree = ast.parse(py_path.read_text(encoding="utf-8"))
    imports: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)

    return imports


def test_production_runtime_does_not_import_training_or_historical_modules():
    offenders: list[str] = []

    for py_path in _iter_python_files():
        imports = _collect_imports(py_path)
        for imp in imports:
            if imp.startswith("src.modeling") and imp not in ALLOWED_MODELING_IMPORTS:
                if imp.startswith(BANNED_MODELING_IMPORT_PREFIXES):
                    rel = py_path.relative_to(PROJECT_ROOT)
                    offenders.append(f"{rel}: {imp}")

    assert offenders == [], "\n".join([
        "Production runtime imports training/historical modules (not allowed):",
        *sorted(offenders),
    ])
