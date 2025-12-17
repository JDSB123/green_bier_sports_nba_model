import importlib
import importlib.util
import sys
from pathlib import Path


def check_import(module_name: str):
    try:
        module = importlib.import_module(module_name)
        print(f"OK import: {module_name}")
        return module
    except Exception as e:
        print(f"FAIL import: {module_name} -> {e}")
        return None


def check_attr(module, attr_name: str):
    if module is None:
        print(f"SKIP attr: {attr_name} (module not loaded)")
        return False
    exists = hasattr(module, attr_name)
    status = 'OK' if exists else 'MISSING'
    print(f"{status} attr: {module.__name__}.{attr_name}")
    return exists


def main():
    ROOT = Path(__file__).resolve().parents[1]
    # Scripts: load from file paths to avoid package issues
    script_dir = ROOT / "scripts"
    script_paths = {
        "scripts.backtest": script_dir / "backtest.py",
        "scripts.collect_api_basketball": script_dir / "collect_api_basketball.py",
        "scripts.collect_balldontlie": script_dir / "collect_balldontlie.py",
        "scripts.collect_odds": script_dir / "collect_odds.py",
        "scripts.collect_the_odds": script_dir / "collect_the_odds.py",
        "scripts.generate_training_data": (
            script_dir / "generate_training_data.py"
        ),
        "scripts.ingest_all": script_dir / "ingest_all.py",
        "scripts.merge_odds": script_dir / "merge_odds.py",
        "scripts.predict": script_dir / "predict.py",
        "scripts.train_models": script_dir / "train_models.py",
    }
    for name, path in script_paths.items():
        try:
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(mod)  # type: ignore
            print(f"OK import: {name}")
        except Exception as e:
            print(f"FAIL import: {name} -> {e}")

    # Ingestion modules and expected callables (load by file)
    ing_dir = ROOT / "src" / "ingestion"
    ingestion_paths = {
        "src.ingestion.api_basketball": ing_dir / "api_basketball.py",
        "src.ingestion.betsapi": ing_dir / "betsapi.py",
        "src.ingestion.the_odds": ing_dir / "the_odds.py",
        "src.ingestion.optional_sources": ing_dir / "optional_sources.py",
    }
    loaded_ing = {}
    for name, path in ingestion_paths.items():
        try:
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(mod)  # type: ignore
            loaded_ing[name] = mod
            print(f"OK import: {name}")
        except Exception as e:
            loaded_ing[name] = None
            print(f"FAIL import: {name} -> {e}")

    api_basketball = loaded_ing.get("src.ingestion.api_basketball")
    check_attr(api_basketball, "fetch_games")
    check_attr(api_basketball, "fetch_teams")

    betsapi = loaded_ing.get("src.ingestion.betsapi")
    check_attr(betsapi, "fetch_events")

    the_odds = loaded_ing.get("src.ingestion.the_odds")
    check_attr(the_odds, "fetch_odds")
    # optional presence already checked above

    # Modeling modules and expected callables/classes (load by file)
    mod_dir = ROOT / "src" / "modeling"
    modeling_paths = {
        "src.modeling.dataset": mod_dir / "dataset.py",
        "src.modeling.features": mod_dir / "features.py",
        "src.modeling.models": mod_dir / "models.py",
    }
    loaded_mod = {}
    for name, path in modeling_paths.items():
        try:
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(mod)  # type: ignore
            loaded_mod[name] = mod
            print(f"OK import: {name}")
        except Exception as e:
            loaded_mod[name] = None
            print(f"FAIL import: {name} -> {e}")

    dataset = loaded_mod.get("src.modeling.dataset")
    check_attr(dataset, "load_training_data")
    check_attr(dataset, "load_predictions_data")

    features = loaded_mod.get("src.modeling.features")
    check_attr(features, "build_features")

    models = loaded_mod.get("src.modeling.models")
    check_attr(models, "train_models")
    check_attr(models, "predict")
    # Config (load by file)
    try:
        cfg_path = ROOT / "src" / "config.py"
        spec = importlib.util.spec_from_file_location("src.config", cfg_path)
        config_mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(config_mod)  # type: ignore
        print("OK import: src.config")
        check_attr(config_mod, "DATA_DIR")
    except Exception as e:
        print(f"FAIL import: src.config -> {e}")

    print("\nSmoke test complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
