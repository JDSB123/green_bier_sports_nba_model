#!/usr/bin/env python3
"""
CLI tool for managing model versions and promotions.

Usage:
    # List all models
    python scripts/model_manage.py list

    # List only production models
    python scripts/model_manage.py list --status production

    # Promote a model to production
    python scripts/model_manage.py promote --name xgboost_model --version 1.0.0

    # Compare two models
    python scripts/model_manage.py compare --model1 xgboost_model:1.0.0 --model2 xgboost_model:1.1.0

    # Show current production model
    python scripts/model_manage.py production
"""
from src.config import settings
from src.modeling.versioning import ModelRegistry
import argparse
import os
import sys
from pathlib import Path
from tabulate import tabulate

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def list_models(args):
    """List models in the registry."""
    models_dir = Path(settings.data_processed_dir) / "models"
    registry = ModelRegistry(models_dir)

    models = registry.list_models(status=args.status, limit=args.limit)

    if not models:
        print("No models found in registry.")
        return

    # Prepare table data
    headers = ["Name", "Version", "Status", "Created", "Key Metrics"]
    rows = []

    for model in models:
        # Format key metrics
        metrics_str = ", ".join([
            f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in list(model.metrics.items())[:3]  # Show first 3 metrics
        ])

        rows.append([
            model.name,
            model.version,
            model.status,
            model.created_at[:10],  # Just the date
            metrics_str,
        ])

    print(tabulate(rows, headers=headers, tablefmt="grid"))
    print(f"\nTotal: {len(models)} model(s)")


def show_production(args):
    """Show the current production model."""
    models_dir = Path(settings.data_processed_dir) / "models"
    registry = ModelRegistry(models_dir)

    prod_model = registry.get_production_model()

    if not prod_model:
        print("No model is currently in production.")
        return

    print(f"Production Model: {prod_model.name} v{prod_model.version}")
    print(f"Status: {prod_model.status}")
    print(f"Created: {prod_model.created_at}")
    print(f"Promoted: {prod_model.promoted_at}")
    print(f"Promoted by: {prod_model.promoted_by}")
    print(f"\nMetrics:")
    for key, value in prod_model.metrics.items():
        print(f"  {key}: {value}")

    if prod_model.notes:
        print(f"\nNotes: {prod_model.notes}")


def promote_model(args):
    """Promote a model to production."""
    models_dir = Path(settings.data_processed_dir) / "models"
    registry = ModelRegistry(models_dir)

    # Check if model exists
    models = registry.list_models()
    target = None
    for m in models:
        if m.name == args.name and m.version == args.version:
            target = m
            break

    if not target:
        print(
            f"Error: Model {args.name} v{args.version} not found in registry.")
        print("\nAvailable models:")
        for m in models:
            print(f"  - {m.name} v{m.version} ({m.status})")
        return

    # Confirm promotion
    if not args.yes:
        print(f"\nAbout to promote {args.name} v{args.version} to production")
        print(f"Current status: {target.status}")
        print(f"\nMetrics:")
        for key, value in target.metrics.items():
            print(f"  {key}: {value}")

        confirm = input("\nContinue? (yes/no): ")
        if confirm.lower() != "yes":
            print("Promotion cancelled.")
            return

    # Promote
    promoted_by = args.promoted_by or os.getenv("USER", "system")
    result = registry.promote_to_production(
        args.name, args.version, promoted_by)

    if result:
        print(
            f"\n✓ Successfully promoted {args.name} v{args.version} to production")
    else:
        print(f"\n✗ Failed to promote model")


def compare_models(args):
    """Compare two models."""
    models_dir = Path(settings.data_processed_dir) / "models"
    registry = ModelRegistry(models_dir)

    # Parse model specifications
    try:
        name1, version1 = args.model1.split(":")
        name2, version2 = args.model2.split(":")
    except ValueError:
        print("Error: Model specifications must be in format 'name:version'")
        return

    comparison = registry.compare_models(name1, version1, name2, version2)

    if "error" in comparison:
        print(f"Error: {comparison['error']}")
        return

    print(f"\nComparing {name1} v{version1} vs {name2} v{version2}\n")
    print("=" * 80)

    # Show metric differences
    if comparison["differences"]:
        headers = ["Metric", "Model 1", "Model 2", "Difference", "% Change"]
        rows = []

        for metric, data in comparison["differences"].items():
            pct_change = data["percentage_change"]
            pct_str = f"{pct_change:+.2f}%" if pct_change is not None else "N/A"

            rows.append([
                metric,
                f"{data['model1']:.4f}" if isinstance(
                    data['model1'], float) else data['model1'],
                f"{data['model2']:.4f}" if isinstance(
                    data['model2'], float) else data['model2'],
                f"{data['difference']:+.4f}" if isinstance(
                    data['difference'], float) else data['difference'],
                pct_str,
            ])

        print(tabulate(rows, headers=headers, tablefmt="grid"))
    else:
        print("No common metrics found for comparison.")


def main():
    parser = argparse.ArgumentParser(description="Manage NBA model versions")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # List command
    list_parser = subparsers.add_parser("list", help="List models")
    list_parser.add_argument("--status", choices=["candidate", "production", "archived"],
                             help="Filter by status")
    list_parser.add_argument(
        "--limit", type=int, help="Limit number of results")

    # Production command
    prod_parser = subparsers.add_parser(
        "production", help="Show current production model")

    # Promote command
    promote_parser = subparsers.add_parser(
        "promote", help="Promote model to production")
    promote_parser.add_argument("--name", required=True, help="Model name")
    promote_parser.add_argument(
        "--version", required=True, help="Model version")
    promote_parser.add_argument("--promoted-by", help="Name of promoter")
    promote_parser.add_argument("--yes", "-y", action="store_true",
                                help="Skip confirmation prompt")

    # Compare command
    compare_parser = subparsers.add_parser(
        "compare", help="Compare two models")
    compare_parser.add_argument("--model1", required=True,
                                help="First model (name:version)")
    compare_parser.add_argument("--model2", required=True,
                                help="Second model (name:version)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "list":
        list_models(args)
    elif args.command == "production":
        show_production(args)
    elif args.command == "promote":
        promote_model(args)
    elif args.command == "compare":
        compare_models(args)


if __name__ == "__main__":
    main()
