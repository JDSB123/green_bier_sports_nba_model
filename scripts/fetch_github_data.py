#!/usr/bin/env python3
"""
Fetch GitHub-hosted NBA data.

This script demonstrates how to use the GitHub data fetcher module to download
historical NBA data from open-source repositories on GitHub.

Usage:
    # Fetch FiveThirtyEight ELO data
    python scripts/fetch_github_data.py --source fivethirtyeight --dataset elo_historical

    # Fetch all FiveThirtyEight datasets
    python scripts/fetch_github_data.py --source fivethirtyeight --all

    # Fetch custom GitHub URL
    python scripts/fetch_github_data.py --url "https://raw.githubusercontent.com/user/repo/branch/file.csv"

    # List available sources
    python scripts/fetch_github_data.py --list-sources
"""

import asyncio
import argparse
import sys
from pathlib import Path
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.github_data import (
    GitHubDataFetcher,
    fetch_fivethirtyeight_elo,
    fetch_fivethirtyeight_all,
    COMMON_NBA_SOURCES,
    FIVETHIRTYEIGHT_URLS,
)


async def main():
    parser = argparse.ArgumentParser(
        description="Fetch NBA data from GitHub repositories"
    )
    parser.add_argument(
        "--source",
        choices=["fivethirtyeight", "custom"],
        help="Data source to fetch from",
    )
    parser.add_argument(
        "--dataset",
        choices=list(FIVETHIRTYEIGHT_URLS.keys()),
        help="Specific dataset to fetch (for FiveThirtyEight)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Fetch all available datasets from source",
    )
    parser.add_argument(
        "--url",
        help="Custom GitHub raw content URL to fetch",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "json", "text", "auto"],
        default="auto",
        help="File format (auto-detects from URL if not specified)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching (always fetch fresh data)",
    )
    parser.add_argument(
        "--list-sources",
        action="store_true",
        help="List all available data sources",
    )
    parser.add_argument(
        "--output",
        help="Output file path (optional, prints to stdout if not specified)",
    )

    args = parser.parse_args()

    # List sources
    if args.list_sources:
        print("Available FiveThirtyEight datasets:")
        for name, url in FIVETHIRTYEIGHT_URLS.items():
            print(f"  - {name}: {url}")
        print("\nCommon NBA sources:")
        for name, url in COMMON_NBA_SOURCES.items():
            print(f"  - {name}: {url}")
        return

    use_cache = not args.no_cache
    fetcher = GitHubDataFetcher()

    try:
        # Fetch FiveThirtyEight data
        if args.source == "fivethirtyeight":
            if args.all:
                print("Fetching all FiveThirtyEight datasets...")
                results = await fetch_fivethirtyeight_all()
                for name, df in results.items():
                    if not df.empty:
                        print(f"✓ {name}: {len(df)} rows")
                        if args.output:
                            output_path = Path(args.output) / f"{name}.csv"
                            output_path.parent.mkdir(parents=True, exist_ok=True)
                            df.to_csv(output_path, index=False)
                            print(f"  Saved to {output_path}")
                    else:
                        print(f"✗ {name}: Failed to fetch")
            elif args.dataset:
                print(f"Fetching FiveThirtyEight dataset: {args.dataset}...")
                df = await fetch_fivethirtyeight_elo(args.dataset, use_cache=use_cache)
                print(f"✓ Fetched {len(df)} rows")
                print(f"\nFirst few rows:")
                print(df.head())
                if args.output:
                    output_path = Path(args.output)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    df.to_csv(output_path, index=False)
                    print(f"\nSaved to {output_path}")
            else:
                print("Error: Specify --dataset or --all for FiveThirtyEight source")
                parser.print_help()
                return

        # Fetch custom URL
        elif args.url:
            print(f"Fetching from custom URL: {args.url}...")
            format = args.format if args.format != "auto" else None
            result = await fetcher.fetch(args.url, format=format, use_cache=use_cache)

            if result.error:
                print(f"✗ Error: {result.error}")
                return

            print(f"✓ Fetched {result.format} data")
            if result.cached:
                print("  (from cache)")

            if isinstance(result.data, pd.DataFrame):
                print(f"  Rows: {len(result.data)}")
                print(f"\nFirst few rows:")
                print(result.data.head())
                if args.output:
                    output_path = Path(args.output)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    result.data.to_csv(output_path, index=False)
                    print(f"\nSaved to {output_path}")
            elif isinstance(result.data, (dict, list)):
                import json

                print(f"  JSON keys: {list(result.data.keys()) if isinstance(result.data, dict) else 'list'}")
                if args.output:
                    output_path = Path(args.output)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(output_path, "w") as f:
                        json.dump(result.data, f, indent=2)
                    print(f"\nSaved to {output_path}")
            else:
                print(f"  Content length: {len(result.data)} bytes")
                if args.output:
                    output_path = Path(args.output)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_bytes(result.data)
                    print(f"\nSaved to {output_path}")

        else:
            print("Error: Specify --source, --url, or --list-sources")
            parser.print_help()

    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
