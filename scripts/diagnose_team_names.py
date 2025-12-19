#!/usr/bin/env python3
"""
Team Name Format Diagnostic Tool

Fetches data from ALL sources and documents:
1. Home/Away team field formats from each source
2. All team variant names found in each source
3. Standardization mapping to ESPN format
4. Validation of standardization coverage

This helps ensure we can properly standardize all team names across sources.
"""
import asyncio
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.ingestion.api_basketball import APIBasketballClient
from src.ingestion.the_odds import (
    fetch_odds,
    fetch_events,
    fetch_participants,
    fetch_event_odds,
)
from src.ingestion.standardize import normalize_team_to_espn
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "data" / "diagnostics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class TeamNameDiagnostic:
    """Diagnostic tool to analyze team name formats from all sources."""

    def __init__(self):
        self.source_formats: Dict[str, Dict] = {}
        self.team_variants: Dict[str, Set[str]] = defaultdict(set)
        self.standardization_results: Dict[str, Dict] = {}

    async def diagnose_api_basketball(self):
        """Diagnose API-Basketball team name formats."""
        print("\n" + "=" * 80)
        print("DIAGNOSING API-BASKETBALL")
        print("=" * 80)

        client = APIBasketballClient(season=settings.current_season)

        # Fetch teams
        print("\n[1] Fetching teams endpoint...")
        teams_result = await client.fetch_teams()
        teams_data = teams_result.data.get("response", [])

        team_names_api_basketball = set()
        team_format_info = []

        for team in teams_data[:5]:  # Sample first 5
            name = team.get("name", "")
            code = team.get("code", "")
            id_val = team.get("id", "")
            team_names_api_basketball.add(name)

            team_format_info.append({
                "source": "API-Basketball",
                "endpoint": "/teams",
                "field": "name",
                "example_value": name,
                "also_has": f"code={code}, id={id_val}",
            })

        # Fetch games - we'll get raw data before standardization
        print("[2] Fetching games endpoint...")
        # First fetch raw data to see format
        raw_data = await client._fetch(
            "games", {"league": client.league_id, "season": client.season}
        )
        games_data = raw_data.get("response", [])

        home_team_formats = set()
        away_team_formats = set()

        for game in games_data[:10]:  # Sample first 10 games
            teams = game.get("teams", {})
            home_obj = teams.get("home", {})
            away_obj = teams.get("away", {})

            home_name = home_obj.get("name", "")
            away_name = away_obj.get("name", "")

            if home_name:
                home_team_formats.add(home_name)
                self.team_variants["api_basketball"].add(home_name)

            if away_name:
                away_team_formats.add(away_name)
                self.team_variants["api_basketball"].add(away_name)

            team_format_info.append({
                "source": "API-Basketball",
                "endpoint": "/games",
                "field": "teams.home.name",
                "example_value": home_name,
                "away_field": "teams.away.name",
                "away_example": away_name,
                "format_structure": "nested: teams.home.name, teams.away.name",
            })

        # Fetch statistics
        print("[3] Fetching statistics endpoint...")
        if teams_data:
            first_team_id = teams_data[0].get("id")
            if first_team_id:
                stats_result = await client.fetch_statistics()
                # Stats don't typically have team names, just ID references

        self.source_formats["api_basketball"] = {
            "source": "API-Basketball",
            "base_url": settings.api_basketball_base_url,
            "endpoints": [
                {
                    "endpoint": "/v1/teams",
                    "team_field": "name",
                    "structure": "flat: {name, code, id}",
                    "examples": list(team_names_api_basketball)[:10],
                },
                {
                    "endpoint": "/v1/games",
                    "home_field": "teams.home.name",
                    "away_field": "teams.away.name",
                    "structure": "nested: teams.home.name, teams.away.name",
                    "home_examples": list(home_team_formats)[:10],
                    "away_examples": list(away_team_formats)[:10],
                },
            ],
            "team_variants_found": list(self.team_variants["api_basketball"]),
            "total_unique_variants": len(self.team_variants["api_basketball"]),
        }

        print(f"✓ Found {len(self.team_variants['api_basketball'])} unique team name variants")
        return team_format_info

    async def diagnose_the_odds_api(self):
        """Diagnose The Odds API team name formats."""
        print("\n" + "=" * 80)
        print("DIAGNOSING THE ODDS API")
        print("=" * 80)

        team_format_info = []

        # Fetch participants
        print("\n[1] Fetching participants endpoint...")
        try:
            participants = await fetch_participants(standardize=False)
            participant_names = set()

            for participant in participants[:10]:  # Sample
                name = participant.get("name") or participant.get("team") or participant.get("id")
                if name:
                    participant_names.add(str(name))
                    self.team_variants["the_odds"].add(str(name))

                team_format_info.append({
                    "source": "The Odds API",
                    "endpoint": "/sports/basketball_nba/participants",
                    "field": "name (or team or id)",
                    "example_value": name,
                    "format_structure": "flat: {name, team, id}",
                })

            print(f"  Found {len(participant_names)} participant names")
        except Exception as e:
            print(f"  ⚠️  Could not fetch participants: {e}")

        # Fetch events
        print("[2] Fetching events endpoint...")
        try:
            events = await fetch_events(standardize=False)
            event_home_teams = set()
            event_away_teams = set()

            for event in events[:10]:  # Sample
                home_team = event.get("home_team", "")
                away_team = event.get("away_team", "")

                if home_team:
                    event_home_teams.add(home_team)
                    self.team_variants["the_odds"].add(home_team)

                if away_team:
                    event_away_teams.add(away_team)
                    self.team_variants["the_odds"].add(away_team)

                team_format_info.append({
                    "source": "The Odds API",
                    "endpoint": "/sports/basketball_nba/events",
                    "home_field": "home_team",
                    "away_field": "away_team",
                    "home_example": home_team,
                    "away_example": away_team,
                    "format_structure": "flat: {home_team, away_team}",
                })

            print(f"  Found {len(event_home_teams)} unique home team names")
            print(f"  Found {len(event_away_teams)} unique away team names")
        except Exception as e:
            print(f"  ⚠️  Could not fetch events: {e}")

        # Fetch odds - fetch raw to see format
        print("[3] Fetching odds endpoint...")
        try:
            # Fetch without standardization by calling API directly
            import httpx
            params = {
                "apiKey": settings.the_odds_api_key,
                "regions": "us",
                "markets": "h2h,spreads,totals",
                "oddsFormat": "american",
                "dateFormat": "iso",
            }
            async with httpx.AsyncClient(timeout=30) as http_client:
                resp = await http_client.get(
                    f"{settings.the_odds_base_url}/sports/basketball_nba/odds",
                    params=params
                )
                resp.raise_for_status()
                odds_data = resp.json()  # Raw data, not standardized
            odds_home_teams = set()
            odds_away_teams = set()

            for game in odds_data[:10]:  # Sample
                home_team = game.get("home_team", "")
                away_team = game.get("away_team", "")

                # Check structure
                if isinstance(home_team, dict):
                    home_name = home_team.get("name", "")
                else:
                    home_name = str(home_team) if home_team else ""

                if isinstance(away_team, dict):
                    away_name = away_team.get("name", "")
                else:
                    away_name = str(away_team) if away_team else ""

                if home_name:
                    odds_home_teams.add(home_name)
                    self.team_variants["the_odds"].add(home_name)

                if away_name:
                    odds_away_teams.add(away_name)
                    self.team_variants["the_odds"].add(away_name)

                team_format_info.append({
                    "source": "The Odds API",
                    "endpoint": "/sports/basketball_nba/odds",
                    "home_field": "home_team (string or dict.name)",
                    "away_field": "away_team (string or dict.name)",
                    "home_example": home_name,
                    "away_example": away_name,
                    "format_structure": "flat: {home_team, away_team} (strings)",
                })

            print(f"  Found {len(odds_home_teams)} unique home team names in odds")
            print(f"  Found {len(odds_away_teams)} unique away team names in odds")
        except Exception as e:
            print(f"  ⚠️  Could not fetch odds: {e}")

        self.source_formats["the_odds"] = {
            "source": "The Odds API",
            "base_url": settings.the_odds_base_url,
            "endpoints": [
                {
                    "endpoint": "/v4/sports/basketball_nba/participants",
                    "team_field": "name (or team or id)",
                    "structure": "flat",
                },
                {
                    "endpoint": "/v4/sports/basketball_nba/events",
                    "home_field": "home_team",
                    "away_field": "away_team",
                    "structure": "flat: {home_team, away_team}",
                },
                {
                    "endpoint": "/v4/sports/basketball_nba/odds",
                    "home_field": "home_team",
                    "away_field": "away_team",
                    "structure": "flat: {home_team, away_team} (strings)",
                },
            ],
            "team_variants_found": list(self.team_variants["the_odds"]),
            "total_unique_variants": len(self.team_variants["the_odds"]),
        }

        print(f"✓ Found {len(self.team_variants['the_odds'])} unique team name variants")
        return team_format_info

    def test_standardization(self):
        """Test standardization of all found variants."""
        print("\n" + "=" * 80)
        print("TESTING STANDARDIZATION")
        print("=" * 80)

        all_variants = set()
        for source_variants in self.team_variants.values():
            all_variants.update(source_variants)

        print(f"\nTesting {len(all_variants)} unique team name variants...")

        standardization_results = {
            "successful": {},
            "failed": {},
        }

        for variant in sorted(all_variants):
            normalized, is_valid = normalize_team_to_espn(variant, source="diagnostic")
            if is_valid:
                standardization_results["successful"][variant] = normalized
            else:
                standardization_results["failed"][variant] = "FAILED"

        print(f"\n✓ Successfully standardized: {len(standardization_results['successful'])}")
        print(f"✗ Failed to standardize: {len(standardization_results['failed'])}")

        if standardization_results["failed"]:
            print("\n⚠️  VARIANTS THAT FAILED STANDARDIZATION:")
            for variant in sorted(standardization_results["failed"].keys()):
                print(f"  - '{variant}'")

        self.standardization_results = standardization_results
        return standardization_results

    def generate_report(self):
        """Generate comprehensive diagnostic report."""
        report = []
        report.append("# Team Name Format Diagnostic Report")
        report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append("---\n")

        # Source formats
        report.append("## Source Team Name Formats\n")
        for source_key, source_info in self.source_formats.items():
            report.append(f"### {source_info['source']}\n")
            report.append(f"**Base URL:** {source_info['base_url']}\n")
            report.append(f"**Total Unique Variants Found:** {source_info['total_unique_variants']}\n")

            for endpoint in source_info["endpoints"]:
                report.append(f"\n**Endpoint:** `{endpoint['endpoint']}`\n")
                report.append(f"- Structure: {endpoint.get('structure', 'N/A')}\n")

                if "home_field" in endpoint:
                    report.append(f"- Home team field: `{endpoint['home_field']}`\n")
                    report.append(f"- Away team field: `{endpoint['away_field']}`\n")

                    if "home_examples" in endpoint:
                        report.append(f"- Home examples: {', '.join(endpoint['home_examples'][:5])}\n")
                        report.append(f"- Away examples: {', '.join(endpoint['away_examples'][:5])}\n")

                elif "team_field" in endpoint:
                    report.append(f"- Team field: `{endpoint['team_field']}`\n")
                    if "examples" in endpoint:
                        report.append(f"- Examples: {', '.join(endpoint['examples'][:10])}\n")

            report.append(f"\n**All Variants Found:**\n")
            for variant in sorted(source_info["team_variants_found"])[:20]:
                report.append(f"- `{variant}`\n")

            report.append("\n---\n")

        # Standardization results
        report.append("## Standardization Coverage\n\n")
        report.append(f"**Total Unique Variants:** {len(self.standardization_results.get('successful', {})) + len(self.standardization_results.get('failed', {}))}\n")
        report.append(f"**Successfully Standardized:** {len(self.standardization_results.get('successful', {}))}\n")
        report.append(f"**Failed to Standardize:** {len(self.standardization_results.get('failed', {}))}\n\n")

        if self.standardization_results.get("failed"):
            report.append("### ⚠️  Variants That Failed Standardization\n\n")
            report.append("| Variant | Status |\n")
            report.append("|---------|--------|\n")
            for variant in sorted(self.standardization_results["failed"].keys()):
                report.append(f"| `{variant}` | FAILED |\n")
            report.append("\n**Action Required:** Add these variants to `src/ingestion/team_mapping.json`\n\n")

        # Standardization mapping
        report.append("## Standardization Mapping\n\n")
        report.append("| Source Variant | ESPN Standard Name |\n")
        report.append("|----------------|-------------------|\n")

        # Group by ESPN name
        by_espn = defaultdict(list)
        for variant, espn_name in self.standardization_results.get("successful", {}).items():
            by_espn[espn_name].append(variant)

        for espn_name in sorted(by_espn.keys()):
            variants = by_espn[espn_name]
            for i, variant in enumerate(sorted(variants)):
                if i == 0:
                    report.append(f"| `{variant}` | **{espn_name}** |\n")
                else:
                    report.append(f"| `{variant}` | {espn_name} |\n")

        # Field format summary
        report.append("\n---\n")
        report.append("## Field Format Summary\n\n")
        report.append("| Source | Home Team Field | Away Team Field | Format Type |\n")
        report.append("|--------|----------------|-----------------|-------------|\n")
        report.append("| API-Basketball | `teams.home.name` | `teams.away.name` | Nested object |\n")
        report.append("| The Odds API (events) | `home_team` | `away_team` | Flat string |\n")
        report.append("| The Odds API (odds) | `home_team` | `away_team` | Flat string |\n")
        report.append("| The Odds API (participants) | N/A | N/A | List of teams |\n")

        report_text = "\n".join(report)

        # Save report
        report_path = OUTPUT_DIR / f"team_name_diagnostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        report_path.write_text(report_text)
        print(f"\n✓ Report saved to: {report_path}")

        # Save JSON data
        json_data = {
            "source_formats": self.source_formats,
            "team_variants": {k: list(v) for k, v in self.team_variants.items()},
            "standardization_results": self.standardization_results,
            "generated_at": datetime.now().isoformat(),
        }
        json_path = OUTPUT_DIR / f"team_name_diagnostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        json_path.write_text(json.dumps(json_data, indent=2))
        print(f"✓ JSON data saved to: {json_path}")

        return report_text

    async def run(self):
        """Run full diagnostic."""
        print("=" * 80)
        print("TEAM NAME FORMAT DIAGNOSTIC")
        print("=" * 80)
        print(f"Output directory: {OUTPUT_DIR}")

        try:
            # Diagnose API-Basketball
            await self.diagnose_api_basketball()

            # Diagnose The Odds API
            await self.diagnose_the_odds_api()

            # Test standardization
            self.test_standardization()

            # Generate report
            self.generate_report()

            print("\n" + "=" * 80)
            print("DIAGNOSTIC COMPLETE")
            print("=" * 80)

        except Exception as e:
            print(f"\n❌ Diagnostic failed: {e}")
            import traceback
            traceback.print_exc()
            raise


async def main():
    """Main entry point."""
    diagnostic = TeamNameDiagnostic()
    await diagnostic.run()


if __name__ == "__main__":
    asyncio.run(main())
