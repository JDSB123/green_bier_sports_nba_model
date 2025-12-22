"""
Validation tests for Single Source of Truth principle fixes.

Ensures all 3 violations have been resolved:
1. Injury data aggregation through single source (fetch_all_injuries)
2. Team name normalization via single source (src.utils.team_names.normalize_team_name)
3. Odds data uses unified fetch_odds() endpoint
"""
import pytest
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestViolation1InjuryDataFixed:
    """Test Fix #1: Injury data aggregation"""

    def test_no_fetch_injuries_espn_in_comprehensive(self):
        """Verify comprehensive.py doesn't directly import fetch_injuries_espn"""
        comp_file = Path(__file__).parent.parent / "src" / "ingestion" / "comprehensive.py"
        with open(comp_file, encoding='utf-8') as f:
            content = f.read()
        
        # Should NOT directly import fetch_injuries_espn
        assert "from src.ingestion.injuries import fetch_injuries_espn" not in content
        assert "import fetch_injuries_espn" not in content
        print("✅ Violation #1 FIXED: comprehensive.py no longer imports fetch_injuries_espn directly")

    def test_comprehensive_uses_fetch_all_injuries(self):
        """Verify comprehensive.py uses fetch_all_injuries"""
        comp_file = Path(__file__).parent.parent / "src" / "ingestion" / "comprehensive.py"
        with open(comp_file, encoding='utf-8') as f:
            content = f.read()
        
        # Should use fetch_all_injuries
        assert "fetch_all_injuries" in content
        print("✅ Fix #1 VERIFIED: comprehensive.py now uses fetch_all_injuries")


class TestViolation2TeamNamesFixed:
    """Test Fix #2: Team name normalization consolidation"""

    def test_team_factors_consolidated(self):
        """Verify team_factors.py uses single source"""
        team_factors_file = Path(__file__).parent.parent / "src" / "modeling" / "team_factors.py"
        with open(team_factors_file, encoding='utf-8') as f:
            content = f.read()
        
        # Should NOT have TEAM_ALIASES dict
        assert "TEAM_ALIASES: Dict[str, str] = {" not in content
        
        # Should NOT have local normalize_team_name function
        assert "def normalize_team_name(team_name: str) -> str:" not in content
        
        # Should import from utils
        assert "from src.utils.team_names import normalize_team_name" in content
        print("✅ Violation #2A FIXED: team_factors.py uses single source")

    def test_dataset_consolidated(self):
        """Verify dataset.py uses single source"""
        dataset_file = Path(__file__).parent.parent / "src" / "modeling" / "dataset.py"
        with open(dataset_file, encoding='utf-8') as f:
            content = f.read()
        
        # Should NOT have TEAM_NAME_MAP dict
        assert "TEAM_NAME_MAP = {" not in content
        
        # Should NOT have local _normalize_team_name method
        assert "def _normalize_team_name(self, name: str)" not in content
        
        # Should import from utils
        assert "from src.utils.team_names import normalize_team_name" in content
        
        # Should use imported function, not self._normalize_team_name
        assert "self._normalize_team_name(" not in content
        print("✅ Violation #2B FIXED: dataset.py uses single source")


class TestViolation3OddsFixed:
    """Test Fix #3: Unified odds endpoint"""

    def test_no_fetch_historical_odds_import_in_training_script(self):
        """Verify build_fresh_training_data.py doesn't import fetch_historical_odds"""
        script_file = Path(__file__).parent.parent / "scripts" / "build_fresh_training_data.py"
        with open(script_file, encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Should NOT import fetch_historical_odds directly
        assert "from src.ingestion.the_odds import fetch_historical_odds" not in content
        assert "fetch_historical_odds," not in content
        print("✅ Violation #3 FIXED: build_fresh_training_data.py no longer imports fetch_historical_odds")

    def test_uses_unified_fetch_odds(self):
        """Verify build_fresh_training_data.py uses unified fetch_odds"""
        script_file = Path(__file__).parent.parent / "scripts" / "build_fresh_training_data.py"
        with open(script_file, encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Should import fetch_odds
        assert "fetch_odds" in content
        
        # Should NOT have historical_available branching logic
        assert "historical_available" not in content
        print("✅ Fix #3 VERIFIED: build_fresh_training_data.py uses unified fetch_odds endpoint")


class TestNoRegressions:
    """Verify fixes don't break existing functionality"""

    def test_normalize_team_name_works(self):
        """Verify normalize_team_name function still works"""
        from src.utils.team_names import normalize_team_name
        
        # Should normalize shorthand names
        result = normalize_team_name("lakers")
        assert result == "nba_lal"
        print(f"✅ normalize_team_name('lakers') = '{result}'")

    def test_team_factors_can_access_normalize(self):
        """Verify team_factors can use the imported normalize_team_name"""
        from src.modeling.team_factors import normalize_team_name
        
        # Should be able to import and use
        result = normalize_team_name("bucks")
        assert result == "nba_mil"
        print(f"✅ team_factors can use normalize_team_name('bucks') = '{result}'")


class TestSummary:
    """Summary of all fixes"""
    
    def test_all_violations_resolved(self):
        """Comprehensive check that all 3 violations are resolved"""
        comp_file = Path(__file__).parent.parent / "src" / "ingestion" / "comprehensive.py"
        team_factors_file = Path(__file__).parent.parent / "src" / "modeling" / "team_factors.py"
        dataset_file = Path(__file__).parent.parent / "src" / "modeling" / "dataset.py"
        script_file = Path(__file__).parent.parent / "scripts" / "build_fresh_training_data.py"
        
        with open(comp_file, encoding='utf-8') as f:
            comp_content = f.read()
        with open(team_factors_file, encoding='utf-8') as f:
            tf_content = f.read()
        with open(dataset_file, encoding='utf-8') as f:
            ds_content = f.read()
        with open(script_file, encoding='utf-8', errors='ignore') as f:
            script_content = f.read()
        
        # Violation #1: Injury data
        assert "from src.ingestion.injuries import fetch_injuries_espn" not in comp_content
        
        # Violation #2: Team names
        assert "TEAM_ALIASES: Dict[str, str] = {" not in tf_content
        assert "def _normalize_team_name(self, name: str)" not in ds_content
        assert "from src.utils.team_names import normalize_team_name" in tf_content
        assert "from src.utils.team_names import normalize_team_name" in ds_content
        
        # Violation #3: Odds
        assert "from src.ingestion.the_odds import fetch_historical_odds" not in script_content
        assert "historical_available" not in script_content
        
        print("\n" + "="*60)
        print("✅ ALL VIOLATIONS RESOLVED")
        print("="*60)
        print("Fix #1 (Injury aggregation):    ✅ COMPLETE")
        print("Fix #2 (Team name consolidation): ✅ COMPLETE")  
        print("Fix #3 (Unified odds endpoint):   ✅ COMPLETE")
        print("="*60)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
