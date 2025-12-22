"""
Validation tests for Single Source of Truth principle.

Ensures that:
1. Injury data is aggregated through fetch_all_injuries() only
2. Team names are normalized via single source (src.utils.team_names.normalize_team_name)
3. Odds data uses unified fetch_odds() endpoint
"""
import pytest
import sys
import os
import ast
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.team_names import normalize_team_name
from src.ingestion.injuries import fetch_all_injuries


class TestInjuryDataSource:
    """Test #1: Injury data aggregation through single source"""

    def test_fetch_all_injuries_is_single_source(self):
        """Verify fetch_all_injuries aggregates all sources"""
        # This is documented in src/ingestion/injuries.py line 274
        import inspect
        source = inspect.getsource(fetch_all_injuries)
        
        # Should contain calls to both ESP and API-Basketball sources
        assert "fetch_injuries_espn" in source
        assert "fetch_injuries_api_basketball" in source
        assert "results.extend" in source or "append" in source
        
    def test_comprehensive_py_uses_fetch_all_injuries(self):
        """Verify comprehensive.py imports and uses fetch_all_injuries()"""
        comp_file = Path(__file__).parent.parent / "src" / "ingestion" / "comprehensive.py"
        with open(comp_file, encoding='utf-8') as f:
            content = f.read()
        
        # Should NOT directly import fetch_injuries_espn
        assert "from src.ingestion.injuries import fetch_injuries_espn" not in content
        
        # Should import fetch_all_injuries
        assert "fetch_all_injuries" in content
        
        # Should call it (not just import)
        # Can be passed as fetch_fn parameter in get_or_fetch or called directly
        assert "fetch_fn=fetch_all_injuries" in content or "await fetch_all_injuries" in content
        
    def test_no_direct_espn_injury_calls_in_scripts(self):
        """Verify scripts don't directly call fetch_injuries_espn"""
        scripts_dir = Path(__file__).parent.parent / "scripts"
        scripts_to_check = list(scripts_dir.glob("*.py"))
        
        violations = []
        for script_file in scripts_to_check:
            if script_file.name == "test_single_source_of_truth.py":
                continue
            with open(script_file, encoding='utf-8') as f:
                content = f.read()
            
            if "from src.ingestion.injuries import fetch_injuries_espn" in content:
                violations.append(f"{script_file.name}: imports fetch_injuries_espn directly")
            if "fetch_injuries_espn(" in content and "fetch_all_injuries" not in content:
                violations.append(f"{script_file.name}: calls fetch_injuries_espn directly")
        
        assert not violations, f"Violations found: {violations}"


class TestTeamNameNormalization:
    """Test #2: Team name normalization through single source"""

    def test_normalize_team_name_single_source(self):
        """Verify normalize_team_name works correctly"""
        # Test shorthand
        assert normalize_team_name("lakers") == "nba_lal"
        assert normalize_team_name("Lakers") == "nba_lal"
        assert normalize_team_name("LA Lakers") == "nba_lal"
        
        # Test that it's consistent
        assert normalize_team_name("lakers") == normalize_team_name("Lakers")
        assert normalize_team_name("lakers") == normalize_team_name("Los Angeles Lakers")

    def test_team_factors_uses_single_source(self):
        """Verify team_factors.py imports normalize_team_name from utils"""
        team_factors_file = Path(__file__).parent.parent / "src" / "modeling" / "team_factors.py"
        with open(team_factors_file, encoding='utf-8') as f:
            content = f.read()
        
        # Should NOT have local TEAM_ALIASES dict
        assert "TEAM_ALIASES: Dict[str, str] = {" not in content
        
        # Should NOT have local normalize_team_name function
        assert "def normalize_team_name(team_name: str) -> str:" not in content
        
        # Should import from utils
        assert "from src.utils.team_names import normalize_team_name" in content

    def test_dataset_uses_single_source(self):
        """Verify dataset.py imports normalize_team_name from utils"""
        dataset_file = Path(__file__).parent.parent / "src" / "modeling" / "dataset.py"
        with open(dataset_file, encoding='utf-8') as f:
            content = f.read()
        
        # Should NOT have local TEAM_NAME_MAP dict
        assert "TEAM_NAME_MAP = {" not in content
        
        # Should NOT have local _normalize_team_name method
        assert "def _normalize_team_name(self, name: str)" not in content
        
        # Should import from utils
        assert "from src.utils.team_names import normalize_team_name" in content
        
        # Should use imported function (can be used with .apply() or direct call)
        assert "normalize_team_name" in content and "from src.utils.team_names import normalize_team_name" in content
        assert "self._normalize_team_name(" not in content

    def test_no_duplicate_team_name_implementations(self):
        """Verify no duplicate team name normalization implementations exist"""
        src_dir = Path(__file__).parent.parent / "src"
        
        duplicate_defs = []
        for py_file in src_dir.rglob("*.py"):
            # Skip the canonical implementation
            if "src/utils/team_names.py" in str(py_file).replace('\\', '/'):
                continue
                
            with open(py_file, encoding='utf-8') as f:
                lines = f.readlines()
            
            # Check for actual function definitions (not imports/comments)
            for line in lines:
                stripped = line.strip()
                if (stripped.startswith('def normalize_team_name(') or 
                    stripped.startswith('def _normalize_team_name(')) and not stripped.startswith('#'):
                    duplicate_defs.append(str(py_file))
                    break
        
        assert not duplicate_defs, f"Duplicate normalize_team_name implementations found in: {duplicate_defs}"


class TestOddsDataSource:
    """Test #3: Odds data through unified source"""

    def test_build_fresh_training_data_uses_single_odds_source(self):
        """Verify build_fresh_training_data.py uses fetch_odds() only"""
        script_file = Path(__file__).parent.parent / "scripts" / "build_fresh_training_data.py"
        with open(script_file, encoding='utf-8') as f:
            content = f.read()
        
        # Should NOT import fetch_historical_odds directly
        assert "from src.ingestion.the_odds import fetch_historical_odds" not in content
        
        # Should import fetch_odds
        assert "from src.ingestion.the_odds import" in content
        assert "fetch_odds" in content
        
        # Should NOT have conditional branches for historical vs current
        # (Check for patterns that indicate dual paths)
        lines = content.split("\n")
        historical_refs = 0
        for i, line in enumerate(lines):
            if "fetch_historical_odds" in line:
                historical_refs += 1
            if "historical_available" in line:
                historical_refs += 1
        
        # Allow some references in comments/documentation, but not in active code
        assert historical_refs < 2, "fetch_historical_odds should be removed from active code"

    def test_the_odds_module_provides_unified_endpoint(self):
        """Verify fetch_odds() in the_odds module handles routing"""
        import inspect
        from src.ingestion.the_odds import fetch_odds
        
        source = inspect.getsource(fetch_odds)
        # Should use standardized team name formatting
        assert "standardize_game_data" in source or "standardize" in source.lower()


class TestArchitecturalConsistency:
    """Test that single source principle is maintained across pipeline"""

    def test_feature_consistency_in_training_vs_prediction(self):
        """Verify team names and odds use same sources in both paths"""
        # Load training script
        train_script = Path(__file__).parent.parent / "scripts" / "build_fresh_training_data.py"
        with open(train_script, encoding='utf-8') as f:
            train_content = f.read()
        
        # Check that training uses standardized team name handling (normalize_team_to_espn)
        assert "normalize_team_to_espn" in train_content, "Training script should use normalize_team_to_espn for consistent team name handling"
        
        assert "fetch_odds" in train_content

    def test_all_ingestion_functions_documented(self):
        """Verify single sources are documented"""
        doc_file = Path(__file__).parent.parent / "docs" / "DATA_SOURCE_OF_TRUTH.md"
        assert doc_file.exists(), "DATA_SOURCE_OF_TRUTH.md should exist and document single sources"
        
        with open(doc_file, encoding='utf-8') as f:
            content = f.read()
        
        # Should document the ingestion single sources
        assert "fetch_all_injuries" in content
        assert "fetch_odds" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
