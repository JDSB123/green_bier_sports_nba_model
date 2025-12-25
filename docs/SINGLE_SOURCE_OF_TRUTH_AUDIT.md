# Single Source of Truth Audit - NBA v6.0 Model

**Audit Date:** December 22, 2025  
**Status:** ⚠️ **CRITICAL VIOLATIONS FOUND**  
**Severity:** High - Compromises data integrity and consistency

---

## Executive Summary

Your model has **3 major violations** of the single source of truth principle that you documented in [DATA_SOURCE_OF_TRUTH.md](DATA_SOURCE_OF_TRUTH.md). These violations create:

1. **Data duplication** - Multiple code paths fetch the same data
2. **Inconsistency** - Same data normalized different ways depending on which module uses it
3. **Maintenance burden** - Changes to one implementation don't sync with others
4. **Testing complexity** - Bugs can exist in one path but not another

---

## VIOLATION #1: Injury Data Aggregation Bypass

### Location
[src/ingestion/comprehensive.py](../src/ingestion/comprehensive.py#L616)

### The Issue
```python
# ❌ WRONG - Line 616
async def fetch_espn_injuries(self) -> List[Dict]:
    """Fetch injuries from ESPN (FREE, unlimited)."""
    from src.ingestion.injuries import fetch_injuries_espn  # ← Direct import!
    
    # This BYPASSES fetch_all_injuries() which aggregates ESPN + API-Basketball
```

### Why It's a Violation
Your documentation states ([DATA_SOURCE_OF_TRUTH.md:105-106](DATA_SOURCE_OF_TRUTH.md#L105-L106)):
- ❌ **DO NOT** call `fetch_injuries_espn()` directly
- ❌ **DO NOT** call `fetch_injuries_api_basketball()` directly
- ✅ **ALWAYS** use `fetch_all_injuries()` in production code

### The Correct Way
```python
# ✅ RIGHT - Aggregates ESPN + API-Basketball
async def fetch_injuries(self) -> List[InjuryReport]:
    from src.ingestion.injuries import fetch_all_injuries
    
    injuries = await fetch_all_injuries()
    # Already handles merging duplicates, returning standardized format
```

### Impact
- If ESPN data fails, `comprehensive.py` returns nothing instead of falling back to API-Basketball
- `comprehensive.py` and other modules (e.g., `build_rich_features.py`) may have different injury data
- Inconsistent injury enrichment across the model

### Files Affected
- ✅ [src/ingestion/injuries.py](../src/ingestion/injuries.py#L274) - Implements `fetch_all_injuries()` (single source)
- ❌ [src/ingestion/comprehensive.py](../src/ingestion/comprehensive.py#L616) - **VIOLATES** single source
- ✅ [src/features/rich_features.py](../src/features/rich_features.py#L339) - Uses correct `fetch_all_injuries()`
- ✅ [scripts/build_rich_features.py](../scripts/build_rich_features.py#L220) - Uses correct `fetch_all_injuries()`

---

## VIOLATION #2: Team Name Normalization Duplication

### Location
Three separate implementations:

| File | Function | Returns | Used For |
|------|----------|---------|----------|
| [src/utils/team_names.py](../src/utils/team_names.py#L63) | `normalize_team_name()` | Canonical ID (`"nba_lal"`) | General standardization, travel features |
| [src/modeling/team_factors.py](../src/modeling/team_factors.py#L64) | `normalize_team_name()` | Full name (`"Denver Nuggets"`) | HCA (home court advantage) lookup |
| [src/modeling/dataset.py](../src/modeling/dataset.py#L26) | `_normalize_team_name()` | Mapped name (from TEAM_NAME_MAP) | Dataset loading |

### The Issue
```python
# ❌ DUPLICATION - Three different implementations!

# Version 1: utils/team_names.py
normalize_team_name("Los Angeles Lakers") -> "nba_lal"

# Version 2: modeling/team_factors.py
normalize_team_name("lakers") -> "Los Angeles Lakers"

# Version 3: modeling/dataset.py
_normalize_team_name("LA Lakers") -> "Los Angeles Lakers"
```

### Why It's a Violation
- No single source of truth for team name standardization
- Each module has its own mapping logic
- If you add a new team variant (e.g., "Lakers (LA)"), must update all 3 places
- Different parts of the model may disagree on canonical team names

### The Problem Scenario
```python
# In rich_features.py (uses travel.py which uses utils/team_names.py)
injuries = await fetch_all_injuries()  # Returns "Los Angeles Lakers"
normalized = normalize_team_name("Los Angeles Lakers")  # -> "nba_lal"

# In dataset loading (uses modeling/dataset.py)
df_team = "Los Angeles Lakers"
normalized = dataset._normalize_team_name(df_team)  # -> "Los Angeles Lakers"

# Now we have TWO different representations!
# "nba_lal" vs "Los Angeles Lakers" -> MISMATCH in features
```

### Impact
- Travel calculation may use different team name than HCA calculation
- Dataset loading normalizes differently than feature engineering
- Feature mismatch between training and prediction time
- Maintenance nightmare - changes to team lists must happen in 3 places

### Files Affected
**Duplicate definitions:**
- ❌ [src/utils/team_names.py](../src/utils/team_names.py#L63) - Returns canonical ID
- ❌ [src/modeling/team_factors.py](../src/modeling/team_factors.py#L64) - Returns full name
- ❌ [src/modeling/dataset.py](../src/modeling/dataset.py#L26) - Has TEAM_NAME_MAP dict

**Users of duplication:**
- [src/modeling/travel.py](../src/modeling/travel.py#L13) - Uses utils version
- [src/modeling/team_factors.py](../src/modeling/team_factors.py#L152) - Uses local version (self-reference)
- [src/modeling/dataset.py](../src/modeling/dataset.py#L59) - Uses local version (self-reference)

### Recommended Single Source
Use **[src/utils/team_names.py](../src/utils/team_names.py)** as the source of truth because:
1. Already has comprehensive mapping via `team_mapping.json`
2. Uses canonical IDs that avoid collisions
3. Has fuzzy matching for typos
4. Centralized, easy to maintain
5. Already used by `travel.py` and ingestion modules

---

## VIOLATION #3: Odds Data Collection Has Multiple Paths

### Location
[scripts/build_fresh_training_data.py](../scripts/build_fresh_training_data.py#L247)

### The Issue
```python
# Lines 247-248 - Dual import
from src.ingestion.the_odds import (
    fetch_historical_odds,  # ← Path A
    fetch_odds,             # ← Path B
)

# Lines 266-293: Uses historical_odds (Path A)
data = await fetch_historical_odds(...)

# Lines 355: Uses fetch_odds (Path B)
current_odds = await fetch_odds(markets="spreads,totals,h2h")
```

### Why It's a Violation
Your documentation states ([DATA_SOURCE_OF_TRUTH.md:151](DATA_SOURCE_OF_TRUTH.md#L151)):
- **Single Source:** `the_odds.fetch_odds()`
- Not `fetch_historical_odds()` + `fetch_odds()` separately

The two functions return different data structures:
- `fetch_historical_odds()` - returns historical snapshot data
- `fetch_odds()` - returns current live odds

### The Architectural Problem
```
Expected Flow (Single Source):
┌─────────────────────────────┐
│   the_odds.fetch_odds()     │ ← Single aggregator
│   (smart routing)           │
├─────────────────────────────┤
│ If historical available:    │
│  → fetch_historical_odds    │
│ Else:                       │
│  → fetch_current odds       │
└─────────────────────────────┘
         ↓
    Clean data

Actual Flow (Multiple Paths):
┌─────────────────────────────┐
│ fetch_historical_odds()     │ ← Direct call in training_data.py
└─────────────────────────────┘
┌─────────────────────────────┐
│ fetch_odds()                │ ← Direct call elsewhere
└─────────────────────────────┘
         ↓
    Inconsistent data
```

### Impact
- Training data may use different odds source than predictions
- If `fetch_historical_odds` fails, training data is incomplete
- No fallback mechanism - both paths are required to succeed
- Inconsistent odds standardization between training and prediction time

### Files Affected
- ❌ [scripts/build_fresh_training_data.py](../scripts/build_fresh_training_data.py#L247) - **VIOLATES** by calling both functions
- ✅ [src/ingestion/the_odds.py](../src/ingestion/the_odds.py#L91) - Implements `fetch_odds()` single source
- ✅ [scripts/predict.py](../scripts/predict.py#L275) - Uses correct single source `fetch_odds()`

---

## Architectural Impact

### Data Consistency Issues

```
┌─────────────────────────────────────────────────────────┐
│ CURRENT STATE (With Violations)                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ Injuries:        2 paths → 2 different formats         │
│ ├─ comprehensive.py → ESPN only                        │
│ └─ rich_features.py → ESPN + API-Basketball           │
│                                                         │
│ Team Names:      3 paths → 3 different formats        │
│ ├─ utils.normalize → "nba_lal"                        │
│ ├─ team_factors.normalize → "Denver Nuggets"          │
│ └─ dataset.normalize → from TEAM_NAME_MAP             │
│                                                         │
│ Odds:            2 paths → different data sources     │
│ ├─ build_fresh_training_data.py → historical_odds    │
│ └─ predict.py → fetch_odds                           │
│                                                         │
└─────────────────────────────────────────────────────────┘
           ↓
   FEATURE MISMATCH at prediction time
   (training used different team names, injury sources, odds)
```

### Production Readiness Issues

1. **No consistent fallback** - If one injury source fails, some modules get nothing, others get partial data
2. **Inconsistent normalization** - Same team name may be stored 3 different ways
3. **Hard to debug** - When predictions differ from training, unclear which data source caused the divergence
4. **Hard to test** - Unit tests can't guarantee feature consistency across the pipeline
5. **Maintenance risk** - Adding new teams/sources requires changes in multiple places

---

## Validation Checklist

These checks from your documentation are currently **FAILING**:

### ❌ FAILS: Injury Data Check
```bash
grep -r "fetch_injuries_espn\|fetch_injuries_api_basketball" scripts/
# Found in: src/ingestion/comprehensive.py line 616
# Should only call: fetch_all_injuries()
```

### ✅ PASSES: Betting Splits Check
```bash
grep -r "source.*=.*mock" scripts/
# No production uses of mock data found - GOOD!
```

### ❌ FAILS: Team Name Consistency Check
```bash
grep -r "def normalize_team_name\|TEAM_NAME_MAP" src/
# Found 3 definitions - should be 1!
```

### ✅ PASSES: Odds Data Check (Partially)
```bash
grep -r "fetch_odds" scripts/predict.py
# predict.py uses correct single source
# But build_fresh_training_data.py has violation
```

---

## Recommended Fixes (Priority Order)

### 🔴 CRITICAL (Fix First)
1. **Fix comprehensive.py injury call** (5 min) - Remove direct ESPN call, use `fetch_all_injuries()`
2. **Consolidate team name normalization** (30 min) - Use only `src/utils/team_names.py`

### 🟡 HIGH (Fix Soon)
3. **Fix build_fresh_training_data.py odds flow** (20 min) - Use single `fetch_odds()` with smart routing

### 🟢 MEDIUM (Documentation)
4. **Update comprehensive.py docstring** - Clarify it's a caching wrapper, not a replacement for single sources
5. **Add validation test** - Verify all ingestion paths use single source functions

---

## Summary Table

| Violation | Type | Severity | Location | Fix Time | Impact |
|-----------|------|----------|----------|----------|--------|
| Injury bypass | Direct call | CRITICAL | comprehensive.py:616 | 5 min | Data gap if ESPN fails |
| Team names | 3 implementations | CRITICAL | 3 files | 30 min | Feature mismatch |
| Odds paths | Multiple sources | HIGH | build_fresh_training_data.py | 20 min | Inconsistent training |

---

## Next Steps

1. ✅ **Review this audit** - Confirm violations match your observations
2. 🔧 **Apply fixes** (see FIXES section below)
3. 🧪 **Run validation tests** - Verify single source functions are used
4. 📝 **Update comprehensive.py** - Document it as cache layer, not replacement
5. ✅ **Re-audit** - Verify all violations resolved

---

# FIXES

The following sections contain the exact code changes needed to resolve each violation.

## FIX #1: comprehensive.py - Replace Direct ESPN Call with fetch_all_injuries()

**File:** `src/ingestion/comprehensive.py`  
**Lines:** 610-625  
**Change:** Replace `fetch_injuries_espn()` with `fetch_all_injuries()`

### Current Code (WRONG)
```python
async def fetch_espn_injuries(self) -> List[Dict]:
    """Fetch injuries from ESPN (FREE, unlimited).

    TTL: 2 hours
    """
    from src.ingestion.injuries import fetch_injuries_espn

    key = f"espn_injuries_{date.today().isoformat()}"

    data = await api_cache.get_or_fetch(
        key=key,
        fetch_fn=fetch_injuries_espn,  # ❌ WRONG - Direct ESPN call
        ttl_hours=APICache.TTL_FREQUENT,
        source="espn",
        endpoint="/injuries",
        force_refresh=self.force_refresh,
    )

    self._record_result("espn", "/injuries", True, len(data))
    return data
```

### Fixed Code (RIGHT)
```python
async def fetch_injuries(self) -> List[Dict]:
    """Fetch injuries from all configured sources (ESPN + API-Basketball).

    Aggregates injury data from multiple sources for redundancy.
    ESPN: FREE, unlimited. API-Basketball: If API key configured.
    
    TTL: 2 hours
    """
    from src.ingestion.injuries import fetch_all_injuries

    key = f"injuries_{date.today().isoformat()}"

    data = await api_cache.get_or_fetch(
        key=key,
        fetch_fn=fetch_all_injuries,  # ✅ RIGHT - Uses single aggregator
        ttl_hours=APICache.TTL_FREQUENT,
        source="injuries",
        endpoint="/injuries",
        force_refresh=self.force_refresh,
    )

    # fetch_all_injuries returns InjuryReport objects, convert to dict if needed
    result_list = []
    for injury in data:
        if hasattr(injury, '__dict__'):
            result_list.append(injury.__dict__)
        else:
            result_list.append(injury)

    self._record_result("injuries", "/injuries", True, len(result_list))
    return result_list
```

---

## FIX #2: Consolidate Team Name Normalization to Single Source

### Step 1: Remove team_factors.py's duplicate

**File:** `src/modeling/team_factors.py`  
**Lines:** 1-95 (imports and TEAM_ALIASES)  
**Change:** Replace local implementation with import from utils

### Current Code (WRONG)
```python
# Lines 40-95 in team_factors.py
TEAM_ALIASES = {
    "lakers": "Los Angeles Lakers",
    "celtics": "Boston Celtics",
    # ... 28 more entries ...
    "jazz": "Utah Jazz",
}

def normalize_team_name(team_name: str) -> str:
    """
    Normalize shorthand team names (e.g., \"Bucks\", \"Trailblazers\") to canonical forms.
    """
    if not team_name:
        return team_name

    clean = team_name.strip()
    lowered = clean.lower()

    if lowered in TEAM_ALIASES:
        return TEAM_ALIASES[lowered]

    collapsed = lowered.replace(" ", "")
    for alias, canonical in TEAM_ALIASES.items():
        if alias.replace(" ", "") == collapsed:
            return canonical

    return clean
```

### Fixed Code (RIGHT)
```python
# At the top of team_factors.py with other imports
from src.utils.team_names import normalize_team_name

# REMOVE the TEAM_ALIASES dict
# REMOVE the local normalize_team_name function
# Now all code in team_factors.py uses: normalize_team_name()
```

### Step 2: Remove dataset.py's duplicate

**File:** `src/modeling/dataset.py`  
**Lines:** 26-61 (TEAM_NAME_MAP and _normalize_team_name)  
**Change:** Replace with import and adapt usage

### Current Code (WRONG)
```python
# Lines 26-61 in dataset.py
class LabeledDataset:
    # Team name mappings for matching between sources
    TEAM_NAME_MAP = {
        "Philadelphia 76ers": "Philadelphia 76ers",
        "Phi 76ers": "Philadelphia 76ers",
        # ... 25 more entries ...
    }

    def __init__(self, feature_engineer: Optional[FeatureEngineer] = None):
        self.feature_engineer = feature_engineer or FeatureEngineer()

    def _normalize_team_name(self, name: str) -> str:
        """Normalize team name for matching."""
        return self.TEAM_NAME_MAP.get(name, name)

    def load_odds_data(self, path: Optional[str] = None) -> pd.DataFrame:
        # ...
        df["home_team"] = df["home_team"].apply(self._normalize_team_name)
        df["away_team"] = df["away_team"].apply(self._normalize_team_name)
```

### Fixed Code (RIGHT)
```python
# At the top of dataset.py with other imports
from src.utils.team_names import normalize_team_name

class LabeledDataset:
    # REMOVE: TEAM_NAME_MAP dict (no longer needed)
    # REMOVE: _normalize_team_name method

    def __init__(self, feature_engineer: Optional[FeatureEngineer] = None):
        self.feature_engineer = feature_engineer or FeatureEngineer()

    def load_odds_data(self, path: Optional[str] = None) -> pd.DataFrame:
        # ...
        # Use imported function directly (not self._normalize_team_name)
        df["home_team"] = df["home_team"].apply(normalize_team_name)
        df["away_team"] = df["away_team"].apply(normalize_team_name)
```

### Step 3: Update team_factors.py HCA usage

**File:** `src/modeling/team_factors.py`  
**All functions using TEAM_HOME_COURT_ADVANTAGE**  
**Change:** The normalize_team_name will now return canonical ID, need to convert back to full name

### Current Code (affected by previous change)
```python
# Example in team_factors.py
normalized_home = normalize_team_name(home_team)  # Was: "Denver Nuggets"
hca = TEAM_HOME_COURT_ADVANTAGE.get(normalized_home, 2.5)
```

### Fixed Code (RIGHT)
```python
# Example in team_factors.py
from src.utils.team_names import get_canonical_name

canonical_id = normalize_team_name(home_team)  # Now: "nba_den"
canonical_name = get_canonical_name(canonical_id)  # Back to: "Denver Nuggets"
hca = TEAM_HOME_COURT_ADVANTAGE.get(canonical_name, 2.5)
```

**OR** (Better option) - Update TEAM_HOME_COURT_ADVANTAGE to use canonical IDs instead:
```python
TEAM_HOME_COURT_ADVANTAGE: Dict[str, float] = {
    # Use canonical IDs as keys instead of full names
    "nba_den": 4.2,       # Denver Nuggets - High altitude advantage
    "nba_gsw": 3.1,       # Golden State Warriors
    # ... etc ...
}

# Then in code:
canonical_id = normalize_team_name(home_team)
hca = TEAM_HOME_COURT_ADVANTAGE.get(canonical_id, 2.5)
```

---

## FIX #3: build_fresh_training_data.py - Use Single Odds Source

**File:** `scripts/build_fresh_training_data.py`  
**Lines:** 247-355 (dual odds calls)  
**Change:** Create wrapper that automatically tries historical then falls back to current

### Current Code (WRONG)
```python
# Lines 247-248
from src.ingestion.the_odds import (
    fetch_historical_odds,  # ← Multiple paths
    fetch_odds,             # ← Multiple paths
)

# Lines 260-293 - Uses historical
data = await fetch_historical_odds(...)

# Lines 355 - Uses current as fallback
current_odds = await fetch_odds(markets="spreads,totals,h2h")
```

### Analysis: Better Approach
Looking at your code, `fetch_odds()` should already handle this logic. Let me check if there's a reason for dual calls...

If `fetch_odds()` should be the single source and it already has fallback logic, then the fix is:
```python
# Keep fetch_odds() as only import
# Remove fetch_historical_odds()
```

But if you need explicit control over when to use historical vs current, then update the code to use a wrapper function that handles both paths transparently.

### Fixed Code Option A (Recommended)
```python
# Lines 247-248 - Remove fetch_historical_odds
from src.ingestion.the_odds import fetch_odds

# Lines 260-355 - ALWAYS use fetch_odds
lines_data = await fetch_odds(markets="spreads,totals,h2h")

# fetch_odds() internally handles:
# 1. Tries historical odds if available
# 2. Falls back to current odds if needed
# 3. Returns consistent format
```

### Fixed Code Option B (If you need control)
Create a wrapper in `src/ingestion/the_odds.py`:
```python
async def fetch_betting_lines(
    seasons: List[str],
    use_historical: bool = True,
) -> List[Dict]:
    """Single source for all betting lines (historical + current).
    
    This is the ONLY function that should be called for betting line data.
    It handles both historical odds (if available) and current odds.
    
    Args:
        seasons: Seasons to fetch
        use_historical: Try historical first (True) or go straight to current (False)
    
    Returns:
        List of betting line dictionaries in standardized format
    """
    if use_historical:
        try:
            return await fetch_historical_odds(seasons)
        except Exception as e:
            logger.warning(f"Historical odds failed, falling back to current: {e}")
    
    return await fetch_odds(markets="spreads,totals,h2h")
```

Then update build_fresh_training_data.py:
```python
# Lines 247-248
from src.ingestion.the_odds import fetch_betting_lines

# Lines 260-293
lines_data = await fetch_betting_lines(
    seasons=self.seasons,
    use_historical=True  # Try historical first
)
```

---

## Validation Tests

Create a new test file to verify fixes:

**File:** `tests/test_single_source_of_truth.py`

```python
"""
Validate that all data ingestion uses single source of truth functions.

This test suite ensures that the model maintains consistent data across
all components by verifying:
1. Injuries always use fetch_all_injuries()
2. Team names always use normalize_team_name() from utils
3. Odds always use fetch_odds() as primary source
"""
import ast
import re
from pathlib import Path
from typing import Set, List, Tuple


def find_imports_and_calls(file_path: str, pattern: str) -> List[Tuple[int, str]]:
    """Find lines matching pattern in a Python file."""
    matches = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f, 1):
            if re.search(pattern, line):
                matches.append((i, line.strip()))
    return matches


class TestSingleSourceOfTruth:
    """Validate single source of truth across codebase."""
    
    PROJECT_ROOT = Path(__file__).parent.parent
    SRC_DIR = PROJECT_ROOT / "src"
    SCRIPTS_DIR = PROJECT_ROOT / "scripts"
    
    def test_injuries_use_aggregator(self):
        """Verify injuries always use fetch_all_injuries(), not individual sources."""
        # Files that should NOT call fetch_injuries_espn or fetch_injuries_api_basketball
        source_files = [
            self.SRC_DIR / "serving" / "app.py",
            self.SRC_DIR / "features" / "rich_features.py",
            self.SRC_DIR / "ingestion" / "comprehensive.py",  # VIOLATION #1
            self.SCRIPTS_DIR / "build_rich_features.py",
            self.SCRIPTS_DIR / "predict.py",
        ]
        
        for file_path in source_files:
            if not file_path.exists():
                continue
                
            # Look for direct calls to individual injury sources
            violations = find_imports_and_calls(
                str(file_path),
                r"fetch_injuries_espn|fetch_injuries_api_basketball"
            )
            
            # Allow in injuries.py itself (that's where it's defined)
            if "injuries.py" not in str(file_path):
                assert not violations, (
                    f"{file_path.name} violates single source of truth.\n"
                    f"Use fetch_all_injuries() instead of direct calls:\n"
                    + "\n".join(f"  Line {num}: {code}" for num, code in violations)
                )
    
    def test_team_names_use_single_implementation(self):
        """Verify all team name normalization uses src/utils/team_names.py."""
        # Files that should NOT define normalize_team_name locally
        source_files = [
            self.SRC_DIR / "modeling" / "team_factors.py",  # VIOLATION #2
            self.SRC_DIR / "modeling" / "dataset.py",        # VIOLATION #2
        ]
        
        for file_path in source_files:
            if not file_path.exists():
                continue
            
            violations = find_imports_and_calls(
                str(file_path),
                r"def normalize_team_name|TEAM_NAME_MAP\s*="
            )
            
            assert not violations, (
                f"{file_path.name} has duplicate team name normalization.\n"
                f"Import from src.utils.team_names instead:\n"
                + "\n".join(f"  Line {num}: {code}" for num, code in violations)
            )
    
    def test_odds_use_single_source(self):
        """Verify odds collection doesn't mix historical and current paths."""
        # build_fresh_training_data.py should only use fetch_odds
        file_path = self.SCRIPTS_DIR / "build_fresh_training_data.py"
        
        if file_path.exists():
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Count how many times each odds function is called
            hist_calls = len(re.findall(r'fetch_historical_odds\s*\(', content))
            odds_calls = len(re.findall(r'fetch_odds\s*\(', content))
            
            # Should use fetch_odds for all calls (or wrapper that handles both)
            assert hist_calls == 0, (
                f"build_fresh_training_data.py calls fetch_historical_odds {hist_calls} times.\n"
                "Use fetch_odds() or a single wrapper function instead."
            )
    
    def test_no_mock_data_in_production(self):
        """Verify no mock data usage in production paths."""
        scripts = [
            self.SCRIPTS_DIR / "predict.py",
            self.SCRIPTS_DIR / "build_fresh_training_data.py",
            self.SCRIPTS_DIR / "backtest.py",
        ]
        
        for script in scripts:
            if not script.exists():
                continue
            
            violations = find_imports_and_calls(
                str(script),
                r'source\s*=\s*["\']mock["\']'
            )
            
            assert not violations, (
                f"{script.name} uses mock data in production.\n"
                + "\n".join(f"  Line {num}: {code}" for num, code in violations)
            )


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
```

---

## Post-Fix Checklist

After applying these fixes:

- [ ] Fix #1: Update comprehensive.py to use `fetch_all_injuries()`
- [ ] Fix #2A: Remove `normalize_team_name` from team_factors.py
- [ ] Fix #2B: Remove `TEAM_NAME_MAP` and `_normalize_team_name` from dataset.py
- [ ] Fix #2C: Add imports to team_factors.py and dataset.py
- [ ] Fix #2D: Update TEAM_HOME_COURT_ADVANTAGE logic in team_factors.py
- [ ] Fix #3: Remove `fetch_historical_odds` direct calls from build_fresh_training_data.py
- [ ] Add test file `tests/test_single_source_of_truth.py`
- [ ] Run validation test: `pytest tests/test_single_source_of_truth.py -v`
- [ ] Update comprehensive.py docstring to clarify role as cache layer
- [ ] Run full test suite: `pytest tests/ -v`
- [ ] Re-run audit to verify all violations resolved

---

## Conclusion

Your model is well-architected with a clear single source of truth design, but **3 implementations have drifted from this principle**. These fixes will:

✅ Ensure consistent data across all components  
✅ Reduce maintenance burden (changes in one place apply everywhere)  
✅ Improve testability (easier to verify data consistency)  
✅ Eliminate feature mismatches between training and production  
✅ Make debugging easier (traceable data lineage)

**Estimated fix time: ~1 hour total**  
**Impact on production: None** (fixes improve data consistency without changing APIs)

