# Single Source of Truth - Quick Fix Guide

**Total Time to Fix: ~1 hour**

---

## Fix #1: comprehensive.py (5 minutes)

**File:** `src/ingestion/comprehensive.py`  
**Problem:** Line 616 calls `fetch_injuries_espn()` directly

### Step 1: Find the method
```bash
grep -n "async def fetch_espn_injuries" src/ingestion/comprehensive.py
# Output: 611:async def fetch_espn_injuries(self) -> List[Dict]:
```

### Step 2: Replace the implementation
Change lines 611-625 from:
```python
async def fetch_espn_injuries(self) -> List[Dict]:
    """Fetch injuries from ESPN (FREE, unlimited).

    TTL: 2 hours
    """
    from src.ingestion.injuries import fetch_injuries_espn  # ← WRONG

    key = f"espn_injuries_{date.today().isoformat()}"

    data = await api_cache.get_or_fetch(
        key=key,
        fetch_fn=fetch_injuries_espn,  # ← WRONG
        ttl_hours=APICache.TTL_FREQUENT,
        source="espn",
        endpoint="/injuries",
        force_refresh=self.force_refresh,
    )

    self._record_result("espn", "/injuries", True, len(data))
    return data
```

To:
```python
async def fetch_injuries(self) -> List[Dict]:
    """Fetch injuries from all configured sources (ESPN + API-Basketball).

    Aggregates injury data from multiple sources for redundancy.
    ESPN: FREE, unlimited. API-Basketball: If API key configured.
    
    TTL: 2 hours
    """
    from src.ingestion.injuries import fetch_all_injuries  # ← RIGHT

    key = f"injuries_{date.today().isoformat()}"

    data = await api_cache.get_or_fetch(
        key=key,
        fetch_fn=fetch_all_injuries,  # ← RIGHT
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

### Step 3: Update any references
Search for other calls to `fetch_espn_injuries`:
```bash
grep -r "fetch_espn_injuries" src/ scripts/
# Should return no results after renaming to fetch_injuries
```

---

## Fix #2: Team Name Consolidation (30 minutes)

### Step 2A: Remove from team_factors.py

**File:** `src/modeling/team_factors.py`

Find and remove these (approximately lines 40-95):
```python
TEAM_ALIASES = {
    "lakers": "Los Angeles Lakers",
    "celtics": "Boston Celtics",
    # ... entire dict ...
}

def normalize_team_name(team_name: str) -> str:
    """Normalize shorthand team names..."""
    # ... entire function ...
```

Add this import at the top:
```python
from src.utils.team_names import normalize_team_name
```

### Step 2B: Remove from dataset.py

**File:** `src/modeling/dataset.py`

Find and remove (approximately lines 26-61):
```python
class LabeledDataset:
    TEAM_NAME_MAP = {
        # ... entire dict ...
    }
    
    def _normalize_team_name(self, name: str) -> str:
        """Normalize team name for matching."""
        # ... method ...
```

Add import at top:
```python
from src.utils.team_names import normalize_team_name
```

Find and replace all `self._normalize_team_name(` with `normalize_team_name(`:
```bash
# Search
grep -n "_normalize_team_name" src/modeling/dataset.py

# Replace (in the file)
df["home_team"] = df["home_team"].apply(self._normalize_team_name)
# becomes
df["home_team"] = df["home_team"].apply(normalize_team_name)
```

### Step 2C: Update team_factors.py HCA logic

**File:** `src/modeling/team_factors.py`

Current code using TEAM_HOME_COURT_ADVANTAGE uses full names like "Denver Nuggets" as keys.  
Since `normalize_team_name()` now returns canonical IDs like "nba_den", need to adapt:

**Option A** (Recommended): Update TEAM_HOME_COURT_ADVANTAGE to use canonical IDs

Find the dict definition (around line 115-150):
```python
TEAM_HOME_COURT_ADVANTAGE: Dict[str, float] = {
    "Denver Nuggets": 4.2,  # ← Change key
    # ... rest of dict ...
}
```

Change to:
```python
from src.utils.team_names import get_canonical_name

TEAM_HOME_COURT_ADVANTAGE: Dict[str, float] = {
    "nba_den": 4.2,  # Denver Nuggets - High altitude advantage
    "nba_gsw": 3.1,  # Golden State Warriors
    "nba_lac": 2.8,  # Los Angeles Clippers
    # ... etc for all 30 teams ...
}
```

Then update all usages:
```python
# OLD
normalized_home = normalize_team_name(home_team)  # Returns "Denver Nuggets"
hca = TEAM_HOME_COURT_ADVANTAGE.get(normalized_home, 2.5)

# NEW
canonical_id = normalize_team_name(home_team)  # Returns "nba_den"
hca = TEAM_HOME_COURT_ADVANTAGE.get(canonical_id, 2.5)
```

**Option B**: Keep TEAM_HOME_COURT_ADVANTAGE with full names, convert back after normalize:
```python
from src.utils.team_names import get_canonical_name

canonical_id = normalize_team_name(home_team)  # Returns "nba_den"
canonical_name = get_canonical_name(canonical_id)  # Returns "Denver Nuggets"
hca = TEAM_HOME_COURT_ADVANTAGE.get(canonical_name, 2.5)
```

### Step 2D: Verify no duplicates remain
```bash
grep -r "def normalize_team_name" src/
# Should only return: src/utils/team_names.py

grep -r "TEAM_NAME_MAP\s*=" src/
# Should return no results (was in dataset.py, now removed)
```

---

## Fix #3: Odds Pipeline (20 minutes)

**File:** `scripts/build_fresh_training_data.py`

### Step 1: Find the imports
```bash
grep -n "from src.ingestion.the_odds import" scripts/build_fresh_training_data.py
# Around line 247-248
```

### Step 2: Remove fetch_historical_odds from imports
**Current:**
```python
from src.ingestion.the_odds import (
    fetch_historical_odds,
    fetch_odds,
    # ... others ...
)
```

**Updated:**
```python
from src.ingestion.the_odds import (
    fetch_odds,
    # ... others ...
)
# Remove fetch_historical_odds from this import
```

### Step 3: Replace all fetch_historical_odds calls
```bash
grep -n "fetch_historical_odds" scripts/build_fresh_training_data.py
# Find all lines where it's called
```

For each occurrence around lines 266-293, replace:
```python
# OLD (around line 266)
data = await fetch_historical_odds(
    sport=sport,
    region="us",
    market=market,
    # ... params ...
)

# NEW - Use fetch_odds instead
data = await fetch_odds(
    markets="spreads,totals,h2h",
    # fetch_odds handles historical vs current internally
)
```

### Step 4: Verify
```bash
grep "fetch_historical_odds" scripts/build_fresh_training_data.py
# Should return NO results (except in comments)

grep "fetch_odds" scripts/build_fresh_training_data.py
# Should return your updated calls
```

---

## Validation (10 minutes)

### Create test file
```bash
# Create tests/test_single_source_of_truth.py
# Copy content from SINGLE_SOURCE_OF_TRUTH_AUDIT.md "Validation Tests" section
```

### Run tests
```bash
cd /path/to/NBA_main
pytest tests/test_single_source_of_truth.py -v
```

All tests should PASS after fixes.

### Manual verification
```bash
# Should return NO matches
grep -r "fetch_injuries_espn\|fetch_injuries_api_basketball" src/ scripts/
grep -r "def normalize_team_name" src/modeling/
grep "fetch_historical_odds" scripts/build_fresh_training_data.py

# If all three commands show nothing, fixes are complete!
```

---

## Order of Fixes

**Recommended order to minimize conflicts:**

1. ✅ **Fix #1** - comprehensive.py (independent, safe)
2. ✅ **Fix #2A** - Remove from team_factors.py (independent)
3. ✅ **Fix #2B** - Remove from dataset.py (independent)
4. ✅ **Fix #2C** - Update HCA logic (depends on 2A + 2B)
5. ✅ **Fix #2D** - Verify no duplicates (validation)
6. ✅ **Fix #3** - Update build_fresh_training_data.py (independent)
7. ✅ **Testing** - Add and run validation tests

---

## Quick Checklist

```bash
# After Fix #1
[ ] comprehensive.py renamed fetch_espn_injuries → fetch_injuries
[ ] comprehensive.py imports fetch_all_injuries instead of fetch_injuries_espn
[ ] Test: grep -r "fetch_injuries_espn\|fetch_injuries_api_basketball" src/

# After Fix #2
[ ] team_factors.py: removed TEAM_ALIASES dict
[ ] team_factors.py: removed normalize_team_name function
[ ] team_factors.py: added import from src.utils.team_names
[ ] dataset.py: removed TEAM_NAME_MAP dict
[ ] dataset.py: removed _normalize_team_name method
[ ] dataset.py: added import from src.utils.team_names
[ ] dataset.py: replaced all self._normalize_team_name( with normalize_team_name(
[ ] team_factors.py: updated TEAM_HOME_COURT_ADVANTAGE logic
[ ] Test: grep -r "def normalize_team_name" src/

# After Fix #3
[ ] build_fresh_training_data.py: removed fetch_historical_odds from imports
[ ] build_fresh_training_data.py: replaced all fetch_historical_odds( calls with fetch_odds(
[ ] Test: grep "fetch_historical_odds" scripts/build_fresh_training_data.py

# Final validation
[ ] pytest tests/test_single_source_of_truth.py -v (ALL PASS)
[ ] pytest tests/ -v (FULL TEST SUITE PASS)
[ ] grep tests show NO violations
```

---

## Common Issues & Solutions

### Issue: "ImportError: cannot import name 'fetch_all_injuries'"
**Solution:** Check spelling - it's `fetch_all_injuries`, not `fetch_all_injury` or `fetch_injuries_all`

### Issue: "normalize_team_name called with 1 required positional argument"
**Solution:** After removing from team_factors.py, function is now module-level (not `self.`), so remove `self.` prefix

### Issue: AttributeError in HCA lookup
**Solution:** Make sure TEAM_HOME_COURT_ADVANTAGE keys match format returned by normalize_team_name (either "nba_den" or "Denver Nuggets" - be consistent!)

### Issue: Tests still fail after fixes
**Solution:** 
1. Run the validation commands manually to see what's left
2. Check for commented-out code that might still have old calls
3. Verify imports are at module level (not conditional)

---

## Files to Check After Each Fix

| Fix | File | Verification Command |
|-----|------|----------------------|
| #1 | comprehensive.py | `grep -n "fetch_all_injuries\|fetch_injuries_espn" src/ingestion/comprehensive.py` |
| #2 | team_factors.py | `grep -n "TEAM_ALIASES\|def normalize_team_name" src/modeling/team_factors.py` |
| #2 | dataset.py | `grep -n "TEAM_NAME_MAP\|_normalize_team_name" src/modeling/dataset.py` |
| #3 | build_fresh_training_data.py | `grep -n "fetch_historical_odds\|fetch_odds" scripts/build_fresh_training_data.py` |

---

## Need Help?

See full details in: [docs/SINGLE_SOURCE_OF_TRUTH_AUDIT.md](docs/SINGLE_SOURCE_OF_TRUTH_AUDIT.md)

