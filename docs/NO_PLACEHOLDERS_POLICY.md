# No Placeholders, No Silent Failures Policy

**Date:** 2025-12-17  
**Status:** ✅ Enforced

---

## Policy

**Absolutely zero:**
- ❌ Fake/mock data
- ❌ Placeholder values (e.g., "unknown", "Unknown", "N/A")
- ❌ Silent failures (exceptions caught and ignored)
- ❌ Default fallbacks that mask data quality issues

---

## Enforcement

### 1. No Placeholder Values

**Examples of FORBIDDEN placeholders:**
- `source: str = "unknown"` ❌
- `team_name = data.get("team", "Unknown")` ❌
- `name = data.get("name", "N/A")` ❌
- Empty strings used as placeholders ❌

**Correct approach:**
- Use `Optional[str] = None` and validate at call site
- Skip records with missing required fields
- Log errors when required data is missing
- Raise exceptions for critical missing data

**Example:**
```python
# ❌ WRONG - placeholder
team_name = data.get("team", "Unknown")

# ✅ CORRECT - explicit validation
team_name = data.get("team")
if not team_name:
    logger.error(f"Missing required field 'team' in data: {data}")
    raise ValueError("Missing required field: team")
    # OR skip this record if field is optional
```

---

### 2. No Silent Failures

**Forbidden patterns:**
```python
# ❌ WRONG - silent failure
try:
    result = fetch_data()
except Exception:
    pass  # Silent failure

# ❌ WRONG - generic exception caught
try:
    result = fetch_data()
except Exception as e:
    return []  # Silent failure - no logging
```

**Required patterns:**
```python
# ✅ CORRECT - explicit error logging
try:
    result = fetch_data()
except httpx.HTTPStatusError as e:
    logger.error(f"HTTP error fetching data: {e.response.status_code} {e.response.reason_phrase}")
    raise  # Or return empty with explicit log
except httpx.RequestError as e:
    logger.error(f"Request error fetching data: {e}")
    raise  # Or return empty with explicit log
except Exception as e:
    logger.error(f"Unexpected error: {type(e).__name__}: {e}", exc_info=True)
    raise  # Or return empty with explicit log
```

---

### 3. No Default Fallbacks That Mask Issues

**Forbidden:**
```python
# ❌ WRONG - fallback masks missing data
status = data.get("status", "questionable")  # If "questionable" is just a default

# ✅ CORRECT - explicit handling
status_raw = data.get("status")
if not status_raw:
    logger.warning(f"Missing status field in data, skipping record")
    continue  # Skip this record
status = normalize_status(status_raw)  # "questionable" is a valid normalized status
```

---

## Data Type Specific Rules

### InjuryReport

**Required fields (no defaults allowed):**
- `player_id: str` - Required
- `player_name: str` - Required
- `team: str` - Required
- `source: str` - Required (must be explicit: "espn", "api_basketball", etc.)

**Optional fields (None allowed):**
- `team_id: Optional[str] = None`
- `injury_type: Optional[str] = None`
- `report_date: Optional[dt.datetime] = None`

**Valid defaults (not placeholders):**
- `status: str = "questionable"` - This is a valid normalized status value
- `ppg: float = 0.0` - Zero is a valid numeric default for missing stats

---

### Team Name Normalization

**Forbidden:**
```python
# ❌ WRONG - placeholder fallback
team_name = data.get("team", "Unknown")
normalized = normalize_team(team_name)
```

**Required:**
```python
# ✅ CORRECT - skip if missing
team_name = data.get("team")
if not team_name:
    logger.warning(f"Skipping record with missing team name")
    continue
normalized = normalize_team(team_name)
```

---

## Error Handling Requirements

### API Fetch Functions

**Required error handling:**
1. Log specific error types (HTTPStatusError, RequestError, etc.)
2. Include URL and status codes in error messages
3. Use `exc_info=True` for unexpected errors
4. Return empty data explicitly (not silently)
5. Log warnings when empty data is returned

**Example:**
```python
async def fetch_injuries_espn() -> List[Dict[str, Any]]:
    url = "https://..."
    try:
        # ... fetch logic
        return injuries
    except httpx.HTTPStatusError as e:
        logger.error(
            f"HTTP error fetching ESPN injuries: "
            f"{e.response.status_code} {e.response.reason_phrase}. URL: {url}"
        )
        return []  # Explicit empty return
    except httpx.RequestError as e:
        logger.error(f"Request error fetching ESPN injuries: {e}. URL: {url}")
        return []  # Explicit empty return
    except Exception as e:
        logger.error(
            f"Unexpected error fetching ESPN injuries: "
            f"{type(e).__name__}: {e}. URL: {url}",
            exc_info=True
        )
        return []  # Explicit empty return
```

---

### Data Validation

**Required validation:**
1. Check required fields before processing
2. Skip records with missing required fields
3. Log warnings for skipped records
4. Raise exceptions for critical validation failures

**Example:**
```python
for item in data:
    player_name = item.get("player_name")
    team_name = item.get("team")
    
    # Skip if required fields missing - no placeholders
    if not player_name:
        logger.warning(f"Skipping record with missing player_name")
        continue
    if not team_name:
        logger.warning(f"Skipping record for player {player_name} with missing team_name")
        continue
    
    # Process valid record
    process_record(player_name, team_name)
```

---

## Verification

### How to Verify Compliance

1. **Search for placeholders:**
   ```bash
   grep -r "Unknown\|unknown\|N/A\|n/a\|placeholder\|Placeholder" src/ --include="*.py"
   ```

2. **Search for silent failures:**
   ```bash
   grep -r "except.*:\s*pass\|except.*:\s*return" src/ --include="*.py"
   ```

3. **Search for default fallbacks:**
   ```bash
   grep -r "\.get(.*Unknown\|\.get(.*unknown\|\.get(.*N/A" src/ --include="*.py"
   ```

---

## Checklist

When writing new code:

- [ ] No placeholder string values ("unknown", "Unknown", "N/A")
- [ ] All exceptions are explicitly logged with context
- [ ] Missing required fields cause records to be skipped or errors raised
- [ ] Default values are only used for valid normalized values, not placeholders
- [ ] Error messages include URLs, status codes, and context
- [ ] Empty returns are explicitly logged as warnings/errors
- [ ] Source fields are explicitly set (not defaulted to "unknown")

---

## Examples of Corrections

### Before (❌ Placeholder)
```python
@dataclass
class InjuryReport:
    source: str = "unknown"  # ❌ Placeholder default
```

### After (✅ Required)
```python
@dataclass
class InjuryReport:
    source: str  # ✅ Required - must be explicitly set
```

---

### Before (❌ Silent Failure)
```python
try:
    data = fetch_data()
except Exception:
    return []  # ❌ Silent failure
```

### After (✅ Explicit Failure)
```python
try:
    data = fetch_data()
except Exception as e:
    logger.error(f"Error fetching data: {e}", exc_info=True)
    return []  # ✅ Explicitly logged
```

---

### Before (❌ Placeholder Fallback)
```python
team_name = data.get("team", "Unknown")  # ❌ Placeholder
```

### After (✅ Explicit Validation)
```python
team_name = data.get("team")
if not team_name:
    logger.warning("Missing team name, skipping record")
    continue  # ✅ Explicit handling
```

---

## Related Documentation

- `docs/DATA_SOURCE_OF_TRUTH.md` - Single source of truth functions
- `docs/PRODUCTION_READINESS_FIXES.md` - Mock data removal fixes

---

**Enforcement:** This policy is enforced in all production code. Any violations should be immediately corrected.
