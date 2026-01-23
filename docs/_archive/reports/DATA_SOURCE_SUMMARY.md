# Data Source Summary - Clarification

## Key Point

**"Standard Format" (formerly called "ESPN format")** = Naming Convention Only  
**ESPN as Data Source** = Used ONLY for schedules

## Actual Data Sources

| Data Type | Primary Source | Secondary Sources | Format Standard |
|-----------|---------------|-------------------|-----------------|
| **Betting Odds** | The Odds API | BetsAPI | Standard format names |
| **Team Records (W-L)** | The Odds API scores (unified) | API-Basketball standings (features) | Standard format names |
| **Game Statistics** | API-Basketball | - | Standard format names |
| **Schedules** | ESPN | - | Standard format names |
| **Standings** | ESPN (available) / API-Basketball (used in features) | - | Standard format names |
| **Injuries** | ESPN + API-Basketball (aggregated) | - | Standard format names |
| **Betting Splits** | Action Network / The Odds API | SBRO, Covers | Standard format names |

## Terminology

### ✅ CORRECT Usage

- "Standard format team names" (or "canonical format")
- "Normalize to standard format"
- "Team names in standard format" (e.g., "Los Angeles Lakers")

### ❌ MISLEADING Usage

- "ESPN format" (suggests using ESPN data)
- "Normalize to ESPN" (suggests using ESPN as source)
- "ESPN team names" (when data comes from other sources)

## Current Status

✅ **Unified Records**: Uses The Odds API (same source as odds) - GOOD  
✅ **Team Name Standardization**: Comprehensive across all sources - GOOD  
✅ **Data Sources**: Multi-source architecture - GOOD  
⚠️ **Terminology**: "ESPN format" is misleading - needs clarification

## Recommendation

Rename terminology from "ESPN format" to "standard format" or "canonical format" to avoid confusion.

**Function naming**: `normalize_team_to_espn()` is fine (historical name), but documentation should clarify it's a naming convention, not an ESPN data dependency.
