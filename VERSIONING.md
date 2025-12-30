# NBA Model Versioning Strategy

**Current Version:** See [VERSION](VERSION) file - `NBA_v33.0.8.0`

**Single Source of Truth:** The `VERSION` file at the repository root is the ONLY authoritative version identifier.

---

## Version Format

```
NBA_v<MAJOR>.<MINOR>.<PATCH>.<BUILD>
```

Example: `NBA_v33.0.8.0`

- **MAJOR (33)**: Model architecture generation or breaking API changes
- **MINOR (0)**: New features, market additions, major logic changes
- **PATCH (8)**: Bug fixes, calibration tweaks, small improvements
- **BUILD (0)**: Build number for hotfixes within same patch (rarely used)

---

## When to Bump Versions

### MAJOR (v33 → v34)
- Complete model retraining with new architecture
- Breaking API changes (removed endpoints, changed response schemas)
- Market structure changes (e.g., Q1/Q2/Q3 → 1H/FG)
- Database schema breaking changes

**Examples:**
- Switching from 9 markets to 4 markets ✅
- Changing from XGBoost to Neural Network ✅
- Removing deprecated `/predict/v2` endpoint ✅

### MINOR (v33.0 → v33.1)
- Adding new markets (e.g., adding player props)
- New endpoints (e.g., `/backtest`, `/compare`)
- Feature engineering improvements
- Adding new data sources

**Examples:**
- Adding 2H markets in addition to 1H ✅
- Adding betting splits integration ✅
- New `/slate/comprehensive` endpoint ✅

### PATCH (v33.0.8 → v33.0.9)
- Bug fixes (prediction errors, calculation bugs)
- Calibration adjustments
- Edge threshold tuning
- Documentation updates
- Performance optimizations (no logic change)

**Examples:**
- Fixing spread sign bug ✅
- Adjusting edge thresholds from 5pts to 3pts ✅
- Adding archiving to predictions ✅
- Fixing Docker secrets handling ✅

### BUILD (v33.0.8.0 → v33.0.8.1)
- Hotfixes for critical production issues
- Docker configuration changes only
- No code changes, just rebuild

**Examples:**
- Emergency fix for API crash ✅
- Secrets rotation with no code change ❌ (don't bump)

---

## Version Bump Checklist

**BEFORE any version bump:**

1. ✅ All tests passing (`pytest tests -v`)
2. ✅ Models trained and validated
3. ✅ Docker builds successfully
4. ✅ Manual testing complete

**TO BUMP VERSION:**

1. **Update VERSION file:**
   ```bash
   echo "NBA_v33.0.9.0" > VERSION
   ```

2. **Update all references (use script below):**
   ```powershell
   python scripts/bump_version.py NBA_v33.0.9.0
   ```
   This updates:
   - [src/serving/app.py](src/serving/app.py)
   - [src/prediction/engine.py](src/prediction/engine.py)
   - [models/production/model_pack.json](models/production/model_pack.json)
   - [tests/test_serving.py](tests/test_serving.py)
   - [.github/copilot-instructions.md](.github/copilot-instructions.md)
   - [README.md](README.md)
   - [infra/nba/main.json](infra/nba/main.json)

3. **Commit with semantic message:**
   ```bash
   git add VERSION src/ models/ tests/ .github/ README.md infra/
   git commit -m "chore: bump version to NBA_v33.0.9.0
   
   - Fixed spread calculation bug
   - Added archive folder for historical tracking
   - Updated deployment docs"
   
   git tag NBA_v33.0.9.0
   git push origin main --tags
   ```

4. **Build and deploy:**
   ```bash
   # Read VERSION file
   $VERSION = Get-Content VERSION -Raw | ForEach-Object { $_.Trim() }
   
   # Build with correct tag
   docker build -t nbagbsacr.azurecr.io/nba-gbsv-api:$VERSION -f Dockerfile.combined .
   
   # Push
   az acr login -n nbagbsacr
   docker push nbagbsacr.azurecr.io/nba-gbsv-api:$VERSION
   
   # Deploy
   az containerapp update -n nba-gbsv-api -g nba-gbsv-model-rg --image nbagbsacr.azurecr.io/nba-gbsv-api:$VERSION
   ```

---

## Version Synchronization

**Files that MUST match VERSION:**

| File | Line/Field | Purpose |
|------|------------|---------|
| `VERSION` | Line 1 | **CANONICAL SOURCE** |
| `src/serving/app.py` | `RELEASE_VERSION` | API version reporting |
| `src/prediction/engine.py` | `MODEL_VERSION` | Model tracking |
| `models/production/model_pack.json` | `version`, `git_tag`, `acr` | Model metadata |
| `tests/test_serving.py` | `test_health_check` | Integration tests |
| `.github/copilot-instructions.md` | Deployment steps | AI agent guidance |
| `README.md` | Versioning section | Documentation |
| `infra/nba/main.json` | `defaultValue` | Azure deployment |

**⚠️ CRITICAL:** Version drift between these files causes deployment failures and confusion.

---

## CI/CD Automation (TODO)

**Planned GitHub Actions:**

1. **Version Validation** (`.github/workflows/version-check.yml`)
   - Trigger: On every PR
   - Action: Verify VERSION matches all references
   - Block merge if mismatched

2. **Auto-Tag on Merge** (`.github/workflows/tag-release.yml`)
   - Trigger: Merge to main
   - Action: Create git tag from VERSION file

3. **Auto-Deploy** (`.github/workflows/deploy.yml`)
   - Trigger: Tag push
   - Action: Build Docker → Push to ACR → Update Container App

---

## Historical Versions

| Version | Date | Changes |
|---------|------|---------|
| NBA_v33.0.8.0 | 2025-12-29 | Added archiving, fixed spread signs, updated docs |
| NBA_v33.0.7.0 | 2025-12-28 | Calibration improvements, edge tuning |
| NBA_v33.0.2.0 | 2025-12-22 | Production deployment with 4 markets |
| NBA_v33.0.0.0 | 2025-12-15 | Major refactor: 9→4 markets (removed Q1) |

---

## FAQ

**Q: Why NBA_v33?**
A: The "33" represents the current model generation. Previous versions (v1-v32) were experimental.

**Q: Why not just use semantic versioning (1.0.0)?**
A: The "NBA_" prefix and generation number help distinguish NBA models from NCAAF/NCAAM models in the same organization.

**Q: What if I forget to bump a file?**
A: CI will catch it (once implemented). For now, run `grep -r "NBA_v33" --include="*.py" --include="*.md" --include="*.json"` to find all references.

**Q: Do I need to bump for README changes?**
A: No. Only bump for code/model/API changes. Documentation-only changes don't require version bumps.

**Q: Should I bump version for experimental branches?**
A: No. Only bump when merging to `main`. Experimental branches keep the current version until proven stable.

---

## Related Docs

- [README.md](README.md) - Project overview
- [.github/copilot-instructions.md](.github/copilot-instructions.md) - Deployment pipeline
- [models/production/model_pack.json](models/production/model_pack.json) - Model metadata
