# Contributing to NBA Basketball Prediction System

Welcome! This document provides guidelines for contributing to this repository.

## 🎯 Project Structure

This is the **SINGLE SOURCE OF TRUTH** for the NBA prediction system:
- **GitHub Repo:** `JDSB123/green_bier_sports_nba_model` (branch `main`)
- **Deployment:** Azure Container App (`nba-gbsv-api`)
- **Version Format:** `NBA_v<MAJOR>.<MINOR>.<PATCH>.<BUILD>`

## 📋 Before You Start

1. **Read the docs:**
   - [README.md](README.md) - Quick start and overview
   - [VERSIONING.md](VERSIONING.md) - Version management rules
   - [ARCHITECTURE_FLOW_AND_ENDPOINTS.md](docs/ARCHITECTURE_FLOW_AND_ENDPOINTS.md) - System design
   - [.github/copilot-instructions.md](.github/copilot-instructions.md) - AI assistant guidance

2. **Check existing issues:** Search for similar problems or features before opening new issues

3. **Test in CI or Codespaces:** Avoid local container runs to prevent conflicting environments

## 🔄 Development Workflow

### 1. Make Changes Locally

```powershell
# Clone if you haven't already
git clone https://github.com/JDSB123/green_bier_sports_nba_model.git
cd NBA_main

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Make your changes
# Edit files...

# Test your changes
pytest tests -v
```

### 2. Version Management

**CRITICAL:** Always update the version when making changes.

```powershell
# Bump version (use appropriate type)
python scripts/bump_version.py patch  # For bug fixes
python scripts/bump_version.py minor  # For new features
python scripts/bump_version.py major  # For breaking changes

# This will update VERSION file and sync across all files
```

### 3. Commit and Push

```powershell
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: add new prediction feature"

# Push to GitHub (triggers Azure build/deploy)
git push origin main
```

### 4. Deploy (If Needed)

```powershell
# Use automated deployment script
.\scripts\deploy.ps1

# This will:
# - Read version from VERSION file
# - Build container image
# - Push to Azure Container Registry
# - Update Azure Container App
# - Verify deployment
```

## 📝 Commit Message Conventions

Follow conventional commits:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation only
- `test:` Adding tests
- `refactor:` Code restructuring
- `chore:` Maintenance tasks

Examples:
```
feat: add Q1 spread prediction endpoint
fix: correct team name mapping for Lakers
docs: update API documentation
test: add unit tests for odds ingestion
```

## 🧪 Testing

### Run All Tests
```powershell
pytest tests -v
```

### Run Specific Test Files
```powershell
pytest tests/test_serving.py -v
```

### Run Tests in CI (recommended)
Push to `main` to trigger CI. For local test runs, use pytest only (no containers).

## 🚨 Critical Rules

### ⚠️ NEVER:
- **Hardcode secrets** in code or `.env` files (use Azure Key Vault)
- **Skip version bumping** before deployment
- **Build Docker images** before pushing to GitHub
- **Force push** to `main` branch without team coordination
- **Commit large files** (CSV, logs, coverage.xml)
- **Use old v6.x version scheme** (always use NBA_v33.x.x.x)

### ✅ ALWAYS:
- **Push to GitHub first** before Docker operations
- **Test locally** before committing
- **Update CHANGELOG.md** for user-facing changes
- **Sync version** across all files (use `bump_version.py`)
- **Run deployment script** instead of manual commands
- **Check `.gitignore`** before committing

## 🔐 Secrets Management

Secrets are stored in **Azure Key Vault** (`nbagbs-keyvault`):

```powershell
# Fetch secrets from Key Vault (Azure-first)
az keyvault secret show --vault-name nbagbs-keyvault --name THE-ODDS-API-KEY --query value -o tsv
az keyvault secret show --vault-name nbagbs-keyvault --name API-BASKETBALL-KEY --query value -o tsv
```

See [docs/SECRETS.md](docs/SECRETS.md) for details.

## 📦 Container Build (Azure)

### Build Images
```powershell
# Build combined API + model image
$VERSION = (Get-Content VERSION -Raw).Trim()
docker build --build-arg MODEL_VERSION=$VERSION -t nbagbsacr.azurecr.io/nba-gbsv-api:$VERSION -f Dockerfile.combined .
```

## 🔧 Maintenance

### Monthly Maintenance Check
```powershell
# Run automated health check
.\scripts\repo-maintenance.ps1

# Run with auto-fix
.\scripts\repo-maintenance.ps1 -Fix
```

This checks:
- Version consistency
- Git tags
- Large files
- CHANGELOG updates
- CI/CD status
- Test coverage
- Repository size

### Deep Cleanup (Rarely Needed)
```powershell
# Remove large files from git history
.\scripts\git-deep-cleanup.ps1

# WARNING: Requires force-push and team re-clone
```

## 🐛 Troubleshooting

### Docker Issues
See [docs/AZURE_OPERATIONS.md](docs/AZURE_OPERATIONS.md)

### Version Conflicts
```powershell
# Check current version
cat VERSION

# Verify consistency
python scripts/bump_version.py --check

# Sync all files
python scripts/bump_version.py $(cat VERSION)
```

### Deployment Failures
```powershell
# Check Container App logs
az containerapp logs -n nba-gbsv-api -g nba-gbsv-model-rg --tail 50

# Check health endpoint
curl https://nba-gbsv-api.ambitiouscoast-4bcd4cd8.eastus.azurecontainerapps.io/health
```

## 📊 Code Review Checklist

Before requesting review:

- [ ] Code follows project structure
- [ ] Tests added/updated and passing
- [ ] Version bumped appropriately
- [ ] CHANGELOG.md updated (if user-facing)
- [ ] Documentation updated (if needed)
- [ ] No secrets committed
- [ ] No large files added
- [ ] Commit messages follow conventions
- [ ] Pushed to GitHub before Docker build
- [ ] Local testing completed

## 🎉 Getting Help

- **Documentation:** Check `docs/` folder
- **Issues:** Open GitHub issue with `question` label
- **AI Assistant:** Use Copilot with `.github/copilot-instructions.md` context
- **Scripts:** See `scripts/README.md` for available automation

## 📅 Release Process

1. **Development:** Make changes on feature branch (optional)
2. **Testing:** Run full test suite
3. **Version Bump:** Use `bump_version.py` with appropriate level
4. **CHANGELOG:** Update with new version entry
5. **Commit & Push:** Push to `main` branch
6. **Deploy:** Run `deploy.ps1` script
7. **Tag:** Git tag created automatically by deployment script
8. **Release:** GitHub release created via `gh release create`

## 📜 License

This is a private repository for Green Bier Sports Ventures. Unauthorized use is prohibited.

## 🙏 Thank You

Thank you for contributing to the NBA prediction system! Your efforts help maintain a clean, reliable, and professional codebase.

---

**Last Updated:** 2025-12-29
**Version:** See `VERSION`
