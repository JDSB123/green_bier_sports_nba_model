# Changelog

All notable changes to the NBA Prediction Model are documented here.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).  
Versioning follows [VERSIONING.md](VERSIONING.md).

---

## [NBA_v33.0.8.0] - 2025-12-29

### 🎉 Major Improvements
This release establishes professional DevOps standards with automated deployment and version management.

### Added
- **Archive folder** for historical output tracking (predictions, picks, slate analysis, odds snapshots)
- **Automated deployment script** ([scripts/deploy.ps1](scripts/deploy.ps1)) - one-command deployment reading VERSION file
- **Version bump automation** ([scripts/bump_version.py](scripts/bump_version.py)) - synchronizes version across 10+ files
- **CI/CD version validation** ([.github/workflows/version-check.yml](.github/workflows/version-check.yml)) - blocks PRs with version drift
- **Complete versioning documentation** ([VERSIONING.md](VERSIONING.md)) - semantic versioning guidelines
- **Git cleanup plan** ([GIT_CLEANUP_PLAN.md](GIT_CLEANUP_PLAN.md)) - repository maintenance guide
- Odds snapshot archiving in API responses

### Fixed
- **Version consistency** - synced all references from NBA_v33.0.2.0 → NBA_v33.0.8.0
- **Deployment pipeline** - now automated with safety checks (git push enforcement, health validation)
- **Git hygiene** - proper tracking of archive/ folder, removed docs/archive from gitignore
- Spread sign calculations in prediction engine
- Archiving functionality in predict.py, run_slate.py, and app.py

### Changed
- Updated .gitignore to properly handle archive/ folder (tracked for audit trail)
- Synced .github/copilot-instructions.md to current version
- Improved error handling in deployment scripts

### Documentation
- [VERSIONING.md](VERSIONING.md) - How to manage versions
- [CLEANUP_SUMMARY_2025-12-29.md](docs/CLEANUP_SUMMARY_2025-12-29.md) - Repository cleanup summary
- [GIT_CLEANUP_PLAN.md](GIT_CLEANUP_PLAN.md) - Git repository maintenance

### Migration Notes
- **Old versioning scheme (v6.x) is DEPRECATED** - all future versions use NBA_v<MAJOR>.<MINOR>.<PATCH>.<BUILD>
- To deploy: `.\scripts\deploy.ps1`
- To bump version: `python scripts\bump_version.py NBA_vX.X.X.X`

---

## [NBA_v33.0.7.0] - 2025-12-28

### Changed
- Calibration improvements for edge calculations
- Edge threshold tuning based on backtest results
- Model confidence adjustments

---

## [NBA_v33.0.2.0] - 2025-12-22

### Added
- Production deployment with 6 markets (1H + FG spreads/totals/moneylines)
- Azure Container App infrastructure

### Changed
- Stabilized prediction API endpoints
- Updated Docker configuration for production

---

## [NBA_v33.0.0.0] - 2025-12-15

### 🎯 Major Refactor
Complete market structure overhaul from 9 markets to 6 markets.

### Changed
- **BREAKING:** Removed Q1/Q2/Q3 markets (too noisy, poor performance)
- Consolidated to 1H + FG only (6 markets total):
  - 1H Spread, 1H Total, 1H Moneyline
  - FG Spread, FG Total, FG Moneyline
- Updated prediction engine for 6-market structure
- Retrained models with new feature set

### Removed
- Q1 spread/total/moneyline markets
- Q2 markets
- Q3 markets

---

## Version History - Old Scheme (DEPRECATED)

These versions used an inconsistent numbering scheme and are **no longer supported**.  
All code and models prior to NBA_v33.0.0.0 are considered legacy.

### [v6.4] - 2025-12-XX
- Details unknown (version not documented)

### [v6.2] - 2025-12-XX
- Details unknown (version not documented)

### [v6.0.1] - 2025-12-XX
- Details unknown (version not documented)

### [v6.0.0-hardened] - 2025-12-XX
- Details unknown (version not documented)

---

## Versioning Guidelines

This project uses **semantic versioning** with the format:

```
NBA_v<MAJOR>.<MINOR>.<PATCH>.<BUILD>
```

- **MAJOR**: Model architecture changes or breaking API changes
- **MINOR**: New features, market additions, major logic changes
- **PATCH**: Bug fixes, calibration tweaks, small improvements
- **BUILD**: Hotfixes within same patch (rarely used)

For complete versioning rules, see [VERSIONING.md](VERSIONING.md).

---

## Links

- [GitHub Repository](https://github.com/JDSB123/green_bier_sports_nba_model)
- [Production API](https://nba-gbsv-api.ambitiouscoast-4bcd4cd8.eastus.azurecontainerapps.io)
- [Versioning Guide](VERSIONING.md)
- [Deployment Guide](scripts/deploy.ps1)

---

## Contributing

See [VERSIONING.md](VERSIONING.md) for:
- How to bump versions
- When to create releases
- Commit message conventions
- Deployment workflow

---

**Latest Release:** [NBA_v33.0.8.0](https://github.com/JDSB123/green_bier_sports_nba_model/releases/tag/NBA_v33.0.8.0)
