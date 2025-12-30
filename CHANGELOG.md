# Changelog

All notable changes to the NBA Prediction Model are documented here.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [VERSIONING.md](VERSIONING.md).

---

## [NBA_v33.0.8.0] - 2025-12-29

### Added
- **Archive folder** for historical output tracking (predictions, picks, slate analysis, odds snapshots)
- **Automated deployment script** ([scripts/deploy.ps1](scripts/deploy.ps1)) - one-command deployment reading VERSION file
- **Version bump automation** ([scripts/bump_version.py](scripts/bump_version.py)) - synchronizes version across all files
- **CI/CD version validation** ([.github/workflows/version-check.yml](.github/workflows/version-check.yml)) - blocks PRs with version drift
- **Complete versioning documentation** ([VERSIONING.md](VERSIONING.md)) - semantic versioning guidelines
- Odds snapshot archiving in API responses

### Fixed
- Version consistency across all source files
- Deployment pipeline with safety checks (git push enforcement, health validation)
- Spread sign calculations in prediction engine
- Archiving functionality in predict.py, run_slate.py, and app.py

### Changed
- Updated .gitignore to properly handle archive/ folder (tracked for audit trail)
- Improved error handling in deployment scripts

---

## [NBA_v33.0.7.0] - 2025-12-28

### Changed
- Calibration improvements for edge calculations
- Edge threshold tuning based on backtest results
- Model confidence adjustments

---

## [NBA_v33.0.2.0] - 2025-12-22

### Added
- Production deployment with 4 markets (1H + FG spreads/totals)
- Azure Container App infrastructure

### Changed
- Stabilized prediction API endpoints
- Updated Docker configuration for production

---

## [NBA_v33.0.0.0] - 2025-12-15

### 🎯 Major Refactor
Complete market structure overhaul to 4 markets.

### Changed
- Consolidated to 1H + FG only (4 markets total):
  - 1H Spread, 1H Total
  - FG Spread, FG Total
- Updated prediction engine for 4-market structure
- Retrained models with new feature set

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
