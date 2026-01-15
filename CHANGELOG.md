# Changelog

All notable changes to the NBA Prediction Model are documented here.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [VERSIONING.md](VERSIONING.md).

---

## [NBA_v33.0.16.0] - 2026-01-14

### Added
- Comprehensive data stack audit script ([scripts/audit_data_stack.py](scripts/audit_data_stack.py)) - validates all 9 markets
- Performance grades and optimization recommendations in model_pack.json
- Data infrastructure documentation in model_pack.json

### Changed
- Updated model_pack.json with latest review (2026-01-14)
- Added threshold optimization recommendations for 1H Spread market
- Documented Azure Blob storage as single source of truth with audit status

### Reviewed
- Backtest results from v33.0.15.0 validated:
  - FG Spread: 60.6% accuracy / 15.7% ROI (Grade A)
  - FG Total: 59.2% accuracy / 13.1% ROI (Grade A-)
  - 1H Spread: 55.9% accuracy / 8.2% ROI (Grade B) - optimization recommended
  - 1H Total: 58.1% accuracy / 11.4% ROI (Grade A-)

---

## [NBA_v33.0.11.0] - 2026-01-05

### Changed
- Version sync to NBA_v33.0.11.0 across runtime, docs, and deployment config.
- Deployment tags aligned to VERSION for ACR/ACA.

---

## [NBA_v33.0.9.0] - 2026-01-05

### Added
- Historical **The Odds** line cache builder ([scripts/cache_theodds_lines.py](scripts/cache_theodds_lines.py))
- Training-data merge utility for The Odds FG + 1H lines with strict label recomputation ([scripts/merge_theodds_lines_into_training_data.py](scripts/merge_theodds_lines_into_training_data.py))
- Leakage-safe, fast walk-forward backtester using precomputed features ([scripts/walkforward_backtest_theodds.py](scripts/walkforward_backtest_theodds.py))
- Confidence-threshold optimizer for ROI-based decisioning ([scripts/optimize_backtest_thresholds.py](scripts/optimize_backtest_thresholds.py))
- Versioned backtest outputs and derived line cache committed under `data/backtest_results/` and `data/historical/derived/`

### Changed
- Improved leakage validation ergonomics (can validate arbitrary training CSV via `--training-path`) ([scripts/validate_leakage.py](scripts/validate_leakage.py))
- Backtest strictness: 1H markets no longer silently approximate 1H lines from FG lines in strict mode ([scripts/backtest.py](scripts/backtest.py))

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

**Latest Release:** [NBA_v33.0.11.0](https://github.com/JDSB123/green_bier_sports_nba_model/releases/tag/NBA_v33.0.11.0)
