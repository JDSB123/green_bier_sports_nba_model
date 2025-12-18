# Deprecated Multi-Service Docker Compose

**Status:** DEPRECATED - Replaced by single-container production image

**Date:** 2025-12-18

**Reason:** Production now uses a single hardened Docker container (`nba-strict-api`) instead of the multi-service compose stack. The single container includes all required models baked-in and has no dependencies on postgres/redis/microservices.

**Replacement:** Use `./run.ps1` to build and run the production Docker container.

**Note:** This compose file is kept for reference only. The backtest compose (`docker-compose.backtest.yml`) remains active for development/testing purposes.
