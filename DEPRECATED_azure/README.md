# Deprecated Azure Function App Deployment

**Status:** DEPRECATED - Removed from production path

**Date:** 2025-12-18

**Reason:** Single-container Docker image is now the production source of truth. Azure Function App deployment path has been removed in favor of the hardened `nba-strict-api` Docker image.

**Replacement:** Use `./run.ps1` to build and run the production Docker container.

**Note:** This directory is kept for historical reference only. Do not use for new deployments.
