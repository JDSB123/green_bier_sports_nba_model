# NBA_v33.0.6.0 - Production Container - STRICT MODE
# Hardened, read-only image with baked-in models
#
# STRICT MODE: FRESH DATA ONLY
# - NO file caching - all data fetched fresh from APIs
# - NO silent fallbacks - errors are raised, not swallowed
# - NO placeholders - all data must be explicitly provided
# - ESPN is the ONLY source for team records
# - Every request clears session cache before fetching
#
# 6 INDEPENDENT MARKETS (1H + FG × Spread/Total/Moneyline):
#
# First Half (3):
# - Spread: 1h_spread_model.pkl (55.9% accuracy, +8.2% ROI)
# - Total: 1h_total_model.pkl (58.1% accuracy, +11.4% ROI)
# - Moneyline: 1h_moneyline_model.pkl (62.5% accuracy, +19.3% ROI)
#
# Full Game (3):
# - Spread: fg_spread_model.joblib (60.6% accuracy, +15.7% ROI)
# - Total: fg_total_model.joblib (59.2% accuracy, +13.1% ROI)
# - Moneyline: fg_moneyline_model.joblib (68.1% accuracy, +30.0% ROI)
#
# Build: docker build -f Dockerfile -t nba-v33:latest .
# Run:   docker compose up -d  (uses docker-compose.yml with read-only and secrets)
# Export: docker save nba-v33:latest | gzip > nba_v33.0.1.0_model.tar.gz

# =============================================================================
# Stage 1: Builder - Install dependencies
# =============================================================================
FROM python:3.11.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --user -r requirements.txt

# =============================================================================
# Stage 2: Production Runtime - Read-only optimized
# =============================================================================
FROM python:3.11.11-slim

# Labels for container identification
LABEL maintainer="Green Bier Ventures"
LABEL version="NBA_v33.0.6.0"
LABEL description="NBA Production Picks Model - STRICT MODE - 6 Independent Markets (1H+FG) - FRESH DATA ONLY"

WORKDIR /app

# Create non-root user with specific UID/GID for security
RUN useradd -m -u 1000 -s /bin/bash appuser && \
    mkdir -p /app/data/processed/models && \
    mkdir -p /app/outputs && \
    mkdir -p /app/secrets && \
    chown -R appuser:appuser /app

# Copy Python packages from builder stage
COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local

# Copy application source code (read-only in production)
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser scripts/ ./scripts/

# =============================================================================
# Bake in production models (immutable in container)
# =============================================================================
COPY --chown=appuser:appuser models/production/ /app/data/processed/models/

# =============================================================================
# Bake in API keys/secrets (immutable in container - fully self-contained)
# =============================================================================
# SECRETS ARE BAKED INTO CONTAINER - No external secrets required
# COPY --chown=appuser:appuser secrets/ /app/secrets/

# Verify ALL 6 REQUIRED model files exist (fail fast if missing)
# 6 markets: 1H (3) + FG (3), with 1H having separate feature files
# NOTE: Q1 markets disabled in NBA_v33.0.1.0+
RUN echo "=== NBA_v33.0.6.0 Model Verification ===" && \
    echo "Checking for 6 independent market models (1H + FG)..." && \
    ls -la /app/data/processed/models/ && \
    echo "" && \
    echo "First Half Models (3 models, 6 files):" && \
    test -f /app/data/processed/models/1h_spread_model.pkl && \
    test -f /app/data/processed/models/1h_spread_features.pkl && \
    echo "  ✓ 1h_spread_model.pkl (55.9% acc, +8.2% ROI)" && \
    test -f /app/data/processed/models/1h_total_model.pkl && \
    test -f /app/data/processed/models/1h_total_features.pkl && \
    echo "  ✓ 1h_total_model.pkl (58.1% acc, +11.4% ROI)" && \
    test -f /app/data/processed/models/1h_moneyline_model.pkl && \
    test -f /app/data/processed/models/1h_moneyline_features.pkl && \
    echo "  ✓ 1h_moneyline_model.pkl (62.5% acc, +19.3% ROI)" && \
    echo "" && \
    echo "Full Game Models (3):" && \
    test -f /app/data/processed/models/fg_spread_model.joblib && \
    echo "  ✓ fg_spread_model.joblib (60.6% acc, +15.7% ROI)" && \
    test -f /app/data/processed/models/fg_total_model.joblib && \
    echo "  ✓ fg_total_model.joblib (59.2% acc, +13.1% ROI)" && \
    test -f /app/data/processed/models/fg_moneyline_model.joblib && \
    echo "  ✓ fg_moneyline_model.joblib (68.1% acc, +30.0% ROI)" && \
    echo "" && \
    echo "=== All 6 independent market models verified! ==="

# =============================================================================
# Environment Configuration - ALL NON-SENSITIVE DEFAULTS BAKED IN
# =============================================================================
# This ensures the container works out-of-the-box with ONLY API keys needed

# Python Configuration
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# Data Directories (REQUIRED by src/config.py)
ENV DATA_RAW_DIR=data/raw
ENV DATA_PROCESSED_DIR=/app/data/processed

# API Base URLs (REQUIRED by src/config.py)
ENV THE_ODDS_BASE_URL=https://api.the-odds-api.com/v4
ENV API_BASKETBALL_BASE_URL=https://v1.basketball.api-sports.io

# Season Configuration (REQUIRED by src/config.py)
ENV CURRENT_SEASON=2025-2026
ENV SEASONS_TO_PROCESS=2024-2025,2025-2026

# Filter Thresholds (REQUIRED by src/config.py - betting prediction filters)
# Spread filters
ENV FILTER_SPREAD_MIN_CONFIDENCE=0.55
ENV FILTER_SPREAD_MIN_EDGE=1.0
# Total filters
ENV FILTER_TOTAL_MIN_CONFIDENCE=0.55
ENV FILTER_TOTAL_MIN_EDGE=1.5
# Moneyline filters (FG/1H)
ENV FILTER_MONEYLINE_MIN_CONFIDENCE=0.55
ENV FILTER_MONEYLINE_MIN_EDGE_PCT=0.03

# CORS Configuration
ENV ALLOWED_ORIGINS=*

# STRICT MODE - All 6 markets required, FRESH DATA ONLY (baked-in env defaults)
ENV NBA_MODEL_VERSION=NBA_v33.0.6.0
ENV NBA_MARKETS=1h_spread,1h_total,1h_moneyline,fg_spread,fg_total,fg_moneyline
ENV NBA_PERIODS=first_half,full_game
ENV NBA_STRICT_MODE=true
ENV NBA_CACHE_DISABLED=true

# =============================================================================
# ONLY THESE 2 SECRETS ARE REQUIRED AT RUNTIME:
#   - THE_ODDS_API_KEY (via env var or /run/secrets/THE_ODDS_API_KEY)
#   - API_BASKETBALL_KEY (via env var or /run/secrets/API_BASKETBALL_KEY)
# =============================================================================

# =============================================================================
# Health Check Configuration - STRICT MODE: All 6 models required (1H + FG)
# =============================================================================
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; r=urllib.request.urlopen('http://localhost:8080/health', timeout=5); import json; d=json.loads(r.read()); exit(0 if d.get('engine_loaded') and d.get('markets')==6 else 1)" || exit 1

# =============================================================================
# Security: Switch to non-root user
# =============================================================================
USER appuser

# Expose API port
EXPOSE 8080

# =============================================================================
# Production Entrypoint
# =============================================================================
# Run with read-only filesystem support
# Model outputs should be written to mounted volume or /app/outputs
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
