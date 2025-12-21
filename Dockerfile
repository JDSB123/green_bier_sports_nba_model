# NBA v6.0 - Production Container (9 Independent Markets)
# Hardened, read-only image with baked-in models
#
# 9 INDEPENDENT MARKETS (Q1 + 1H + FG × Spread/Total/Moneyline):
#
# First Quarter (3):
# - Spread: q1_spread_model.joblib
# - Total: q1_total_model.joblib
# - Moneyline: q1_moneyline_model.joblib
#
# First Half (3):
# - Spread: 1h_spread_model.pkl (55.9% accuracy, +8.2% ROI)
# - Total: 1h_total_model.pkl (58.1% accuracy, +11.4% ROI)
# - Moneyline: 1h_moneyline_model.pkl
#
# Full Game (3):
# - Spread: fg_spread_model.joblib (60.6% accuracy, +15.7% ROI)
# - Total: fg_total_model.joblib (59.2% accuracy, +13.1% ROI)
# - Moneyline: fg_moneyline_model.joblib (65.5% accuracy, +25.1% ROI)
#
# Build: docker build -f Dockerfile -t nba-v60:latest .
# Run:   docker compose up -d  (uses docker-compose.yml with read-only and secrets)
# Export: docker save nba-v60:latest | gzip > nba_v6.0_model.tar.gz

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
LABEL version="6.0"
LABEL description="NBA Production Picks Model - 9 Independent Markets (Q1+1H+FG)"

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

# Verify ALL 9 REQUIRED model files exist (fail fast if missing)
# 9 markets: Q1 (3) + 1H (3) + FG (3), with 1H having separate feature files
RUN echo "=== NBA v6.0 Model Verification ===" && \
    echo "Checking for 9 independent market models (Q1 + 1H + FG)..." && \
    ls -la /app/data/processed/models/ && \
    echo "" && \
    echo "First Quarter Models (3):" && \
    test -f /app/data/processed/models/q1_spread_model.joblib && \
    echo "  ✓ q1_spread_model.joblib" && \
    test -f /app/data/processed/models/q1_total_model.joblib && \
    echo "  ✓ q1_total_model.joblib" && \
    test -f /app/data/processed/models/q1_moneyline_model.joblib && \
    echo "  ✓ q1_moneyline_model.joblib" && \
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
    echo "  ✓ 1h_moneyline_model.pkl" && \
    echo "" && \
    echo "Full Game Models (3):" && \
    test -f /app/data/processed/models/fg_spread_model.joblib && \
    echo "  ✓ fg_spread_model.joblib (60.6% acc, +15.7% ROI)" && \
    test -f /app/data/processed/models/fg_total_model.joblib && \
    echo "  ✓ fg_total_model.joblib (59.2% acc, +13.1% ROI)" && \
    test -f /app/data/processed/models/fg_moneyline_model.joblib && \
    echo "  ✓ fg_moneyline_model.joblib (65.5% acc, +25.1% ROI)" && \
    echo "" && \
    echo "=== All 9 independent market models verified! ==="

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
ENV CURRENT_SEASON=2024-2025
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
# Q1-specific filters (STRICTER for profitability)
ENV FILTER_Q1_MIN_CONFIDENCE=0.60
ENV FILTER_Q1_MIN_EDGE_PCT=0.05

# CORS Configuration
ENV ALLOWED_ORIGINS=*

# v6.4 STRICT MODE - All 9 markets required (baked-in env defaults)
ENV NBA_MODEL_VERSION=6.4-STRICT
ENV NBA_MARKETS=q1_spread,q1_total,q1_moneyline,1h_spread,1h_total,1h_moneyline,fg_spread,fg_total,fg_moneyline
ENV NBA_PERIODS=first_quarter,first_half,full_game
ENV NBA_STRICT_MODE=true

# =============================================================================
# ONLY THESE 2 SECRETS ARE REQUIRED AT RUNTIME:
#   - THE_ODDS_API_KEY (via env var or /run/secrets/THE_ODDS_API_KEY)
#   - API_BASKETBALL_KEY (via env var or /run/secrets/API_BASKETBALL_KEY)
# =============================================================================

# =============================================================================
# Health Check Configuration - STRICT MODE: All 9 models required
# =============================================================================
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; r=urllib.request.urlopen('http://localhost:8080/health', timeout=5); import json; d=json.loads(r.read()); exit(0 if d.get('engine_loaded') and d.get('markets')==9 else 1)" || exit 1

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
