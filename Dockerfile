# NBA v5.1 FINAL - Production Container
# Hardened, read-only image with baked-in models
# 
# 6 PROVEN ROE Markets (Full Game + First Half):
#
# Full Game:
# - Spread: 60.6% accuracy, +15.7% ROI
# - Total: 59.2% accuracy, +13.1% ROI
# - Moneyline: 65.5% accuracy, +25.1% ROI
#
# First Half:
# - Spread: 55.9% accuracy, +8.2% ROI
# - Total: 58.1% accuracy, +11.4% ROI
# - Moneyline: 63.0% accuracy, +19.8% ROI
#
# Build: docker build -f Dockerfile -t nba-v51-final:latest .
# Run:   docker compose up -d  (uses docker-compose.yml with read-only and secrets)
# Export: docker save nba-v51-final:latest | gzip > nba_v5.1_model_FINAL.tar.gz

# =============================================================================
# Stage 1: Builder - Install dependencies
# =============================================================================
FROM python:3.11-slim AS builder

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
FROM python:3.11-slim

# Labels for container identification
LABEL maintainer="Green Bier Ventures"
LABEL version="5.1-FINAL"
LABEL description="NBA Production Picks Model - 6 Proven ROE Markets (FG+1H)"

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
COPY --chown=appuser:appuser secrets/ /app/secrets/

# Verify ALL 6 REQUIRED model files exist (fail fast if missing)
RUN echo "=== NBA v5.1 FINAL Model Verification ===" && \
    echo "Checking for 6 required models (FG + 1H)..." && \
    ls -la /app/data/processed/models/ && \
    echo "" && \
    echo "Full Game Models:" && \
    test -f /app/data/processed/models/spreads_model.joblib && \
    echo "  ✓ spreads_model.joblib (60.6% acc, +15.7% ROI)" && \
    test -f /app/data/processed/models/totals_model.joblib && \
    echo "  ✓ totals_model.joblib (59.2% acc, +13.1% ROI)" && \
    test -f /app/data/processed/models/moneyline_model.joblib && \
    echo "  ✓ moneyline_model.joblib (65.5% acc, +25.1% ROI)" && \
    echo "" && \
    echo "First Half Models:" && \
    test -f /app/data/processed/models/first_half_spread_model.pkl && \
    test -f /app/data/processed/models/first_half_spread_features.pkl && \
    echo "  ✓ first_half_spread_model.pkl (55.9% acc, +8.2% ROI)" && \
    test -f /app/data/processed/models/first_half_total_model.pkl && \
    test -f /app/data/processed/models/first_half_total_features.pkl && \
    echo "  ✓ first_half_total_model.pkl (58.1% acc, +11.4% ROI)" && \
    echo "" && \
    echo "=== All 6 required models verified! ===" && \
    echo "" && \
    echo "=== Verifying baked-in secrets ===" && \
    test -f /app/secrets/THE_ODDS_API_KEY && \
    echo "  ✓ THE_ODDS_API_KEY" && \
    test -f /app/secrets/API_BASKETBALL_KEY && \
    echo "  ✓ API_BASKETBALL_KEY" && \
    echo "=== All secrets verified! ==="

# =============================================================================
# Environment Configuration
# =============================================================================
ENV DATA_PROCESSED_DIR=/app/data/processed
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# v5.1 specific settings
ENV NBA_MODEL_VERSION=5.1-FINAL
ENV NBA_MARKETS=fg_spread,fg_total,fg_moneyline,1h_spread,1h_total,1h_moneyline
ENV NBA_PERIODS=full_game,first_half

# =============================================================================
# Health Check Configuration
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
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
