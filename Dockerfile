# NBA v5.0 BETA - Production Single-Container Dockerfile
# Hardened, production-ready image with baked-in models
# 
# Build: docker build -t nba-strict-api:latest .
# Run:   docker run -p 8090:8080 --env-file .env nba-strict-api:latest

# Multi-stage build for smaller, more secure images
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --user -r requirements.txt

# Final stage - production runtime
FROM python:3.11-slim

WORKDIR /app

# Create non-root user first (security best practice)
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/data/processed/models && \
    chown -R appuser:appuser /app

# Copy only the installed packages from builder to appuser's home
COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local

# Copy application code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser scripts/ ./scripts/

# Copy production model pack (baked into image for reproducibility)
COPY --chown=appuser:appuser models/production/ /app/data/processed/models/

# Verify model files exist (fail fast if missing)
RUN ls -la /app/data/processed/models/ && \
    test -f /app/data/processed/models/spreads_model.joblib && \
    test -f /app/data/processed/models/totals_model.joblib && \
    test -f /app/data/processed/models/first_half_spread_model.pkl && \
    test -f /app/data/processed/models/first_half_spread_features.pkl && \
    test -f /app/data/processed/models/first_half_total_model.pkl && \
    test -f /app/data/processed/models/first_half_total_features.pkl && \
    echo "✓ All required model files verified"

# Set environment variables
ENV DATA_PROCESSED_DIR=/app/data/processed
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Healthcheck using stdlib (no external dependencies)
# Uses urllib.request instead of requests to avoid dependency issues
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health', timeout=5).read()" || exit 1

# Switch to non-root user (security)
USER appuser

EXPOSE 8080

# Production command
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8080"]
