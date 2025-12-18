# Multi-stage build for smaller, more secure images
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
COPY services/prediction-service-python/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --user -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# Final stage
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/models && \
    chown -R appuser:appuser /app

# Copy only installed packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy application code
COPY --chown=appuser:appuser services/prediction-service-python/app /app/app
COPY --chown=appuser:appuser src /app/src

ENV PYTHONPATH="/app:/app/src" \
    PATH="/home/appuser/.local/bin:$PATH"

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8082/health', timeout=5)" || exit 1

# Switch to non-root user
USER appuser

EXPOSE 8082

# Run service
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8082"]
