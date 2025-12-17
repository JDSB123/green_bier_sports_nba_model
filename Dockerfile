# Multi-stage build for smaller, more secure images
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
COPY requirements.txt pyproject.toml setup.py ./
RUN pip install --no-cache-dir --user -r requirements.txt

# Final stage
FROM python:3.11-slim

WORKDIR /app

# Copy only the installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/

# Create necessary directories
RUN mkdir -p /app/data/processed/models

# Set environment variables
ENV DATA_PROCESSED_DIR=/app/data/processed
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')" || exit 1

# Run as non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8080"]
