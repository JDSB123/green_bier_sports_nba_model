# Multi-stage build for smaller, more secure images
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
COPY requirements.txt pyproject.toml setup.py ./
RUN pip install --no-cache-dir --user -r requirements.txt

# Final stage
FROM python:3.11-slim

WORKDIR /app

# Create non-root user first
RUN useradd -m -u 1000 appuser

# Copy only the installed packages from builder to appuser's home
COPY --from=builder /root/.local /home/appuser/.local

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/

# Create necessary directories
RUN mkdir -p /app/data/processed/models

# Set ownership
RUN chown -R appuser:appuser /app /home/appuser/.local

# Set environment variables
ENV DATA_PROCESSED_DIR=/app/data/processed
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')" || exit 1

# Switch to non-root user
USER appuser

EXPOSE 8080

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8080"]
