FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies
COPY services/prediction-service-python/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# Copy application code + v4 source tree
COPY services/prediction-service-python/app /app/app
COPY src /app/src
RUN mkdir -p /app/models

ENV PYTHONPATH="/app:/app/src"

# Expose port
EXPOSE 8082

# Run service
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8082"]
