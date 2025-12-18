# Odds Ingestion Service (Rust)
# Multi-stage build
FROM rust:1.75-slim AS builder
WORKDIR /app

# Create a new empty project to leverage dependency caching
RUN cargo new --bin odds-ingestion
WORKDIR /app/odds-ingestion

# Copy manifest and fetch deps
COPY Cargo.toml ./Cargo.toml
RUN cargo fetch

# Copy source
COPY src ./src

# Build release
RUN cargo build --release

# Runtime image - minimal and secure
FROM debian:bookworm-slim

# Install only essential runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

WORKDIR /app

# Copy binary with proper ownership
COPY --from=builder --chown=appuser:appuser /app/odds-ingestion/target/release/odds-ingestion /app/odds-ingestion

# Environment variables (set at runtime via docker-compose)
ENV POLL_INTERVAL_SECONDS=30 \
    RUST_LOG=info

# Switch to non-root user
USER appuser

ENTRYPOINT ["/app/odds-ingestion"]
