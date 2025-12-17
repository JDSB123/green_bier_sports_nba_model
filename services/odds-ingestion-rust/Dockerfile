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

# Runtime image
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# Copy binary
COPY --from=builder /app/odds-ingestion/target/release/odds-ingestion /app/odds-ingestion

ENV THE_ODDS_API_KEY= \
    DATABASE_URL=postgres://nba:nba@postgres:5432/nba \
    REDIS_URL=redis://redis:6379 \
    POLL_INTERVAL_SECONDS=30 \
    RUST_LOG=info

ENTRYPOINT ["/app/odds-ingestion"]
