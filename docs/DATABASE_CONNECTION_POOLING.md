# Database Connection Pooling Configuration

## Overview

This document describes the database connection pooling configuration for the NBA v5.0 BETA system.

## PostgreSQL Connection Pooling

### TimescaleDB/PostgreSQL

The system uses TimescaleDB (PostgreSQL extension) for time-series data storage. Connection pooling is handled at the application level.

#### Python Services (FastAPI)

For Python services using async database access:

**Recommended Configuration:**
- **Pool Size:** 5-20 connections per service
- **Max Overflow:** 10 connections
- **Pool Timeout:** 30 seconds
- **Pool Recycle:** 3600 seconds (1 hour)

**Example Configuration (asyncpg):**
```python
import asyncpg

pool = await asyncpg.create_pool(
    database_url,
    min_size=5,
    max_size=20,
    max_queries=50000,
    max_inactive_connection_lifetime=3600,
    timeout=30
)
```

#### Rust Services

For Rust services using sqlx:

**Recommended Configuration:**
- **Max Connections:** 10-20 per service
- **Connection Timeout:** 10 seconds
- **Idle Timeout:** 600 seconds (10 minutes)

**Example Configuration:**
```rust
let pool = sqlx::PgPool::connect_with(
    database_url.parse()?
        .max_connections(20)
        .acquire_timeout(Duration::from_secs(10))
        .idle_timeout(Duration::from_secs(600))
).await?;
```

### Connection Pool Monitoring

Monitor connection pool health:

1. **Active Connections:** Track number of active connections
2. **Idle Connections:** Monitor idle connection count
3. **Wait Time:** Track connection acquisition wait time
4. **Connection Errors:** Monitor connection failures

### Best Practices

1. **Size Appropriately:** Don't over-provision connections
2. **Monitor Usage:** Track connection pool metrics
3. **Handle Failures:** Implement retry logic for connection failures
4. **Connection Lifecycle:** Recycle connections periodically
5. **Resource Limits:** Set appropriate limits per service

### Environment Variables

Configure connection pooling via environment variables:

```env
# Database connection pool settings
DB_POOL_MIN_SIZE=5
DB_POOL_MAX_SIZE=20
DB_POOL_TIMEOUT=30
DB_POOL_RECYCLE=3600
```

## Redis Connection Pooling

Redis connections are managed by the Redis client library with connection pooling built-in.

**Recommended Configuration:**
- **Max Connections:** 10-50 per service
- **Connection Timeout:** 5 seconds
- **Retry Policy:** Exponential backoff

## Production Considerations

1. **Load Testing:** Test connection pool behavior under load
2. **Monitoring:** Set up alerts for connection pool exhaustion
3. **Scaling:** Adjust pool sizes based on actual usage
4. **Failover:** Configure connection pool for high availability
