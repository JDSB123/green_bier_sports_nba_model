# Docker Secrets Guide - NBA v5.0 BETA

**Last Updated:** 2025-12-18  
**Status:** Production-Ready Secrets Management

---

## Overview

NBA v5.0 BETA now supports Docker secrets for secure API key and credential management. This provides a production-grade solution for managing sensitive data.

**Features:**
- ✅ Docker Swarm secrets (production)
- ✅ Docker Compose secret files (development)
- ✅ Automatic fallback to environment variables
- ✅ Secret management scripts
- ✅ Never committed to version control

---

## Architecture

### Secret Resolution Priority

The application reads secrets in this order:

1. **Docker Swarm secrets** (`/run/secrets/{name}`) - Production
2. **Docker Compose secret files** (`/run/secrets/{name}`) - Development
3. **Local secret files** (`./secrets/{name}`) - Local development
4. **Environment variables** (`{NAME}`) - Fallback
5. **Default value** - If none found

### Secret Sources

| Source | Use Case | Location |
|--------|----------|----------|
| Docker Swarm Secrets | Production deployment | `/run/secrets/` (in container) |
| Secret Files (Compose) | Development/testing | `./secrets/` (mounted to `/run/secrets/`) |
| Environment Variables | Quick setup | `.env` file or shell environment |

---

## Quick Start

### Development (Docker Compose)

1. **Create secrets from your existing `.env` file:**
   ```powershell
   python scripts/manage_secrets.py create-from-env
   ```

2. **Or create secrets manually:**
   ```powershell
   # Create secrets directory
   mkdir secrets
   
   # Create secret files (one per secret)
   echo "your_odds_api_key_here" > secrets\THE_ODDS_API_KEY
   echo "your_basketball_api_key_here" > secrets\API_BASKETBALL_KEY
   echo "your_secure_password" > secrets\DB_PASSWORD
   ```

3. **Start the stack:**
   ```powershell
   docker compose up -d
   ```

   Secrets are automatically mounted from `./secrets/` to `/run/secrets/` in containers.

### Production (Docker Swarm)

1. **Initialize Swarm (if not already):**
   ```powershell
   docker swarm init
   ```

2. **Create secrets in Swarm:**
   ```powershell
   # Create secrets from secret files
   python scripts/manage_secrets.py create-swarm
   
   # Or create manually:
   echo "your_key" | docker secret create THE_ODDS_API_KEY -
   echo "your_key" | docker secret create API_BASKETBALL_KEY -
   echo "your_password" | docker secret create DB_PASSWORD -
   ```

3. **Deploy the stack:**
   ```powershell
   docker stack deploy -c docker-compose.swarm.yml nba
   ```

4. **Check services:**
   ```powershell
   docker stack services nba
   docker service logs nba_strict-api
   ```

---

## Secret Management Script

The `scripts/manage_secrets.py` script provides utilities for managing secrets:

### Commands

**Create secrets from .env file:**
```powershell
python scripts/manage_secrets.py create-from-env
```

**Create Docker Swarm secrets:**
```powershell
python scripts/manage_secrets.py create-swarm
```

**List all secret files:**
```powershell
python scripts/manage_secrets.py list
```

**Validate secrets:**
```powershell
python scripts/manage_secrets.py validate
```

---

## Required Secrets

| Secret Name | Description | Required |
|-------------|-------------|----------|
| `THE_ODDS_API_KEY` | The Odds API key | ✅ Yes |
| `API_BASKETBALL_KEY` | API-Basketball key | ✅ Yes |
| `DB_PASSWORD` | Database password | ✅ Yes |

### Optional Secrets

| Secret Name | Description | Required |
|-------------|-------------|----------|
| `SERVICE_API_KEY` | API authentication key | ❌ No |
| `ACTION_NETWORK_USERNAME` | Action Network username | ❌ No |
| `ACTION_NETWORK_PASSWORD` | Action Network password | ❌ No |
| `BETSAPI_KEY` | BetsAPI key | ❌ No |
| `KAGGLE_API_TOKEN` | Kaggle API token | ❌ No |

---

## Migration from .env

### Step 1: Create Secrets

```powershell
# Create secrets from existing .env file
python scripts/manage_secrets.py create-from-env
```

This reads your `.env` file and creates secret files in `./secrets/`.

### Step 2: Verify Secrets

```powershell
# List created secrets
python scripts/manage_secrets.py list

# Validate all required secrets exist
python scripts/manage_secrets.py validate
```

### Step 3: Test

```powershell
# Start stack (secrets will be used automatically)
docker compose up -d

# Check health (should show secrets are loaded)
curl http://localhost:8090/health
```

### Step 4: Remove .env (Optional)

Once secrets are working, you can remove the `.env` file:

```powershell
# Backup first
copy .env .env.backup

# Remove (secrets will be used instead)
del .env
```

---

## Docker Compose Configuration

The `docker-compose.yml` file mounts secrets as volumes:

```yaml
volumes:
  - ./secrets:/run/secrets:ro  # Read-only mount
```

Services automatically read from `/run/secrets/` in the container.

### Example Service Configuration

```yaml
strict-api:
  volumes:
    - ./secrets:/run/secrets:ro
  environment:
    # Secrets take precedence over env vars
    - THE_ODDS_API_KEY=${THE_ODDS_API_KEY:-}
```

---

## Docker Swarm Configuration

The `docker-compose.swarm.yml` file uses native Docker secrets:

```yaml
services:
  strict-api:
    secrets:
      - THE_ODDS_API_KEY
      - API_BASKETBALL_KEY

secrets:
  THE_ODDS_API_KEY:
    external: true  # Must be created in Swarm first
```

### Creating Swarm Secrets

```powershell
# From secret files
python scripts/manage_secrets.py create-swarm

# Or manually
cat secrets/THE_ODDS_API_KEY | docker secret create THE_ODDS_API_KEY -
```

---

## Application Code

The application automatically reads secrets via `src/utils/secrets.py`:

```python
from src.utils.secrets import read_secret

# Read secret (tries Docker secrets, then env vars)
api_key = read_secret("THE_ODDS_API_KEY", default="")
```

The `src/config.py` uses secrets automatically:

```python
the_odds_api_key: str = field(
    default_factory=lambda: _secret_or_env("THE_ODDS_API_KEY", "THE_ODDS_API_KEY", "")
)
```

---

## Security Best Practices

### 1. Secret File Permissions

On Unix-like systems, secret files should have restrictive permissions:

```bash
chmod 600 secrets/THE_ODDS_API_KEY
```

The management script sets these automatically.

### 2. Never Commit Secrets

- ✅ Secret files are in `.gitignore`
- ✅ Only `.example` files are committed
- ✅ Never commit actual secret values

### 3. Rotate Secrets Regularly

```powershell
# Update secret file
echo "new_key" > secrets/THE_ODDS_API_KEY

# For Swarm, update the secret
echo "new_key" | docker secret create THE_ODDS_API_KEY_new -
# Update service to use new secret
# Remove old secret
docker secret rm THE_ODDS_API_KEY
```

### 4. Use Different Secrets per Environment

- Development: `secrets-dev/`
- Staging: `secrets-staging/`
- Production: Docker Swarm secrets

---

## Troubleshooting

### Secrets Not Found

**Error:** `Secret not found` or service fails to start

**Solution:**
1. Check secret files exist:
   ```powershell
   python scripts/manage_secrets.py list
   ```

2. Validate secrets:
   ```powershell
   python scripts/manage_secrets.py validate
   ```

3. Check file permissions (Unix):
   ```bash
   ls -la secrets/
   ```

### Swarm Secrets Not Working

**Error:** `secret not found` in Swarm

**Solution:**
1. Check Swarm is initialized:
   ```powershell
   docker info | Select-String Swarm
   ```

2. List Swarm secrets:
   ```powershell
   docker secret ls
   ```

3. Create missing secrets:
   ```powershell
   python scripts/manage_secrets.py create-swarm
   ```

### Fallback to Environment Variables

If secrets aren't working, the system automatically falls back to environment variables. Check:

1. `.env` file exists and has values
2. Environment variables are set in shell
3. Docker Compose is reading `.env` file

---

## Comparison: Secrets vs .env

| Feature | Docker Secrets | .env File |
|---------|---------------|-----------|
| Security | ✅ Encrypted at rest (Swarm) | ⚠️ Plain text |
| Version Control | ✅ Never committed | ⚠️ Risk of committing |
| Production Ready | ✅ Yes | ❌ No |
| Development | ✅ Works | ✅ Works |
| Swarm Support | ✅ Native | ❌ No |
| Rotation | ✅ Easy | ⚠️ Manual |

---

## Summary

✅ **Docker secrets implemented** - Production-grade secret management  
✅ **Backward compatible** - Falls back to `.env` for development  
✅ **Management scripts** - Easy secret creation and validation  
✅ **Swarm support** - Native Docker Swarm secrets  
✅ **Security hardened** - Secrets never logged or exposed  

**Next Steps:**
1. Create secrets from your `.env` file
2. Test with Docker Compose
3. Deploy to Swarm for production
4. Remove `.env` file once secrets are working

---

## References

- [Docker Secrets Documentation](https://docs.docker.com/engine/swarm/secrets/)
- [Docker Compose Secrets](https://docs.docker.com/compose/use-secrets/)
- [Security Hardening Guide](SECURITY_HARDENING.md)
