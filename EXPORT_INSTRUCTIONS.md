# NBA v5.1 FINAL - Container Export Instructions

## Hardened Production Container

This container is fully hardened for production deployment:
- ✅ Read-only root filesystem
- ✅ Non-root user (appuser, UID 1000)
- ✅ No new privileges security option
- ✅ Resource limits enforced
- ✅ Strict Docker secrets mode (no fallbacks)
- ✅ Models baked into image
- ✅ Health checks configured

## Export Container for Distribution

### Step 1: Prepare Secrets (REQUIRED BEFORE BUILD)

**IMPORTANT**: Secrets must exist in `./secrets/` directory BEFORE building the container.

The Dockerfile copies secrets from `./secrets/` into the container. Only `.example` files are committed to git - you must create the actual secret files:

```powershell
# Create secrets directory (if it doesn't exist)
if (-not (Test-Path secrets)) { New-Item -ItemType Directory -Path secrets }

# Create actual API key files (replace with your real API keys)
echo "your_actual_odds_api_key" | Out-File -FilePath secrets\THE_ODDS_API_KEY -Encoding utf8 -NoNewline
echo "your_actual_basketball_api_key" | Out-File -FilePath secrets\API_BASKETBALL_KEY -Encoding utf8 -NoNewline
```

**CRITICAL**: The build will FAIL if these files don't exist or are empty. `.example` files are NOT valid secrets.

### Step 2: Build the Container

```powershell
docker compose build
```

This creates the image: `nba-v51-final:latest`

The build process will verify that both secret files exist and are non-empty before completing.

### Step 3: Export Container Image

```powershell
# Export as compressed tar.gz
docker save nba-v51-final:latest | gzip > nba_v5.1_model_FINAL.tar.gz
```

### Step 4: Package for Distribution

Include these files in your distribution package:

1. **Container image**: `nba_v5.1_model_FINAL.tar.gz` (includes secrets baked-in)
2. **Docker Compose config**: `docker-compose.yml` (optional)
3. **This file**: `EXPORT_INSTRUCTIONS.md`

**Note**: Secrets are BAKED INTO the container image - no separate secrets setup required!

### Step 5: Load Container on Target System

```bash
# Load the container image
docker load -i nba_v5.1_model_FINAL.tar.gz

# Verify image loaded
docker images | grep nba-v51-final
```

### Step 6: Deploy Container

**No secrets setup needed - they're already in the container!**

```powershell
# Start container with docker-compose (recommended)
docker compose up -d

# Or manually with docker run (NO SECRETS VOLUME NEEDED)
docker run -d `
  --name nba-v51-final `
  -p 8090:8080 `
  --read-only `
  --tmpfs /tmp:size=100M,mode=1777 `
  --tmpfs /app/outputs:size=50M,uid=1000,gid=1000,mode=0755 `
  nba-v51-final:latest
```

### Step 7: Verify Deployment

```powershell
# Check container status
docker ps --filter "name=nba-v51"

# Check health
curl http://localhost:8090/health

# View logs
docker logs nba-v51-final
```

## Security Features

The exported container includes:

1. **Read-only filesystem**: Prevents modification of container files
2. **Non-root execution**: Runs as `appuser` (UID 1000)
3. **No privilege escalation**: `no-new-privileges:true`
4. **Resource limits**: CPU (2.0) and Memory (2G) limits
5. **Strict secrets**: Only reads from `/run/secrets/`, fails if missing
6. **Health checks**: Automatic health monitoring
7. **Immutable models**: Models baked into image at build time

## Container Contents

- **Models**: All 7 model files baked into `/app/data/processed/models/`
- **Secrets**: API keys baked into `/app/secrets/` (fully self-contained)
- **Code**: Application code in `/app/src/`
- **Dependencies**: Python packages in `/home/appuser/.local`
- **Configuration**: Environment variables set in Dockerfile

**Everything is included - no external dependencies required!**

## What's NOT Included

- Local data files (generated at runtime in `/app/outputs`)
- Development tools

## Production Deployment Checklist

- [ ] Container image exported (includes secrets)
- [ ] Container image loaded on target system
- [ ] Port 8090 available
- [ ] Container started (docker compose up -d or docker run)
- [ ] Health check passing
- [ ] Logs reviewed for errors

**No secrets setup required - everything is in the container!**

## Troubleshooting

### Container fails to start
- Check logs: `docker logs nba-v51-final`
- Verify container image includes secrets (they're baked in at build time)
- Ensure secrets directory exists in source before building: `./secrets/THE_ODDS_API_KEY` and `./secrets/API_BASKETBALL_KEY`

### Health check failing
- Verify models loaded: `curl http://localhost:8090/health`
- Check API keys are valid
- Review container logs for errors

### Build fails with "Required secret files missing or empty"
- Ensure actual secret files exist in `./secrets/` before building (not just `.example` files)
- Verify files are non-empty: `Get-Content secrets\THE_ODDS_API_KEY` should show your actual API key
- Check that `.gitignore` in `secrets/` is not preventing files from being present locally
- Note: Only `.example` files are in git - you must create actual secret files locally before building

### Secrets not found at runtime
- Secrets should be baked into container at `/app/secrets/` during build
- If container starts but fails with SecretNotFoundError, secrets may have been missing/empty at build time
- Rebuild container with valid secret files in `./secrets/` directory
