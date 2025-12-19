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

### Step 1: Build the Container

```powershell
docker compose build
```

This creates the image: `nba-v51-final:latest`

### Step 2: Export Container Image

```powershell
# Export as compressed tar.gz
docker save nba-v51-final:latest | gzip > nba_v5.1_model_FINAL.tar.gz
```

### Step 3: Package for Distribution

Include these files in your distribution package:

1. **Container image**: `nba_v5.1_model_FINAL.tar.gz`
2. **Docker Compose config**: `docker-compose.yml`
3. **Environment template**: `.env.example` (optional, for reference)
4. **Secrets examples**: `secrets/*.example` files
5. **This file**: `EXPORT_INSTRUCTIONS.md`

### Step 4: Load Container on Target System

```bash
# Load the container image
docker load -i nba_v5.1_model_FINAL.tar.gz

# Verify image loaded
docker images | grep nba-v51-final
```

### Step 5: Setup Secrets (REQUIRED)

**Create secret files on target system:**

```powershell
# Create secrets directory
mkdir secrets

# Create API key files (replace with actual keys)
echo "your_odds_api_key" | Out-File -FilePath secrets\THE_ODDS_API_KEY -Encoding utf8 -NoNewline
echo "your_basketball_api_key" | Out-File -FilePath secrets\API_BASKETBALL_KEY -Encoding utf8 -NoNewline
```

**Important**: Secrets are NOT included in the container export. They must be created on the target system.

### Step 6: Deploy Container

```powershell
# Start container with docker-compose (recommended)
docker compose up -d

# Or manually with docker run
docker run -d `
  --name nba-v51-final `
  -p 8090:8080 `
  --read-only `
  --tmpfs /tmp:size=100M,mode=1777 `
  --tmpfs /app/outputs:size=50M,uid=1000,gid=1000,mode=0755 `
  -v ${PWD}/secrets:/run/secrets:ro `
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
- **Code**: Application code in `/app/src/`
- **Dependencies**: Python packages in `/home/appuser/.local`
- **Configuration**: Environment variables set in Dockerfile

## What's NOT Included

- API keys/secrets (must be provided via mounted secrets)
- `.env` file (not used - container uses Docker secrets only)
- Local data files
- Development tools

## Production Deployment Checklist

- [ ] Container image exported
- [ ] Secrets directory created on target system
- [ ] API keys added to secret files
- [ ] docker-compose.yml configured (if using compose)
- [ ] Port 8090 available
- [ ] Health check passing
- [ ] Logs reviewed for errors

## Troubleshooting

### Container fails to start
- Check secrets are mounted: `docker inspect nba-v51-final | grep -A 10 Mounts`
- Verify secret files exist and contain valid API keys
- Check logs: `docker logs nba-v51-final`

### Health check failing
- Verify models loaded: `curl http://localhost:8090/health`
- Check API keys are valid
- Review container logs for errors

### Secrets not found
- Ensure secrets directory is mounted: `./secrets:/run/secrets:ro`
- Verify secret file names match exactly: `THE_ODDS_API_KEY`, `API_BASKETBALL_KEY`
- Check file permissions (should be readable)
