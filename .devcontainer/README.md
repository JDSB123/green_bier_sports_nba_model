# Running NBA Prediction System on GitHub Codespaces

This configuration allows you to run the entire NBA prediction system online through GitHub Codespaces.

## 🚀 Quick Start

1. **Open in Codespaces:**
   - Go to your GitHub repository
   - Click "Code" > "Codespaces" > "Create codespace on main"

2. **Environment setup (automatic):**
   - Post-create runs `scripts/setup_codespace.sh`, which creates/uses a repo-local `.venv`, installs `requirements.txt`, and syncs Codespaces secrets into `.env`/`secrets/`.
   - VS Code is pointed at `.venv` by default to avoid global installs and cross-project bleed.

3. **API Keys Configuration:**
   - API keys are handled via Docker secrets (`./secrets/` directory) or `.env` file
   - The containers automatically read from mounted secrets at `/run/secrets/`
   - Ensure your API keys are in `./secrets/` or `.env` before starting

4. **Start the API:**
   ```bash
   docker compose up -d
   ```

5. **Access the API:**
   - The API will be available on port 8090
   - GitHub Codespaces will automatically forward the port
   - Click the port notification or go to "Ports" tab to get the public URL

## 📋 Usage

### Health Check
```bash
curl http://localhost:8090/health
```

### Get Today's Predictions
```bash
curl http://localhost:8090/slate/today
```

### Run Analysis Script
```bash
python scripts/run_slate.py
```

## 🔧 API Key Configuration

The Docker containers read API keys from:

1. **Docker Secrets** (preferred): `./secrets/` directory
   - Mounted to `/run/secrets/` in containers
   - Files: `THE_ODDS_API_KEY`, `API_BASKETBALL_KEY`, etc.

2. **Environment Variables**: `.env` file
   - Read via `env_file` in docker-compose.yml

Both methods are self-contained within the containers - no external configuration needed.

## ⚠️ Important Notes

1. **Models:** Models are baked into the Docker image
2. **Storage:** Codespaces have limited storage (typically 32GB)
3. **Timeout:** Codespaces pause after 30 minutes of inactivity (free tier)

## 🔗 Resources

- [GitHub Codespaces Docs](https://docs.github.com/en/codespaces)
