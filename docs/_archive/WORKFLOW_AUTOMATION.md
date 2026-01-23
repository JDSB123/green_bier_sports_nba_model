# NBA Model: Automated CI/CD & Dev Workflow

## 🚀 How It Works
- **Every push to `main`** triggers a full build, test, and deploy to Azure Container Apps.
- **Manual deploys** can be triggered from the GitHub Actions UI.
- **All secrets** (API keys, Action Network creds) are managed in GitHub → Settings → Secrets.
- **Azure authentication** uses OIDC (federated credential) – no static Azure secrets needed.
- **Post-deploy smoke tests** ensure the API is healthy and predictions are live.

## 🟢 How to Work Safely
1. **Edit code as needed** in your Codespace or local dev environment.
2. **Run tests locally** (`pytest tests -v`) before pushing.
3. **Stage, commit, and push** to `main`:
   ```sh
   git add .
   git commit -m "your message"
   git push origin main
   ```
4. **That’s it!** The workflow will:
   - Build and test
   - Deploy to Azure
   - Run health checks
   - Fail loudly if anything is wrong

## 🛡️ What’s Automated
- **No manual Azure login or secret copying**
- **No manual Docker build/push**
- **No manual deployment steps**
- **No silent failures** – all errors are visible in GitHub Actions

## 🧑‍💻 For Agents & Automation
- Agents can safely stage/commit/push at any time.
- No need to manually run deployment scripts.
- All environment and deployment state is synced to GitHub and Azure automatically.

## 🔒 Requirements
- GitHub repo secrets: THE_ODDS_API_KEY, API_BASKETBALL_KEY (and Action Network creds if needed)
- Azure federated credential (OIDC) set up for this repo/branch/app registration

## 📝 Troubleshooting
- If a deploy fails, check GitHub Actions logs for the error.
- If OIDC/Azure login fails, verify federated credential in Azure App Registration.
- If secrets are missing, add them in GitHub → Settings → Secrets.

---

**This workflow is designed for zero-friction, robust, production-grade NBA model operations.**

For details, see `.github/workflows/gbs-nba-deploy.yml` and `docs/AZURE_CONFIG.md`.
