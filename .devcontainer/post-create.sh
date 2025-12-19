#!/bin/bash
# Post-create script for GitHub Codespaces

set -e

echo "🚀 Setting up NBA Prediction System in Codespaces..."

# Create secrets directory if needed (for Docker secrets mount)
mkdir -p secrets

# Make scripts executable
chmod +x scripts/*.py 2>/dev/null || true

echo ""
echo "✅ Setup complete!"
echo ""
echo "📋 API keys are configured via:"
echo "   - ./secrets/ directory (mounted to /run/secrets in containers)"
echo "   - .env file (for docker-compose env_file)"
echo ""
echo "🚀 Start the API:"
echo "   docker compose up -d"
echo ""
echo "🏥 Check health:"
echo "   curl http://localhost:8090/health"
echo ""
echo "📊 Get predictions:"
echo "   curl http://localhost:8090/slate/today"
echo ""
echo "💡 Port 8090 will be automatically forwarded by Codespaces"
