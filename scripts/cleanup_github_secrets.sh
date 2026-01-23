#!/bin/bash
# GitHub Secrets Cleanup Script
# Run this script to remove deprecated/unused secrets from the repository
#
# Prerequisites: gh CLI installed and authenticated with admin:org scope
# Install: https://cli.github.com/
# Auth: gh auth login --scopes "admin:org,repo,workflow"

set -e

echo "🔍 GitHub Secrets Cleanup - NBA Model Repository"
echo "=================================================="
echo ""

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI (gh) is not installed"
    echo "📦 Install from: https://cli.github.com/"
    exit 1
fi

# Check authentication
if ! gh auth status &> /dev/null; then
    echo "❌ Not authenticated with GitHub CLI"
    echo "🔐 Run: gh auth login --scopes 'admin:org,repo,workflow'"
    exit 1
fi

echo "✅ GitHub CLI authenticated"
echo ""

# Function to safely remove a secret
remove_secret() {
    local secret_name=$1
    echo "🗑️  Removing: $secret_name"
    if gh secret remove "$secret_name" 2>/dev/null; then
        echo "   ✅ Removed successfully"
    else
        echo "   ⚠️  Not found or already removed"
    fi
}

echo "Phase 1: Remove Registry Credentials (Security Risk)"
echo "----------------------------------------------------"
remove_secret "NBAGBSVAPI_REGISTRY_PASSWORD"
remove_secret "NBAGBSVAPI_REGISTRY_USERNAME"
echo ""

echo "Phase 2: Remove Duplicate Azure Credentials"
echo "--------------------------------------------"
remove_secret "NBAGBSVAPI_AZURE_CLIENT_ID"
remove_secret "NBAGBSVAPI_AZURE_TENANT_ID"
remove_secret "NBAGBSVAPI_AZURE_SUBSCRIPTION_ID"
echo ""

echo "Phase 3: Remove Action Network Credentials (Optional)"
echo "-------------------------------------------------------"
echo "⚠️  Only removing if not actively used"
read -p "Remove ACTION_NETWORK_USERNAME and ACTION_NETWORK_PASSWORD? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    remove_secret "ACTION_NETWORK_USERNAME"
    remove_secret "ACTION_NETWORK_PASSWORD"
    echo "✅ Action Network secrets removed"
else
    echo "⏭️  Skipping Action Network secrets"
fi
echo ""

echo "Phase 4: Remove Legacy Azure Credentials (After OIDC Migration)"
echo "----------------------------------------------------------------"
echo "⚠️  Only remove this AFTER verifying deploy.yml works with OIDC"
read -p "Remove AZURE_CREDENTIALS (legacy auth)? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    remove_secret "AZURE_CREDENTIALS"
    echo "✅ Legacy Azure credentials removed"
else
    echo "⏭️  Keeping AZURE_CREDENTIALS (remove after testing OIDC)"
fi
echo ""

echo "✅ Cleanup Complete!"
echo ""
echo "📋 Remaining Required Secrets:"
echo "   - THE_ODDS_API_KEY"
echo "   - API_BASKETBALL_KEY"
echo "   - AZURE_CLIENT_ID"
echo "   - AZURE_TENANT_ID"
echo "   - AZURE_SUBSCRIPTION_ID"
echo ""
echo "🔍 Verify with: gh secret list"
