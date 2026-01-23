# Azure Credentials Guide for clause.com / ACR

This guide explains where to find the Azure credentials needed for clause.com to run your code through Azure Container Registry (ACR).

## Required Credentials

You need to provide clause.com with these four values:
- `AZURE_CLIENT_ID` - Application (client) ID
- `AZURE_CLIENT_SECRET` - Client secret value
- `AZURE_TENANT_ID` - Directory (tenant) ID
- `AZURE_SUBSCRIPTION_ID` - Subscription ID

---

## Step-by-Step: Where to Find Each Credential

### 1. AZURE_SUBSCRIPTION_ID

**Location:** Azure Portal → Subscriptions

1. Go to [Azure Portal](https://portal.azure.com)
2. In the search bar at the top, type "Subscriptions" and select it
3. You'll see a list of your subscriptions
4. Click on the subscription you want to use (likely the one containing `nba-gbsv-model-rg`)
5. The **Subscription ID** is displayed at the top of the Overview page (format: `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`)

**Alternative (Azure CLI):**
```bash
az account show --query id -o tsv
```

---

### 2. AZURE_TENANT_ID

**Location:** Azure Portal → Azure Active Directory → Overview

1. Go to [Azure Portal](https://portal.azure.com)
2. In the search bar, type "Azure Active Directory" (or "Microsoft Entra ID") and select it
3. On the Overview page, you'll see **Tenant ID** listed (format: `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`)

**Alternative (Azure CLI):**
```bash
az account show --query tenantId -o tsv
```

---

### 3. AZURE_CLIENT_ID (Application ID)

**Location:** Azure Portal → Azure Active Directory → App registrations

You need to create an App Registration (Service Principal) if you don't have one:

#### Option A: Create a New App Registration

1. Go to [Azure Portal](https://portal.azure.com)
2. Navigate to **Azure Active Directory** → **App registrations**
3. Click **+ New registration**
4. Enter a name (e.g., "clause-com-acr-access")
5. Select **Accounts in this organizational directory only**
6. Click **Register**
7. After creation, you'll see the **Application (client) ID** on the Overview page - this is your `AZURE_CLIENT_ID`

#### Option B: Use Existing App Registration

1. Go to **Azure Active Directory** → **App registrations**
2. Find your existing app (e.g., `gbs-nba-github` if you have one)
3. Click on it
4. The **Application (client) ID** is on the Overview page

**Alternative (Azure CLI):**
```bash
# List all app registrations
az ad app list --query "[].{DisplayName:displayName, AppId:appId}" -o table

# Get specific app's client ID
az ad app show --id <app-id> --query appId -o tsv
```

---

### 4. AZURE_CLIENT_SECRET

**Location:** Azure Portal → Azure Active Directory → App registrations → [Your App] → Certificates & secrets

**Important:** You must create a new client secret if one doesn't exist or if the existing one has expired.

1. Go to **Azure Active Directory** → **App registrations**
2. Click on your app registration (the one you're using for `AZURE_CLIENT_ID`)
3. In the left menu, click **Certificates & secrets**
4. Under **Client secrets**, click **+ New client secret**
5. Enter a description (e.g., "clause.com ACR access")
6. Choose an expiration (recommended: 12 or 24 months)
7. Click **Add**
8. **IMPORTANT:** Copy the **Value** immediately - it will only be shown once!
   - The value shown is your `AZURE_CLIENT_SECRET`
   - If you lose it, you'll need to create a new one

**Alternative (Azure CLI):**
```bash
# Create a new client secret
az ad app credential reset --id <client-id> --append

# Note: The output will show the secret value - copy it immediately!
```

---

## Granting Permissions to the Service Principal

After creating/identifying your App Registration, you need to grant it permissions to access ACR:

### Grant ACR Access (Required for clause.com)

1. Go to **Azure Portal** → **Container registries**
2. Click on your registry (e.g., `nbagbsacr`)
3. Click **Access control (IAM)** in the left menu
4. Click **+ Add** → **Add role assignment**
5. Select role: **AcrPush** (or **Contributor** for broader access)
6. Click **Next**
7. Under **Assign access to**, select **Managed identity, user, or service principal**
8. Click **+ Select members**
9. Search for your app registration name and select it
10. Click **Select**, then **Review + assign**

**Alternative (Azure CLI):**
```bash
# Get your ACR resource ID
ACR_NAME="nbagbsacr"
ACR_ID=$(az acr show --name $ACR_NAME --query id -o tsv)

# Assign AcrPush role to your service principal
az role assignment create \
  --assignee <your-client-id> \
  --role AcrPush \
  --scope $ACR_ID
```

---

## Verify Your Credentials

Test that your credentials work:

```bash
# Set your credentials as environment variables
export AZURE_CLIENT_ID="<your-client-id>"
export AZURE_CLIENT_SECRET="<your-secret>"
export AZURE_TENANT_ID="<your-tenant-id>"
export AZURE_SUBSCRIPTION_ID="<your-subscription-id>"

# Login using service principal
az login --service-principal \
  -u $AZURE_CLIENT_ID \
  -p $AZURE_CLIENT_SECRET \
  --tenant $AZURE_TENANT_ID

# Verify you can access ACR
az acr login --name nbagbsacr
```

---

## Security Best Practices

1. **Never commit secrets to git** - Use environment variables or secret management
2. **Set appropriate expiration** - Don't create secrets that never expire
3. **Use least privilege** - Grant only the minimum permissions needed (AcrPush for ACR access)
4. **Rotate secrets regularly** - Create new secrets before old ones expire
5. **Monitor usage** - Check Azure AD sign-in logs for service principal activity

---

## Quick Reference: Your Current Setup

Based on your codebase, you may already have these values:

- **Subscription ID:** Check `docs/WORKFLOW_FIX_SUMMARY.md` (may contain: `3a1a4a94-45a5-4f7c-8ada-97978221052c`)
- **Tenant ID:** Check `docs/WORKFLOW_FIX_SUMMARY.md` (may contain: `18ee0910-417d-4a81-a3f5-7945bdbd5a78`)
- **Client ID:** You may have a service principal named `gbs-nba-github` (ID: `971db985-be14-4352-bb1d-144d8e8b198c`)

**Note:** You can reuse an existing service principal or create a new one specifically for clause.com. If creating a new one, follow the steps above.

---

## Troubleshooting

### "Authentication failed" error
- Verify the client secret hasn't expired
- Check that the service principal has the correct role assignments
- Ensure tenant ID and subscription ID are correct

### "Access denied" to ACR
- Verify the service principal has **AcrPush** or **Contributor** role on the ACR
- Check that you're using the correct subscription ID

### "Subscription not found"
- Verify the subscription ID is correct
- Ensure the service principal has access to the subscription (may need Contributor role at subscription level)

