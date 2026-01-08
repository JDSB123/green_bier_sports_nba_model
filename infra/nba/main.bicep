// Green Bier Sports - NBA Model Infrastructure
// Single-source-of-truth deployment - ALL resources in one file.
//
// Usage:
//   az deployment group create -g nba-gbsv-model-rg -f infra/nba/main.bicep `
//     -p theOddsApiKey=... apiBasketballKey=... imageTag=<tag>

targetScope = 'resourceGroup'

// =============================================================================
// PARAMETERS
// =============================================================================

@description('Azure region')
@minLength(1)
@maxLength(50)
param location string = resourceGroup().location

@description('Environment')
@allowed(['dev', 'staging', 'prod'])
param environment string = 'prod'

@description('NBA app semantic version (tag + resource tagging)')
param versionTag string = 'NBA_v33.0.11.0'

@description('Container image tag (defaults to versionTag)')
param imageTag string = versionTag

@description('Application identifier for tagging')
param appTag string = 'nba-model'

@description('Owner tag value')
param ownerTag string = 'sports-analytics'

@description('Cost center tag value')
param costCenterTag string = 'sports-nba'

@description('Compliance tag value')
param complianceTag string = 'internal'

@description('Additional tags to merge onto all resources')
param extraTags object = {}

// Resource names (match actual Azure resources)
@description('Container App name')
param appName string = 'nba-gbsv-api'

@description('Container Apps Environment name')
param containerAppEnvName string = 'nba-gbsv-model-env'

@description('Container Registry name')
param acrName string = 'nbagbsacr'

@description('Key Vault name')
param keyVaultName string = 'nbagbs-keyvault'

@description('Storage account name')
param storageAccountName string = 'nbagbsvstrg'

// Teams Bot resource names
@description('Function App name for Teams bot trigger')
param functionAppName string = 'nba-picks-trigger'

@description('Bot Service name')
param botServiceName string = 'nba-picks-bot'

@description('App Service Plan name for Function App')
param appServicePlanName string = 'nba-gbsv-func-plan'

@description('Deploy Teams Bot resources (Function App, Bot Service). Requires Azure quota for Dynamic VMs.')
param deployTeamsBot bool = false

// API Keys (required)
@description('The Odds API Key (required)')
@secure()
param theOddsApiKey string

@description('API-Basketball Key (required)')
@secure()
param apiBasketballKey string

// Teams Bot credentials (required for bot)
@description('Microsoft App ID for Teams Bot')
param microsoftAppId string = ''

@description('Microsoft App Tenant ID for Teams Bot')
param microsoftAppTenantId string = ''

@description('Microsoft App Password for Teams Bot')
@secure()
param microsoftAppPassword string = ''

// Optional integrations
@description('Database URL (optional)')
@secure()
param databaseUrl string = ''

@description('Website domain for CORS')
param websiteDomain string = 'greenbiersportventures.com'

@description('Comma-separated origins for app-level CORS (matches ACA ingress)')
param allowedOrigins string = 'https://${websiteDomain},https://www.${websiteDomain}'

// Application security/auth
@description('Require API authentication for the service. Defaults to false for safety. If set to true, serviceApiKey MUST be provided or deployment will fail at runtime.')
param requireApiAuth bool = false

@description('Service API key for authenticated access. REQUIRED if requireApiAuth=true. If requireApiAuth=true and this is empty, the application will fail to start with a validation error.')
@secure()
param serviceApiKey string = ''

// Application runtime configuration
@description('Current NBA season (e.g., 2025-2026)')
param currentSeason string = '2025-2026'

@description('Comma-separated list of seasons to process')
param seasonsToProcess string = '2024-2025,2025-2026'

@description('Prediction feature validation mode: strict|warn|silent')
param predictionFeatureMode string = 'strict'

// Filter thresholds (string values passed as env vars)
@description('Min confidence for FG spreads (e.g., 0.62)')
param filterSpreadMinConfidence string = '0.62'

@description('Min edge (points) for FG spreads (e.g., 2.0)')
param filterSpreadMinEdge string = '2.0'

@description('Min confidence for FG totals (e.g., 0.72)')
param filterTotalMinConfidence string = '0.72'

@description('Min edge (points) for FG totals (e.g., 3.0)')
param filterTotalMinEdge string = '3.0'

@description('Min confidence for 1H spreads (e.g., 0.68)')
param filter1hSpreadMinConfidence string = '0.68'

@description('Min edge (points) for 1H spreads (e.g., 1.5)')
param filter1hSpreadMinEdge string = '1.5'

@description('Min confidence for 1H totals (e.g., 0.66)')
param filter1hTotalMinConfidence string = '0.66'

@description('Min edge (points) for 1H totals (e.g., 2.0)')
param filter1hTotalMinEdge string = '2.0'

// Optional premium integrations
@description('Action Network username (optional)')
param actionNetworkUsername string = ''

@description('Action Network password (optional)')
@secure()
param actionNetworkPassword string = ''

// Scaling
@description('Minimum replicas')
param minReplicas int = 1

@description('Maximum replicas')
param maxReplicas int = 3

@description('Concurrent requests per replica before scaling out')
param concurrentRequests string = '50'

@description('CPU cores for the container')
param containerCpu string = '0.5'

@description('Memory for the container')
param containerMemory string = '1Gi'

// =============================================================================
// TAGS
// =============================================================================

var requiredTags = {
  enterprise: 'green-bier-sports-ventures'
  app: appTag
  environment: environment
  owner: ownerTag
  cost_center: costCenterTag
  compliance: complianceTag
  version: versionTag
  managedBy: 'bicep'
}

var tags = union(requiredTags, extraTags)

// =============================================================================
// PLATFORM RESOURCES (ACR, Key Vault, Log Analytics, App Insights, CA Env)
// =============================================================================

// Container Registry
resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: acrName
  location: location
  tags: tags
  sku: { name: 'Basic' }
  properties: { adminUserEnabled: true }
}

// Key Vault
resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: keyVaultName
  location: location
  tags: tags
  properties: {
    sku: { family: 'A', name: 'standard' }
    tenantId: subscription().tenantId
    enableRbacAuthorization: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 7
    enablePurgeProtection: true
  }
}

// Log Analytics Workspace
resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: 'gbs-logs-${environment}'
  location: location
  tags: tags
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
  }
}

// Application Insights
resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: 'gbs-insights-${environment}'
  location: location
  tags: tags
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logAnalytics.id
  }
}

// Container Apps Environment
resource containerAppEnv 'Microsoft.App/managedEnvironments@2023-05-01' = {
  name: containerAppEnvName
  location: location
  tags: tags
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
    workloadProfiles: [
      {
        name: 'Consumption'
        workloadProfileType: 'Consumption'
      }
    ]
  }
}

// =============================================================================
// DATA LAYER (Storage)
// =============================================================================

module storage '../modules/storage.bicep' = {
  name: 'data-storage'
  params: {
    name: storageAccountName
    app: 'nba'
    environment: environment
    location: location
    tags: tags
    containers: ['models', 'predictions', 'results', 'nbahistoricaldata']
  }
}

// =============================================================================
// COMPUTE LAYER (Container App)
// =============================================================================

// Secrets
var apiSecrets = concat(
  [
    { name: 'acr-password', value: acr.listCredentials().passwords[0].value }
    { name: 'the-odds-api-key', value: theOddsApiKey }
    { name: 'api-basketball-key', value: apiBasketballKey }
    { name: 'app-insights-connection-string', value: appInsights.properties.ConnectionString }
  ],
  databaseUrl == '' ? [] : [{ name: 'database-url', value: databaseUrl }],
  serviceApiKey == '' ? [] : [{ name: 'service-api-key', value: serviceApiKey }],
  actionNetworkUsername == '' ? [] : [{ name: 'action-network-username', value: actionNetworkUsername }],
  actionNetworkPassword == '' ? [] : [{ name: 'action-network-password', value: actionNetworkPassword }]
)

// Environment variables
var appEnvVars = concat(
  [
    { name: 'THE_ODDS_API_KEY', secretRef: 'the-odds-api-key' }
    { name: 'API_BASKETBALL_KEY', secretRef: 'api-basketball-key' }
    { name: 'APPLICATIONINSIGHTS_CONNECTION_STRING', secretRef: 'app-insights-connection-string' }
    { name: 'ALLOWED_ORIGINS', value: allowedOrigins }
    { name: 'REQUIRE_API_AUTH', value: string(requireApiAuth) }
    { name: 'GBS_SPORT', value: 'nba' }
    { name: 'NBA_MODEL_VERSION', value: versionTag }
    { name: 'NBA_MARKETS', value: '1h_spread,1h_total,fg_spread,fg_total' }
    { name: 'NBA_PERIODS', value: 'first_half,full_game' }
    { name: 'AZURE_STORAGE_CONNECTION_STRING', value: storage.outputs.connectionString }
    { name: 'CURRENT_SEASON', value: currentSeason }
    { name: 'SEASONS_TO_PROCESS', value: seasonsToProcess }
    { name: 'PREDICTION_FEATURE_MODE', value: predictionFeatureMode }
    // FG thresholds
    { name: 'FILTER_SPREAD_MIN_CONFIDENCE', value: filterSpreadMinConfidence }
    { name: 'FILTER_SPREAD_MIN_EDGE', value: filterSpreadMinEdge }
    { name: 'FILTER_TOTAL_MIN_CONFIDENCE', value: filterTotalMinConfidence }
    { name: 'FILTER_TOTAL_MIN_EDGE', value: filterTotalMinEdge }
    // 1H thresholds
    { name: 'FILTER_1H_SPREAD_MIN_CONFIDENCE', value: filter1hSpreadMinConfidence }
    { name: 'FILTER_1H_SPREAD_MIN_EDGE', value: filter1hSpreadMinEdge }
    { name: 'FILTER_1H_TOTAL_MIN_CONFIDENCE', value: filter1hTotalMinConfidence }
    { name: 'FILTER_1H_TOTAL_MIN_EDGE', value: filter1hTotalMinEdge }
  ],
  databaseUrl == '' ? [] : [{ name: 'DATABASE_URL', secretRef: 'database-url' }],
  serviceApiKey == '' ? [] : [{ name: 'SERVICE_API_KEY', secretRef: 'service-api-key' }],
  actionNetworkUsername == '' ? [] : [{ name: 'ACTION_NETWORK_USERNAME', secretRef: 'action-network-username' }],
  actionNetworkPassword == '' ? [] : [{ name: 'ACTION_NETWORK_PASSWORD', secretRef: 'action-network-password' }]
)

module containerApp '../modules/containerApp.bicep' = {
  name: 'compute-app'
  params: {
    name: appName
    location: location
    managedEnvironmentId: containerAppEnv.id
    tags: tags
    image: '${acr.properties.loginServer}/nba-gbsv-api:${imageTag}'
    envVars: appEnvVars
    secrets: apiSecrets
    registries: [
      {
        server: acr.properties.loginServer
        username: acr.listCredentials().username
        passwordSecretRef: 'acr-password'
      }
    ]
    ingressOrigins: [
      'http://localhost:3000'
      'https://*.azurewebsites.net'
      'https://${websiteDomain}'
      'https://www.${websiteDomain}'
    ]
    targetPort: 8090
    transport: 'auto'
    minReplicas: minReplicas
    maxReplicas: maxReplicas
    httpConcurrentRequests: concurrentRequests
    cpu: containerCpu
    memory: containerMemory
    activeRevisionsMode: 'Single'
  }
}

// =============================================================================
// TEAMS BOT LAYER (Function App + Bot Service) - OPTIONAL
// Requires Azure quota for Dynamic VMs in the target region.
// Set deployTeamsBot=true to enable.
// =============================================================================

// App Service Plan (Consumption/Dynamic for Function App)
resource funcAppServicePlan 'Microsoft.Web/serverfarms@2023-01-01' = if (deployTeamsBot) {
  name: appServicePlanName
  location: location
  tags: tags
  kind: 'functionapp'
  sku: {
    name: 'Y1'
    tier: 'Dynamic'
    size: 'Y1'
    family: 'Y'
    capacity: 0
  }
  properties: {
    reserved: true // Linux
  }
}

// Function App for Teams Bot trigger
resource functionApp 'Microsoft.Web/sites@2023-01-01' = if (deployTeamsBot) {
  name: functionAppName
  location: location
  tags: tags
  kind: 'functionapp,linux'
  properties: {
    serverFarmId: funcAppServicePlan.id
    reserved: true
    siteConfig: {
      linuxFxVersion: 'Python|3.11'
      appSettings: [
        { name: 'FUNCTIONS_WORKER_RUNTIME', value: 'python' }
        { name: 'FUNCTIONS_EXTENSION_VERSION', value: '~4' }
        { name: 'AzureWebJobsStorage', value: storage.outputs.connectionString }
        { name: 'APPLICATIONINSIGHTS_CONNECTION_STRING', value: appInsights.properties.ConnectionString }
        { name: 'NBA_API_URL', value: 'https://${containerApp.outputs.containerAppFqdn}' }
        { name: 'MICROSOFT_APP_ID', value: microsoftAppId }
        { name: 'MICROSOFT_APP_TENANT_ID', value: microsoftAppTenantId }
        { name: 'MICROSOFT_APP_PASSWORD', value: microsoftAppPassword }
      ]
    }
    httpsOnly: true
  }
}

// Bot Service
resource botService 'Microsoft.BotService/botServices@2022-09-15' = if (deployTeamsBot && microsoftAppId != '') {
  name: botServiceName
  location: 'global'
  tags: tags
  kind: 'azurebot'
  sku: {
    name: 'F0'
  }
  properties: {
    displayName: 'NBA Picks Bot'
    endpoint: 'https://${functionApp.properties.defaultHostName}/api/bot'
    msaAppId: microsoftAppId
    msaAppTenantId: microsoftAppTenantId
    msaAppType: 'SingleTenant'
  }
}

// =============================================================================
// OUTPUTS
// =============================================================================

output containerAppFqdn string = containerApp.outputs.containerAppFqdn
output containerAppUrl string = containerApp.outputs.containerAppUrl
output storageAccountName string = storage.outputs.storageAccountName
output acrLoginServer string = acr.properties.loginServer
output keyVaultUri string = keyVault.properties.vaultUri
output appInsightsConnectionString string = appInsights.properties.ConnectionString
output functionAppUrl string = deployTeamsBot ? 'https://${functionApp.properties.defaultHostName}' : ''
output botEndpoint string = deployTeamsBot && microsoftAppId != '' ? 'https://${functionApp.properties.defaultHostName}/api/bot' : ''

