// Green Bier Sports - NBA Model Infrastructure
// Single-source-of-truth deployment for the NBA resource group.
//
// Usage:
//   az deployment group create -g <nba-rg> -f infra/nba/main.bicep `
//     -p theOddsApiKey=... apiBasketballKey=... imageTag=<tag>

targetScope = 'resourceGroup'

@description('Azure region')
@minLength(1)
@maxLength(50)
param location string = resourceGroup().location

@description('Environment')
@allowed([
  'dev'
  'staging'
  'prod'
])
param environment string = 'prod'

@description('NBA app semantic version (tag + resource tagging)')
param versionTag string = 'NBA_v33.0.8.0'

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

@description('Container App name')
param appName string = 'nba-gbsv-api'

@description('Container Apps Environment name')
param containerAppEnvName string = 'nbagbsvmodel-env'

@description('ACR name (existing)')
param acrName string = 'nbagbsacr'

@description('Storage account name override (optional)')
param storageAccountName string = ''

@description('Database URL (optional)')
@secure()
param databaseUrl string = ''

@description('Application Insights connection string (optional)')
@secure()
param appInsightsConnectionString string = ''

// =============================================================================
// API KEYS - Required for the NBA Picks API to function
// =============================================================================
@description('The Odds API Key (required)')
@secure()
param theOddsApiKey string

@description('API-Basketball Key (required)')
@secure()
param apiBasketballKey string

@description('Website domain for CORS (e.g., greenbiersportventures.com)')
param websiteDomain string = 'greenbiersportventures.com'

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

// Existing shared resources in the same RG
resource containerAppEnv 'Microsoft.App/managedEnvironments@2023-05-01' existing = {
  name: containerAppEnvName
}

resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' existing = {
  name: acrName
}

// Required tag policy (merges with extraTags)
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

// Secrets list (optional entries appended)
var apiSecrets = concat(
  [
    {
      name: 'acr-password'
      value: acr.listCredentials().passwords[0].value
    }
    {
      name: 'the-odds-api-key'
      value: theOddsApiKey
    }
    {
      name: 'api-basketball-key'
      value: apiBasketballKey
    }
  ],
  databaseUrl == '' ? [] : [
    {
      name: 'database-url'
      value: databaseUrl
    }
  ],
  appInsightsConnectionString == '' ? [] : [
    {
      name: 'app-insights-connection-string'
      value: appInsightsConnectionString
    }
  ]
)

// Data layer (Storage)
module storage '../modules/storage.bicep' = {
  name: 'data-storage'
  params: {
    name: storageAccountName
    app: 'nba'
    environment: environment
    location: location
    tags: tags
    containers: [
      'models'
      'predictions'
      'results'
    ]
  }
}

// Environment variables (optional entries appended)
var appEnvVars = concat(
  [
    {
      name: 'THE_ODDS_API_KEY'
      secretRef: 'the-odds-api-key'
    }
    {
      name: 'API_BASKETBALL_KEY'
      secretRef: 'api-basketball-key'
    }
    {
      name: 'GBS_SPORT'
      value: 'nba'
    }
    {
      name: 'NBA_MODEL_VERSION'
      value: versionTag
    }
    {
      name: 'NBA_MARKETS'
      value: '1h_spread,1h_total,fg_spread,fg_total'
    }
    {
      name: 'NBA_PERIODS'
      value: 'first_half,full_game'
    }
    {
      name: 'AZURE_STORAGE_CONNECTION_STRING'
      value: storage.outputs.connectionString
    }
  ],
  databaseUrl == '' ? [] : [
    {
      name: 'DATABASE_URL'
      secretRef: 'database-url'
    }
  ],
  appInsightsConnectionString == '' ? [] : [
    {
      name: 'APPLICATIONINSIGHTS_CONNECTION_STRING'
      secretRef: 'app-insights-connection-string'
    }
  ]
)

// Compute layer (Container App)
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
      'https://*.greenbier.com'
      'https://${websiteDomain}'
      'https://www.${websiteDomain}'
    ]
    targetPort: 8080
    transport: 'auto'
    minReplicas: minReplicas
    maxReplicas: maxReplicas
    httpConcurrentRequests: concurrentRequests
    cpu: containerCpu
    memory: containerMemory
    revisionMode: 'Single'
  }
}

// Outputs
output containerAppFqdn string = containerApp.outputs.containerAppFqdn
output containerAppUrl string = containerApp.outputs.containerAppUrl
output storageAccountName string = storage.outputs.storageAccountName
