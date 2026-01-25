// Green Bier Sports - Infrastructure Entry Point
//
// SINGLE ENTRYPOINT: deploy the NBA picks API (Container App) and optional trigger adapters.
// This wrapper exists so CI/CD or humans can always deploy from infra/main.bicep.
//
// Usage:
//   az deployment group create -g nba-gbsv-model-rg -f infra/main.bicep \
//     -p imageTag=<tag> theOddsApiKey=... apiBasketballKey=...
//
// Notes:
// - The Container App API is the single prediction surface.
// - Optional trigger adapters (Teams Bot / Function App) should only call the API,
//   not duplicate prediction logic.

targetScope = 'resourceGroup'

@description('Azure region')
@minLength(1)
@maxLength(50)
param location string = resourceGroup().location

@description('Environment')
@allowed(['dev', 'staging', 'prod'])
param environment string = 'prod'

@description('Container image tag (used for version tagging + deployment)')
param imageTag string

@description('Deploy Teams Bot resources (Function App, Bot Service).')
param deployTeamsBot bool = false

@description('The Odds API Key (required)')
@secure()
param theOddsApiKey string

@description('API-Basketball Key (required)')
@secure()
param apiBasketballKey string

@description('Action Network username (optional; required for premium splits)')
@secure()
param actionNetworkUsername string = ''

@description('Action Network password (optional; required for premium splits)')
@secure()
param actionNetworkPassword string = ''

@description('Microsoft App ID for Teams Bot')
param microsoftAppId string = ''

@description('Microsoft App Tenant ID for Teams Bot')
param microsoftAppTenantId string = ''

@description('Microsoft App Password for Teams Bot')
@secure()
param microsoftAppPassword string = ''

@description('Website domain for CORS')
param websiteDomain string = 'greenbiersportventures.com'

@description('Allowed origins for CORS (passed through to container app and API)')
param allowedOrigins array = [
  'https://*.azurewebsites.net'
  'https://${websiteDomain}'
  'https://www.${websiteDomain}'
]

@description('Require API authentication for the container app')
param requireApiAuth bool = false

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

module nba 'nba/main.bicep' = {
  name: 'nba'
  params: {
    location: location
    environment: environment
    imageTag: imageTag
    deployTeamsBot: deployTeamsBot
    theOddsApiKey: theOddsApiKey
    apiBasketballKey: apiBasketballKey
    actionNetworkUsername: actionNetworkUsername
    actionNetworkPassword: actionNetworkPassword
    microsoftAppId: microsoftAppId
    microsoftAppTenantId: microsoftAppTenantId
    microsoftAppPassword: microsoftAppPassword
    websiteDomain: websiteDomain
    allowedOrigins: allowedOrigins
    requireApiAuth: requireApiAuth
    minReplicas: minReplicas
    maxReplicas: maxReplicas
    concurrentRequests: concurrentRequests
    containerCpu: containerCpu
    containerMemory: containerMemory
  }
}

output containerAppFqdn string = nba.outputs.containerAppFqdn
output containerAppUrl string = nba.outputs.containerAppUrl
output acrLoginServer string = nba.outputs.acrLoginServer
output keyVaultUri string = nba.outputs.keyVaultUri
