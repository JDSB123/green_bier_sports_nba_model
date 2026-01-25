// Reusable Container App module
@description('Container App name')
param name string

@description('Azure region')
param location string

@description('Managed Environment resource ID')
param managedEnvironmentId string

@description('Tags to apply to the container app')
param tags object

@description('Container image (registry/repository:tag)')
param image string

@description('Container environment variables (name/value or name/secretRef objects)')
param envVars array

@description('Container app secrets array [{ name, value }]')
param secrets array

@description('Registry configuration array [{ server, username, passwordSecretRef }]')
param registries array

@description('Ingress allowed origins for CORS')
param ingressOrigins array = []

@description('Ingress target port')
param targetPort int = 8090

@description('Ingress transport')
param transport string = 'auto'

@description('Minimum replicas')
param minReplicas int = 1

@description('Maximum replicas')
param maxReplicas int = 3

@description('Concurrent HTTP requests per replica for scale rules')
param httpConcurrentRequests string = '50'

@description('CPU cores')
param cpu string = '0.5'

@description('Memory allocation')
param memory string = '1Gi'

@description('Active revisions mode (Single or Multiple)')
param activeRevisionsMode string = 'Single'

@description('Identity type')
param identityType string = 'SystemAssigned'

resource containerApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: name
  location: location
  tags: tags
  identity: {
    type: identityType
  }
  properties: {
    managedEnvironmentId: managedEnvironmentId
    configuration: {
      ingress: {
        external: true
        targetPort: targetPort
        transport: transport
        corsPolicy: {
          allowedOrigins: ingressOrigins
          allowedMethods: [
            'GET'
            'POST'
            'OPTIONS'
          ]
          allowedHeaders: ['*']
        }
      }
      registries: registries
      secrets: secrets
      activeRevisionsMode: activeRevisionsMode
    }
    template: {
      containers: [
        {
          name: name
          image: image
          resources: {
            cpu: json(cpu)
            memory: memory
          }
          env: envVars
          probes: [
            {
              type: 'Liveness'
              httpGet: {
                path: '/health'
                port: targetPort
              }
              initialDelaySeconds: 10
              periodSeconds: 30
            }
            {
              type: 'Readiness'
              httpGet: {
                path: '/health'
                port: targetPort
              }
              initialDelaySeconds: 5
              periodSeconds: 10
            }
          ]
        }
      ]
      scale: {
        minReplicas: minReplicas
        maxReplicas: maxReplicas
        rules: [
          {
            name: 'http-rule'
            http: {
              metadata: {
                concurrentRequests: httpConcurrentRequests
              }
            }
          }
        ]
      }
    }
  }
}

output containerAppFqdn string = containerApp.properties.configuration.ingress.fqdn
output containerAppUrl string = 'https://${containerApp.properties.configuration.ingress.fqdn}'
