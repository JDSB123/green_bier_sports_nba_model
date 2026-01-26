// Reusable Container App Job module
@description('Container App Job name')
param name string

@description('Azure region')
param location string

@description('Managed Environment resource ID')
param managedEnvironmentId string

@description('Tags to apply to the job')
param tags object

@description('Container image (registry/repository:tag)')
param image string

@description('Container environment variables (name/value or name/secretRef objects)')
param envVars array

@description('Container job secrets array [{ name, value }]')
param secrets array

@description('Registry configuration array [{ server, username, passwordSecretRef }]')
param registries array

@description('Schedule cron expression (UTC)')
param scheduleCron string = '*/5 * * * *'

@description('Parallelism for scheduled job')
param parallelism int = 1

@description('Replica completion count')
param replicaCompletionCount int = 1

@description('Replica timeout in seconds')
param replicaTimeout int = 1800

@description('Replica retry limit')
param replicaRetryLimit int = 1

@description('CPU cores')
param cpu string = '0.25'

@description('Memory allocation')
param memory string = '0.5Gi'

@description('Container command (entrypoint)')
param command array = []

resource containerJob 'Microsoft.App/jobs@2023-05-01' = {
  name: name
  location: location
  tags: tags
  properties: {
    environmentId: managedEnvironmentId
    configuration: {
      triggerType: 'Schedule'
      scheduleTriggerConfig: {
        cronExpression: scheduleCron
        parallelism: parallelism
        replicaCompletionCount: replicaCompletionCount
      }
      registries: registries
      secrets: secrets
      replicaTimeout: replicaTimeout
      replicaRetryLimit: replicaRetryLimit
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
          command: command
        }
      ]
    }
  }
}

output jobName string = containerJob.name
