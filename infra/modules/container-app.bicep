param appHostName string
param containerImage string
param containerAppName string
param containerAppsEnvironmentResourceId string
param entraClientId string
param enablePublicIngress bool
param isPlaceholder bool
param keyVaultSecretUri string
param location string
param registryLoginServer string
param tags object
param tenantId string
param mlflowStorageName string = ''

var mlflowEnabled = !empty(mlflowStorageName)
var appContainerImage = isPlaceholder ? 'mcr.microsoft.com/azuredocs/containerapps-helloworld:latest' : containerImage

module app 'br/public:avm/res/app/container-app:0.23.0' = {
  name: 'app'
  params: {
    activeRevisionsMode: 'Single'
    authConfig: isPlaceholder
      ? null
      : {
          globalValidation: {
            unauthenticatedClientAction: 'RedirectToLoginPage'
          }
          httpSettings: {
            requireHttps: true
          }
          identityProviders: {
            azureActiveDirectory: {
              enabled: true
              registration: {
                clientId: entraClientId
                clientSecretSettingName: 'entra-client-secret'
                openIdIssuer: '${environment().authentication.loginEndpoint}${tenantId}/v2.0'
              }
              validation: {
                allowedAudiences: [
                  entraClientId
                ]
              }
            }
          }
          platform: {
            enabled: true
          }
        }
    containers: [
      {
        image: appContainerImage
        name: 'robotics-demo'
        resources: {
          cpu: 1
          memory: '2Gi'
        }
        env: concat([
          {
            name: 'ROBOTICS_DEMO_HOST'
            value: '0.0.0.0'
          }
          {
            name: 'ROBOTICS_DEMO_PORT'
            value: '8765'
          }
          {
            name: 'ROBOTICS_DEMO_ALLOWED_HOSTS'
            value: appHostName
          }
        ], mlflowEnabled ? [
          {
            name: 'MLFLOW_TRACKING_URI'
            value: 'file:/mnt/mlflow/mlruns'
          }
        ] : [])
        volumeMounts: mlflowEnabled ? [
          {
            volumeName: 'mlflow'
            mountPath: '/mnt/mlflow'
          }
        ] : []
        probes: isPlaceholder
          ? []
          : [
              {
                type: 'Readiness'
                httpGet: {
                  path: '/api/health'
                  port: 8765
                  scheme: 'HTTP'
                }
                initialDelaySeconds: 10
                periodSeconds: 10
              }
            ]
      }
    ]
    disableIngress: !enablePublicIngress
    enableTelemetry: false
    environmentResourceId: containerAppsEnvironmentResourceId
    ingressAllowInsecure: false
    ingressExternal: true
    ingressTargetPort: 8765
    location: location
    managedIdentities: {
      systemAssigned: true
    }
    name: containerAppName
    registries: isPlaceholder
      ? []
      : [
          {
            identity: 'system'
            server: registryLoginServer
          }
        ]
    scaleSettings: {
      maxReplicas: 1
      minReplicas: 0
    }
    secrets: isPlaceholder
      ? []
      : [
          {
            identity: 'system'
            keyVaultUrl: keyVaultSecretUri
            name: 'entra-client-secret'
          }
        ]
    tags: tags
    volumes: mlflowEnabled ? [
      {
        name: 'mlflow'
        storageType: 'AzureFile'
        storageName: mlflowStorageName
      }
    ] : []
  }
}

resource deployedApp 'Microsoft.App/containerApps@2026-01-01' existing = {
  name: containerAppName
}

output name string = app.outputs.name
output fqdn string = app.outputs.fqdn
output systemAssignedMIPrincipalId string = app.outputs.systemAssignedMIPrincipalId!
output resourceId string = deployedApp.id
