targetScope = 'resourceGroup'

// Evidently UI over the drift snapshots the app writes to the MLflow Azure Files share
// (EVIDENTLY_WORKSPACE=/mnt/mlflow/evidently). Reuses the MLflow identity and environment
// storage from mlflow.bicep, so it needs no new role assignments or storage account.

param location string = resourceGroup().location
param containerAppsEnvironmentName string
param evidentlyAppName string
param registryName string
param keyVaultName string
param entraClientId string
param tenantId string
param identityName string
param envStorageName string = 'mlflowshare'
param appImage string
param tags object = {}

resource containerEnv 'Microsoft.App/managedEnvironments@2026-07-01' existing = {
  name: containerAppsEnvironmentName
}

resource identity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' existing = {
  name: identityName
}

resource registry 'Microsoft.ContainerRegistry/registries@2025-11-01' existing = {
  name: registryName
}

resource keyVault 'Microsoft.KeyVault/vaults@2026-05-15' existing = {
  name: keyVaultName
}

var evidentlyHost = '${evidentlyAppName}.${containerEnv.properties.defaultDomain}'

resource evidently 'Microsoft.App/containerApps@2026-01-01' = {
  name: evidentlyAppName
  location: location
  tags: tags
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${identity.id}': {}
    }
  }
  properties: {
    managedEnvironmentId: containerEnv.id
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: 8000
        allowInsecure: false
      }
      registries: [
        {
          server: registry.properties.loginServer
          identity: identity.id
        }
      ]
      secrets: [
        {
          name: 'entra-client-secret'
          keyVaultUrl: '${keyVault.properties.vaultUri}secrets/entra-client-secret'
          identity: identity.id
        }
      ]
    }
    template: {
      volumes: [
        {
          name: 'mlflow'
          storageType: 'AzureFile'
          storageName: envStorageName
        }
      ]
      containers: [
        {
          name: 'evidently'
          // The app image already ships Evidently at the same version that writes the snapshots.
          image: appImage
          command: [
            'evidently'
            'ui'
            '--workspace'
            '/mnt/mlflow/evidently'
            '--host'
            '0.0.0.0'
            '--port'
            '8000'
          ]
          resources: {
            cpu: json('0.5')
            memory: '1Gi'
          }
          volumeMounts: [
            {
              volumeName: 'mlflow'
              mountPath: '/mnt/mlflow'
            }
          ]
          probes: [
            {
              type: 'Startup'
              tcpSocket: { port: 8000 }
              initialDelaySeconds: 5
              periodSeconds: 5
              failureThreshold: 24
            }
          ]
        }
      ]
      scale: {
        minReplicas: 0
        maxReplicas: 1
      }
    }
  }
}

resource evidentlyAuth 'Microsoft.App/containerApps/authConfigs@2026-01-01' = {
  parent: evidently
  name: 'current'
  properties: {
    platform: {
      enabled: true
    }
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
  }
}

output evidentlyUrl string = 'https://${evidentlyHost}'
