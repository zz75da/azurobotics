targetScope = 'resourceGroup'

param location string = resourceGroup().location
param containerAppsEnvironmentName string
param mlflowAppName string
param registryName string
param keyVaultName string
param entraClientId string
param tenantId string
param identityName string
param storageAccountName string
param shareName string = 'mlflow'
param shareQuotaGiB int = 5
param mlflowImage string = ''
param tags object = {}

var acrPullRoleId = '7f951dda-4ed3-4680-a7ca-43fe172d538d'
var keyVaultSecretsUserRoleId = '4633458b-17de-408a-b874-0445c86b69e6'
var envStorageName = 'mlflowshare'

resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: storageAccountName
  location: location
  tags: tags
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    minimumTlsVersion: 'TLS1_2'
    supportsHttpsTrafficOnly: true
    allowBlobPublicAccess: false
    // Container Apps Azure Files mounts authenticate with the account key.
    allowSharedKeyAccess: true
  }
}

resource fileService 'Microsoft.Storage/storageAccounts/fileServices@2023-05-01' = {
  parent: storage
  name: 'default'
}

resource share 'Microsoft.Storage/storageAccounts/fileServices/shares@2023-05-01' = {
  parent: fileService
  name: shareName
  properties: {
    shareQuota: shareQuotaGiB
    accessTier: 'TransactionOptimized'
  }
}

resource containerEnv 'Microsoft.App/managedEnvironments@2026-07-01' existing = {
  name: containerAppsEnvironmentName
}

resource envStorage 'Microsoft.App/managedEnvironments/storages@2026-07-01' = {
  parent: containerEnv
  name: envStorageName
  dependsOn: [
    share
  ]
  properties: {
    azureFile: {
      accountName: storage.name
      accountKey: storage.listKeys().keys[0].value
      shareName: shareName
      accessMode: 'ReadWrite'
    }
  }
}

resource identity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: identityName
  location: location
  tags: tags
}

resource registry 'Microsoft.ContainerRegistry/registries@2025-11-01' existing = {
  name: registryName
}

resource keyVault 'Microsoft.KeyVault/vaults@2026-05-15' existing = {
  name: keyVaultName
}

resource acrPull 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(registry.id, identity.id, acrPullRoleId)
  scope: registry
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', acrPullRoleId)
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

resource secretsUser 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(keyVault.id, identity.id, keyVaultSecretsUserRoleId)
  scope: keyVault
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', keyVaultSecretsUserRoleId)
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

var mlflowHost = '${mlflowAppName}.${containerEnv.properties.defaultDomain}'

// Deployed in a second run (mlflowImage set) once the role assignments have propagated.
resource mlflow 'Microsoft.App/containerApps@2026-01-01' = if (!empty(mlflowImage)) {
  name: mlflowAppName
  location: location
  tags: tags
  dependsOn: [
    envStorage
  ]
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
        targetPort: 5000
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
          name: 'mlflow'
          image: mlflowImage
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
              httpGet: { path: '/health', port: 5000 }
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

resource mlflowAuth 'Microsoft.App/containerApps/authConfigs@2026-01-01' = if (!empty(mlflowImage)) {
  parent: mlflow
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

output mlflowUrl string = 'https://${mlflowHost}'
output envStorageName string = envStorageName
