targetScope = 'resourceGroup'

// Azure Machine Learning workspace for the training pipeline and model registry.
// The workspace gets its own storage account and Key Vault, so it has no access to the app's
// storage (and its MLflow share) or the app's Key Vault. It reuses the container registry, with
// AcrPush/AcrPull through its managed identity, and the Log Analytics workspace through a new
// workspace-based Application Insights. Training runs on serverless compute, so nothing here
// bills while idle. Role assignments need Owner or User Access Administrator, so this template is
// deployed by a person, not by the GitHub Actions identity.

param location string = resourceGroup().location
param workspaceName string
param storageAccountName string
param keyVaultName string
param applicationInsightsName string
param registryName string
param logAnalyticsName string
param containerAppName string

@description('Object id of the GitHub Actions service principal that submits pipelines.')
param githubPrincipalId string

@description('Object id of a person who also submits pipelines from a workstation; empty to skip.')
param operatorPrincipalId string = ''

param tags object = {}

// Built-in role definition ids.
var roles = {
  acrPush: '8311e382-0749-4cb8-b61a-899b2cf8f0dc'
  acrPull: '7f951dda-4ed3-4680-a7ca-43fe172d538d'
  keyVaultAdministrator: '00482a5a-887f-4fb3-b363-3b7fe8e74483'
  storageBlobDataContributor: 'ba92f5b4-2d11-453d-a403-e96b0029c9fe'
  storageBlobDataReader: '2a2b9908-6ea1-4ae2-8e65-a410df84e7d1'
  storageFileDataPrivilegedContributor: '69566ab7-960f-475b-8e7c-b3118f30c6bd'
  azureMlDataScientist: 'f6c7c914-8db3-469d-8ca1-694a8f32e121'
}

resource registry 'Microsoft.ContainerRegistry/registries@2025-11-01' existing = {
  name: registryName
}

resource logWorkspace 'Microsoft.OperationalInsights/workspaces@2026-03-01' existing = {
  name: logAnalyticsName
}

resource containerApp 'Microsoft.App/containerApps@2026-01-01' existing = {
  name: containerAppName
}

resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: storageAccountName
  location: location
  tags: tags
  kind: 'StorageV2'
  sku: {
    name: 'Standard_LRS'
  }
  properties: {
    minimumTlsVersion: 'TLS1_2'
    supportsHttpsTrafficOnly: true
    allowBlobPublicAccess: false
    isHnsEnabled: false
  }
}

resource keyVault 'Microsoft.KeyVault/vaults@2026-05-15' = {
  name: keyVaultName
  location: location
  tags: tags
  properties: {
    tenantId: subscription().tenantId
    sku: {
      family: 'A'
      name: 'standard'
    }
    enableRbacAuthorization: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 7
    // No purge protection, so the teardown script can remove the vault completely.
    publicNetworkAccess: 'Enabled'
  }
}

resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: applicationInsightsName
  location: location
  tags: tags
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logWorkspace.id
    IngestionMode: 'LogAnalytics'
  }
}

resource workspace 'Microsoft.MachineLearningServices/workspaces@2025-09-01' = {
  name: workspaceName
  location: location
  tags: tags
  identity: {
    type: 'SystemAssigned'
  }
  sku: {
    name: 'Basic'
    tier: 'Basic'
  }
  properties: {
    friendlyName: 'Robotics anomaly detection (personal project)'
    storageAccount: storage.id
    keyVault: keyVault.id
    containerRegistry: registry.id
    applicationInsights: appInsights.id
    publicNetworkAccess: 'Enabled'
    // Datastores use Entra identities instead of the storage account key.
    systemDatastoresAuthMode: 'identity'
  }
}

// Workspace identity: what Azure ML needs on its own storage and vault, and on the shared registry.
resource wsKeyVault 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(keyVault.id, workspace.id, roles.keyVaultAdministrator)
  scope: keyVault
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.keyVaultAdministrator)
    principalId: workspace.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

resource wsBlob 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(storage.id, workspace.id, roles.storageBlobDataContributor)
  scope: storage
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.storageBlobDataContributor)
    principalId: workspace.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

resource wsFile 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(storage.id, workspace.id, roles.storageFileDataPrivilegedContributor)
  scope: storage
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.storageFileDataPrivilegedContributor)
    principalId: workspace.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

resource wsAcrPush 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(registry.id, workspace.id, roles.acrPush)
  scope: registry
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.acrPush)
    principalId: workspace.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

resource wsAcrPull 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(registry.id, workspace.id, roles.acrPull)
  scope: registry
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.acrPull)
    principalId: workspace.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

// GitHub Actions: submit pipelines, and upload code and data with its own identity
// (identity-based datastores).
resource githubWorkspace 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(workspace.id, githubPrincipalId, roles.azureMlDataScientist)
  scope: workspace
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.azureMlDataScientist)
    principalId: githubPrincipalId
    principalType: 'ServicePrincipal'
  }
}

resource githubBlob 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(storage.id, githubPrincipalId, roles.storageBlobDataContributor)
  scope: storage
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.storageBlobDataContributor)
    principalId: githubPrincipalId
    principalType: 'ServicePrincipal'
  }
}

resource operatorWorkspace 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(operatorPrincipalId)) {
  name: guid(workspace.id, operatorPrincipalId, roles.azureMlDataScientist)
  scope: workspace
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.azureMlDataScientist)
    principalId: operatorPrincipalId
    principalType: 'User'
  }
}

resource operatorBlob 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(operatorPrincipalId)) {
  name: guid(storage.id, operatorPrincipalId, roles.storageBlobDataContributor)
  scope: storage
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.storageBlobDataContributor)
    principalId: operatorPrincipalId
    principalType: 'User'
  }
}

// The app reads the latest registered model at startup.
resource appWorkspace 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(workspace.id, containerApp.id, roles.azureMlDataScientist)
  scope: workspace
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.azureMlDataScientist)
    principalId: containerApp.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

resource appBlob 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(storage.id, containerApp.id, roles.storageBlobDataReader)
  scope: storage
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.storageBlobDataReader)
    principalId: containerApp.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

output workspaceName string = workspace.name
output mlflowTrackingUri string = workspace.properties.mlFlowTrackingUri
