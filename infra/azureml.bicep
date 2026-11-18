targetScope = 'resourceGroup'

// Azure Machine Learning workspace for the training pipeline and model registry.
// It reuses the existing Key Vault, storage account, container registry and Log Analytics, and
// adds only the workspace and a workspace-based Application Insights. Training runs on serverless
// compute, so nothing here bills while idle. Role assignments need Owner or User Access
// Administrator, so this template is deployed by a person, not by the GitHub Actions identity.

param location string = resourceGroup().location
param workspaceName string
param applicationInsightsName string
param storageAccountName string
param keyVaultName string
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
  contributor: 'b24988ac-6180-42a0-ab88-20f7382dd24c'
  keyVaultAdministrator: '00482a5a-887f-4fb3-b363-3b7fe8e74483'
  storageBlobDataContributor: 'ba92f5b4-2d11-453d-a403-e96b0029c9fe'
  storageBlobDataReader: '2a2b9908-6ea1-4ae2-8e65-a410df84e7d1'
  storageFileDataPrivilegedContributor: '69566ab7-960f-475b-8e7c-b3118f30c6bd'
  azureMlDataScientist: 'f6c7c914-8db3-469d-8ca1-694a8f32e121'
}

resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' existing = {
  name: storageAccountName
}

resource keyVault 'Microsoft.KeyVault/vaults@2026-05-15' existing = {
  name: keyVaultName
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

// Workspace identity: what Azure ML needs on the resources it reuses.
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

resource wsRegistry 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(registry.id, workspace.id, roles.contributor)
  scope: registry
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.contributor)
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
