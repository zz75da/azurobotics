targetScope = 'resourceGroup'

// Management rights for the Azure ML workspace identity over its own dependencies only.
// On creation Azure ML grants its identity Azure AI Administrator on the whole resource group,
// which would reach the app's storage account and Key Vault. That assignment is removed and
// replaced by Contributor on the workspace's own Key Vault, storage account, Application Insights
// and the workspace itself, which is what workspace updates need (for example writing access
// policies on its vault). Deploy after infra/azureml.bicep has created the workspace; it is a
// separate template because the workspace update itself depends on these rights.

param workspaceName string
param storageAccountName string
param keyVaultName string
param applicationInsightsName string

var contributorRoleId = 'b24988ac-6180-42a0-ab88-20f7382dd24c'

resource workspace 'Microsoft.MachineLearningServices/workspaces@2025-09-01' existing = {
  name: workspaceName
}

resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' existing = {
  name: storageAccountName
}

resource keyVault 'Microsoft.KeyVault/vaults@2026-05-15' existing = {
  name: keyVaultName
}

resource appInsights 'Microsoft.Insights/components@2020-02-02' existing = {
  name: applicationInsightsName
}

resource onWorkspace 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(workspace.id, workspace.id, contributorRoleId)
  scope: workspace
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', contributorRoleId)
    principalId: workspace.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

resource onStorage 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(storage.id, workspace.id, contributorRoleId)
  scope: storage
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', contributorRoleId)
    principalId: workspace.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

resource onKeyVault 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(keyVault.id, workspace.id, contributorRoleId)
  scope: keyVault
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', contributorRoleId)
    principalId: workspace.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

resource onAppInsights 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(appInsights.id, workspace.id, contributorRoleId)
  scope: appInsights
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', contributorRoleId)
    principalId: workspace.identity.principalId
    principalType: 'ServicePrincipal'
  }
}
