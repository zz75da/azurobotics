param location string
param name string
param tags object
param workspaceResourceId string

// The workspace disables shared-key ingestion (AVM default), so logs go through
// Azure Monitor diagnostic settings, which authenticate with Entra instead.
resource environment 'Microsoft.App/managedEnvironments@2026-07-01' = {
  name: name
  location: location
  tags: tags
  properties: {
    appLogsConfiguration: {
      destination: 'azure-monitor'
    }
  }
}

resource logsToWorkspace 'Microsoft.Insights/diagnosticSettings@2021-05-01-preview' = {
  name: 'logs-to-workspace'
  scope: environment
  properties: {
    workspaceId: workspaceResourceId
    logAnalyticsDestinationType: 'Dedicated'
    logs: [
      {
        categoryGroup: 'allLogs'
        enabled: true
      }
    ]
  }
}

output resourceId string = environment.id
output defaultDomain string = environment.properties.defaultDomain
