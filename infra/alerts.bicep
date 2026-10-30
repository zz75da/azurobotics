targetScope = 'resourceGroup'

// Azure Monitor alerts on the health of the main app. Metric-based, so they work behind Entra
// and with scale-to-zero (zero replicas is normal here, so the replica count is not alerted on).

param location string = 'global'
param containerAppName string
param tags object = {}

var ownerRoleId = '8e3af657-a8ff-443c-a75c-2fe8c4bcb635'

resource app 'Microsoft.App/containerApps@2026-01-01' existing = {
  name: containerAppName
}

resource actionGroup 'Microsoft.Insights/actionGroups@2023-01-01' = {
  name: '${containerAppName}-alerts'
  location: 'global'
  tags: tags
  properties: {
    enabled: true
    groupShortName: 'robotics'
    // Mails the subscription owners, so no personal address lives in the template.
    armRoleReceivers: [
      {
        name: 'owner'
        roleId: ownerRoleId
        useCommonAlertSchema: true
      }
    ]
  }
}

resource serverErrors 'Microsoft.Insights/metricAlerts@2018-03-01' = {
  name: '${containerAppName}-5xx'
  location: location
  tags: tags
  properties: {
    description: 'The app answered more than 5 HTTP 5xx in 5 minutes.'
    severity: 1
    enabled: true
    scopes: [
      app.id
    ]
    evaluationFrequency: 'PT1M'
    windowSize: 'PT5M'
    criteria: {
      'odata.type': 'Microsoft.Azure.Monitor.SingleResourceMultipleMetricCriteria'
      allOf: [
        {
          name: 'server-errors'
          criterionType: 'StaticThresholdCriterion'
          metricName: 'Requests'
          metricNamespace: 'Microsoft.App/containerApps'
          operator: 'GreaterThan'
          threshold: 5
          timeAggregation: 'Total'
          dimensions: [
            {
              name: 'statusCodeCategory'
              operator: 'Include'
              values: [
                '5xx'
              ]
            }
          ]
        }
      ]
    }
    actions: [
      {
        actionGroupId: actionGroup.id
      }
    ]
  }
}

resource restarts 'Microsoft.Insights/metricAlerts@2018-03-01' = {
  name: '${containerAppName}-restarts'
  location: location
  tags: tags
  properties: {
    description: 'The container restarted at least 3 times in 15 minutes (crash loop).'
    severity: 1
    enabled: true
    scopes: [
      app.id
    ]
    evaluationFrequency: 'PT5M'
    windowSize: 'PT15M'
    criteria: {
      'odata.type': 'Microsoft.Azure.Monitor.SingleResourceMultipleMetricCriteria'
      allOf: [
        {
          name: 'restarts'
          criterionType: 'StaticThresholdCriterion'
          metricName: 'RestartCount'
          metricNamespace: 'Microsoft.App/containerApps'
          operator: 'GreaterThanOrEqual'
          threshold: 3
          timeAggregation: 'Maximum'
        }
      ]
    }
    actions: [
      {
        actionGroupId: actionGroup.id
      }
    ]
  }
}
