targetScope = 'resourceGroup'

// Monthly cost budget on this resource group. Budgets only notify; they never stop resources.
// The amount is in the billing account's currency.

@description('Monthly budget amount, in the billing currency.')
param amount int = 10

@description('Addresses notified when spend crosses a threshold. Passed at deploy time so no address lives in the repository.')
@minLength(1)
param contactEmails array

@description('First day of the budget period; must be the first of a month.')
param startDate string = utcNow('yyyy-MM-01')

resource budget 'Microsoft.Consumption/budgets@2023-11-01' = {
  name: 'budget-${resourceGroup().name}'
  properties: {
    category: 'Cost'
    amount: amount
    timeGrain: 'Monthly'
    timePeriod: {
      startDate: startDate
    }
    notifications: {
      actual50: {
        enabled: true
        operator: 'GreaterThanOrEqualTo'
        threshold: 50
        thresholdType: 'Actual'
        contactEmails: contactEmails
      }
      actual100: {
        enabled: true
        operator: 'GreaterThanOrEqualTo'
        threshold: 100
        thresholdType: 'Actual'
        contactEmails: contactEmails
      }
      forecast100: {
        enabled: true
        operator: 'GreaterThanOrEqualTo'
        threshold: 100
        thresholdType: 'Forecasted'
        contactEmails: contactEmails
      }
    }
  }
}

output budgetName string = budget.name
