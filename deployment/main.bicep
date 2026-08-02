@description('The location to deploy resources to.')
param location string = resourceGroup().location

@description('The name of the Container App.')
param containerAppName string = 'mcp-kql-server'

@description('The name of the Container Registry.')
param containerRegistryName string

@description('The image to deploy (e.g. myregistry.azurecr.io/mcp-kql-server:latest).')
param containerImage string

@description('The Kusto Cluster URL.')
param kustoClusterUrl string

@description('The Kusto Database Name.')
param kustoDatabaseName string = 'DefaultDB'

resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: '${containerAppName}-logs'
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
  }
}

resource containerAppEnv 'Microsoft.App/managedEnvironments@2023-05-01' = {
  name: '${containerAppName}-env'
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
  }
}

resource managedIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: '${containerAppName}-id'
  location: location
}

resource containerRegistry 'Microsoft.ContainerRegistry/registries@2023-01-01-preview' existing = {
  name: containerRegistryName
}

resource containerApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: containerAppName
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${managedIdentity.id}': {}
    }
  }
  properties: {
    managedEnvironmentId: containerAppEnv.id
    configuration: {
      ingress: {
        external: false
        targetPort: 8000
        transport: 'auto'
      }
      registries: [
        {
          server: containerRegistry.properties.loginServer
          identity: managedIdentity.id
        }
      ]
      secrets: []
    }
    template: {
      containers: [
        {
          name: containerAppName
          image: containerImage
          env: [
            {
              name: 'KUSTO_CLUSTER_URL'
              value: kustoClusterUrl
            }
            {
              name: 'KUSTO_DATABASE'
              value: kustoDatabaseName
            }
            {
              name: 'AZURE_CLIENT_ID'
              value: managedIdentity.properties.clientId
            }
            {
              name: 'MCP_KQL_AUTH_MODE'
              value: 'managed-identity'
            }
            {
              name: 'MCP_KQL_ALLOW_UNAUTHENTICATED_REMOTE_HTTP'
              value: 'true'
            }
            {
              name: 'FASTMCP_TRANSPORT'
              value: 'streamable-http'
            }
            {
              name: 'FASTMCP_HOST'
              value: '0.0.0.0'
            }
            {
              name: 'FASTMCP_PORT'
              value: '8000'
            }
            {
              name: 'MCP_KQL_LOG_LEVEL'
              value: 'INFO'
            }
            {
              name: 'MCP_KQL_CACHE_RESULTS'
              value: 'false'
            }
          ]
          resources: {
            cpu: json('0.5')
            memory: '1.0Gi'
          }
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 5
        rules: [
          {
            name: 'http-requests'
            custom: {
              type: 'http'
              metadata: {
                concurrentRequests: '10'
              }
            }
          }
        ]
      }
    }
  }
}

// Output the Managed Identity Principal ID for RBAC assignment
output managedIdentityPrincipalId string = managedIdentity.properties.principalId
output managedIdentityClientId string = managedIdentity.properties.clientId
output containerAppUrl string = containerApp.properties.configuration.ingress.fqdn
