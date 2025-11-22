<#
.SYNOPSIS
    Deploys the MCP KQL Server to Azure Container Apps.
.DESCRIPTION
    This script automates the deployment of the MCP KQL Server.
    It creates a Resource Group, Azure Container Registry, builds the Docker image,
    pushes it to ACR, and deploys the Container App using Bicep.
.PARAMETER SubscriptionId
    The Azure Subscription ID to deploy to.
.PARAMETER ResourceGroupName
    The name of the Resource Group to create/use.
.PARAMETER ClusterUrl
    The URL of your Azure Data Explorer (Kusto) cluster.
.PARAMETER Location
    The Azure region to deploy to (default: eastus).
.EXAMPLE
    .\deploy.ps1 -SubscriptionId "guid" -ResourceGroupName "mcp-prod" -ClusterUrl "https://mycluster.kusto.windows.net"
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$SubscriptionId,

    [Parameter(Mandatory=$true)]
    [string]$ResourceGroupName,

    [Parameter(Mandatory=$true)]
    [string]$ClusterUrl,

    [string]$Location = "eastus",
    
    [string]$DatabaseName = "DefaultDB"
)

$ErrorActionPreference = "Stop"

Write-Host "🚀 Starting MCP KQL Server Deployment..." -ForegroundColor Cyan

# 1. Login and Set Subscription
Write-Host "Step 1: Authenticating to Azure..." -ForegroundColor Yellow
az login --output none
az account set --subscription $SubscriptionId
Write-Host "✅ Authenticated." -ForegroundColor Green

# 2. Create Resource Group
Write-Host "Step 2: Creating Resource Group '$ResourceGroupName' in '$Location'..." -ForegroundColor Yellow
az group create --name $ResourceGroupName --location $Location --output none
Write-Host "✅ Resource Group ready." -ForegroundColor Green

# 3. Create Azure Container Registry (ACR)
$acrName = "mcpkql" + (Get-Random -Minimum 1000 -Maximum 9999)
Write-Host "Step 3: Creating Azure Container Registry '$acrName'..." -ForegroundColor Yellow
# Check if ACR exists in RG, if not create
$acrExists = az acr list --resource-group $ResourceGroupName --query "[?name=='$acrName']" --output tsv
if (-not $acrExists) {
    az acr create --resource-group $ResourceGroupName --name $acrName --sku Basic --admin-enabled true --output none
}
$acrLoginServer = az acr show --name $acrName --resource-group $ResourceGroupName --query "loginServer" --output tsv
Write-Host "✅ ACR '$acrLoginServer' ready." -ForegroundColor Green

# 4. Build and Push Docker Image
Write-Host "Step 4: Building and Pushing Docker Image..." -ForegroundColor Yellow
# We need to build from the root directory, so we assume script is run from deployment/ folder
# and we reference the parent directory context.
$imageTag = "$acrLoginServer/mcp-kql-server:latest"
az acr build --registry $acrName --image $imageTag --file Dockerfile ..
Write-Host "✅ Image pushed to '$imageTag'." -ForegroundColor Green

# 5. Deploy Infrastructure (Bicep)
Write-Host "Step 5: Deploying Container App Infrastructure..." -ForegroundColor Yellow
$deploymentName = "mcp-deploy-" + (Get-Date -Format "yyyyMMddHHmm")
$deployment = az deployment group create `
    --resource-group $ResourceGroupName `
    --name $deploymentName `
    --template-file main.bicep `
    --parameters containerRegistryName=$acrName containerImage=$imageTag kustoClusterUrl=$ClusterUrl kustoDatabaseName=$DatabaseName `
    --output json | ConvertFrom-Json

$principalId = $deployment.properties.outputs.managedIdentityPrincipalId.value
$appUrl = $deployment.properties.outputs.containerAppUrl.value

Write-Host "✅ Infrastructure deployed." -ForegroundColor Green

# 6. Assign RBAC Roles (Optional/Best Effort)
Write-Host "Step 6: Attempting to assign 'Database User' role to Managed Identity..." -ForegroundColor Yellow
Write-Host "   Principal ID: $principalId"
Write-Host "   Note: This requires you to have 'User Access Administrator' or 'Owner' on the Kusto Cluster."

try {
    # Note: This is a simplified assignment. In production, you might scope this to the specific database.
    # We try to assign 'Viewer' on the cluster/database scope if possible via ARM, 
    # but Kusto roles are often managed inside Kusto (.add database user).
    # Here we attempt a generic Reader role on the resource for demonstration, 
    # but REAL Kusto access needs the Kusto-specific command or role assignment.
    
    # Using Azure CLI to execute Kusto command if current user has permissions
    $kustoQuery = ".add database ['$DatabaseName'] users ('aadapp=$principalId')"
    # This part is tricky to automate without knowing the user's permissions on Kusto.
    # We will output instructions instead to be safe.
    Write-Host "⚠️  AUTOMATION SKIPPED: Please run the following KQL command on your cluster to grant access:" -ForegroundColor Magenta
    Write-Host "   $kustoQuery" -ForegroundColor White
}
catch {
    Write-Warning "Failed to assign roles automatically. Please configure Kusto permissions manually."
}

Write-Host "`n🎉 Deployment Complete!" -ForegroundColor Cyan
Write-Host "--------------------------------------------------"
Write-Host "Server URL: https://$appUrl" -ForegroundColor Green
Write-Host "--------------------------------------------------"
