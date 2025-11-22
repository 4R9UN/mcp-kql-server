#!/bin/bash
set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}🚀 Starting MCP KQL Server Deployment...${NC}"

# Default values
LOCATION="eastus"
DATABASE_NAME="DefaultDB"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --subscription) SUBSCRIPTION_ID="$2"; shift ;;
        --resource-group) RESOURCE_GROUP="$2"; shift ;;
        --cluster-url) CLUSTER_URL="$2"; shift ;;
        --location) LOCATION="$2"; shift ;;
        --database) DATABASE_NAME="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [[ -z "$SUBSCRIPTION_ID" || -z "$RESOURCE_GROUP" || -z "$CLUSTER_URL" ]]; then
    echo "Usage: ./deploy.sh --subscription <id> --resource-group <name> --cluster-url <url> [--location <region>]"
    exit 1
fi

# 1. Login
echo -e "${YELLOW}Step 1: Authenticating to Azure...${NC}"
az login --output none
az account set --subscription "$SUBSCRIPTION_ID"
echo -e "${GREEN}✅ Authenticated.${NC}"

# 2. Create Resource Group
echo -e "${YELLOW}Step 2: Creating Resource Group '$RESOURCE_GROUP' in '$LOCATION'...${NC}"
az group create --name "$RESOURCE_GROUP" --location "$LOCATION" --output none
echo -e "${GREEN}✅ Resource Group ready.${NC}"

# 3. Create ACR
ACR_NAME="mcpkql$RANDOM"
echo -e "${YELLOW}Step 3: Creating Azure Container Registry '$ACR_NAME'...${NC}"
az acr create --resource-group "$RESOURCE_GROUP" --name "$ACR_NAME" --sku Basic --admin-enabled true --output none
ACR_LOGIN_SERVER=$(az acr show --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" --query "loginServer" --output tsv)
echo -e "${GREEN}✅ ACR '$ACR_LOGIN_SERVER' ready.${NC}"

# 4. Build and Push
echo -e "${YELLOW}Step 4: Building and Pushing Docker Image...${NC}"
IMAGE_TAG="$ACR_LOGIN_SERVER/mcp-kql-server:latest"
# Build from parent directory
az acr build --registry "$ACR_NAME" --image "$IMAGE_TAG" --file Dockerfile ..
echo -e "${GREEN}✅ Image pushed to '$IMAGE_TAG'.${NC}"

# 5. Deploy Bicep
echo -e "${YELLOW}Step 5: Deploying Container App Infrastructure...${NC}"
DEPLOYMENT_NAME="mcp-deploy-$(date +%Y%m%d%H%M)"
OUTPUT_JSON=$(az deployment group create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$DEPLOYMENT_NAME" \
    --template-file main.bicep \
    --parameters containerRegistryName="$ACR_NAME" containerImage="$IMAGE_TAG" kustoClusterUrl="$CLUSTER_URL" kustoDatabaseName="$DATABASE_NAME" \
    --output json)

PRINCIPAL_ID=$(echo "$OUTPUT_JSON" | jq -r '.properties.outputs.managedIdentityPrincipalId.value')
APP_URL=$(echo "$OUTPUT_JSON" | jq -r '.properties.outputs.containerAppUrl.value')

echo -e "${GREEN}✅ Infrastructure deployed.${NC}"

# 6. Instructions
echo -e "${YELLOW}Step 6: Access Configuration${NC}"
echo "The Managed Identity Principal ID is: $PRINCIPAL_ID"
echo -e "${CYAN}⚠️  ACTION REQUIRED:${NC} Run the following KQL command on your cluster to grant access:"
echo "   .add database ['$DATABASE_NAME'] users ('aadapp=$PRINCIPAL_ID')"

echo -e "\n${CYAN}🎉 Deployment Complete!${NC}"
echo "--------------------------------------------------"
echo -e "Server URL: ${GREEN}https://$APP_URL${NC}"
echo "--------------------------------------------------"
