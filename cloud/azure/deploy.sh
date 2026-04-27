#!/bin/bash
# Azure deployment script for HistoCore

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

ENVIRONMENT=${1:-"production"}
AZURE_REGION=${2:-"East US"}

echo -e "${GREEN}Deploying HistoCore to Azure AKS...${NC}"
echo "Environment: $ENVIRONMENT"
echo "Region: $AZURE_REGION"

# Check tools
for tool in terraform az kubectl; do
    if ! command -v $tool &> /dev/null; then
        echo -e "${RED}$tool not found${NC}"
        exit 1
    fi
done

# Azure login check
if ! az account show &> /dev/null; then
    echo -e "${YELLOW}Please login to Azure CLI first:${NC}"
    echo "az login"
    exit 1
fi

# Terraform init
cd terraform
echo -e "${GREEN}Initializing Terraform...${NC}"
terraform init

# Plan
echo -e "${GREEN}Planning infrastructure...${NC}"
terraform plan -var="environment=$ENVIRONMENT" -var="azure_region=$AZURE_REGION"

# Apply
echo -e "${YELLOW}Apply infrastructure? (y/N)${NC}"
read -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    terraform apply -var="environment=$ENVIRONMENT" -var="azure_region=$AZURE_REGION" -auto-approve
else
    echo "Deployment cancelled"
    exit 0
fi

# Get outputs
RESOURCE_GROUP=$(terraform output -raw resource_group_name)
CLUSTER_NAME=$(terraform output -raw aks_cluster_name)
REDIS_HOSTNAME=$(terraform output -raw redis_hostname)
REDIS_PORT=$(terraform output -raw redis_port)
STORAGE_ACCOUNT=$(terraform output -raw storage_account_name)
STORAGE_CONTAINER=$(terraform output -raw storage_container_name)
KEY_VAULT=$(terraform output -raw key_vault_name)

# Configure kubectl
echo -e "${GREEN}Configuring kubectl...${NC}"
az aks get-credentials --resource-group $RESOURCE_GROUP --name $CLUSTER_NAME --overwrite-existing

# Install NVIDIA device plugin
echo -e "${GREEN}Installing NVIDIA device plugin...${NC}"
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.1/nvidia-device-plugin.yml

# Deploy HistoCore
cd ../../../k8s
echo -e "${GREEN}Deploying HistoCore to AKS...${NC}"

# Update ConfigMap with Azure resources
kubectl create configmap histocore-config \
    --from-literal=REDIS_URL="redis://$REDIS_HOSTNAME:$REDIS_PORT" \
    --from-literal=AZURE_STORAGE_ACCOUNT="$STORAGE_ACCOUNT" \
    --from-literal=AZURE_STORAGE_CONTAINER="$STORAGE_CONTAINER" \
    --from-literal=AZURE_KEY_VAULT="$KEY_VAULT" \
    --dry-run=client -o yaml | kubectl apply -f -

# Deploy
kubectl apply -f namespace.yaml
kubectl apply -f secret.yaml
kubectl apply -f streaming.yaml
kubectl apply -f redis.yaml
kubectl apply -f monitoring.yaml
kubectl apply -f hpa.yaml
kubectl apply -f ingress.yaml

# Wait for deployment
echo -e "${GREEN}Waiting for deployment...${NC}"
kubectl wait --for=condition=available --timeout=600s deployment/histocore-streaming -n histocore

# Get load balancer IP
LB_IP=$(kubectl get service histocore-streaming -n histocore -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

echo -e "${GREEN}Deployment complete!${NC}"
echo "Resource Group: $RESOURCE_GROUP"
echo "Cluster: $CLUSTER_NAME"
echo "Redis: $REDIS_HOSTNAME:$REDIS_PORT"
echo "Storage: $STORAGE_ACCOUNT/$STORAGE_CONTAINER"
echo "Load Balancer IP: $LB_IP"
echo
echo "Access via:"
echo "http://$LB_IP"
echo "kubectl port-forward -n histocore svc/histocore-streaming 8000:8000"