#!/bin/bash
# AWS deployment script for HistoCore

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

ENVIRONMENT=${1:-"production"}
AWS_REGION=${2:-"us-west-2"}

echo -e "${GREEN}Deploying HistoCore to AWS EKS...${NC}"
echo "Environment: $ENVIRONMENT"
echo "Region: $AWS_REGION"

# Check tools
for tool in terraform aws kubectl; do
    if ! command -v $tool &> /dev/null; then
        echo -e "${RED}$tool not found${NC}"
        exit 1
    fi
done

# Terraform init
cd terraform
echo -e "${GREEN}Initializing Terraform...${NC}"
terraform init

# Plan
echo -e "${GREEN}Planning infrastructure...${NC}"
terraform plan -var="environment=$ENVIRONMENT" -var="aws_region=$AWS_REGION"

# Apply
echo -e "${YELLOW}Apply infrastructure? (y/N)${NC}"
read -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    terraform apply -var="environment=$ENVIRONMENT" -var="aws_region=$AWS_REGION" -auto-approve
else
    echo "Deployment cancelled"
    exit 0
fi

# Get outputs
CLUSTER_NAME=$(terraform output -raw cluster_name)
REDIS_ENDPOINT=$(terraform output -raw redis_endpoint)
S3_BUCKET=$(terraform output -raw s3_bucket_name)

# Configure kubectl
echo -e "${GREEN}Configuring kubectl...${NC}"
aws eks update-kubeconfig --region $AWS_REGION --name $CLUSTER_NAME

# Install NVIDIA device plugin
echo -e "${GREEN}Installing NVIDIA device plugin...${NC}"
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.1/nvidia-device-plugin.yml

# Deploy HistoCore
cd ../../../k8s
echo -e "${GREEN}Deploying HistoCore to EKS...${NC}"

# Update ConfigMap with AWS resources
kubectl create configmap histocore-config \
    --from-literal=REDIS_URL="redis://$REDIS_ENDPOINT:6379" \
    --from-literal=S3_BUCKET="$S3_BUCKET" \
    --from-literal=AWS_REGION="$AWS_REGION" \
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

# Get ALB endpoint
ALB_ENDPOINT=$(kubectl get ingress histocore-ingress -n histocore -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')

echo -e "${GREEN}Deployment complete!${NC}"
echo "Cluster: $CLUSTER_NAME"
echo "Redis: $REDIS_ENDPOINT"
echo "S3: $S3_BUCKET"
echo "ALB: $ALB_ENDPOINT"
echo
echo "Access via:"
echo "https://$ALB_ENDPOINT"
echo "kubectl port-forward -n histocore svc/histocore-streaming 8000:8000"