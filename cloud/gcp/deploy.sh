#!/bin/bash
# GCP deployment script for HistoCore

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

PROJECT_ID=${1}
ENVIRONMENT=${2:-"production"}
GCP_REGION=${3:-"us-central1"}

if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}Usage: $0 <PROJECT_ID> [ENVIRONMENT] [REGION]${NC}"
    echo "Example: $0 my-histocore-project production us-central1"
    exit 1
fi

echo -e "${GREEN}Deploying HistoCore to Google Cloud...${NC}"
echo "Project: $PROJECT_ID"
echo "Environment: $ENVIRONMENT"
echo "Region: $GCP_REGION"

# Check tools
for tool in terraform gcloud kubectl; do
    if ! command -v $tool &> /dev/null; then
        echo -e "${RED}$tool not found${NC}"
        exit 1
    fi
done

# GCP authentication check
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n1 &> /dev/null; then
    echo -e "${YELLOW}Please authenticate with Google Cloud first:${NC}"
    echo "gcloud auth login"
    echo "gcloud auth application-default login"
    exit 1
fi

# Set project
gcloud config set project $PROJECT_ID

# Generate PostgreSQL password
POSTGRES_PASSWORD=$(openssl rand -base64 32)

# Terraform init
cd terraform
echo -e "${GREEN}Initializing Terraform...${NC}"
terraform init

# Plan
echo -e "${GREEN}Planning infrastructure...${NC}"
terraform plan \
    -var="project_id=$PROJECT_ID" \
    -var="environment=$ENVIRONMENT" \
    -var="gcp_region=$GCP_REGION" \
    -var="postgres_password=$POSTGRES_PASSWORD"

# Apply
echo -e "${YELLOW}Apply infrastructure? (y/N)${NC}"
read -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    terraform apply \
        -var="project_id=$PROJECT_ID" \
        -var="environment=$ENVIRONMENT" \
        -var="gcp_region=$GCP_REGION" \
        -var="postgres_password=$POSTGRES_PASSWORD" \
        -auto-approve
else
    echo "Deployment cancelled"
    exit 0
fi

# Get outputs
CLUSTER_NAME=$(terraform output -raw gke_cluster_name)
REDIS_HOST=$(terraform output -raw redis_host)
REDIS_PORT=$(terraform output -raw redis_port)
REDIS_AUTH=$(terraform output -raw redis_auth_string)
STORAGE_BUCKET=$(terraform output -raw storage_bucket_name)
POSTGRES_CONNECTION=$(terraform output -raw postgres_connection_name)
LB_IP=$(terraform output -raw load_balancer_ip)

# Configure kubectl
echo -e "${GREEN}Configuring kubectl...${NC}"
gcloud container clusters get-credentials $CLUSTER_NAME --region $GCP_REGION --project $PROJECT_ID

# Install NVIDIA device plugin
echo -e "${GREEN}Installing NVIDIA device plugin...${NC}"
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml

# Deploy HistoCore
cd ../../../k8s
echo -e "${GREEN}Deploying HistoCore to GKE...${NC}"

# Update ConfigMap with GCP resources
kubectl create configmap histocore-config \
    --from-literal=REDIS_URL="redis://:$REDIS_AUTH@$REDIS_HOST:$REDIS_PORT" \
    --from-literal=GCS_BUCKET="$STORAGE_BUCKET" \
    --from-literal=POSTGRES_CONNECTION="$POSTGRES_CONNECTION" \
    --from-literal=POSTGRES_PASSWORD="$POSTGRES_PASSWORD" \
    --from-literal=GCP_PROJECT="$PROJECT_ID" \
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

echo -e "${GREEN}Deployment complete!${NC}"
echo "Project: $PROJECT_ID"
echo "Cluster: $CLUSTER_NAME"
echo "Redis: $REDIS_HOST:$REDIS_PORT"
echo "Storage: gs://$STORAGE_BUCKET"
echo "Load Balancer IP: $LB_IP"
echo
echo "Access via:"
echo "http://$LB_IP"
echo "kubectl port-forward -n histocore svc/histocore-streaming 8000:8000"
echo
echo "PostgreSQL password saved in Terraform state. Retrieve with:"
echo "terraform output -raw postgres_password"