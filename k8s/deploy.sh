#!/bin/bash
# Kubernetes deployment script for HistoCore streaming

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Config
NAMESPACE="histocore"
KUBECTL_CMD="kubectl"

echo -e "${GREEN}Deploying HistoCore to Kubernetes...${NC}"

# Check kubectl
if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}kubectl not found. Install kubectl first.${NC}"
    exit 1
fi

# Check cluster connection
if ! kubectl cluster-info &> /dev/null; then
    echo -e "${RED}Cannot connect to Kubernetes cluster.${NC}"
    exit 1
fi

# Check GPU nodes
GPU_NODES=$(kubectl get nodes -l accelerator=nvidia-tesla-k80 --no-headers 2>/dev/null | wc -l)
if [ "$GPU_NODES" -eq 0 ]; then
    echo -e "${YELLOW}Warning: No GPU nodes found. Update node selector in streaming.yaml${NC}"
fi

# Deploy in order
echo -e "${GREEN}Creating namespace...${NC}"
kubectl apply -f namespace.yaml

echo -e "${GREEN}Creating ConfigMaps and Secrets...${NC}"
kubectl apply -f configmap.yaml
kubectl apply -f secret.yaml
kubectl apply -f prometheus-config.yaml
kubectl apply -f grafana-config.yaml

echo -e "${GREEN}Deploying Redis...${NC}"
kubectl apply -f redis.yaml

echo -e "${GREEN}Deploying monitoring...${NC}"
kubectl apply -f monitoring.yaml

echo -e "${GREEN}Deploying streaming service...${NC}"
kubectl apply -f streaming.yaml

echo -e "${GREEN}Creating ingress...${NC}"
kubectl apply -f ingress.yaml

echo -e "${GREEN}Setting up autoscaling...${NC}"
kubectl apply -f hpa.yaml

# Wait for deployments
echo -e "${GREEN}Waiting for deployments...${NC}"
kubectl wait --for=condition=available --timeout=300s deployment/histocore-redis -n $NAMESPACE
kubectl wait --for=condition=available --timeout=300s deployment/histocore-streaming -n $NAMESPACE
kubectl wait --for=condition=available --timeout=300s deployment/prometheus -n $NAMESPACE
kubectl wait --for=condition=available --timeout=300s deployment/grafana -n $NAMESPACE

# Show status
echo -e "${GREEN}Deployment complete!${NC}"
echo
echo "Services:"
kubectl get svc -n $NAMESPACE

echo
echo "Pods:"
kubectl get pods -n $NAMESPACE

echo
echo "Access URLs (update /etc/hosts or DNS):"
echo "Dashboard: https://histocore.example.com"
echo "API: https://api.histocore.example.com"
echo "Grafana: http://grafana-service-ip:3000 (admin/histocore123)"
echo "Prometheus: http://prometheus-service-ip:9090"

echo
echo "Port forward for local access:"
echo "kubectl port-forward -n $NAMESPACE svc/histocore-streaming 8000:8000"
echo "kubectl port-forward -n $NAMESPACE svc/grafana 3000:3000"