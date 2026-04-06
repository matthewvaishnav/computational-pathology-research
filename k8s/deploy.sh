#!/bin/bash
# Kubernetes deployment script for Computational Pathology API

set -e

# Colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="pathology"
DEPLOYMENT_TYPE="${1:-cpu}"  # cpu or gpu
DRY_RUN="${2:-false}"

echo -e "${BLUE}Computational Pathology API - Kubernetes Deployment${NC}"
echo -e "${BLUE}====================================================${NC}"
echo ""

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}Error: kubectl not found${NC}"
    exit 1
fi

if ! kubectl cluster-info &> /dev/null; then
    echo -e "${RED}Error: Cannot connect to Kubernetes cluster${NC}"
    exit 1
fi

echo -e "${GREEN}✓ kubectl configured${NC}"
echo -e "${GREEN}✓ Cluster accessible${NC}"
echo ""

# Show cluster info
echo -e "${YELLOW}Cluster information:${NC}"
kubectl cluster-info | head -n 1
echo ""

# Confirm deployment
echo -e "${YELLOW}Deployment configuration:${NC}"
echo "  Namespace: ${NAMESPACE}"
echo "  Type: ${DEPLOYMENT_TYPE}"
echo "  Dry run: ${DRY_RUN}"
echo ""

if [ "$DRY_RUN" != "true" ]; then
    read -p "Continue with deployment? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Deployment cancelled"
        exit 0
    fi
fi

# Deployment steps
echo -e "${BLUE}Starting deployment...${NC}"
echo ""

# Step 1: Create namespace
echo -e "${YELLOW}Step 1: Creating namespace...${NC}"
if [ "$DRY_RUN" = "true" ]; then
    kubectl apply -f namespace.yaml --dry-run=client
else
    kubectl apply -f namespace.yaml
fi
echo -e "${GREEN}✓ Namespace created${NC}"
echo ""

# Step 2: Create ConfigMap
echo -e "${YELLOW}Step 2: Creating ConfigMap...${NC}"
if [ "$DRY_RUN" = "true" ]; then
    kubectl apply -f configmap.yaml --dry-run=client
else
    kubectl apply -f configmap.yaml
fi
echo -e "${GREEN}✓ ConfigMap created${NC}"
echo ""

# Step 3: Create Secrets
echo -e "${YELLOW}Step 3: Creating Secrets...${NC}"
echo -e "${RED}Warning: Update secret.yaml with real values before production!${NC}"
if [ "$DRY_RUN" = "true" ]; then
    kubectl apply -f secret.yaml --dry-run=client
else
    kubectl apply -f secret.yaml
fi
echo -e "${GREEN}✓ Secrets created${NC}"
echo ""

# Step 4: Create PVCs
echo -e "${YELLOW}Step 4: Creating Persistent Volume Claims...${NC}"
if [ "$DRY_RUN" = "true" ]; then
    kubectl apply -f pvc.yaml --dry-run=client
else
    kubectl apply -f pvc.yaml
    echo "Waiting for PVCs to be bound..."
    kubectl wait --for=condition=Bound pvc/pathology-models-pvc -n ${NAMESPACE} --timeout=60s || true
fi
echo -e "${GREEN}✓ PVCs created${NC}"
echo ""

# Step 5: Deploy application
echo -e "${YELLOW}Step 5: Deploying application...${NC}"
if [ "$DEPLOYMENT_TYPE" = "gpu" ]; then
    echo "Deploying GPU version..."
    if [ "$DRY_RUN" = "true" ]; then
        kubectl apply -f gpu-deployment.yaml --dry-run=client
    else
        kubectl apply -f gpu-deployment.yaml
    fi
else
    echo "Deploying CPU version..."
    if [ "$DRY_RUN" = "true" ]; then
        kubectl apply -f deployment.yaml --dry-run=client
    else
        kubectl apply -f deployment.yaml
    fi
fi
echo -e "${GREEN}✓ Deployment created${NC}"
echo ""

# Step 6: Create Service
echo -e "${YELLOW}Step 6: Creating Service...${NC}"
if [ "$DRY_RUN" = "true" ]; then
    kubectl apply -f service.yaml --dry-run=client
else
    kubectl apply -f service.yaml
fi
echo -e "${GREEN}✓ Service created${NC}"
echo ""

# Step 7: Create Ingress (optional)
if [ -f "ingress.yaml" ]; then
    echo -e "${YELLOW}Step 7: Creating Ingress...${NC}"
    echo -e "${RED}Note: Update ingress.yaml with your domain before production!${NC}"
    if [ "$DRY_RUN" = "true" ]; then
        kubectl apply -f ingress.yaml --dry-run=client
    else
        read -p "Create Ingress? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            kubectl apply -f ingress.yaml
            echo -e "${GREEN}✓ Ingress created${NC}"
        else
            echo "Skipping Ingress creation"
        fi
    fi
    echo ""
fi

# Step 8: Create HPA (optional)
if [ -f "hpa.yaml" ]; then
    echo -e "${YELLOW}Step 8: Creating Horizontal Pod Autoscaler...${NC}"
    if [ "$DRY_RUN" = "true" ]; then
        kubectl apply -f hpa.yaml --dry-run=client
    else
        read -p "Enable autoscaling? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            kubectl apply -f hpa.yaml
            echo -e "${GREEN}✓ HPA created${NC}"
        else
            echo "Skipping HPA creation"
        fi
    fi
    echo ""
fi

# Wait for deployment
if [ "$DRY_RUN" != "true" ]; then
    echo -e "${YELLOW}Waiting for deployment to be ready...${NC}"
    if [ "$DEPLOYMENT_TYPE" = "gpu" ]; then
        kubectl rollout status deployment/pathology-api-gpu -n ${NAMESPACE} --timeout=300s
    else
        kubectl rollout status deployment/pathology-api -n ${NAMESPACE} --timeout=300s
    fi
    echo -e "${GREEN}✓ Deployment ready${NC}"
    echo ""
fi

# Show deployment status
echo -e "${BLUE}Deployment Status${NC}"
echo -e "${BLUE}=================${NC}"
echo ""

if [ "$DRY_RUN" != "true" ]; then
    echo -e "${YELLOW}Pods:${NC}"
    kubectl get pods -n ${NAMESPACE}
    echo ""
    
    echo -e "${YELLOW}Services:${NC}"
    kubectl get svc -n ${NAMESPACE}
    echo ""
    
    echo -e "${YELLOW}Ingress:${NC}"
    kubectl get ingress -n ${NAMESPACE} 2>/dev/null || echo "No ingress configured"
    echo ""
fi

# Show access information
echo -e "${BLUE}Access Information${NC}"
echo -e "${BLUE}==================${NC}"
echo ""

if [ "$DRY_RUN" != "true" ]; then
    EXTERNAL_IP=$(kubectl get svc pathology-api -n ${NAMESPACE} -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")
    
    if [ "$EXTERNAL_IP" != "pending" ] && [ -n "$EXTERNAL_IP" ]; then
        echo -e "${GREEN}API URL: http://${EXTERNAL_IP}${NC}"
        echo -e "${GREEN}Health check: http://${EXTERNAL_IP}/health${NC}"
        echo -e "${GREEN}API docs: http://${EXTERNAL_IP}/docs${NC}"
    else
        echo -e "${YELLOW}External IP pending... Check with:${NC}"
        echo "  kubectl get svc pathology-api -n ${NAMESPACE}"
    fi
    echo ""
    
    echo -e "${YELLOW}Port forward (for testing):${NC}"
    echo "  kubectl port-forward svc/pathology-api 8000:8000 -n ${NAMESPACE}"
    echo "  curl http://localhost:8000/health"
    echo ""
fi

echo -e "${GREEN}Deployment complete!${NC}"
echo ""

echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Check pod logs: kubectl logs -f -l app=pathology-api -n ${NAMESPACE}"
echo "  2. Test API: curl http://<EXTERNAL-IP>/health"
echo "  3. Monitor: kubectl get pods -n ${NAMESPACE} --watch"
echo "  4. Scale: kubectl scale deployment pathology-api -n ${NAMESPACE} --replicas=5"
echo ""
