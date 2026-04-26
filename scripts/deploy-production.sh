#!/bin/bash

# Production Deployment Script for Federated Learning System
set -e

echo "🚀 Starting Production FL System Deployment"

# Configuration
NAMESPACE="federated-learning"
DOCKER_REGISTRY="your-registry.com"
VERSION=${1:-"latest"}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed"
        exit 1
    fi
    
    # Check if docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "docker is not installed"
        exit 1
    fi
    
    # Check if helm is installed
    if ! command -v helm &> /dev/null; then
        log_error "helm is not installed"
        exit 1
    fi
    
    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    log_info "Prerequisites check passed ✅"
}

# Create namespace
create_namespace() {
    log_info "Creating namespace: $NAMESPACE"
    
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    
    # Label namespace for network policies
    kubectl label namespace $NAMESPACE name=$NAMESPACE --overwrite
}

# Generate TLS certificates
generate_certificates() {
    log_info "Generating TLS certificates..."
    
    # Create certificates directory
    mkdir -p ./certs
    
    # Generate CA private key
    openssl genrsa -out ./certs/ca-key.pem 4096
    
    # Generate CA certificate
    openssl req -new -x509 -days 365 -key ./certs/ca-key.pem -out ./certs/ca-cert.pem \
        -subj "/C=US/ST=CA/L=San Francisco/O=HistoCore FL/CN=HistoCore FL CA"
    
    # Generate server private key
    openssl genrsa -out ./certs/server-key.pem 4096
    
    # Generate server certificate signing request
    openssl req -new -key ./certs/server-key.pem -out ./certs/server.csr \
        -subj "/C=US/ST=CA/L=San Francisco/O=HistoCore FL/CN=fl-coordinator"
    
    # Generate server certificate
    openssl x509 -req -days 365 -in ./certs/server.csr -CA ./certs/ca-cert.pem \
        -CAkey ./certs/ca-key.pem -CAcreateserial -out ./certs/server-cert.pem \
        -extensions v3_req -extfile <(cat <<EOF
[v3_req]
keyUsage = keyEncipherment, dataEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[alt_names]
DNS.1 = fl-coordinator
DNS.2 = fl-coordinator.federated-learning.svc.cluster.local
DNS.3 = localhost
IP.1 = 127.0.0.1
EOF
)
    
    # Clean up CSR
    rm ./certs/server.csr
    
    log_info "TLS certificates generated ✅"
}

# Create secrets
create_secrets() {
    log_info "Creating Kubernetes secrets..."
    
    # Generate random passwords
    DB_PASSWORD=$(openssl rand -base64 32)
    REDIS_PASSWORD=$(openssl rand -base64 32)
    SECRET_KEY=$(openssl rand -base64 64)
    GRAFANA_PASSWORD=$(openssl rand -base64 16)
    
    # Create password files
    mkdir -p ./secrets
    echo -n "$DB_PASSWORD" > ./secrets/db_password.txt
    echo -n "$REDIS_PASSWORD" > ./secrets/redis_password.txt
    echo -n "$GRAFANA_PASSWORD" > ./secrets/grafana_password.txt
    
    # Create TLS secret
    kubectl create secret tls fl-tls-certs \
        --cert=./certs/server-cert.pem \
        --key=./certs/server-key.pem \
        --namespace=$NAMESPACE \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Create application secrets
    kubectl create secret generic fl-secrets \
        --from-literal=database-url="postgresql://fluser:$DB_PASSWORD@postgres:5432/federated_learning" \
        --from-literal=redis-url="redis://:$REDIS_PASSWORD@redis:6379" \
        --from-literal=secret-key="$SECRET_KEY" \
        --from-literal=sentry-dsn="${SENTRY_DSN:-}" \
        --namespace=$NAMESPACE \
        --dry-run=client -o yaml | kubectl apply -f -
    
    log_info "Secrets created ✅"
    log_warn "Database password: $DB_PASSWORD"
    log_warn "Redis password: $REDIS_PASSWORD"
    log_warn "Grafana password: $GRAFANA_PASSWORD"
    log_warn "Please save these passwords securely!"
}

# Build and push Docker images
build_images() {
    log_info "Building Docker images..."
    
    # Build coordinator image
    docker build -f docker/Dockerfile.coordinator -t $DOCKER_REGISTRY/fl-coordinator:$VERSION .
    docker push $DOCKER_REGISTRY/fl-coordinator:$VERSION
    
    # Build client image
    docker build -f docker/Dockerfile.client -t $DOCKER_REGISTRY/fl-client:$VERSION .
    docker push $DOCKER_REGISTRY/fl-client:$VERSION
    
    log_info "Docker images built and pushed ✅"
}

# Deploy infrastructure
deploy_infrastructure() {
    log_info "Deploying infrastructure components..."
    
    # Deploy PostgreSQL
    helm repo add bitnami https://charts.bitnami.com/bitnami
    helm repo update
    
    helm upgrade --install postgres bitnami/postgresql \
        --namespace $NAMESPACE \
        --set auth.postgresPassword="$(cat ./secrets/db_password.txt)" \
        --set auth.username=fluser \
        --set auth.password="$(cat ./secrets/db_password.txt)" \
        --set auth.database=federated_learning \
        --set primary.persistence.size=100Gi \
        --set primary.resources.requests.memory=2Gi \
        --set primary.resources.requests.cpu=1000m \
        --set primary.resources.limits.memory=4Gi \
        --set primary.resources.limits.cpu=2000m \
        --wait
    
    # Deploy Redis
    helm upgrade --install redis bitnami/redis \
        --namespace $NAMESPACE \
        --set auth.password="$(cat ./secrets/redis_password.txt)" \
        --set master.persistence.size=10Gi \
        --set master.resources.requests.memory=1Gi \
        --set master.resources.requests.cpu=500m \
        --wait
    
    # Deploy monitoring stack
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo add grafana https://grafana.github.io/helm-charts
    helm repo update
    
    # Deploy Prometheus
    helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
        --namespace $NAMESPACE \
        --set prometheus.prometheusSpec.retention=30d \
        --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=50Gi \
        --set grafana.adminPassword="$(cat ./secrets/grafana_password.txt)" \
        --wait
    
    log_info "Infrastructure deployed ✅"
}

# Deploy FL application
deploy_application() {
    log_info "Deploying FL application..."
    
    # Update image tags in deployment files
    sed -i "s|fl-coordinator:latest|$DOCKER_REGISTRY/fl-coordinator:$VERSION|g" kubernetes/coordinator-deployment.yaml
    sed -i "s|fl-client:latest|$DOCKER_REGISTRY/fl-client:$VERSION|g" kubernetes/client-deployment.yaml
    
    # Apply Kubernetes manifests
    kubectl apply -f kubernetes/ --namespace=$NAMESPACE
    
    # Wait for deployments to be ready
    kubectl rollout status deployment/fl-coordinator --namespace=$NAMESPACE --timeout=300s
    
    log_info "FL application deployed ✅"
}

# Run database migrations
run_migrations() {
    log_info "Running database migrations..."
    
    # Create migration job
    kubectl create job fl-migrate-$(date +%s) \
        --image=$DOCKER_REGISTRY/fl-coordinator:$VERSION \
        --namespace=$NAMESPACE \
        -- python -m alembic upgrade head
    
    # Wait for migration to complete
    kubectl wait --for=condition=complete job/fl-migrate-* --namespace=$NAMESPACE --timeout=300s
    
    log_info "Database migrations completed ✅"
}

# Verify deployment
verify_deployment() {
    log_info "Verifying deployment..."
    
    # Check pod status
    kubectl get pods --namespace=$NAMESPACE
    
    # Check services
    kubectl get services --namespace=$NAMESPACE
    
    # Test health endpoints
    COORDINATOR_IP=$(kubectl get service fl-coordinator --namespace=$NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    
    if [ -n "$COORDINATOR_IP" ]; then
        log_info "Testing health endpoint..."
        curl -k https://$COORDINATOR_IP/health || log_warn "Health check failed - service may still be starting"
    else
        log_warn "LoadBalancer IP not yet assigned"
    fi
    
    log_info "Deployment verification completed ✅"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary files..."
    rm -rf ./secrets
    rm -rf ./certs
}

# Main deployment flow
main() {
    log_info "🏥 Production Federated Learning System Deployment"
    log_info "Version: $VERSION"
    log_info "Namespace: $NAMESPACE"
    log_info "Registry: $DOCKER_REGISTRY"
    
    check_prerequisites
    create_namespace
    generate_certificates
    create_secrets
    build_images
    deploy_infrastructure
    deploy_application
    run_migrations
    verify_deployment
    
    log_info "🎉 Deployment completed successfully!"
    log_info ""
    log_info "Next steps:"
    log_info "1. Configure DNS to point to the LoadBalancer IP"
    log_info "2. Set up monitoring dashboards in Grafana"
    log_info "3. Configure backup schedules"
    log_info "4. Register FL clients"
    log_info ""
    log_info "Access URLs:"
    log_info "- FL Coordinator: https://$COORDINATOR_IP"
    log_info "- Grafana: http://$COORDINATOR_IP:3000 (admin/$(cat ./secrets/grafana_password.txt))"
    log_info "- Prometheus: http://$COORDINATOR_IP:9090"
    
    cleanup
}

# Handle script interruption
trap cleanup EXIT

# Run main function
main "$@"