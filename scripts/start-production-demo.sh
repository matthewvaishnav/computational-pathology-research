#!/bin/bash

# Start Production FL Demo
set -e

echo "🚀 Starting Production FL Demo"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null && ! command -v docker &> /dev/null; then
    log_warn "Docker Compose not found. Please install Docker and Docker Compose."
    exit 1
fi

# Use docker compose or docker-compose
DOCKER_COMPOSE="docker compose"
if ! docker compose version &> /dev/null; then
    DOCKER_COMPOSE="docker-compose"
fi

# Create necessary directories
log_info "Creating necessary directories..."
mkdir -p certs secrets data/hospital_a data/hospital_b data/hospital_c logs

# Generate certificates if they don't exist
if [ ! -f "certs/server-cert.pem" ]; then
    log_info "Generating TLS certificates..."
    
    # Generate CA private key
    openssl genrsa -out certs/ca-key.pem 4096
    
    # Generate CA certificate
    openssl req -new -x509 -days 365 -key certs/ca-key.pem -out certs/ca-cert.pem \
        -subj "/C=US/ST=CA/L=San Francisco/O=HistoCore FL/CN=HistoCore FL CA"
    
    # Generate server private key
    openssl genrsa -out certs/server-key.pem 4096
    
    # Generate server certificate signing request
    openssl req -new -key certs/server-key.pem -out certs/server.csr \
        -subj "/C=US/ST=CA/L=San Francisco/O=HistoCore FL/CN=localhost"
    
    # Generate server certificate
    openssl x509 -req -days 365 -in certs/server.csr -CA certs/ca-cert.pem \
        -CAkey certs/ca-key.pem -CAcreateserial -out certs/server-cert.pem \
        -extensions v3_req -extfile <(cat <<EOF
[v3_req]
keyUsage = keyEncipherment, dataEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
DNS.2 = coordinator
DNS.3 = client_hospital_a
DNS.4 = client_hospital_b
DNS.5 = client_hospital_c
IP.1 = 127.0.0.1
EOF
)
    
    # Clean up CSR
    rm certs/server.csr
    
    log_info "✅ TLS certificates generated"
fi

# Generate secrets if they don't exist
if [ ! -f "secrets/db_password.txt" ]; then
    log_info "Generating secrets..."
    
    openssl rand -base64 32 > secrets/db_password.txt
    openssl rand -base64 32 > secrets/redis_password.txt
    openssl rand -base64 16 > secrets/grafana_password.txt
    
    log_info "✅ Secrets generated"
fi

# Create sample data
log_info "Creating sample data..."
python3 -c "
import numpy as np
import os

# Create sample pathology data for each hospital
for hospital in ['hospital_a', 'hospital_b', 'hospital_c']:
    data_dir = f'data/{hospital}'
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate synthetic pathology image features (784 features like MNIST)
    np.random.seed(hash(hospital) % 2**32)
    
    # Different data distributions for each hospital
    if hospital == 'hospital_a':
        # Hospital A: More benign cases
        benign_samples = np.random.normal(0.3, 0.2, (7000, 784))
        malignant_samples = np.random.normal(0.7, 0.2, (3000, 784))
    elif hospital == 'hospital_b':
        # Hospital B: Balanced distribution
        benign_samples = np.random.normal(0.4, 0.25, (5000, 784))
        malignant_samples = np.random.normal(0.6, 0.25, (5000, 784))
    else:
        # Hospital C: More malignant cases
        benign_samples = np.random.normal(0.2, 0.15, (3000, 784))
        malignant_samples = np.random.normal(0.8, 0.15, (7000, 784))
    
    # Combine and save
    X = np.vstack([benign_samples, malignant_samples])
    y = np.hstack([np.zeros(len(benign_samples)), np.ones(len(malignant_samples))])
    
    np.save(f'{data_dir}/features.npy', X)
    np.save(f'{data_dir}/labels.npy', y)
    
    print(f'Generated {len(X)} samples for {hospital}')

print('Sample data created successfully!')
"

# Start the production stack
log_info "Starting production FL stack..."

$DOCKER_COMPOSE -f docker-compose.prod.yml up -d

# Wait for services to be ready
log_info "Waiting for services to start..."
sleep 30

# Check service health
log_info "Checking service health..."

services=("postgres" "redis" "coordinator")
for service in "${services[@]}"; do
    if $DOCKER_COMPOSE -f docker-compose.prod.yml ps "$service" | grep -q "Up"; then
        log_info "✅ $service is running"
    else
        log_warn "⚠️ $service may not be running properly"
    fi
done

# Display access information
echo ""
log_info "🎉 Production FL Demo Started!"
echo ""
log_info "Access URLs:"
log_info "- FL Coordinator API: https://localhost:8080"
log_info "- FL Coordinator Health: https://localhost:8080/health"
log_info "- FL Coordinator Metrics: https://localhost:8080/metrics"
log_info "- Hospital A Client: https://localhost:8081"
log_info "- Hospital B Client: https://localhost:8082"
log_info "- Hospital C Client: https://localhost:8083"
log_info "- Prometheus: http://localhost:9090"
log_info "- Grafana: http://localhost:3000 (admin/$(cat secrets/grafana_password.txt))"
log_info "- Kibana: http://localhost:5601"
echo ""
log_info "Database Credentials:"
log_info "- Database: federated_learning"
log_info "- Username: fluser"
log_info "- Password: $(cat secrets/db_password.txt)"
echo ""
log_info "To run tests: ./scripts/test-production.sh"
log_info "To stop demo: docker-compose -f docker-compose.prod.yml down"
echo ""
log_info "🏥 Ready for federated learning across 3 hospitals!"

# Show logs
log_info "Showing coordinator logs (Ctrl+C to exit):"
$DOCKER_COMPOSE -f docker-compose.prod.yml logs -f coordinator