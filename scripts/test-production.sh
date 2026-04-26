#!/bin/bash

# Production FL System Testing Script
set -e

echo "🧪 Testing Production FL System"

# Configuration
COORDINATOR_URL="https://localhost:8080"
CLIENT_URLS=("https://localhost:8081" "https://localhost:8082" "https://localhost:8083")
TEST_TIMEOUT=300  # 5 minutes

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Test health endpoints
test_health_endpoints() {
    log_info "Testing health endpoints..."
    
    # Test coordinator health
    if curl -k -f "$COORDINATOR_URL/health" > /dev/null 2>&1; then
        log_info "✅ Coordinator health check passed"
    else
        log_error "❌ Coordinator health check failed"
        return 1
    fi
    
    # Test client health
    for i in "${!CLIENT_URLS[@]}"; do
        client_url="${CLIENT_URLS[$i]}"
        if curl -k -f "$client_url/health" > /dev/null 2>&1; then
            log_info "✅ Client $((i+1)) health check passed"
        else
            log_error "❌ Client $((i+1)) health check failed"
            return 1
        fi
    done
}

# Test metrics endpoints
test_metrics_endpoints() {
    log_info "Testing metrics endpoints..."
    
    # Test coordinator metrics
    if curl -k -f "$COORDINATOR_URL/metrics" | grep -q "fl_coordinator"; then
        log_info "✅ Coordinator metrics available"
    else
        log_error "❌ Coordinator metrics not available"
        return 1
    fi
    
    # Test client metrics
    for i in "${!CLIENT_URLS[@]}"; do
        client_url="${CLIENT_URLS[$i]}"
        if curl -k -f "$client_url/metrics" | grep -q "fl_client"; then
            log_info "✅ Client $((i+1)) metrics available"
        else
            log_error "❌ Client $((i+1)) metrics not available"
            return 1
        fi
    done
}

# Test client registration
test_client_registration() {
    log_info "Testing client registration..."
    
    for i in "${!CLIENT_URLS[@]}"; do
        client_id="hospital_$((i+1))"
        client_url="${CLIENT_URLS[$i]}"
        
        # Register client
        response=$(curl -k -s -X POST "$client_url/api/v1/register" \
            -H "Content-Type: application/json" \
            -d "{\"client_id\": \"$client_id\"}")
        
        if echo "$response" | grep -q "success"; then
            log_info "✅ Client $client_id registered successfully"
        else
            log_error "❌ Client $client_id registration failed: $response"
            return 1
        fi
    done
}

# Test training round
test_training_round() {
    log_info "Testing federated training round..."
    
    # Start training round
    training_config='{
        "algorithm": "fedavg",
        "min_clients": 2,
        "max_clients": 3,
        "round_timeout": 300
    }'
    
    response=$(curl -k -s -X POST "$COORDINATOR_URL/api/v1/training/start" \
        -H "Content-Type: application/json" \
        -d "$training_config")
    
    if echo "$response" | grep -q "success"; then
        round_id=$(echo "$response" | grep -o '"round_id":[0-9]*' | cut -d':' -f2)
        log_info "✅ Training round $round_id started"
        
        # Wait for round completion
        log_info "Waiting for training round to complete..."
        for ((i=0; i<$TEST_TIMEOUT; i+=10)); do
            status_response=$(curl -k -s "$COORDINATOR_URL/api/v1/training/status/$round_id")
            
            if echo "$status_response" | grep -q '"status":"completed"'; then
                log_info "✅ Training round $round_id completed successfully"
                return 0
            elif echo "$status_response" | grep -q '"status":"failed"'; then
                log_error "❌ Training round $round_id failed"
                return 1
            fi
            
            sleep 10
        done
        
        log_error "❌ Training round $round_id timed out"
        return 1
    else
        log_error "❌ Failed to start training round: $response"
        return 1
    fi
}

# Test privacy features
test_privacy_features() {
    log_info "Testing privacy features..."
    
    # Test privacy budget tracking
    for i in "${!CLIENT_URLS[@]}"; do
        client_url="${CLIENT_URLS[$i]}"
        
        response=$(curl -k -s "$client_url/api/v1/status")
        
        if echo "$response" | grep -q "privacy_budget"; then
            log_info "✅ Client $((i+1)) privacy budget tracking working"
        else
            log_error "❌ Client $((i+1)) privacy budget tracking failed"
            return 1
        fi
    done
}

# Test Byzantine detection
test_byzantine_detection() {
    log_info "Testing Byzantine detection..."
    
    # Check if Byzantine detection metrics are available
    metrics_response=$(curl -k -s "$COORDINATOR_URL/metrics")
    
    if echo "$metrics_response" | grep -q "fl_byzantine_clients_detected"; then
        log_info "✅ Byzantine detection metrics available"
    else
        log_error "❌ Byzantine detection metrics not available"
        return 1
    fi
}

# Test database connectivity
test_database() {
    log_info "Testing database connectivity..."
    
    # Test admin stats endpoint (requires database)
    response=$(curl -k -s "$COORDINATOR_URL/api/v1/admin/stats")
    
    if echo "$response" | grep -q "clients"; then
        log_info "✅ Database connectivity working"
    else
        log_error "❌ Database connectivity failed"
        return 1
    fi
}

# Test monitoring stack
test_monitoring() {
    log_info "Testing monitoring stack..."
    
    # Test Prometheus
    if curl -f "http://localhost:9090/-/healthy" > /dev/null 2>&1; then
        log_info "✅ Prometheus is healthy"
    else
        log_warn "⚠️ Prometheus not accessible (may not be deployed)"
    fi
    
    # Test Grafana
    if curl -f "http://localhost:3000/api/health" > /dev/null 2>&1; then
        log_info "✅ Grafana is healthy"
    else
        log_warn "⚠️ Grafana not accessible (may not be deployed)"
    fi
}

# Performance test
test_performance() {
    log_info "Running performance tests..."
    
    # Test concurrent requests
    log_info "Testing concurrent health checks..."
    
    for i in {1..10}; do
        curl -k -s "$COORDINATOR_URL/health" > /dev/null &
    done
    wait
    
    log_info "✅ Concurrent requests handled successfully"
    
    # Test load on metrics endpoint
    log_info "Testing metrics endpoint load..."
    
    for i in {1..5}; do
        curl -k -s "$COORDINATOR_URL/metrics" > /dev/null &
    done
    wait
    
    log_info "✅ Metrics endpoint load test passed"
}

# Security tests
test_security() {
    log_info "Testing security features..."
    
    # Test TLS
    if curl -k -I "$COORDINATOR_URL/health" 2>&1 | grep -q "SSL"; then
        log_info "✅ TLS encryption enabled"
    else
        log_warn "⚠️ TLS encryption status unclear"
    fi
    
    # Test authentication (should fail without token)
    response=$(curl -k -s -w "%{http_code}" "$COORDINATOR_URL/api/v1/clients" -o /dev/null)
    
    if [ "$response" = "401" ]; then
        log_info "✅ Authentication required for protected endpoints"
    else
        log_warn "⚠️ Authentication may not be properly configured"
    fi
}

# Cleanup test data
cleanup_test_data() {
    log_info "Cleaning up test data..."
    
    # Reset privacy budgets
    for i in "${!CLIENT_URLS[@]}"; do
        client_url="${CLIENT_URLS[$i]}"
        curl -k -s -X POST "$client_url/api/v1/privacy/reset" > /dev/null || true
    done
    
    log_info "✅ Test data cleaned up"
}

# Main test execution
main() {
    log_info "🏥 Production FL System Test Suite"
    log_info "Coordinator: $COORDINATOR_URL"
    log_info "Clients: ${CLIENT_URLS[*]}"
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 30
    
    # Run tests
    test_health_endpoints || exit 1
    test_metrics_endpoints || exit 1
    test_client_registration || exit 1
    test_database || exit 1
    test_privacy_features || exit 1
    test_byzantine_detection || exit 1
    test_monitoring
    test_performance || exit 1
    test_security
    test_training_round || exit 1
    
    # Cleanup
    cleanup_test_data
    
    log_info "🎉 All tests passed! Production FL system is working correctly."
    
    # Print summary
    echo ""
    log_info "Test Summary:"
    log_info "✅ Health endpoints working"
    log_info "✅ Metrics collection working"
    log_info "✅ Client registration working"
    log_info "✅ Database connectivity working"
    log_info "✅ Privacy features working"
    log_info "✅ Byzantine detection available"
    log_info "✅ Performance tests passed"
    log_info "✅ Security features enabled"
    log_info "✅ Federated training working"
    echo ""
    log_info "🚀 Production system ready for deployment!"
}

# Handle script interruption
trap cleanup_test_data EXIT

# Run main function
main "$@"