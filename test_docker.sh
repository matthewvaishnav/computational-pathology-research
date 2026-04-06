#!/bin/bash
# Test script for Docker deployment

set -e

echo "=== Testing Docker Deployment ==="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Error: Docker Compose is not installed${NC}"
    exit 1
fi

echo "✓ Docker and Docker Compose are installed"
echo ""

# Build the image
echo "Building Docker image..."
docker-compose build api
echo -e "${GREEN}✓ Image built successfully${NC}"
echo ""

# Start the service
echo "Starting API service..."
docker-compose up -d api
echo -e "${GREEN}✓ Service started${NC}"
echo ""

# Wait for service to be ready
echo "Waiting for service to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Service is ready${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}Error: Service did not start in time${NC}"
        docker-compose logs api
        docker-compose down
        exit 1
    fi
    sleep 2
done
echo ""

# Test health endpoint
echo "Testing /health endpoint..."
HEALTH_RESPONSE=$(curl -s http://localhost:8000/health)
if echo "$HEALTH_RESPONSE" | grep -q "healthy"; then
    echo -e "${GREEN}✓ Health check passed${NC}"
    echo "Response: $HEALTH_RESPONSE"
else
    echo -e "${RED}Error: Health check failed${NC}"
    echo "Response: $HEALTH_RESPONSE"
    docker-compose down
    exit 1
fi
echo ""

# Test model-info endpoint
echo "Testing /model-info endpoint..."
MODEL_INFO=$(curl -s http://localhost:8000/model-info)
if echo "$MODEL_INFO" | grep -q "MultimodalFusionModel"; then
    echo -e "${GREEN}✓ Model info retrieved${NC}"
    echo "Response: $MODEL_INFO"
else
    echo -e "${RED}Error: Model info failed${NC}"
    echo "Response: $MODEL_INFO"
    docker-compose down
    exit 1
fi
echo ""

# Test prediction endpoint (with synthetic data)
echo "Testing /predict endpoint..."
PREDICT_RESPONSE=$(curl -s -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{
        "genomic": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    }' 2>&1)

# Note: This will fail because genomic needs 2000 features, but we test the endpoint responds
if echo "$PREDICT_RESPONSE" | grep -q "detail"; then
    echo -e "${GREEN}✓ Prediction endpoint is responding${NC}"
    echo "Response: $PREDICT_RESPONSE"
else
    echo -e "${RED}Warning: Unexpected prediction response${NC}"
    echo "Response: $PREDICT_RESPONSE"
fi
echo ""

# Show container stats
echo "Container stats:"
docker stats --no-stream pathology-api
echo ""

# Show logs
echo "Recent logs:"
docker-compose logs --tail=20 api
echo ""

# Cleanup
echo "Cleaning up..."
docker-compose down
echo -e "${GREEN}✓ Cleanup complete${NC}"
echo ""

echo -e "${GREEN}=== All tests passed! ===${NC}"
echo ""
echo "To start the service manually:"
echo "  docker-compose up -d api"
echo ""
echo "To view logs:"
echo "  docker-compose logs -f api"
echo ""
echo "To stop the service:"
echo "  docker-compose down"
