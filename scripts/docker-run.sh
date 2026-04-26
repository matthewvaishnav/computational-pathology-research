#!/bin/bash
# Docker run script for HistoCore streaming

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Config
IMAGE_NAME="histocore/streaming"
VERSION=${1:-"latest"}
MODE=${2:-"dev"}

echo -e "${GREEN}Starting HistoCore streaming...${NC}"

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_FLAG="--gpus all"
    echo -e "${GREEN}✓ GPU support enabled${NC}"
else
    GPU_FLAG=""
    echo -e "${YELLOW}⚠ No GPU detected${NC}"
fi

# Development mode
if [[ "$MODE" == "dev" ]]; then
    echo "Mode: Development"
    docker run -it --rm \
        $GPU_FLAG \
        -p 8000:8000 \
        -p 8001:8001 \
        -p 8002:8002 \
        -v $(pwd)/data:/app/data \
        -v $(pwd)/logs:/app/logs \
        -v $(pwd)/cache:/app/cache \
        -e HISTOCORE_LOG_LEVEL=DEBUG \
        $IMAGE_NAME:$VERSION

# Production mode
elif [[ "$MODE" == "prod" ]]; then
    echo "Mode: Production (use docker-compose)"
    docker-compose up -d
    echo "Services starting..."
    echo "Dashboard: http://localhost:8000"
    echo "Grafana: http://localhost:3000 (admin/histocore123)"
    echo "Prometheus: http://localhost:9090"

# Interactive shell
elif [[ "$MODE" == "shell" ]]; then
    echo "Mode: Interactive shell"
    docker run -it --rm \
        $GPU_FLAG \
        -v $(pwd):/app \
        $IMAGE_NAME:$VERSION \
        /bin/bash

else
    echo "Usage: $0 [version] [dev|prod|shell]"
    exit 1
fi