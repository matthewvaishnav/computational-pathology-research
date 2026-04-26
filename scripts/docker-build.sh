#!/bin/bash
# Docker build script for HistoCore streaming

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Config
IMAGE_NAME="histocore/streaming"
VERSION=${1:-"latest"}
BUILD_TARGET=${2:-"production"}

echo -e "${GREEN}Building HistoCore Docker image...${NC}"
echo "Image: $IMAGE_NAME:$VERSION"
echo "Target: $BUILD_TARGET"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker not found. Install Docker first.${NC}"
    exit 1
fi

# Check NVIDIA Docker
if ! docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}Warning: NVIDIA Docker not available. GPU features disabled.${NC}"
fi

# Build image
echo -e "${GREEN}Building Docker image...${NC}"
docker build \
    --target $BUILD_TARGET \
    --tag $IMAGE_NAME:$VERSION \
    --tag $IMAGE_NAME:latest \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    .

# Verify build
if docker images | grep -q $IMAGE_NAME; then
    echo -e "${GREEN}✓ Build successful${NC}"
    docker images | grep $IMAGE_NAME
else
    echo -e "${RED}✗ Build failed${NC}"
    exit 1
fi

# Optional: Run quick test
if [[ "$3" == "test" ]]; then
    echo -e "${GREEN}Running quick test...${NC}"
    docker run --rm $IMAGE_NAME:$VERSION python -c "import src.streaming; print('✓ Import successful')"
fi

echo -e "${GREEN}Done. Run with: docker-compose up${NC}"