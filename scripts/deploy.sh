#!/usr/bin/env bash
# Start the local computational-pathology research stack.
# This is development infrastructure, not a clinical or production deployment.

set -euo pipefail

if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is required." >&2
    exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
    echo "Docker Compose v2 ('docker compose') is required." >&2
    exit 1
fi

if [[ ! -f .env ]]; then
    cp .env.docker.example .env
    echo "Created .env from .env.docker.example."
    echo "Replace every CHANGE_ME value, then run this script again."
    exit 1
fi

if grep -q "CHANGE_ME" .env; then
    echo ".env still contains CHANGE_ME placeholders. Refusing to start." >&2
    exit 1
fi

mkdir -p logs/api logs/nginx data/uploads data/exports

echo "Validating Compose configuration..."
docker compose config >/dev/null

echo "Building and starting local research services..."
docker compose up -d --build

echo
echo "Local stack:"
echo "  API:        http://localhost:8000"
echo "  API docs:   http://localhost:8000/docs"
echo "  Grafana:    http://localhost:3000"
echo "  Prometheus: http://localhost:9090"
echo
docker compose ps
echo
echo "Stop with: docker compose down"
