# Docker Deployment Guide

This guide covers containerized deployment of the computational pathology API using Docker and Docker Compose.

## Quick Start

### Prerequisites

- Docker 20.10+ installed
- Docker Compose 2.0+ installed
- At least 4GB RAM available for containers
- (Optional) NVIDIA Docker for GPU support

### Basic Deployment

```bash
# Build and start the API service
docker-compose up -d api

# Check service status
docker-compose ps

# View logs
docker-compose logs -f api

# Test the API
curl http://localhost:8000/health
```

The API will be available at `http://localhost:8000`.

### With Jupyter Notebook

```bash
# Start both API and notebook services
docker-compose up -d

# Access Jupyter at http://localhost:8888
# Access API at http://localhost:8000
```

## Docker Images

### Building the Image

```bash
# Build the image
docker build -t pathology-api:latest .

# Build with specific Python version
docker build --build-arg PYTHON_VERSION=3.11 -t pathology-api:py311 .

# Build without cache (clean build)
docker build --no-cache -t pathology-api:latest .
```

### Image Size Optimization

The Dockerfile uses multi-stage builds and slim base images to minimize size:

- Base image: `python:3.10-slim` (~150MB)
- Final image: ~2-3GB (includes PyTorch and dependencies)

**Further optimization options**:

```dockerfile
# Use Alpine for smaller base (more complex)
FROM python:3.10-alpine

# Use distroless for security
FROM gcr.io/distroless/python3

# Remove unnecessary files
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip
```

## Docker Compose

### Services

The `docker-compose.yml` defines two services:

1. **api**: FastAPI server for inference
2. **notebook**: Jupyter notebook for development (optional)

### Configuration

```yaml
# docker-compose.yml
services:
  api:
    build: .
    ports:
      - "8000:8000"  # Map host:container
    volumes:
      - ./models:/app/models:ro  # Mount trained models (read-only)
    environment:
      - LOG_LEVEL=info
    restart: unless-stopped
```

### Volume Mounts

**Models directory** (required for trained weights):
```yaml
volumes:
  - ./models:/app/models:ro
```

**Data directory** (optional, for batch processing):
```yaml
volumes:
  - ./data:/app/data:ro
```

**Results directory** (for saving outputs):
```yaml
volumes:
  - ./results:/app/results:rw
```

### Environment Variables

Configure the container via environment variables:

```yaml
environment:
  - PYTHONUNBUFFERED=1        # Real-time logging
  - LOG_LEVEL=info            # Logging level
  - MODEL_PATH=/app/models/best_model.pth  # Model checkpoint
  - DEVICE=cuda               # cpu or cuda
  - MAX_BATCH_SIZE=32         # Maximum batch size
```

## GPU Support

### NVIDIA Docker Setup

1. Install NVIDIA Docker runtime:
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

2. Update `docker-compose.yml`:
```yaml
services:
  api:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

3. Or use Docker CLI:
```bash
docker run --gpus all -p 8000:8000 pathology-api:latest
```

### Verify GPU Access

```bash
# Inside container
docker exec -it pathology-api python -c "import torch; print(torch.cuda.is_available())"
```

## Production Deployment

### Security Hardening

1. **Run as non-root user**:
```dockerfile
# Add to Dockerfile
RUN useradd -m -u 1000 appuser
USER appuser
```

2. **Read-only filesystem**:
```yaml
services:
  api:
    read_only: true
    tmpfs:
      - /tmp
```

3. **Resource limits**:
```yaml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G
```

4. **Network isolation**:
```yaml
networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge
    internal: true
```

### Health Checks

The Dockerfile includes a health check:

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1
```

Monitor health:
```bash
docker inspect --format='{{.State.Health.Status}}' pathology-api
```

### Logging

**View logs**:
```bash
# Follow logs
docker-compose logs -f api

# Last 100 lines
docker-compose logs --tail=100 api

# Since timestamp
docker-compose logs --since 2024-01-01T00:00:00 api
```

**Configure logging driver**:
```yaml
services:
  api:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

### Monitoring

**Prometheus metrics** (add to `deploy/api.py`):
```python
from prometheus_client import Counter, Histogram, make_asgi_app

# Add metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
```

**Container stats**:
```bash
docker stats pathology-api
```

## Kubernetes Deployment

### Basic Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pathology-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: pathology-api
  template:
    metadata:
      labels:
        app: pathology-api
    spec:
      containers:
      - name: api
        image: pathology-api:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

### Service

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: pathology-api
spec:
  selector:
    app: pathology-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Deploy to Kubernetes

```bash
# Apply configurations
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Check status
kubectl get pods
kubectl get services

# View logs
kubectl logs -f deployment/pathology-api

# Scale replicas
kubectl scale deployment pathology-api --replicas=5
```

## Cloud Deployment

### AWS ECS

1. **Push image to ECR**:
```bash
aws ecr create-repository --repository-name pathology-api
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker tag pathology-api:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/pathology-api:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/pathology-api:latest
```

2. **Create ECS task definition** (see `deploy/README.md` for details)

### Google Cloud Run

```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/PROJECT_ID/pathology-api

# Deploy to Cloud Run
gcloud run deploy pathology-api \
  --image gcr.io/PROJECT_ID/pathology-api \
  --platform managed \
  --region us-central1 \
  --memory 8Gi \
  --cpu 4 \
  --max-instances 10
```

### Azure Container Instances

```bash
# Push to ACR
az acr create --resource-group myResourceGroup --name myregistry --sku Basic
az acr login --name myregistry
docker tag pathology-api:latest myregistry.azurecr.io/pathology-api:latest
docker push myregistry.azurecr.io/pathology-api:latest

# Deploy to ACI
az container create \
  --resource-group myResourceGroup \
  --name pathology-api \
  --image myregistry.azurecr.io/pathology-api:latest \
  --cpu 4 \
  --memory 8 \
  --ports 8000
```

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker-compose logs api

# Common issues:
# 1. Port already in use
sudo lsof -i :8000
# Kill process or change port in docker-compose.yml

# 2. Missing model file
# Ensure models/best_model.pth exists or update MODEL_PATH

# 3. Out of memory
# Increase Docker memory limit in Docker Desktop settings
```

### Slow Performance

```bash
# Check resource usage
docker stats pathology-api

# Solutions:
# 1. Increase CPU/memory limits in docker-compose.yml
# 2. Enable GPU support
# 3. Use model quantization
# 4. Reduce batch size
```

### Connection Refused

```bash
# Check if container is running
docker ps

# Check if port is exposed
docker port pathology-api

# Test from inside container
docker exec -it pathology-api curl http://localhost:8000/health

# Check firewall rules
sudo ufw status
```

### GPU Not Detected

```bash
# Verify NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Check Docker daemon configuration
cat /etc/docker/daemon.json
# Should include: {"runtimes": {"nvidia": {"path": "nvidia-container-runtime"}}}

# Restart Docker
sudo systemctl restart docker
```

## Development Workflow

### Local Development with Docker

```bash
# Mount source code for live reloading
docker run -it --rm \
  -v $(pwd)/src:/app/src \
  -v $(pwd)/deploy:/app/deploy \
  -p 8000:8000 \
  pathology-api:latest \
  uvicorn deploy.api:app --host 0.0.0.0 --port 8000 --reload
```

### Running Tests in Container

```bash
# Run tests
docker-compose run --rm api pytest tests/ -v

# With coverage
docker-compose run --rm api pytest tests/ --cov=src --cov-report=html
```

### Interactive Shell

```bash
# Open bash shell in running container
docker exec -it pathology-api bash

# Or start new container with shell
docker run -it --rm pathology-api:latest bash
```

## Best Practices

1. **Use specific tags**: Avoid `latest` in production
   ```bash
   docker tag pathology-api:latest pathology-api:v1.0.0
   ```

2. **Multi-stage builds**: Separate build and runtime dependencies

3. **Layer caching**: Order Dockerfile commands from least to most frequently changed

4. **Security scanning**: Scan images for vulnerabilities
   ```bash
   docker scan pathology-api:latest
   ```

5. **Resource limits**: Always set CPU and memory limits in production

6. **Health checks**: Implement proper health check endpoints

7. **Logging**: Use structured logging and centralized log aggregation

8. **Secrets management**: Never hardcode secrets in images
   ```bash
   docker run -e API_KEY=$(cat api_key.txt) pathology-api:latest
   ```

## Performance Optimization

### Image Size

```bash
# Check image size
docker images pathology-api

# Analyze layers
docker history pathology-api:latest

# Use dive for detailed analysis
dive pathology-api:latest
```

### Build Cache

```bash
# Use BuildKit for better caching
DOCKER_BUILDKIT=1 docker build -t pathology-api:latest .

# Cache from registry
docker build --cache-from pathology-api:latest -t pathology-api:latest .
```

### Runtime Performance

1. **Use tmpfs for temporary files**:
```yaml
tmpfs:
  - /tmp
  - /app/cache
```

2. **Optimize Python startup**:
```dockerfile
ENV PYTHONOPTIMIZE=1
ENV PYTHONDONTWRITEBYTECODE=1
```

3. **Use gunicorn for production**:
```dockerfile
CMD ["gunicorn", "deploy.api:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
```

## References

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
