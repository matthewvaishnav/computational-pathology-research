# HistoCore Docker Deployment

GPU-enabled Docker containers for real-time WSI streaming.

## Quick Start

```bash
# Build image
./scripts/docker-build.sh

# Run development
./scripts/docker-run.sh latest dev

# Run production
docker-compose up -d
```

## Requirements

- Docker 20.10+
- Docker Compose 2.0+
- NVIDIA Docker (for GPU)
- 8GB+ RAM
- 4GB+ GPU memory

## Services

| Service | Port | Description |
|---------|------|-------------|
| Streaming | 8000 | Web dashboard |
| API | 8001 | REST API |
| Metrics | 8002 | Prometheus metrics |
| Grafana | 3000 | Monitoring dashboards |
| Prometheus | 9090 | Metrics collection |
| Redis | 6379 | Cache |

## GPU Setup

### NVIDIA Docker (Linux)
```bash
# Install NVIDIA Docker
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### Windows WSL2
```powershell
# Install Docker Desktop with WSL2
# Enable GPU support in Docker Desktop settings
# Verify: docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi
```

## Configuration

### Environment Variables
```bash
# GPU settings
CUDA_VISIBLE_DEVICES=0
HISTOCORE_MAX_MEMORY_GB=8
HISTOCORE_BATCH_SIZE=32

# Performance
HISTOCORE_WORKERS=4
HISTOCORE_TIMEOUT=300

# Cache
REDIS_URL=redis://redis:6379
HISTOCORE_CACHE_TTL=3600
```

### Production Config
Edit `docker/production.env` for production settings.

## Build Targets

- `base`: CUDA runtime + Python
- `deps`: Dependencies installed
- `app`: Application code
- `production`: Optimized for production

## Health Checks

All services include health checks:
- HTTP endpoints: `/health`
- Redis: `redis-cli ping`
- Automatic restart on failure

## Monitoring

### Grafana Dashboards
- GPU utilization
- Memory usage
- Processing throughput
- Error rates
- Cache hit rates

### Prometheus Metrics
- `histocore_processing_time_seconds`
- `histocore_memory_usage_bytes`
- `histocore_gpu_utilization_percent`
- `histocore_cache_hits_total`

## Troubleshooting

### GPU Not Detected
```bash
# Check NVIDIA drivers
nvidia-smi

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi

# Check container GPU access
docker exec histocore-streaming nvidia-smi
```

### Memory Issues
```bash
# Check container memory
docker stats histocore-streaming

# Adjust memory limits
export HISTOCORE_MAX_MEMORY_GB=4
docker-compose up -d
```

### Performance Issues
```bash
# Check logs
docker logs histocore-streaming

# Monitor metrics
curl http://localhost:8002/metrics

# Check GPU utilization
docker exec histocore-streaming nvidia-smi
```

## Development

### Local Development
```bash
# Mount source code
docker run -it --rm \
  --gpus all \
  -v $(pwd):/app \
  -p 8000:8000 \
  histocore/streaming:latest \
  /bin/bash
```

### Debug Mode
```bash
# Enable debug logging
export HISTOCORE_LOG_LEVEL=DEBUG
docker-compose up
```

## Production Deployment

### Security
- Change default passwords
- Use secrets management
- Enable TLS/SSL
- Configure firewall rules

### Scaling
```yaml
# docker-compose.override.yml
services:
  histocore-streaming:
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 8G
        reservations:
          devices:
            - driver: nvidia
              count: 1
```

### Backup
```bash
# Backup Redis data
docker exec histocore-redis redis-cli BGSAVE

# Backup Grafana dashboards
docker cp histocore-grafana:/var/lib/grafana ./backup/
```