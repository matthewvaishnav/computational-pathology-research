# Docker Deployment Guide - Medical AI Platform

Complete production-ready Docker deployment stack for the Medical AI Revolution platform.

## 🚀 Quick Start

### Prerequisites
- Docker Desktop (Windows/Mac) or Docker Engine (Linux)
- Docker Compose v2.0+
- 8GB+ RAM available
- 10GB+ disk space

### 1. Deploy the Stack

**Windows:**
```cmd
scripts\deploy.bat
```

**Linux/Mac:**
```bash
chmod +x scripts/deploy.sh
./scripts/deploy.sh
```

### 2. Access Services

| Service | URL | Credentials |
|---------|-----|-------------|
| **API Server** | http://localhost:8000 | - |
| **API Documentation** | http://localhost:8000/docs | - |
| **Grafana Monitoring** | http://localhost:3000 | admin/admin123 |
| **Prometheus Metrics** | http://localhost:9090 | - |
| **Database** | localhost:5432 | medai/medai123 |

## 📋 Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Nginx         │    │   API Server    │    │   Worker        │
│   (Load Bal.)   │────│   (FastAPI)     │────│   (Celery)      │
│   Port: 80      │    │   Port: 8000    │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         │              │   PostgreSQL    │              │
         └──────────────│   Database      │──────────────┘
                        │   Port: 5432    │
                        └─────────────────┘
                                 │
                        ┌─────────────────┐
                        │   Redis Cache   │
                        │   Port: 6379    │
                        └─────────────────┘
```

## 🔧 Configuration

### Environment Variables (.env)

```bash
# Database
DATABASE_URL=postgresql://medai:medai123@postgres:5432/medical_ai

# Model Configuration
MODEL_PATH=/app/checkpoints/pcam_real/best_model.pth
CONFIDENCE_THRESHOLD=0.8

# Security (CHANGE IN PRODUCTION!)
SECRET_KEY=your-secret-key-change-in-production
JWT_EXPIRE_MINUTES=30

# File Uploads
MAX_UPLOAD_SIZE_MB=100
SUPPORTED_FORMATS=svs,ndpi,tiff,dcm
```

### Volume Mounts

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `./checkpoints` | `/app/checkpoints` | AI model files |
| `./data` | `/app/data` | Upload/export data |
| `./logs` | `/app/logs` | Application logs |

## 🏥 API Endpoints

### Core Endpoints

```bash
# Health Check
GET /health

# Upload Image for Analysis
POST /api/v1/analyze/upload
Content-Type: multipart/form-data
Body: file=@image.svs

# Get Analysis Results
GET /api/v1/analyze/{analysis_id}

# List User's Analyses
GET /api/v1/analyze/list

# DICOM Integration
POST /api/v1/dicom/query
GET /api/v1/dicom/retrieve/{study_id}
```

### Example Usage

```bash
# Upload and analyze an image
curl -X POST "http://localhost:8000/api/v1/analyze/upload" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@test_image.svs"

# Check analysis status
curl "http://localhost:8000/api/v1/analyze/12345"
```

## 📊 Monitoring & Observability

### Grafana Dashboards

1. **System Overview**: CPU, memory, disk usage
2. **API Performance**: Request rates, response times, errors
3. **Model Metrics**: Inference time, confidence scores
4. **Database Performance**: Query times, connections

### Prometheus Metrics

- `api_requests_total`: Total API requests
- `api_request_duration_seconds`: Request duration
- `model_inference_duration_seconds`: Model inference time
- `database_connections_active`: Active DB connections

### Log Aggregation

```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f api
docker-compose logs -f worker
docker-compose logs -f nginx
```

## 🔒 Security Features

### Network Security
- Internal Docker network isolation
- Nginx reverse proxy with rate limiting
- CORS protection
- Security headers (XSS, CSRF, etc.)

### Data Security
- PostgreSQL with encrypted connections
- JWT token authentication
- File upload validation
- Audit logging for all actions

### Production Hardening
```bash
# Change default passwords
# Update .env file:
POSTGRES_PASSWORD=strong-random-password
SECRET_KEY=long-random-secret-key

# Enable HTTPS (add SSL certificates)
# Uncomment HTTPS server block in nginx.conf
```

## 🚀 Scaling & Performance

### Horizontal Scaling

```yaml
# Scale API servers
docker-compose up -d --scale api=3

# Scale workers
docker-compose up -d --scale worker=2
```

### Performance Tuning

```bash
# API Server
API_WORKERS=4  # Adjust based on CPU cores

# Database
# Edit docker-compose.yml postgres service:
command: postgres -c max_connections=200 -c shared_buffers=256MB

# Redis
# Add to docker-compose.yml redis service:
command: redis-server --maxmemory 512mb --maxmemory-policy allkeys-lru
```

## 🐛 Troubleshooting

### Common Issues

**1. Services won't start**
```bash
# Check Docker daemon
docker info

# Check port conflicts
netstat -tulpn | grep :8000

# View service logs
docker-compose logs api
```

**2. Database connection errors**
```bash
# Check database status
docker-compose exec postgres pg_isready -U medai

# Reset database
docker-compose down -v
docker-compose up -d postgres
```

**3. Model loading errors**
```bash
# Check model file exists
ls -la checkpoints/pcam_real/best_model.pth

# Check API logs
docker-compose logs api | grep -i model
```

**4. High memory usage**
```bash
# Monitor resource usage
docker stats

# Limit container memory
# Add to docker-compose.yml:
deploy:
  resources:
    limits:
      memory: 2G
```

### Health Checks

```bash
# API Health
curl http://localhost:8000/health

# Database Health
docker-compose exec postgres pg_isready -U medai

# Redis Health
docker-compose exec redis redis-cli ping
```

## 🔄 Maintenance

### Backup & Restore

```bash
# Backup database
docker-compose exec postgres pg_dump -U medai medical_ai > backup.sql

# Restore database
docker-compose exec -T postgres psql -U medai medical_ai < backup.sql

# Backup volumes
docker run --rm -v medical-ai_postgres_data:/data -v $(pwd):/backup alpine tar czf /backup/postgres_backup.tar.gz /data
```

### Updates & Upgrades

```bash
# Update images
docker-compose pull
docker-compose up -d

# Rebuild after code changes
docker-compose build --no-cache
docker-compose up -d
```

### Log Rotation

```bash
# Configure log rotation in docker-compose.yml:
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

## 🌐 Cloud Deployment

### AWS ECS
```bash
# Install ECS CLI
# Configure cluster
ecs-cli configure --cluster medical-ai --region us-west-2

# Deploy
ecs-cli compose up --create-log-groups
```

### Kubernetes
```bash
# Convert to Kubernetes manifests
kompose convert

# Deploy to cluster
kubectl apply -f .
```

### Docker Swarm
```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml medical-ai
```

## 📞 Support

### Getting Help
- **Documentation**: Check `/docs` endpoint for API docs
- **Logs**: Use `docker-compose logs -f` for debugging
- **Monitoring**: Check Grafana dashboards for system health
- **Issues**: Create GitHub issues for bugs or feature requests

### Performance Optimization
- Monitor Grafana dashboards for bottlenecks
- Scale services based on load
- Optimize database queries
- Use Redis caching for frequent requests

---

**🎉 Congratulations! You now have a production-ready Medical AI platform running in Docker.**

The platform is ready for:
- ✅ Image upload and analysis
- ✅ Real-time monitoring
- ✅ Horizontal scaling
- ✅ Production deployment
- ✅ Clinical workflow integration