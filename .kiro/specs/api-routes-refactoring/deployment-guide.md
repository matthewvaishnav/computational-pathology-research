# Deployment Guide: API Routes Refactoring

## Overview

This guide covers deployment considerations for the refactored FastAPI application, including environment configuration, security settings, monitoring, and scaling strategies.

## Environment Configuration

### Required Environment Variables

```bash
# Database Configuration
DATABASE_URL=postgresql://user:password@host:5432/dbname
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30

# Security Configuration
SECRET_KEY=your-cryptographically-secure-secret-key-here
JWT_SECRET_KEY=your-jwt-secret-key-here
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=30

# CORS Configuration
ALLOWED_ORIGINS=https://app.example.com,https://mobile.example.com
ALLOWED_METHODS=GET,POST,PUT,DELETE,OPTIONS
ALLOWED_HEADERS=Authorization,Content-Type,X-Requested-With

# OAuth Configuration (Optional)
AZURE_CLIENT_ID=your-azure-client-id
AZURE_CLIENT_SECRET=your-azure-client-secret
AZURE_TENANT_ID=your-azure-tenant-id
OAUTH_REDIRECT_URI=https://api.example.com/api/v1/auth/oauth/callback

# File Storage Configuration
UPLOAD_DIR=/app/uploads
MAX_UPLOAD_SIZE=104857600  # 100MB in bytes
ALLOWED_FILE_TYPES=image/png,image/jpeg,image/tiff,application/dicom

# Rate Limiting Configuration
RATE_LIMIT_ENABLED=true
LOGIN_RATE_LIMIT=5/minute
UPLOAD_RATE_LIMIT=10/minute
DEFAULT_RATE_LIMIT=100/minute

# Monitoring Configuration (Optional)
JAEGER_ENDPOINT=http://jaeger:14268/api/traces
OTLP_ENDPOINT=http://otel-collector:4317
PROMETHEUS_METRICS_ENABLED=true

# Application Configuration
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
WORKERS=4
```

### Environment-Specific Configurations

#### Development Environment

```bash
# .env.development
DATABASE_URL=sqlite:///./dev.db
SECRET_KEY=dev-secret-key-not-for-production
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8080
DEBUG=true
LOG_LEVEL=DEBUG
ENVIRONMENT=development
```

#### Staging Environment

```bash
# .env.staging
DATABASE_URL=postgresql://user:pass@staging-db:5432/histocore_staging
SECRET_KEY=staging-secret-key-change-in-production
ALLOWED_ORIGINS=https://staging.example.com
DEBUG=false
LOG_LEVEL=INFO
ENVIRONMENT=staging
RATE_LIMIT_ENABLED=true
```

#### Production Environment

```bash
# .env.production
DATABASE_URL=postgresql://user:pass@prod-db:5432/histocore_prod
SECRET_KEY=production-secret-key-from-secrets-manager
ALLOWED_ORIGINS=https://app.example.com,https://mobile.example.com
DEBUG=false
LOG_LEVEL=WARNING
ENVIRONMENT=production
RATE_LIMIT_ENABLED=true
JWT_EXPIRE_MINUTES=15
```

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY alembic/ ./alembic/
COPY alembic.ini .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app
USER app

# Create upload directory
RUN mkdir -p /app/uploads

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/histocore
      - SECRET_KEY=${SECRET_KEY}
      - ALLOWED_ORIGINS=${ALLOWED_ORIGINS}
    depends_on:
      - db
      - redis
    volumes:
      - ./uploads:/app/uploads
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=histocore
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - api
    restart: unless-stopped

volumes:
  postgres_data:
```

### Production Docker Compose

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  api:
    build: .
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - SECRET_KEY=${SECRET_KEY}
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}
      - ALLOWED_ORIGINS=${ALLOWED_ORIGINS}
      - ENVIRONMENT=production
    volumes:
      - /var/app/uploads:/app/uploads
    restart: unless-stopped
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
        reservations:
          cpus: '0.5'
          memory: 512M

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.prod.conf:/etc/nginx/nginx.conf
      - /etc/letsencrypt:/etc/letsencrypt:ro
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 256M
```

## Kubernetes Deployment

### Deployment Manifest

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: histocore-api
  labels:
    app: histocore-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: histocore-api
  template:
    metadata:
      labels:
        app: histocore-api
    spec:
      containers:
      - name: api
        image: histocore/api:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: histocore-secrets
              key: database-url
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: histocore-secrets
              key: secret-key
        - name: ALLOWED_ORIGINS
          valueFrom:
            configMapKeyRef:
              name: histocore-config
              key: allowed-origins
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /api/v1/system/readiness
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
        volumeMounts:
        - name: uploads
          mountPath: /app/uploads
      volumes:
      - name: uploads
        persistentVolumeClaim:
          claimName: histocore-uploads
```

### Service Manifest

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: histocore-api-service
spec:
  selector:
    app: histocore-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
```

### Ingress Manifest

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: histocore-api-ingress
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
spec:
  tls:
  - hosts:
    - api.example.com
    secretName: histocore-tls
  rules:
  - host: api.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: histocore-api-service
            port:
              number: 80
```

### ConfigMap and Secrets

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: histocore-config
data:
  allowed-origins: "https://app.example.com,https://mobile.example.com"
  environment: "production"
  log-level: "INFO"
  rate-limit-enabled: "true"

---
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: histocore-secrets
type: Opaque
data:
  database-url: <base64-encoded-database-url>
  secret-key: <base64-encoded-secret-key>
  jwt-secret-key: <base64-encoded-jwt-secret-key>
```

## Load Balancer Configuration

### Nginx Configuration

```nginx
# nginx.conf
upstream histocore_api {
    least_conn;
    server api1:8000 max_fails=3 fail_timeout=30s;
    server api2:8000 max_fails=3 fail_timeout=30s;
    server api3:8000 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name api.example.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.example.com;

    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/api.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.example.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;

    # Security Headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    # Rate Limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=auth:10m rate=5r/m;

    # File Upload Limits
    client_max_body_size 100M;
    client_body_timeout 60s;

    location / {
        limit_req zone=api burst=20 nodelay;
        
        proxy_pass http://histocore_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }

    location /api/v1/auth/login {
        limit_req zone=auth burst=5 nodelay;
        
        proxy_pass http://histocore_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /health {
        access_log off;
        proxy_pass http://histocore_api;
    }

    location /metrics {
        allow 10.0.0.0/8;
        allow 172.16.0.0/12;
        allow 192.168.0.0/16;
        deny all;
        
        proxy_pass http://histocore_api;
    }
}
```

## Database Configuration

### PostgreSQL Production Settings

```sql
-- postgresql.conf optimizations
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200
work_mem = 4MB
min_wal_size = 1GB
max_wal_size = 4GB
max_worker_processes = 8
max_parallel_workers_per_gather = 2
max_parallel_workers = 8
max_parallel_maintenance_workers = 2
```

### Database Migration

```bash
# Run database migrations
alembic upgrade head

# Create initial admin user
python scripts/create_admin_user.py --email admin@example.com --password secure_password
```

### Connection Pooling

```python
# src/database/connection.py
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False
)
```

## Monitoring and Logging

### Prometheus Metrics

```python
# src/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge

# Request metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Active database connections')

# Business metrics
USER_REGISTRATIONS = Counter('user_registrations_total', 'Total user registrations')
IMAGE_UPLOADS = Counter('image_uploads_total', 'Total image uploads')
ANALYSIS_REQUESTS = Counter('analysis_requests_total', 'Total analysis requests')
```

### Structured Logging

```python
# src/logging/config.py
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
            
        return json.dumps(log_entry)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

for handler in logging.root.handlers:
    handler.setFormatter(JSONFormatter())
```

### Health Checks

```python
# Enhanced health check endpoint
@router.get("/health")
async def health_check(db: Session = Depends(get_db_session)):
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "components": {}
    }
    
    try:
        # Database health
        db.execute("SELECT 1")
        health_status["components"]["database"] = "healthy"
    except Exception as e:
        health_status["components"]["database"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"
    
    try:
        # Redis health (if using Redis)
        redis_client.ping()
        health_status["components"]["redis"] = "healthy"
    except Exception as e:
        health_status["components"]["redis"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"
    
    # Return appropriate status code
    status_code = 200 if health_status["status"] == "healthy" else 503
    return JSONResponse(content=health_status, status_code=status_code)
```

## Security Configuration

### HTTPS and TLS

```bash
# Generate SSL certificate with Let's Encrypt
certbot certonly --webroot -w /var/www/html -d api.example.com

# Auto-renewal cron job
0 12 * * * /usr/bin/certbot renew --quiet
```

### Security Headers

```python
# src/middleware/security.py
from fastapi import FastAPI
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware

def add_security_middleware(app: FastAPI):
    # HTTPS redirect in production
    if ENVIRONMENT == "production":
        app.add_middleware(HTTPSRedirectMiddleware)
    
    # Trusted hosts
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["api.example.com", "*.example.com"]
    )
```

### Rate Limiting

```python
# src/middleware/rate_limiting.py
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
```

## Scaling Strategies

### Horizontal Scaling

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: histocore-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: histocore-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Database Scaling

```python
# Read replicas configuration
MASTER_DATABASE_URL = "postgresql://user:pass@master-db:5432/histocore"
REPLICA_DATABASE_URL = "postgresql://user:pass@replica-db:5432/histocore"

# Connection routing
class DatabaseRouter:
    def db_for_read(self, model, **hints):
        return 'replica'
    
    def db_for_write(self, model, **hints):
        return 'master'
```

### Caching Strategy

```python
# Redis caching
import redis
from functools import wraps

redis_client = redis.Redis(host='redis', port=6379, db=0)

def cache_result(expiration=300):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached_result = redis_client.get(cache_key)
            if cached_result:
                return json.loads(cached_result)
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            redis_client.setex(cache_key, expiration, json.dumps(result))
            
            return result
        return wrapper
    return decorator
```

## Backup and Recovery

### Database Backup

```bash
#!/bin/bash
# backup_database.sh

BACKUP_DIR="/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="histocore_backup_${TIMESTAMP}.sql"

# Create backup
pg_dump $DATABASE_URL > "${BACKUP_DIR}/${BACKUP_FILE}"

# Compress backup
gzip "${BACKUP_DIR}/${BACKUP_FILE}"

# Upload to S3 (optional)
aws s3 cp "${BACKUP_DIR}/${BACKUP_FILE}.gz" s3://histocore-backups/

# Clean up old backups (keep last 7 days)
find $BACKUP_DIR -name "histocore_backup_*.sql.gz" -mtime +7 -delete
```

### File Storage Backup

```bash
#!/bin/bash
# backup_uploads.sh

UPLOAD_DIR="/app/uploads"
BACKUP_DIR="/backups/uploads"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Sync uploads to backup directory
rsync -av --delete $UPLOAD_DIR/ "${BACKUP_DIR}/${TIMESTAMP}/"

# Upload to S3
aws s3 sync "${BACKUP_DIR}/${TIMESTAMP}/" s3://histocore-uploads-backup/
```

## Disaster Recovery

### Recovery Procedures

```bash
# Database recovery
pg_restore --clean --if-exists -d $DATABASE_URL backup_file.sql

# File recovery
aws s3 sync s3://histocore-uploads-backup/ /app/uploads/

# Application restart
kubectl rollout restart deployment/histocore-api
```

### Monitoring and Alerting

```yaml
# prometheus/alerts.yaml
groups:
- name: histocore-api
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: High error rate detected
      
  - alert: DatabaseConnectionFailure
    expr: up{job="histocore-api"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: Database connection failure
      
  - alert: HighMemoryUsage
    expr: container_memory_usage_bytes / container_spec_memory_limit_bytes > 0.9
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: High memory usage
```

## Production Checklist

### Pre-Deployment

- [ ] Environment variables configured
- [ ] SSL certificates installed
- [ ] Database migrations applied
- [ ] Security headers configured
- [ ] Rate limiting enabled
- [ ] Monitoring and logging configured
- [ ] Backup procedures tested
- [ ] Load testing completed
- [ ] Security audit performed

### Post-Deployment

- [ ] Health checks passing
- [ ] Metrics being collected
- [ ] Logs being aggregated
- [ ] Alerts configured
- [ ] Backup jobs running
- [ ] Performance monitoring active
- [ ] Security monitoring enabled
- [ ] Documentation updated

### Ongoing Maintenance

- [ ] Regular security updates
- [ ] Database maintenance
- [ ] Log rotation
- [ ] Backup verification
- [ ] Performance optimization
- [ ] Capacity planning
- [ ] Incident response procedures
- [ ] Disaster recovery testing

This deployment guide provides comprehensive coverage of production deployment considerations for the refactored FastAPI application, ensuring security, scalability, and reliability in production environments.