# the platform Real-Time WSI Streaming - Technical Administrator Guide

**For System Administrators, DevOps Engineers, and IT Staff**

## 🎯 Overview

Production deployment and administration guide for the platform Real-Time WSI Streaming system. Covers installation, configuration, monitoring, troubleshooting, and maintenance.

---

## 📋 Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Security Setup](#security-setup)
5. [PACS Integration](#pacs-integration)
6. [Monitoring](#monitoring)
7. [Backup & Recovery](#backup--recovery)
8. [Troubleshooting](#troubleshooting)
9. [Maintenance](#maintenance)
10. [Performance Tuning](#performance-tuning)

---

## System Requirements

### Hardware Requirements

**Minimum (Development/Testing)**:
- CPU: 4 cores (Intel Xeon or AMD EPYC)
- RAM: 16GB
- GPU: NVIDIA GPU with 8GB VRAM (Tesla T4, RTX 3070)
- Storage: 100GB SSD
- Network: 1Gbps

**Recommended (Production)**:
- CPU: 16+ cores (Intel Xeon Gold or AMD EPYC)
- RAM: 64GB+
- GPU: NVIDIA A100 (40GB) or 4x V100 (32GB)
- Storage: 1TB+ NVMe SSD (RAID 10)
- Network: 10Gbps

**High-Availability (Enterprise)**:
- CPU: 32+ cores per node
- RAM: 128GB+ per node
- GPU: 8x A100 (80GB) across nodes
- Storage: 10TB+ distributed storage (Ceph, GlusterFS)
- Network: 25Gbps with redundancy

### Software Requirements

**Operating System**:
- Ubuntu 20.04/22.04 LTS (recommended)
- RHEL 8/9
- CentOS 8/9
- Windows Server 2019/2022 (limited support)

**Container Runtime**:
- Docker 20.10+ or Podman 4.0+
- Kubernetes 1.24+ (for orchestration)

**GPU Drivers**:
- NVIDIA Driver 525.60.13+
- CUDA 11.8+
- cuDNN 8.6+

**Optional Components**:
- Redis 7.0+ (caching)
- PostgreSQL 14+ (audit logs)
- Prometheus 2.40+ (monitoring)
- Grafana 9.0+ (dashboards)

---

## Installation

### Method 1: Docker (Recommended)

**Single GPU Setup**:

```bash
# Pull image
docker pull the platform/streaming:latest

# Run container
docker run -d \
  --name the platform-streaming \
  --gpus all \
  -p 8000:8000 \
  -p 8001:8001 \
  -v /data/models:/models \
  -v /data/cache:/cache \
  -v /data/logs:/logs \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e MODEL_PATH=/models/the platform_v1.pth \
  the platform/streaming:latest
```

**Multi-GPU Setup**:

```bash
docker run -d \
  --name the platform-streaming \
  --gpus '"device=0,1,2,3"' \
  -p 8000:8000 \
  -p 8001:8001 \
  -v /data/models:/models \
  -v /data/cache:/cache \
  -v /data/logs:/logs \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -e ENABLE_MULTI_GPU=true \
  the platform/streaming:latest
```

**With Redis Caching**:

```bash
# Start Redis
docker run -d \
  --name redis \
  -p 6379:6379 \
  redis:7-alpine

# Start the platform with Redis
docker run -d \
  --name the platform-streaming \
  --gpus all \
  --link redis:redis \
  -p 8000:8000 \
  -e REDIS_HOST=redis \
  -e REDIS_PORT=6379 \
  -e ENABLE_CACHING=true \
  the platform/streaming:latest
```

### Method 2: Kubernetes

**Deploy with Helm**:

```bash
# Add Helm repo
helm repo add the platform https://charts.the platform.ai
helm repo update

# Install
helm install the platform-streaming the platform/streaming \
  --namespace the platform \
  --create-namespace \
  --set gpu.count=4 \
  --set replicaCount=2 \
  --set persistence.enabled=true \
  --set persistence.size=1Ti
```

**Manual Deployment**:

```bash
# Apply manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml

# Verify deployment
kubectl get pods -n the platform
kubectl logs -f deployment/the platform-streaming -n the platform
```

### Method 3: Bare Metal

**Install Dependencies**:

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.10
sudo apt install python3.10 python3.10-venv python3-pip -y

# Install CUDA (if not already installed)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install cuda-11-8 -y

# Install OpenSlide
sudo apt install openslide-tools python3-openslide -y
```

**Install the platform**:

```bash
# Clone repository
git clone https://github.com/the platform/streaming.git
cd streaming

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install the platform
pip install -e .

# Download model
mkdir -p models
wget https://models.the platform.ai/v1/the platform_v1.pth -O models/the platform_v1.pth
```

**Run Service**:

```bash
# Start server
python -m src.streaming.server \
  --host 0.0.0.0 \
  --port 8000 \
  --model-path models/the platform_v1.pth \
  --gpu-ids 0,1,2,3 \
  --workers 4
```

**Systemd Service** (recommended):

```bash
# Create service file
sudo nano /etc/systemd/system/the platform-streaming.service
```

```ini
[Unit]
Description=the platform Real-Time WSI Streaming
After=network.target

[Service]
Type=simple
User=the platform
WorkingDirectory=/opt/the platform/streaming
Environment="PATH=/opt/the platform/streaming/venv/bin"
ExecStart=/opt/the platform/streaming/venv/bin/python -m src.streaming.server --config /etc/the platform/config.yaml
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable the platform-streaming
sudo systemctl start the platform-streaming
sudo systemctl status the platform-streaming
```

---

## Configuration

### Configuration File

**Location**: `/etc/the platform/config.yaml` (bare metal) or ConfigMap (K8s)

**Minimal Configuration**:

```yaml
# Core settings
model:
  path: /models/the platform_v1.pth
  device: cuda
  
gpu:
  ids: [0]
  enable_multi_gpu: false
  
server:
  host: 0.0.0.0
  port: 8000
  workers: 4
```

**Production Configuration**:

```yaml
# Core settings
model:
  path: /models/the platform_v1.pth
  device: cuda
  enable_fp16: true
  enable_tensorrt: true
  
gpu:
  ids: [0, 1, 2, 3]
  enable_multi_gpu: true
  memory_limit_gb: 30
  
# Performance
processing:
  batch_size: 64
  max_concurrent_slides: 10
  tile_size: 224
  overlap: 0
  
# Caching
cache:
  enabled: true
  backend: redis
  redis_host: redis.the platform.svc.cluster.local
  redis_port: 6379
  ttl_seconds: 86400
  compression: true
  
# Storage
storage:
  backend: s3
  s3_bucket: the platform-production
  s3_region: us-east-1
  local_cache_dir: /cache
  cleanup_after_days: 7
  
# Security
security:
  enable_tls: true
  tls_cert: /etc/the platform/certs/server.crt
  tls_key: /etc/the platform/certs/server.key
  enable_at_rest_encryption: true
  encryption_key_path: /etc/the platform/keys/master.key
  key_rotation_days: 90
  
# Authentication
auth:
  enabled: true
  jwt_secret: ${JWT_SECRET}
  access_token_expire_minutes: 30
  refresh_token_expire_days: 7
  enable_sso: true
  sso_provider: saml
  
# PACS Integration
pacs:
  enabled: true
  ae_title: the platform
  host: pacs.hospital.org
  port: 11112
  enable_tls: true
  query_retrieve_level: STUDY
  
# Monitoring
monitoring:
  enable_prometheus: true
  prometheus_port: 9090
  enable_tracing: true
  tracing_endpoint: http://jaeger:14268/api/traces
  log_level: INFO
  
# Compliance
compliance:
  enable_hipaa: true
  enable_gdpr: true
  audit_log_path: /logs/audit.log
  retention_days: 2555  # 7 years
  enable_anonymization: true
```

### Environment Variables

**Core**:
- `MODEL_PATH`: Path to model file
- `CUDA_VISIBLE_DEVICES`: GPU IDs (e.g., "0,1,2,3")
- `ENABLE_MULTI_GPU`: Enable multi-GPU (true/false)

**Security**:
- `JWT_SECRET`: JWT signing secret (required)
- `ENCRYPTION_KEY`: Master encryption key
- `TLS_CERT_PATH`: TLS certificate path
- `TLS_KEY_PATH`: TLS private key path

**Caching**:
- `REDIS_HOST`: Redis hostname
- `REDIS_PORT`: Redis port (default: 6379)
- `REDIS_PASSWORD`: Redis password

**Storage**:
- `AWS_ACCESS_KEY_ID`: AWS access key
- `AWS_SECRET_ACCESS_KEY`: AWS secret key
- `S3_BUCKET`: S3 bucket name

**PACS**:
- `PACS_HOST`: PACS server hostname
- `PACS_PORT`: PACS server port
- `PACS_AE_TITLE`: Application Entity title

---

## Security Setup

### TLS/SSL Configuration

**Generate Self-Signed Certificate** (testing only):

```bash
openssl req -x509 -newkey rsa:4096 \
  -keyout server.key -out server.crt \
  -days 365 -nodes \
  -subj "/CN=the platform.hospital.org"
```

**Use Let's Encrypt** (production):

```bash
# Install certbot
sudo apt install certbot -y

# Generate certificate
sudo certbot certonly --standalone \
  -d the platform.hospital.org \
  --email admin@hospital.org \
  --agree-tos

# Certificates at: /etc/letsencrypt/live/the platform.hospital.org/
```

**Configure TLS**:

```yaml
security:
  enable_tls: true
  tls_cert: /etc/letsencrypt/live/the platform.hospital.org/fullchain.pem
  tls_key: /etc/letsencrypt/live/the platform.hospital.org/privkey.pem
  tls_version: TLSv1.3
```

### Encryption Key Management

**Generate Master Key**:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

**Store Securely**:

```bash
# Option 1: File (restrict permissions)
echo "YOUR_KEY_HERE" > /etc/the platform/keys/master.key
chmod 600 /etc/the platform/keys/master.key
chown the platform:the platform /etc/the platform/keys/master.key

# Option 2: Environment variable
export ENCRYPTION_KEY="YOUR_KEY_HERE"

# Option 3: Kubernetes Secret
kubectl create secret generic the platform-keys \
  --from-literal=master-key=YOUR_KEY_HERE \
  -n the platform
```

### Authentication Setup

**JWT Secret**:

```bash
# Generate strong secret
openssl rand -hex 32

# Set environment variable
export JWT_SECRET="your_generated_secret"
```

**SSO Integration** (SAML):

```yaml
auth:
  enable_sso: true
  sso_provider: saml
  saml_metadata_url: https://idp.hospital.org/metadata
  saml_entity_id: the platform.hospital.org
  saml_acs_url: https://the platform.hospital.org/auth/saml/acs
```

**Create Admin User**:

```bash
python -m src.streaming.cli create-user \
  --username admin \
  --email admin@hospital.org \
  --role admin \
  --password-prompt
```

### Firewall Configuration

**Allow Required Ports**:

```bash
# HTTP/HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# API
sudo ufw allow 8000/tcp

# WebSocket
sudo ufw allow 8001/tcp

# Prometheus (internal only)
sudo ufw allow from 10.0.0.0/8 to any port 9090

# PACS (internal only)
sudo ufw allow from 10.0.0.0/8 to any port 11112

# Enable firewall
sudo ufw enable
```

---

## PACS Integration

### DICOM Configuration

**Configure PACS Connection**:

```yaml
pacs:
  enabled: true
  ae_title: the platform
  host: pacs.hospital.org
  port: 11112
  called_ae_title: PACS_SERVER
  enable_tls: true
  tls_cert: /etc/the platform/certs/dicom-client.crt
  tls_key: /etc/the platform/certs/dicom-client.key
  query_retrieve_level: STUDY
  timeout_seconds: 30
  max_pdu_length: 16384
```

**Test PACS Connection**:

```bash
python -m src.streaming.cli test-pacs \
  --host pacs.hospital.org \
  --port 11112 \
  --ae-title the platform \
  --called-ae-title PACS_SERVER
```

### PACS Server Configuration

**Add the platform to PACS Whitelist**:

1. Login to PACS admin console
2. Navigate to AE Title management
3. Add new AE Title:
   - **AE Title**: the platform
   - **Hostname**: the platform.hospital.org
   - **Port**: 11112
   - **Permissions**: Query, Retrieve, Store
4. Save and restart PACS service

**Verify Configuration**:

```bash
# From PACS server, test echo
dcmecho the platform@the platform.hospital.org:11112
```

---

## Monitoring

### Prometheus Setup

**Scrape Configuration** (`prometheus.yml`):

```yaml
scrape_configs:
  - job_name: 'the platform-streaming'
    static_configs:
      - targets: ['the platform.hospital.org:9090']
    scrape_interval: 15s
    metrics_path: /metrics
```

**Key Metrics**:

- `the platform_processing_time_seconds`: Slide processing time
- `the platform_throughput_patches_per_second`: Processing throughput
- `the platform_gpu_memory_used_bytes`: GPU memory usage
- `the platform_gpu_utilization_percent`: GPU utilization
- `the platform_cache_hit_rate`: Cache hit rate
- `the platform_active_slides`: Currently processing slides
- `the platform_queue_length`: Queued slides

### Grafana Dashboards

**Import Dashboard**:

1. Login to Grafana
2. Navigate to Dashboards → Import
3. Upload `grafana/the platform-dashboard.json`
4. Select Prometheus data source
5. Click Import

**Dashboard Panels**:

- Processing time trends
- GPU utilization and memory
- Throughput (patches/second)
- Cache performance
- Queue length
- Error rates
- System health

### Alerting Rules

**Prometheus Alerts** (`alerts.yml`):

```yaml
groups:
  - name: the platform
    interval: 30s
    rules:
      - alert: HighProcessingTime
        expr: the platform_processing_time_seconds > 60
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Slide processing time exceeds 60 seconds"
          
      - alert: HighGPUMemory
        expr: the platform_gpu_memory_used_bytes / the platform_gpu_memory_total_bytes > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "GPU memory usage above 90%"
          
      - alert: ServiceDown
        expr: up{job="the platform-streaming"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "the platform service is down"
```

### Log Management

**Centralized Logging** (ELK Stack):

```yaml
# Filebeat configuration
filebeat.inputs:
  - type: log
    enabled: true
    paths:
      - /var/log/the platform/*.log
    json.keys_under_root: true
    json.add_error_key: true
    
output.elasticsearch:
  hosts: ["elasticsearch:9200"]
  index: "the platform-%{+yyyy.MM.dd}"
```

**Log Rotation**:

```bash
# /etc/logrotate.d/the platform
/var/log/the platform/*.log {
    daily
    rotate 30
    compress
    delaycompress
    notifempty
    create 0640 the platform the platform
    sharedscripts
    postrotate
        systemctl reload the platform-streaming
    endscript
}
```

---

## Backup & Recovery

### Backup Strategy

**What to Backup**:

1. **Configuration files**: `/etc/the platform/`
2. **Encryption keys**: `/etc/the platform/keys/`
3. **Audit logs**: `/var/log/the platform/audit.log`
4. **Model files**: `/models/`
5. **Database** (if using PostgreSQL for audit logs)

**Backup Script**:

```bash
#!/bin/bash
# /usr/local/bin/the platform-backup.sh

BACKUP_DIR="/backup/the platform"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/the platform_backup_$DATE.tar.gz"

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup files
tar -czf $BACKUP_FILE \
  /etc/the platform/ \
  /var/log/the platform/audit.log \
  /models/

# Backup database (if applicable)
pg_dump the platform_audit > $BACKUP_DIR/audit_db_$DATE.sql

# Encrypt backup
gpg --encrypt --recipient admin@hospital.org $BACKUP_FILE

# Upload to S3
aws s3 cp $BACKUP_FILE.gpg s3://the platform-backups/

# Cleanup old backups (keep 30 days)
find $BACKUP_DIR -name "*.tar.gz*" -mtime +30 -delete

echo "Backup completed: $BACKUP_FILE.gpg"
```

**Cron Schedule**:

```bash
# Daily backup at 2 AM
0 2 * * * /usr/local/bin/the platform-backup.sh
```

### Disaster Recovery

**Recovery Procedure**:

1. **Restore Configuration**:
```bash
# Download backup
aws s3 cp s3://the platform-backups/the platform_backup_YYYYMMDD.tar.gz.gpg .

# Decrypt
gpg --decrypt the platform_backup_YYYYMMDD.tar.gz.gpg > the platform_backup.tar.gz

# Extract
tar -xzf the platform_backup.tar.gz -C /
```

2. **Restore Database**:
```bash
psql the platform_audit < audit_db_YYYYMMDD.sql
```

3. **Restart Service**:
```bash
systemctl restart the platform-streaming
```

4. **Verify**:
```bash
curl https://the platform.hospital.org/health
```

**RTO/RPO**:
- **Recovery Time Objective (RTO)**: <1 hour
- **Recovery Point Objective (RPO)**: <24 hours

---

## Troubleshooting

### Common Issues

**1. Service Won't Start**

```bash
# Check logs
journalctl -u the platform-streaming -n 100

# Common causes:
# - Missing model file
# - GPU not available
# - Port already in use
# - Configuration error

# Verify GPU
nvidia-smi

# Check port
netstat -tulpn | grep 8000

# Validate config
python -m src.streaming.cli validate-config --config /etc/the platform/config.yaml
```

**2. Slow Processing**

```bash
# Check GPU utilization
nvidia-smi -l 1

# Check system resources
htop

# Check network (if using PACS)
ping pacs.hospital.org

# Review metrics
curl http://localhost:9090/metrics | grep processing_time

# Common causes:
# - GPU memory full (reduce batch size)
# - Network latency (check PACS connection)
# - CPU bottleneck (increase workers)
# - Disk I/O (use faster storage)
```

**3. PACS Connection Failed**

```bash
# Test connectivity
telnet pacs.hospital.org 11112

# Test DICOM echo
python -m src.streaming.cli test-pacs

# Check firewall
sudo ufw status

# Verify AE Title configuration
# Check PACS logs for rejected connections
```

**4. High Memory Usage**

```bash
# Check GPU memory
nvidia-smi

# Check system memory
free -h

# Reduce batch size
# Edit config: processing.batch_size = 32

# Enable FP16
# Edit config: model.enable_fp16 = true

# Restart service
systemctl restart the platform-streaming
```

**5. Authentication Errors**

```bash
# Verify JWT secret is set
echo $JWT_SECRET

# Check token expiration
# Tokens expire after 30 minutes by default

# Test authentication
curl -X POST https://the platform.hospital.org/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"password"}'

# Check audit logs
tail -f /var/log/the platform/audit.log | grep AUTH
```

### Diagnostic Commands

```bash
# System health check
python -m src.streaming.cli health-check

# GPU diagnostics
python -m src.streaming.cli gpu-info

# PACS connectivity test
python -m src.streaming.cli test-pacs

# Configuration validation
python -m src.streaming.cli validate-config

# Performance benchmark
python -m src.streaming.cli benchmark --slides 10

# Cache statistics
python -m src.streaming.cli cache-stats

# Audit log query
python -m src.streaming.cli audit-query --user admin --days 7
```

---

## Maintenance

### Regular Maintenance Tasks

**Daily**:
- Monitor system health dashboard
- Review error logs
- Check processing queue length
- Verify backup completion

**Weekly**:
- Review performance metrics
- Check disk space usage
- Update security patches
- Test disaster recovery

**Monthly**:
- Rotate encryption keys (if policy requires)
- Review audit logs for anomalies
- Update model if new version available
- Capacity planning review

**Quarterly**:
- Full system backup test
- Security audit
- Performance benchmarking
- User access review

### Update Procedure

**Minor Updates** (patch releases):

```bash
# Docker
docker pull the platform/streaming:latest
docker stop the platform-streaming
docker rm the platform-streaming
docker run -d ... the platform/streaming:latest

# Kubernetes
kubectl set image deployment/the platform-streaming \
  the platform=the platform/streaming:latest \
  -n the platform

# Bare metal
cd /opt/the platform/streaming
git pull
pip install -r requirements.txt
systemctl restart the platform-streaming
```

**Major Updates** (version upgrades):

1. **Backup current system**
2. **Test in staging environment**
3. **Schedule maintenance window**
4. **Notify users**
5. **Perform update**
6. **Run validation tests**
7. **Monitor for issues**
8. **Rollback if needed**

### Model Updates

```bash
# Download new model
wget https://models.the platform.ai/v2/the platform_v2.pth -O /models/the platform_v2.pth

# Update configuration
# Edit config: model.path = /models/the platform_v2.pth

# Restart service
systemctl restart the platform-streaming

# Verify
curl https://the platform.hospital.org/health
```

---

## Performance Tuning

### GPU Optimization

**Batch Size Tuning**:

```yaml
# Start with default
processing:
  batch_size: 64

# If GPU memory < 16GB
processing:
  batch_size: 32

# If GPU memory > 32GB
processing:
  batch_size: 128
```

**Multi-GPU Configuration**:

```yaml
gpu:
  ids: [0, 1, 2, 3]
  enable_multi_gpu: true
  strategy: data_parallel  # or pipeline_parallel
```

**TensorRT Optimization**:

```yaml
model:
  enable_tensorrt: true
  tensorrt_precision: fp16  # or int8
  tensorrt_workspace_gb: 4
```

### Caching Optimization

**Redis Configuration**:

```yaml
cache:
  enabled: true
  backend: redis
  redis_host: redis
  redis_port: 6379
  redis_max_connections: 50
  ttl_seconds: 86400
  compression: true
  compression_level: 6
```

**Cache Warming** (preload frequently accessed slides):

```bash
python -m src.streaming.cli cache-warm \
  --slide-list frequent_slides.txt \
  --workers 4
```

### Network Optimization

**PACS Connection Pooling**:

```yaml
pacs:
  connection_pool_size: 10
  connection_timeout: 30
  max_retries: 3
  retry_delay: 5
```

**Bandwidth Optimization**:

```yaml
pacs:
  enable_compression: true
  transfer_syntax: JPEG2000Lossless
  max_concurrent_transfers: 5
```

### Database Optimization

**PostgreSQL Tuning** (for audit logs):

```sql
-- Increase shared buffers
ALTER SYSTEM SET shared_buffers = '4GB';

-- Increase work memory
ALTER SYSTEM SET work_mem = '256MB';

-- Enable parallel queries
ALTER SYSTEM SET max_parallel_workers_per_gather = 4;

-- Reload configuration
SELECT pg_reload_conf();
```

---

## Appendix

### CLI Reference

```bash
# Health check
python -m src.streaming.cli health-check

# Create user
python -m src.streaming.cli create-user --username USER --role ROLE

# Test PACS
python -m src.streaming.cli test-pacs --host HOST --port PORT

# Benchmark
python -m src.streaming.cli benchmark --slides N

# Cache management
python -m src.streaming.cli cache-stats
python -m src.streaming.cli cache-clear

# Audit logs
python -m src.streaming.cli audit-query --user USER --days N
python -m src.streaming.cli audit-export --output FILE

# Configuration
python -m src.streaming.cli validate-config --config FILE
python -m src.streaming.cli show-config
```

### API Endpoints

**Health & Status**:
- `GET /health` - Health check
- `GET /status` - System status
- `GET /metrics` - Prometheus metrics

**Authentication**:
- `POST /auth/login` - Login
- `POST /auth/refresh` - Refresh token
- `POST /auth/logout` - Logout

**Processing**:
- `POST /process` - Process slide
- `GET /process/{id}` - Get processing status
- `DELETE /process/{id}` - Cancel processing

**Results**:
- `GET /results/{id}` - Get results
- `GET /results/{id}/heatmap` - Get attention heatmap
- `POST /results/{id}/report` - Generate report

**PACS**:
- `GET /pacs/worklist` - Get worklist
- `POST /pacs/retrieve` - Retrieve study
- `POST /pacs/store` - Store results

### Support Resources

**Documentation**: https://docs.the platform.ai  
**Technical Support**: support@the platform.ai | 1-800-the platform  
**Security Issues**: security@the platform.ai  
**Community Forum**: https://community.the platform.ai

---

**Document Version**: 1.0.0  
**Last Updated**: 2026-04-27  
**For**: System Administrators, DevOps Engineers, IT Staff
