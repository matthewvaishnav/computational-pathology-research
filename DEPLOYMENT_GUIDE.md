# HistoCore Deployment Guide

Complete guide for deploying HistoCore in production environments.

## 🚀 Quick Deployment

### 1. One-Click Setup
```bash
# Download and install
git clone https://github.com/matthewvaishnav/histocore.git
cd histocore
python install.py

# Launch
python histocore.py
```

### 2. Choose Interface
- **Desktop GUI**: Best for individual users
- **Web Interface**: Best for teams and remote access  
- **CLI**: Best for automation and scripting

## 📋 System Requirements

### Minimum Requirements
- **OS**: Windows 10+, macOS 10.15+, Ubuntu 18.04+
- **Python**: 3.9+
- **RAM**: 4GB
- **Storage**: 10GB free space
- **Network**: Internet for initial setup

### Recommended Requirements
- **RAM**: 16GB+ (for large WSI files)
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **Storage**: SSD with 50GB+ free space
- **CPU**: 8+ cores for batch processing

### Enterprise Requirements
- **RAM**: 32GB+
- **GPU**: Multiple GPUs for parallel processing
- **Storage**: High-speed NVMe SSD array
- **Network**: Gigabit ethernet for PACS integration

## 🏥 Clinical Deployment

### Hospital Integration

**PACS Connectivity**:
```bash
# Configure PACS settings
cp .kiro/pacs/config.development.yaml .kiro/pacs/config.production.yaml
# Edit with your PACS server details
```

**DICOM Compliance**:
- Supports DICOM C-FIND/C-MOVE/C-STORE
- Multi-vendor compatibility (GE, Philips, Siemens, Agfa)
- TLS 1.3 encryption
- HIPAA audit logging

**Regulatory Compliance**:
- FDA/CE marking support
- Risk management (ISO 14971)
- Audit trails and version control
- Privacy protection (HIPAA/GDPR)

### Security Configuration

**Network Security**:
```bash
# Enable HTTPS
export HISTOCORE_USE_HTTPS=true
export HISTOCORE_SSL_CERT=/path/to/cert.pem
export HISTOCORE_SSL_KEY=/path/to/key.pem

# Configure firewall
# Allow ports: 5000 (web), 8080 (API), 443 (HTTPS)
```

**Data Protection**:
- AES-256 encryption for data at rest
- TLS 1.3 for data in transit
- Role-based access controls
- Automatic session timeout

## 🐳 Docker Deployment

### Single Container
```bash
# Build image
docker build -t histocore:latest .

# Run container
docker run -d \
  --name histocore \
  -p 5000:5000 \
  -v /data/wsi:/app/data \
  -v /results:/app/results \
  histocore:latest
```

### Docker Compose
```yaml
version: '3.8'
services:
  histocore:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./data:/app/data
      - ./results:/app/results
    environment:
      - HISTOCORE_ENV=production
      - HISTOCORE_GPU=true
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## ☸️ Kubernetes Deployment

### Basic Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: histocore
spec:
  replicas: 3
  selector:
    matchLabels:
      app: histocore
  template:
    metadata:
      labels:
        app: histocore
    spec:
      containers:
      - name: histocore
        image: histocore:latest
        ports:
        - containerPort: 5000
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "16Gi"
            cpu: "8"
            nvidia.com/gpu: 1
---
apiVersion: v1
kind: Service
metadata:
  name: histocore-service
spec:
  selector:
    app: histocore
  ports:
  - port: 80
    targetPort: 5000
  type: LoadBalancer
```

### Helm Chart
```bash
# Install with Helm
helm repo add histocore https://charts.histocore.io
helm install histocore histocore/histocore \
  --set gpu.enabled=true \
  --set replicas=3 \
  --set storage.size=100Gi
```

## 🌐 Cloud Deployment

### AWS Deployment
```bash
# Deploy to ECS
aws ecs create-cluster --cluster-name histocore-cluster

# Deploy to EKS
eksctl create cluster --name histocore --region us-west-2 --nodegroup-name gpu-nodes --node-type p3.2xlarge
```

### Azure Deployment
```bash
# Deploy to ACI
az container create \
  --resource-group histocore-rg \
  --name histocore \
  --image histocore:latest \
  --ports 5000 \
  --gpu-count 1 \
  --gpu-sku V100
```

### Google Cloud Deployment
```bash
# Deploy to GKE
gcloud container clusters create histocore-cluster \
  --accelerator type=nvidia-tesla-v100,count=1 \
  --enable-autoscaling \
  --num-nodes=3
```

## 📊 Monitoring & Logging

### Prometheus Metrics
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'histocore'
    static_configs:
      - targets: ['histocore:5000']
    metrics_path: '/metrics'
```

### Grafana Dashboard
- Processing throughput (slides/hour)
- GPU utilization
- Memory usage
- Error rates
- Response times

### Centralized Logging
```bash
# ELK Stack integration
export HISTOCORE_LOG_LEVEL=INFO
export HISTOCORE_LOG_FORMAT=json
export HISTOCORE_ELASTICSEARCH_URL=http://elasticsearch:9200
```

## 🔧 Performance Tuning

### Optimization Script
```bash
# Run performance optimization
python optimize_performance.py

# Apply optimized config
cp optimized_config.json config/production.json
```

### GPU Optimization
```bash
# Enable mixed precision
export HISTOCORE_MIXED_PRECISION=true

# Enable model compilation
export HISTOCORE_COMPILE_MODEL=true

# Set memory fraction
export HISTOCORE_GPU_MEMORY_FRACTION=0.8
```

### Batch Processing
```bash
# Configure for high throughput
export HISTOCORE_BATCH_SIZE=64
export HISTOCORE_NUM_WORKERS=8
export HISTOCORE_PREFETCH_FACTOR=4
```

## 🔄 CI/CD Pipeline

### GitHub Actions
```yaml
name: Deploy HistoCore
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Build and push Docker image
      run: |
        docker build -t histocore:${{ github.sha }} .
        docker push histocore:${{ github.sha }}
    - name: Deploy to production
      run: |
        kubectl set image deployment/histocore histocore=histocore:${{ github.sha }}
```

### Automated Testing
```bash
# Run test suite before deployment
python test_wsi_processing.py
pytest tests/ --cov=src --cov-report=html
```

## 🚨 Troubleshooting

### Common Issues

**GPU Not Detected**:
```bash
# Check CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA drivers
# NVIDIA website: https://developer.nvidia.com/cuda-downloads
```

**Memory Errors**:
```bash
# Reduce batch size
export HISTOCORE_BATCH_SIZE=16

# Enable streaming mode
export HISTOCORE_STREAMING_MODE=true
```

**PACS Connection Issues**:
```bash
# Test DICOM connectivity
python -c "from pynetdicom import AE; ae = AE(); ae.add_requested_context('1.2.840.10008.1.1'); assoc = ae.associate('PACS_IP', 104)"
```

**Web Interface Not Loading**:
```bash
# Check port availability
netstat -an | grep 5000

# Check firewall settings
sudo ufw allow 5000
```

### Performance Issues

**Slow Processing**:
1. Enable GPU acceleration
2. Increase batch size
3. Use SSD storage
4. Add more RAM
5. Enable model compilation

**High Memory Usage**:
1. Reduce batch size
2. Enable streaming mode
3. Clear cache regularly
4. Use memory mapping

### Log Analysis
```bash
# View application logs
tail -f logs/histocore.log

# Search for errors
grep ERROR logs/histocore.log

# Monitor resource usage
htop
nvidia-smi -l 1
```

## 📞 Support

### Documentation
- **User Guide**: [USER_INTERFACES.md](USER_INTERFACES.md)
- **API Reference**: [docs/API_REFERENCE.md](docs/API_REFERENCE.md)
- **Architecture**: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

### Community
- **GitHub Issues**: https://github.com/matthewvaishnav/histocore/issues
- **Discussions**: https://github.com/matthewvaishnav/histocore/discussions
- **Wiki**: https://github.com/matthewvaishnav/histocore/wiki

### Enterprise Support
- **Professional Services**: Available for large deployments
- **Custom Integration**: PACS/EMR integration support
- **Training**: On-site training for clinical teams
- **SLA**: 24/7 support with guaranteed response times

## 🎯 Success Metrics

### Performance Targets
- **Processing Speed**: <10 seconds per slide
- **Throughput**: 100+ slides per hour
- **Accuracy**: >90% sensitivity, >80% specificity
- **Uptime**: 99.9% availability

### Monitoring KPIs
- Slides processed per day
- Average processing time
- Error rate (<1%)
- User satisfaction score
- System resource utilization

### ROI Metrics
- Cost per slide analysis
- Time savings vs manual review
- Diagnostic accuracy improvement
- Pathologist productivity increase

---

**Ready for Production**: HistoCore is enterprise-ready with comprehensive deployment options, monitoring, and support.