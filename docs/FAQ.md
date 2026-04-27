# HistoCore Frequently Asked Questions (FAQ)

Common questions about HistoCore Real-Time WSI Streaming.

## Table of Contents

- [General](#general)
- [Installation](#installation)
- [Configuration](#configuration)
- [Performance](#performance)
- [GPU and Hardware](#gpu-and-hardware)
- [PACS Integration](#pacs-integration)
- [Security and Compliance](#security-and-compliance)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)

## General

### What is HistoCore?

HistoCore is a production-grade computational pathology framework providing real-time whole-slide image (WSI) streaming and analysis. It processes gigapixel slides in under 30 seconds using GPU-accelerated streaming architecture with progressive confidence updates.

### What makes HistoCore different from other solutions?

- **Real-time processing**: <30 second processing for gigapixel slides
- **Streaming architecture**: Process tiles as they load, not after full download
- **Progressive confidence**: Real-time confidence updates with early stopping
- **Production-ready**: Complete PACS integration, monitoring, and deployment tools
- **Clinical workflow**: DICOM/FHIR support, clinical reporting, regulatory compliance

### What are the main use cases?

- **Clinical deployment**: Real-time pathology analysis in hospital workflows
- **Research**: Computational pathology research with attention-based MIL models
- **PACS integration**: Automated WSI processing from hospital PACS systems
- **Live demos**: Hospital demonstrations with real-time confidence visualization
- **Batch processing**: High-throughput slide processing for research studies

### Is HistoCore open source?

Yes, HistoCore is open source under the MIT license. You can use, modify, and distribute it freely.

### What WSI formats are supported?

- **.svs** (Aperio)
- **.tiff** (Generic TIFF)
- **.ndpi** (Hamamatsu)
- **DICOM** (DICOM WSI)
- Other formats supported by OpenSlide

### What models are included?

- **AttentionMIL**: Gated attention mechanism
- **CLAM**: Clustering-constrained attention
- **TransMIL**: Transformer-based MIL
- **Custom models**: Easy integration of custom architectures

## Installation

### What are the system requirements?

**Minimum**:
- CPU: 4 cores
- RAM: 16GB
- GPU: NVIDIA GPU with 8GB VRAM (RTX 3080+)
- Storage: 100GB SSD
- OS: Linux (Ubuntu 20.04+) or Windows Server 2019+

**Recommended**:
- CPU: 16+ cores
- RAM: 64GB+
- GPU: NVIDIA GPU with 16GB+ VRAM (RTX 4090, A100)
- Storage: 500GB+ NVMe SSD
- OS: Linux (Ubuntu 22.04+)

### Do I need a GPU?

Yes, a NVIDIA GPU with CUDA support is required for production use. CPU-only mode is available for development/testing but is significantly slower.

### How do I install HistoCore?

```bash
# Clone repository
git clone https://github.com/histocore/histocore.git
cd histocore

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

See [Installation Guide](INSTALLATION.md) for detailed instructions.

### Can I use Docker?

Yes, Docker is the recommended deployment method:

```bash
# Build image
./scripts/docker-build.sh

# Start services
docker-compose up -d
```

See [Docker Deployment](deployment/DEPLOYMENT_GUIDE.md#docker-deployment) for details.

### What Python version is required?

Python 3.9 or higher is required. Python 3.10+ is recommended.

### How do I install CUDA?

See [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) for your platform.

For Ubuntu:
```bash
sudo apt-get install nvidia-cuda-toolkit
```

## Configuration

### How do I configure HistoCore?

Configuration can be done via:
1. **Environment variables**: `export HISTOCORE_BATCH_SIZE=32`
2. **Configuration files**: `config.yaml`
3. **Command-line arguments**: `--batch-size 32`

See [Configuration Guide](deployment/CONFIGURATION_GUIDE.md) for complete reference.

### What is the default configuration?

```yaml
streaming:
  tile_size: 1024
  batch_size: 32
  memory_budget_gb: 2.0
  target_time: 30.0
  confidence_threshold: 0.95

gpu:
  device_ids: [0]
  enable_amp: true
```

### How do I optimize for my hardware?

**For high-end GPU (RTX 4090, A100)**:
```yaml
streaming:
  batch_size: 128
  memory_budget_gb: 16.0
gpu:
  device_ids: [0, 1, 2, 3]
```

**For limited memory**:
```yaml
streaming:
  batch_size: 16
  memory_budget_gb: 4.0
gpu:
  memory_fraction: 0.8
```

See [Performance Tuning](deployment/CONFIGURATION_GUIDE.md#performance-tuning) for details.

### Can I update configuration without restart?

Yes, hot reload is supported:

```bash
# Update config
kubectl edit configmap histocore-config -n histocore

# Trigger reload
curl -X POST http://localhost:8001/admin/config/reload \
  -H "Authorization: Bearer <token>"
```

### Where are configuration files located?

- **Docker**: `/app/config.yaml`
- **Kubernetes**: ConfigMap `histocore-config`
- **Local**: `config.yaml` in project root

## Performance

### How fast is HistoCore?

- **Processing time**: <30 seconds for 100K+ patch gigapixel slides
- **Throughput**: >3000 patches/second on RTX 4090
- **Memory usage**: <2GB RAM during processing
- **Latency**: <100ms for real-time updates

### What affects processing speed?

1. **GPU performance**: Faster GPU = faster processing
2. **Batch size**: Larger batches = higher throughput
3. **Memory**: More memory = larger batches
4. **Storage**: Faster storage = faster tile loading
5. **Network**: Faster network = faster PACS retrieval

### How can I improve performance?

1. **Increase batch size**:
```yaml
streaming:
  batch_size: 64
```

2. **Enable AMP**:
```yaml
gpu:
  enable_amp: true
```

3. **Use multiple GPUs**:
```yaml
gpu:
  device_ids: [0, 1, 2, 3]
```

4. **Optimize workers**:
```yaml
performance:
  processing:
    num_workers: 8
```

See [Performance Tuning](deployment/CONFIGURATION_GUIDE.md#performance-tuning) for more.

### What is early stopping?

Early stopping terminates processing when confidence reaches a threshold, reducing processing time:

```yaml
processing:
  early_stopping:
    enabled: true
    confidence_threshold: 0.95
    min_patches: 1000
```

### How accurate is early stopping?

Early stopping maintains 95%+ accuracy compared to full processing while reducing processing time by 30-50% on average.

## GPU and Hardware

### What GPUs are supported?

Any NVIDIA GPU with CUDA support. Recommended:
- **Development**: RTX 3080, RTX 4070 (8-12GB VRAM)
- **Production**: RTX 4090, A100, H100 (16-80GB VRAM)

### Can I use multiple GPUs?

Yes, multi-GPU support is built-in:

```yaml
gpu:
  device_ids: [0, 1, 2, 3]
  strategy: data_parallel
```

### Can I use AMD GPUs?

No, CUDA is required. AMD ROCm support is not currently available.

### Can I run without a GPU?

Yes, but performance will be significantly slower (10-100x). CPU-only mode is suitable for development/testing only.

### How much GPU memory do I need?

- **Minimum**: 8GB (RTX 3080)
- **Recommended**: 16GB+ (RTX 4090, A100)
- **Optimal**: 24GB+ (RTX 4090, A100 40GB)

### What if I run out of GPU memory?

HistoCore automatically reduces batch size on OOM:

```yaml
gpu:
  memory_management:
    oom_retry: true
    oom_batch_reduction: 0.5
```

### How much system RAM do I need?

- **Minimum**: 16GB
- **Recommended**: 32GB+
- **Optimal**: 64GB+

### What storage is recommended?

- **Development**: SATA SSD (500 MB/s)
- **Production**: NVMe SSD (3000+ MB/s)
- **Optimal**: NVMe RAID (10000+ MB/s)

## PACS Integration

### What is PACS integration?

PACS (Picture Archiving and Communication System) integration allows HistoCore to:
- Query PACS for WSI studies
- Retrieve slides automatically
- Process slides in real-time
- Store results back to PACS

### What PACS systems are supported?

HistoCore supports any DICOM-compliant PACS:
- GE Healthcare Centricity
- Philips IntelliSpace
- Siemens syngo
- Agfa Enterprise Imaging
- Generic DICOM PACS

### How do I configure PACS integration?

```yaml
pacs:
  enabled: true
  remote_host: pacs.hospital.org
  remote_port: 104
  remote_ae_title: PACS_SERVER
  ae_title: HISTOCORE
```

See [PACS Configuration](deployment/CONFIGURATION_GUIDE.md#pacs-integration) for details.

### Is PACS integration secure?

Yes, HistoCore supports:
- **TLS 1.3 encryption**
- **Mutual authentication**
- **HIPAA-compliant audit logging**
- **Access control**

### Can I test PACS integration without a real PACS?

Yes, use a PACS simulator:

```bash
# Install dcmtk
sudo apt-get install dcmtk

# Start PACS simulator
storescp -v -aet PACS_SERVER 104
```

### How do I troubleshoot PACS connection issues?

```bash
# Test PACS connectivity
python -m src.pacs.test_connection

# Test DICOM echo
dcmecho -v PACS_SERVER 104

# Check logs
docker logs histocore-streaming | grep PACS
```

See [PACS Troubleshooting](TROUBLESHOOTING.md#pacs-integration-issues) for more.

## Security and Compliance

### Is HistoCore HIPAA compliant?

HistoCore provides HIPAA-compliant features:
- **Encryption**: TLS 1.3 in transit, AES-256 at rest
- **Audit logging**: 7-year retention
- **Access control**: RBAC with OAuth 2.0
- **Data protection**: Secure deletion, anonymization

However, HIPAA compliance requires proper deployment and configuration. See [Security Guide](deployment/CONFIGURATION_GUIDE.md#security-configuration).

### Is HistoCore FDA approved?

HistoCore is a research framework and is not FDA approved. For clinical use, you must:
1. Validate the system for your specific use case
2. Follow FDA 510(k) pathway for medical devices
3. Implement required quality management systems

See [Regulatory Compliance](deployment/CONFIGURATION_GUIDE.md#security-configuration) for guidance.

### How is data encrypted?

- **In transit**: TLS 1.3 encryption
- **At rest**: AES-256-GCM encryption
- **PACS**: DICOM TLS encryption

```yaml
security:
  encryption:
    at_rest:
      enabled: true
      algorithm: AES-256-GCM
    in_transit:
      tls_enabled: true
      tls_version: TLSv1.3
```

### How is authentication handled?

OAuth 2.0 with JWT tokens:

```bash
# Get token
curl -X POST https://api.histocore.ai/auth/token \
  -d '{"client_id":"...","client_secret":"..."}'

# Use token
curl -H "Authorization: Bearer <token>" \
  https://api.histocore.ai/v1/process/wsi
```

### What audit logging is provided?

All operations are logged:
- Authentication/authorization
- Data access
- Configuration changes
- Admin actions

Logs are retained for 7 years (HIPAA requirement).

### Can I use HistoCore in the cloud?

Yes, cloud deployment is supported:
- **AWS**: EKS with GPU instances
- **Azure**: AKS with GPU VMs
- **GCP**: GKE with GPU nodes

See [Cloud Deployment](deployment/DEPLOYMENT_GUIDE.md#cloud-deployment) for details.

## Deployment

### What deployment options are available?

1. **Docker**: Single server deployment
2. **Kubernetes**: Multi-server production deployment
3. **Cloud**: AWS/Azure/GCP managed deployment

See [Deployment Guide](deployment/DEPLOYMENT_GUIDE.md) for comparison.

### Which deployment method should I use?

- **Development**: Docker
- **Production (single server)**: Docker with docker-compose
- **Production (multi-server)**: Kubernetes
- **Enterprise**: Cloud (AWS/Azure/GCP)

### How do I deploy to Kubernetes?

```bash
cd k8s
./deploy.sh
```

See [Kubernetes Deployment](deployment/DEPLOYMENT_GUIDE.md#kubernetes-deployment) for details.

### How do I deploy to AWS?

```bash
cd cloud/aws
./deploy.sh production us-west-2
```

See [AWS Deployment](deployment/DEPLOYMENT_GUIDE.md#aws-deployment) for details.

### How do I scale HistoCore?

**Horizontal scaling** (Kubernetes):
```bash
kubectl scale deployment histocore-streaming --replicas=10 -n histocore
```

**Auto-scaling**:
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
spec:
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        averageUtilization: 70
```

### How do I monitor HistoCore?

Built-in monitoring with:
- **Prometheus**: Metrics collection
- **Grafana**: Visualization dashboards
- **Jaeger**: Distributed tracing
- **Alertmanager**: Alert notifications

```bash
# Start monitoring stack
cd monitoring
docker-compose up -d

# Access Grafana
open http://localhost:3000
```

### How do I backup HistoCore?

```bash
# Backup Redis data
docker exec histocore-redis redis-cli BGSAVE

# Backup persistent volumes
kubectl get pv
# Use volume snapshots

# Backup configuration
kubectl get configmap histocore-config -o yaml > backup/config.yaml
```

### How do I update HistoCore?

**Docker**:
```bash
docker-compose pull
docker-compose up -d
```

**Kubernetes**:
```bash
kubectl set image deployment/histocore-streaming \
  histocore-streaming=histocore/streaming:v1.1.0 -n histocore
```

## Troubleshooting

### Where can I find logs?

**Docker**:
```bash
docker logs histocore-streaming
```

**Kubernetes**:
```bash
kubectl logs deployment/histocore-streaming -n histocore
```

**Application logs**:
```bash
tail -f /var/log/histocore/app.log
```

### How do I enable debug logging?

```bash
export HISTOCORE_LOG_LEVEL=DEBUG
docker-compose restart histocore-streaming
```

### How do I report a bug?

1. **Check existing issues**: https://github.com/histocore/histocore/issues
2. **Collect diagnostics**:
```bash
python -m src.streaming.diagnostics > diagnostics.txt
```
3. **Create GitHub issue** with:
   - Description of the problem
   - Steps to reproduce
   - Diagnostics output
   - Logs (sanitize sensitive data)

### How do I get help?

- **Documentation**: [docs/](.)
- **Troubleshooting**: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **GitHub Issues**: https://github.com/histocore/histocore/issues
- **Email**: support@histocore.ai
- **Slack**: https://histocore.slack.com

### What if my question isn't answered here?

1. Check the [full documentation](.)
2. Search [GitHub issues](https://github.com/histocore/histocore/issues)
3. Ask on [Slack](https://histocore.slack.com)
4. Email support@histocore.ai

## Contributing

### How can I contribute?

Contributions are welcome! See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

### How do I submit a pull request?

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### What should I work on?

Check the [GitHub issues](https://github.com/histocore/histocore/issues) for:
- **Good first issues**: Easy tasks for new contributors
- **Help wanted**: Tasks that need contributors
- **Feature requests**: New features to implement

### How do I run tests?

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_streaming.py

# Run with coverage
pytest --cov=src tests/
```

## License

### What license is HistoCore under?

MIT License - you can use, modify, and distribute HistoCore freely.

### Can I use HistoCore commercially?

Yes, the MIT license allows commercial use.

### Do I need to cite HistoCore?

While not required, citations are appreciated:

```bibtex
@software{histocore2024,
  title={HistoCore: Real-Time WSI Streaming Framework},
  author={HistoCore Contributors},
  year={2024},
  url={https://github.com/histocore/histocore}
}
```

## Support

### How do I get support?

- **Documentation**: [docs/](.)
- **GitHub Issues**: https://github.com/histocore/histocore/issues
- **Email**: support@histocore.ai
- **Slack**: https://histocore.slack.com

### Is commercial support available?

Yes, commercial support is available for:
- **Installation and deployment**
- **Custom development**
- **Training and consulting**
- **Priority support**

Contact sales@histocore.ai for details.

### What is the SLA for support?

**Community support** (free):
- Best effort response
- GitHub issues and Slack

**Enterprise support** (paid):
- 24/7 availability
- <4 hour response time
- Dedicated support engineer

Contact sales@histocore.ai for enterprise support.
