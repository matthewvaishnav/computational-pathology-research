# HistoCore Troubleshooting Guide

Comprehensive troubleshooting guide for HistoCore Real-Time WSI Streaming.

## Table of Contents

- [Quick Diagnostics](#quick-diagnostics)
- [Installation Issues](#installation-issues)
- [GPU Issues](#gpu-issues)
- [Processing Issues](#processing-issues)
- [Memory Issues](#memory-issues)
- [Network Issues](#network-issues)
- [PACS Integration Issues](#pacs-integration-issues)
- [Performance Issues](#performance-issues)
- [Deployment Issues](#deployment-issues)
- [Error Messages](#error-messages)

## Quick Diagnostics

### Health Check

```bash
# Basic health check
curl http://localhost:8000/health

# Detailed health check
curl -H "Authorization: Bearer <token>" \
  http://localhost:8000/health/detailed

# Check all components
python -m src.streaming.health_check
```

### System Information

```bash
# Check GPU
nvidia-smi

# Check Python environment
python --version
pip list | grep torch

# Check Docker
docker --version
docker ps

# Check Kubernetes
kubectl version
kubectl get pods -n histocore
```

### Log Analysis

```bash
# Docker logs
docker logs histocore-streaming --tail 100

# Kubernetes logs
kubectl logs deployment/histocore-streaming -n histocore --tail 100

# Application logs
tail -f /var/log/histocore/app.log

# Error logs only
grep ERROR /var/log/histocore/app.log
```

## Installation Issues

### Issue: pip install fails

**Symptoms**:
```
ERROR: Could not find a version that satisfies the requirement torch>=2.0
```

**Solutions**:

1. **Update pip**:
```bash
python -m pip install --upgrade pip
```

2. **Install PyTorch separately**:
```bash
# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

3. **Use conda**:
```bash
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
```

### Issue: OpenSlide not found

**Symptoms**:
```
ImportError: cannot import name 'OpenSlide' from 'openslide'
```

**Solutions**:

**Linux**:
```bash
sudo apt-get update
sudo apt-get install openslide-tools python3-openslide
pip install openslide-python
```

**macOS**:
```bash
brew install openslide
pip install openslide-python
```

**Windows**:
```powershell
# Download OpenSlide binaries from https://openslide.org/download/
# Extract to C:\OpenSlide
# Add to PATH
$env:PATH += ";C:\OpenSlide\bin"
pip install openslide-python
```

### Issue: CUDA not available

**Symptoms**:
```python
>>> import torch
>>> torch.cuda.is_available()
False
```

**Solutions**:

1. **Check NVIDIA driver**:
```bash
nvidia-smi
# Should show driver version and GPU info
```

2. **Install CUDA toolkit**:
```bash
# Ubuntu
sudo apt-get install nvidia-cuda-toolkit

# Check version
nvcc --version
```

3. **Reinstall PyTorch with CUDA**:
```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

4. **Verify installation**:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
```

## GPU Issues

### Issue: GPU not detected

**Symptoms**:
```
RuntimeError: No CUDA GPUs are available
```

**Diagnostics**:
```bash
# Check GPU visibility
nvidia-smi

# Check CUDA devices
python -c "import torch; print(torch.cuda.device_count())"

# Check Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi

# Check Kubernetes GPU
kubectl get nodes -o json | jq '.items[].status.allocatable."nvidia.com/gpu"'
```

**Solutions**:

1. **Docker**: Install NVIDIA Docker runtime
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

2. **Kubernetes**: Install NVIDIA device plugin
```bash
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/main/nvidia-device-plugin.yml
```

3. **Environment variable**: Set CUDA_VISIBLE_DEVICES
```bash
export CUDA_VISIBLE_DEVICES=0
```

### Issue: GPU out of memory

**Symptoms**:
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Diagnostics**:
```bash
# Check GPU memory usage
nvidia-smi

# Monitor GPU memory
watch -n 1 nvidia-smi

# Check application memory usage
curl http://localhost:8002/metrics | grep gpu_memory
```

**Solutions**:

1. **Reduce batch size**:
```bash
export HISTOCORE_BATCH_SIZE=16
docker-compose restart histocore-streaming
```

2. **Enable automatic batch size reduction**:
```yaml
# config.yaml
gpu:
  memory_management:
    oom_retry: true
    oom_batch_reduction: 0.5
```

3. **Enable garbage collection**:
```yaml
performance:
  memory:
    gc_enabled: true
    gc_interval: 50
```

4. **Use gradient checkpointing**:
```yaml
gpu:
  gradient_checkpointing: true
```

5. **Clear GPU cache manually**:
```python
import torch
torch.cuda.empty_cache()
```

### Issue: Low GPU utilization

**Symptoms**:
- GPU utilization <50%
- Slow processing despite available GPU

**Diagnostics**:
```bash
# Monitor GPU utilization
nvidia-smi dmon -s u

# Check batch size
curl http://localhost:8002/metrics | grep batch_size

# Check processing throughput
curl http://localhost:8002/metrics | grep throughput
```

**Solutions**:

1. **Increase batch size**:
```bash
export HISTOCORE_BATCH_SIZE=64
```

2. **Enable AMP (Automatic Mixed Precision)**:
```yaml
gpu:
  enable_amp: true
```

3. **Increase number of workers**:
```yaml
performance:
  processing:
    num_workers: 8
```

4. **Use multiple GPUs**:
```yaml
gpu:
  device_ids: [0, 1, 2, 3]
```

## Processing Issues

### Issue: Processing timeout

**Symptoms**:
```
TimeoutError: Processing exceeded 300 seconds
```

**Solutions**:

1. **Increase timeout**:
```bash
export HISTOCORE_TIMEOUT=600
```

2. **Enable early stopping**:
```yaml
processing:
  early_stopping:
    enabled: true
    confidence_threshold: 0.95
```

3. **Optimize processing**:
```yaml
streaming:
  batch_size: 64
gpu:
  enable_amp: true
```

### Issue: Low confidence predictions

**Symptoms**:
- Confidence scores consistently <0.7
- Early stopping not triggered

**Diagnostics**:
```bash
# Check confidence distribution
curl http://localhost:8002/metrics | grep confidence

# Check model quality
python -m src.streaming.validate_model
```

**Solutions**:

1. **Check model weights**:
```bash
# Verify model file exists and is valid
ls -lh /models/best_model.pth
python -c "import torch; torch.load('/models/best_model.pth')"
```

2. **Adjust confidence threshold**:
```yaml
processing:
  confidence_threshold: 0.85
```

3. **Increase processing coverage**:
```yaml
processing:
  early_stopping:
    min_patches: 5000
```

### Issue: Incorrect predictions

**Symptoms**:
- Predictions don't match expected results
- High error rate

**Diagnostics**:
```bash
# Run validation
python -m src.streaming.validate --data-dir /data/validation

# Check model metrics
python -m src.streaming.evaluate_model
```

**Solutions**:

1. **Verify input data**:
```python
from src.data.wsi_pipeline import WSIProcessor
processor = WSIProcessor()
metadata = processor.get_metadata('slide.svs')
print(metadata)
```

2. **Check preprocessing**:
```yaml
preprocessing:
  normalization: true
  stain_normalization: true
```

3. **Retrain or update model**:
```bash
python experiments/train_pcam.py --config config.yaml
```

## Memory Issues

### Issue: System out of memory

**Symptoms**:
```
MemoryError: Unable to allocate array
```

**Diagnostics**:
```bash
# Check system memory
free -h

# Monitor memory usage
watch -n 1 free -h

# Check application memory
curl http://localhost:8002/metrics | grep memory_usage
```

**Solutions**:

1. **Reduce memory budget**:
```bash
export HISTOCORE_MAX_MEMORY_GB=4
```

2. **Enable memory management**:
```yaml
performance:
  memory:
    gc_enabled: true
    gc_threshold: 0.8
```

3. **Reduce buffer sizes**:
```yaml
streaming:
  buffer_size: 8
  tile_cache_size: 500
```

4. **Use swap space** (not recommended for production):
```bash
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Issue: Memory leak

**Symptoms**:
- Memory usage increases over time
- Eventually crashes with OOM

**Diagnostics**:
```bash
# Monitor memory over time
watch -n 5 'ps aux | grep histocore'

# Profile memory usage
python -m memory_profiler src/streaming/main.py
```

**Solutions**:

1. **Enable periodic cleanup**:
```yaml
performance:
  memory:
    gc_enabled: true
    gc_interval: 100
```

2. **Restart periodically**:
```yaml
# Kubernetes
spec:
  template:
    spec:
      containers:
      - name: histocore-streaming
        lifecycle:
          preStop:
            exec:
              command: ["/bin/sh", "-c", "sleep 15"]
```

3. **Report issue with memory profile**:
```bash
python -m memory_profiler src/streaming/main.py > memory_profile.txt
# Attach to GitHub issue
```

## Network Issues

### Issue: Connection refused

**Symptoms**:
```
ConnectionRefusedError: [Errno 111] Connection refused
```

**Diagnostics**:
```bash
# Check if service is running
docker ps
kubectl get pods -n histocore

# Check port binding
netstat -tulpn | grep 8000

# Test connection
curl -v http://localhost:8000/health
```

**Solutions**:

1. **Check service status**:
```bash
# Docker
docker-compose ps
docker-compose logs histocore-streaming

# Kubernetes
kubectl get pods -n histocore
kubectl logs deployment/histocore-streaming -n histocore
```

2. **Check firewall**:
```bash
# Linux
sudo ufw status
sudo ufw allow 8000/tcp

# Check iptables
sudo iptables -L -n
```

3. **Check port binding**:
```yaml
# docker-compose.yml
ports:
  - "8000:8000"  # Ensure correct mapping
```

### Issue: Slow network performance

**Symptoms**:
- High latency
- Slow data transfer

**Diagnostics**:
```bash
# Test network speed
iperf3 -c server_ip

# Check network metrics
curl http://localhost:8002/metrics | grep network

# Monitor network traffic
sudo iftop
```

**Solutions**:

1. **Enable compression**:
```yaml
performance:
  network:
    compression_enabled: true
```

2. **Increase connection pool**:
```yaml
performance:
  network:
    connection_pool_size: 20
```

3. **Use faster network**:
- Upgrade to 10 Gbps network
- Use dedicated network for data transfer

## PACS Integration Issues

### Issue: PACS connection failed

**Symptoms**:
```
PACSConnectionError: Failed to connect to PACS server
```

**Diagnostics**:
```bash
# Test PACS connectivity
python -m src.pacs.test_connection

# Check PACS configuration
cat .kiro/pacs/config.yaml

# Test DICOM echo
dcmecho -v PACS_SERVER 104
```

**Solutions**:

1. **Verify PACS configuration**:
```yaml
pacs:
  remote_host: pacs.hospital.org
  remote_port: 104
  remote_ae_title: PACS_SERVER
  ae_title: HISTOCORE
```

2. **Check network connectivity**:
```bash
# Ping PACS server
ping pacs.hospital.org

# Test port
telnet pacs.hospital.org 104
```

3. **Check firewall rules**:
```bash
sudo ufw allow 104/tcp
```

4. **Enable TLS if required**:
```yaml
pacs:
  security:
    tls_enabled: true
    cert_file: /certs/client.crt
    key_file: /certs/client.key
```

### Issue: DICOM transfer failed

**Symptoms**:
```
DICOMTransferError: C-MOVE operation failed
```

**Solutions**:

1. **Check PACS permissions**:
- Verify AE title is registered with PACS
- Check C-MOVE permissions

2. **Increase timeouts**:
```yaml
pacs:
  connection_timeout: 60
  network_timeout: 120
```

3. **Enable retry logic**:
```yaml
pacs:
  workflow:
    retry_failed: true
    retry_attempts: 3
```

## Performance Issues

### Issue: Slow processing

**Symptoms**:
- Processing time >60 seconds
- Low throughput

**Diagnostics**:
```bash
# Check processing metrics
curl http://localhost:8002/metrics | grep processing_duration

# Profile performance
python -m cProfile -o profile.stats src/streaming/main.py

# Analyze profile
python -m pstats profile.stats
```

**Solutions**:

1. **Optimize configuration**:
```yaml
streaming:
  batch_size: 64
  buffer_size: 32

gpu:
  enable_amp: true
  device_ids: [0, 1]

performance:
  processing:
    num_workers: 8
    max_concurrent_batches: 4
```

2. **Use faster hardware**:
- Upgrade to faster GPU (RTX 4090, A100)
- Use NVMe SSD for data storage
- Increase RAM

3. **Enable caching**:
```yaml
performance:
  network:
    cache_enabled: true
    cache_ttl: 3600
```

### Issue: High latency

**Symptoms**:
- API response time >1 second
- Slow dashboard updates

**Solutions**:

1. **Enable caching**:
```yaml
performance:
  network:
    cache_enabled: true
```

2. **Optimize database queries**:
```yaml
database:
  connection_pool_size: 20
  query_timeout: 5
```

3. **Use CDN for static assets**:
```yaml
web:
  cdn_enabled: true
  cdn_url: https://cdn.histocore.ai
```

## Deployment Issues

### Issue: Docker build fails

**Symptoms**:
```
ERROR: failed to solve: process "/bin/sh -c pip install -r requirements.txt" did not complete successfully
```

**Solutions**:

1. **Clear Docker cache**:
```bash
docker system prune -a
docker-compose build --no-cache
```

2. **Check Dockerfile**:
```dockerfile
# Use specific base image version
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install dependencies separately
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
RUN pip install -r requirements.txt
```

3. **Build with verbose output**:
```bash
docker-compose build --progress=plain
```

### Issue: Kubernetes pod not starting

**Symptoms**:
```
kubectl get pods -n histocore
NAME                                  READY   STATUS             RESTARTS
histocore-streaming-xxx               0/1     CrashLoopBackOff   5
```

**Diagnostics**:
```bash
# Check pod status
kubectl describe pod histocore-streaming-xxx -n histocore

# Check logs
kubectl logs histocore-streaming-xxx -n histocore

# Check events
kubectl get events -n histocore --sort-by='.lastTimestamp'
```

**Solutions**:

1. **Check resource limits**:
```yaml
resources:
  requests:
    memory: "4Gi"
    cpu: "2000m"
    nvidia.com/gpu: 1
  limits:
    memory: "8Gi"
    cpu: "4000m"
    nvidia.com/gpu: 1
```

2. **Check image pull**:
```bash
# Verify image exists
docker pull histocore/streaming:latest

# Check image pull secrets
kubectl get secrets -n histocore
```

3. **Check configuration**:
```bash
# Verify configmap
kubectl get configmap histocore-config -n histocore -o yaml

# Verify secrets
kubectl get secret histocore-secrets -n histocore -o yaml
```

## Error Messages

### Common Error Messages and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `CUDA out of memory` | GPU memory exhausted | Reduce batch size, enable AMP |
| `Connection refused` | Service not running | Start service, check firewall |
| `File not found` | Missing file | Check file path, verify mount |
| `Permission denied` | Insufficient permissions | Check file permissions, run as root |
| `Timeout` | Operation took too long | Increase timeout, optimize processing |
| `Invalid configuration` | Config syntax error | Validate YAML syntax |
| `Model not found` | Missing model file | Download model, check path |
| `GPU not available` | CUDA not installed | Install CUDA, check drivers |

### Getting Help

If you can't resolve the issue:

1. **Collect diagnostics**:
```bash
# Run diagnostic script
python -m src.streaming.diagnostics > diagnostics.txt

# Collect logs
docker logs histocore-streaming > logs.txt
kubectl logs deployment/histocore-streaming -n histocore > logs.txt

# Export configuration
kubectl get configmap histocore-config -n histocore -o yaml > config.yaml
```

2. **Create GitHub issue**:
- Include diagnostics output
- Include logs (sanitize sensitive data)
- Include configuration (sanitize secrets)
- Describe steps to reproduce

3. **Contact support**:
- Email: support@histocore.ai
- Include issue number
- Include diagnostics

## Support Resources

- **Documentation**: [docs/](.)
- **API Reference**: [docs/api/](api/)
- **FAQ**: [docs/FAQ.md](FAQ.md)
- **GitHub Issues**: https://github.com/histocore/histocore/issues
- **Email**: support@histocore.ai
