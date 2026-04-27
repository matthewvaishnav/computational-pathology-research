# HistoCore Configuration Guide

Complete reference for configuring HistoCore Real-Time WSI Streaming.

## Table of Contents

- [Overview](#overview)
- [Configuration Methods](#configuration-methods)
- [Core Settings](#core-settings)
- [GPU Configuration](#gpu-configuration)
- [Processing Configuration](#processing-configuration)
- [PACS Integration](#pacs-integration)
- [Monitoring Configuration](#monitoring-configuration)
- [Security Configuration](#security-configuration)
- [Performance Tuning](#performance-tuning)
- [Environment-Specific Configs](#environment-specific-configs)

## Overview

HistoCore supports multiple configuration methods with the following precedence (highest to lowest):

1. **Environment variables** - Runtime overrides
2. **Configuration files** - YAML/JSON configs
3. **Command-line arguments** - CLI flags
4. **Default values** - Built-in defaults

## Configuration Methods

### Environment Variables

Set via shell or Docker/K8s:

```bash
# Shell
export HISTOCORE_BATCH_SIZE=32
export CUDA_VISIBLE_DEVICES=0

# Docker
docker run -e HISTOCORE_BATCH_SIZE=32 histocore/streaming

# Kubernetes
kubectl set env deployment/histocore-streaming HISTOCORE_BATCH_SIZE=32
```

### Configuration Files

**config.yaml**:
```yaml
streaming:
  tile_size: 1024
  batch_size: 32
  memory_budget_gb: 8.0

gpu:
  device_ids: [0]
  enable_amp: true

monitoring:
  prometheus:
    enabled: true
    port: 8002
```

**Load configuration**:
```bash
# Command line
python -m src.streaming.main --config config.yaml

# Environment variable
export HISTOCORE_CONFIG=config.yaml
python -m src.streaming.main
```

### Command-Line Arguments

```bash
python -m src.streaming.main \
  --batch-size 32 \
  --memory-budget 8.0 \
  --gpu-ids 0 1 \
  --log-level INFO
```

### Hot Reload

Update configuration without restart:

```python
from src.streaming.config_manager import ConfigManager

# Initialize
config_manager = ConfigManager('config.yaml')

# Enable hot reload
config_manager.enable_hot_reload()

# Update config
config_manager.update_config({
    'streaming': {
        'batch_size': 64
    }
})

# Get current config
config = config_manager.get_config()
```

## Core Settings

### Application Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `HISTOCORE_ENV` | string | `development` | Environment (development/staging/production) |
| `HISTOCORE_LOG_LEVEL` | string | `INFO` | Logging level (DEBUG/INFO/WARNING/ERROR) |
| `HISTOCORE_WORKERS` | int | `4` | Number of worker processes |
| `HISTOCORE_TIMEOUT` | int | `300` | Request timeout (seconds) |

**Example**:
```yaml
application:
  environment: production
  log_level: INFO
  workers: 4
  timeout: 300
```

### Storage Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `HISTOCORE_DATA_DIR` | string | `/data` | Data directory path |
| `HISTOCORE_CACHE_DIR` | string | `/cache` | Cache directory path |
| `HISTOCORE_TEMP_DIR` | string | `/tmp` | Temporary files directory |
| `HISTOCORE_MAX_DISK_USAGE` | float | `0.9` | Max disk usage (0.0-1.0) |

**Example**:
```yaml
storage:
  data_dir: /data/histocore
  cache_dir: /cache/histocore
  temp_dir: /tmp/histocore
  max_disk_usage: 0.9
```

## GPU Configuration

### Basic GPU Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `CUDA_VISIBLE_DEVICES` | string | `0` | GPU device IDs (comma-separated) |
| `HISTOCORE_GPU_MEMORY_FRACTION` | float | `0.9` | GPU memory fraction to use |
| `HISTOCORE_ENABLE_AMP` | bool | `true` | Enable automatic mixed precision |
| `HISTOCORE_GPU_ALLOW_GROWTH` | bool | `true` | Allow GPU memory growth |

**Example**:
```yaml
gpu:
  device_ids: [0, 1]
  memory_fraction: 0.9
  enable_amp: true
  allow_growth: true
```

### Multi-GPU Configuration

**Data Parallelism**:
```yaml
gpu:
  device_ids: [0, 1, 2, 3]
  strategy: data_parallel
  batch_size_per_gpu: 16
```

**Model Parallelism**:
```yaml
gpu:
  device_ids: [0, 1]
  strategy: model_parallel
  encoder_gpu: 0
  attention_gpu: 1
```

### GPU Memory Management

```yaml
gpu:
  memory_management:
    # Pre-allocate memory pool
    preallocate: true
    pool_size_gb: 8.0
    
    # Automatic cleanup
    cleanup_interval: 100  # batches
    cleanup_threshold: 0.85  # 85% usage
    
    # OOM handling
    oom_retry: true
    oom_batch_reduction: 0.5  # Reduce by 50%
```

## Processing Configuration

### Streaming Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `HISTOCORE_TILE_SIZE` | int | `1024` | Tile size for streaming (pixels) |
| `HISTOCORE_BATCH_SIZE` | int | `32` | Batch size for processing |
| `HISTOCORE_BUFFER_SIZE` | int | `16` | Tile buffer size |
| `HISTOCORE_MAX_MEMORY_GB` | float | `2.0` | Maximum memory usage (GB) |

**Example**:
```yaml
streaming:
  tile_size: 1024
  batch_size: 32
  buffer_size: 16
  memory_budget_gb: 2.0
```

### Processing Parameters

```yaml
processing:
  # Target performance
  target_time: 30.0  # seconds
  confidence_threshold: 0.95
  
  # Early stopping
  early_stopping:
    enabled: true
    min_patches: 1000
    confidence_window: 100
    
  # Quality control
  quality:
    min_tissue_percent: 0.1
    blur_threshold: 100
    artifact_detection: true
```

### Attention Configuration

```yaml
attention:
  # Model selection
  model: AttentionMIL  # AttentionMIL, CLAM, TransMIL
  
  # Attention parameters
  feature_dim: 2048
  hidden_dim: 256
  num_heads: 8
  dropout: 0.1
  
  # Aggregation
  aggregation: weighted  # weighted, max, mean
  temperature: 1.0
```

## PACS Integration

### Connection Settings

```yaml
pacs:
  enabled: true
  
  # DICOM settings
  ae_title: HISTOCORE
  ae_port: 104
  
  # Remote PACS
  remote_ae_title: PACS_SERVER
  remote_host: pacs.hospital.org
  remote_port: 104
  
  # Timeouts
  connection_timeout: 30
  network_timeout: 60
  acse_timeout: 30
  dimse_timeout: 60
```

### Security Settings

```yaml
pacs:
  security:
    # TLS encryption
    tls_enabled: true
    tls_version: TLSv1.3
    cert_file: /certs/client.crt
    key_file: /certs/client.key
    ca_file: /certs/ca.crt
    
    # Authentication
    require_auth: true
    username: histocore
    password: ${PACS_PASSWORD}
```

### Workflow Configuration

```yaml
pacs:
  workflow:
    # Polling
    poll_enabled: true
    poll_interval: 60  # seconds
    
    # Query filters
    query:
      modality: SM  # Slide Microscopy
      study_date_range: 7  # days
      priority_keywords: [urgent, stat]
    
    # Processing
    auto_process: true
    max_concurrent: 10
    retry_failed: true
    retry_attempts: 3
```

## Monitoring Configuration

### Prometheus Metrics

```yaml
monitoring:
  prometheus:
    enabled: true
    port: 8002
    path: /metrics
    
    # Metric collection
    collect_interval: 15  # seconds
    histogram_buckets: [0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0]
    
    # Custom metrics
    custom_metrics:
      - name: slides_processed
        type: counter
        help: Total slides processed
      - name: processing_duration
        type: histogram
        help: Processing duration in seconds
```

### OpenTelemetry Tracing

```yaml
monitoring:
  tracing:
    enabled: true
    
    # Jaeger exporter
    jaeger:
      endpoint: http://jaeger:14268/api/traces
      service_name: histocore-streaming
      
    # Sampling
    sampling:
      type: probabilistic  # always, probabilistic, rate_limiting
      rate: 0.1  # 10% of traces
      
    # Instrumentation
    auto_instrument:
      - requests
      - asyncio
      - torch
```

### Logging Configuration

```yaml
monitoring:
  logging:
    # Log level
    level: INFO
    
    # Format
    format: json  # json, text
    
    # Output
    handlers:
      - type: console
        level: INFO
      - type: file
        level: DEBUG
        filename: /logs/histocore.log
        max_bytes: 104857600  # 100MB
        backup_count: 10
      - type: syslog
        level: WARNING
        address: /dev/log
    
    # Structured logging
    structured:
      enabled: true
      include_timestamp: true
      include_correlation_id: true
      include_component: true
```

### Health Checks

```yaml
monitoring:
  health_checks:
    # Endpoints
    enabled: true
    port: 8000
    
    # Check intervals
    liveness_interval: 10  # seconds
    readiness_interval: 5
    
    # Thresholds
    thresholds:
      cpu_percent: 90
      memory_percent: 90
      gpu_memory_percent: 95
      disk_percent: 90
```

## Security Configuration

### Authentication

```yaml
security:
  authentication:
    # OAuth 2.0
    oauth:
      enabled: true
      issuer: https://auth.histocore.ai
      client_id: histocore-api
      client_secret: ${OAUTH_CLIENT_SECRET}
      scopes:
        - read:wsi
        - write:wsi
        - admin:config
    
    # JWT
    jwt:
      secret_key: ${JWT_SECRET_KEY}
      algorithm: RS256
      expiration: 3600  # seconds
      refresh_enabled: true
      refresh_expiration: 86400  # 24 hours
```

### Authorization

```yaml
security:
  authorization:
    # RBAC
    rbac:
      enabled: true
      roles:
        - name: admin
          permissions: [read, write, delete, admin]
        - name: user
          permissions: [read, write]
        - name: viewer
          permissions: [read]
    
    # API keys
    api_keys:
      enabled: true
      header_name: X-API-Key
      rate_limit: 1000  # requests/hour
```

### Encryption

```yaml
security:
  encryption:
    # Data at rest
    at_rest:
      enabled: true
      algorithm: AES-256-GCM
      key_file: /keys/encryption.key
    
    # Data in transit
    in_transit:
      tls_enabled: true
      tls_version: TLSv1.3
      cert_file: /certs/server.crt
      key_file: /certs/server.key
      ca_file: /certs/ca.crt
```

### Audit Logging

```yaml
security:
  audit:
    enabled: true
    
    # Events to log
    events:
      - authentication
      - authorization
      - data_access
      - configuration_change
      - admin_action
    
    # Storage
    storage:
      type: file  # file, database, syslog
      path: /logs/audit.log
      retention_days: 2555  # 7 years (HIPAA)
      
    # Format
    format: json
    include_request_body: false
    include_response_body: false
```

## Performance Tuning

### Memory Optimization

```yaml
performance:
  memory:
    # Garbage collection
    gc_enabled: true
    gc_interval: 100  # batches
    gc_threshold: 0.8  # 80% usage
    
    # Memory pools
    use_memory_pool: true
    pool_size_gb: 8.0
    
    # Caching
    feature_cache_size: 10000
    tile_cache_size: 1000
```

### Processing Optimization

```yaml
performance:
  processing:
    # Parallelism
    num_workers: 4
    prefetch_factor: 2
    
    # Batch processing
    dynamic_batching: true
    min_batch_size: 8
    max_batch_size: 128
    
    # Async processing
    async_enabled: true
    max_concurrent_batches: 4
```

### Network Optimization

```yaml
performance:
  network:
    # Connection pooling
    connection_pool_size: 10
    connection_timeout: 30
    
    # Compression
    compression_enabled: true
    compression_level: 6
    
    # Caching
    cache_enabled: true
    cache_ttl: 3600  # seconds
```

## Environment-Specific Configs

### Development

```yaml
# config.development.yaml
application:
  environment: development
  log_level: DEBUG
  workers: 2

streaming:
  batch_size: 16
  memory_budget_gb: 4.0

gpu:
  device_ids: [0]
  enable_amp: false

monitoring:
  prometheus:
    enabled: false
  tracing:
    enabled: false

security:
  authentication:
    oauth:
      enabled: false
```

### Staging

```yaml
# config.staging.yaml
application:
  environment: staging
  log_level: INFO
  workers: 4

streaming:
  batch_size: 32
  memory_budget_gb: 8.0

gpu:
  device_ids: [0]
  enable_amp: true

monitoring:
  prometheus:
    enabled: true
  tracing:
    enabled: true
    sampling:
      rate: 0.5

security:
  authentication:
    oauth:
      enabled: true
```

### Production

```yaml
# config.production.yaml
application:
  environment: production
  log_level: WARNING
  workers: 8

streaming:
  batch_size: 64
  memory_budget_gb: 16.0

gpu:
  device_ids: [0, 1, 2, 3]
  enable_amp: true

monitoring:
  prometheus:
    enabled: true
  tracing:
    enabled: true
    sampling:
      rate: 0.1
  logging:
    level: INFO
    format: json

security:
  authentication:
    oauth:
      enabled: true
  authorization:
    rbac:
      enabled: true
  encryption:
    at_rest:
      enabled: true
    in_transit:
      tls_enabled: true
  audit:
    enabled: true
```

## Configuration Validation

### Schema Validation

```python
from src.streaming.config_manager import ConfigManager

# Load and validate
config_manager = ConfigManager('config.yaml')
config_manager.validate()

# Export schema
schema = config_manager.export_schema()
print(schema)
```

### Configuration Testing

```bash
# Validate configuration
python -m src.streaming.config_manager validate config.yaml

# Test configuration
python -m src.streaming.config_manager test config.yaml

# Compare configurations
python -m src.streaming.config_manager diff config1.yaml config2.yaml
```

## Configuration Examples

### High-Throughput Configuration

Optimize for maximum throughput:

```yaml
streaming:
  batch_size: 128
  buffer_size: 32
  memory_budget_gb: 16.0

gpu:
  device_ids: [0, 1, 2, 3]
  enable_amp: true
  memory_fraction: 0.95

processing:
  target_time: 20.0
  early_stopping:
    enabled: false

performance:
  processing:
    num_workers: 8
    max_concurrent_batches: 8
```

### Low-Latency Configuration

Optimize for minimum latency:

```yaml
streaming:
  batch_size: 16
  buffer_size: 4
  memory_budget_gb: 4.0

gpu:
  device_ids: [0]
  enable_amp: true

processing:
  target_time: 10.0
  early_stopping:
    enabled: true
    min_patches: 500

performance:
  processing:
    num_workers: 2
    prefetch_factor: 1
```

### Memory-Constrained Configuration

Optimize for limited memory:

```yaml
streaming:
  batch_size: 8
  buffer_size: 4
  memory_budget_gb: 2.0

gpu:
  device_ids: [0]
  enable_amp: true
  memory_fraction: 0.8

processing:
  early_stopping:
    enabled: true

performance:
  memory:
    gc_enabled: true
    gc_interval: 50
    feature_cache_size: 5000
```

## Troubleshooting

### Configuration Issues

**Invalid configuration**:
```bash
# Validate configuration
python -m src.streaming.config_manager validate config.yaml

# Check for errors
python -m src.streaming.config_manager check config.yaml
```

**Configuration not loading**:
```bash
# Check file path
ls -la config.yaml

# Check permissions
chmod 644 config.yaml

# Check syntax
python -c "import yaml; yaml.safe_load(open('config.yaml'))"
```

### Performance Issues

**Slow processing**:
```yaml
# Increase batch size
streaming:
  batch_size: 64

# Enable AMP
gpu:
  enable_amp: true

# Increase workers
performance:
  processing:
    num_workers: 8
```

**High memory usage**:
```yaml
# Reduce batch size
streaming:
  batch_size: 16

# Enable garbage collection
performance:
  memory:
    gc_enabled: true
    gc_interval: 50

# Reduce cache sizes
performance:
  memory:
    feature_cache_size: 5000
```

## Support

- **Documentation**: [docs/](../)
- **API Reference**: [docs/api/](../api/)
- **GitHub Issues**: https://github.com/histocore/histocore/issues
- **Email**: support@histocore.ai
