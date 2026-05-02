# Federated Learning Configuration Guide

## Overview

FL system uses YAML config files for coordinator + clients. All settings have defaults but should be customized for production.

## Coordinator Configuration

### Basic Config (`configs/federated/coordinator.yaml`)

```yaml
# Server settings
server:
  host: "0.0.0.0"
  port: 8080
  tls_cert: "./certs/coordinator-cert.pem"
  tls_key: "./certs/coordinator-key.pem"
  enable_mutual_auth: true

# Training settings
training:
  num_rounds: 100
  min_clients: 3
  max_clients: 10
  client_timeout_seconds: 600  # 10 minutes
  
  # Synchronization mode
  sync_mode: "semi_synchronous"  # synchronous | semi_synchronous | fully_asynchronous
  min_client_percentage: 0.8  # For semi-sync mode

# Aggregation settings
aggregation:
  algorithm: "fedavg"  # fedavg | fedprox | fedadam
  
  # FedProx settings (if algorithm=fedprox)
  fedprox_mu: 0.01
  
  # FedAdam settings (if algorithm=fedadam)
  fedadam_beta1: 0.9
  fedadam_beta2: 0.999
  fedadam_epsilon: 1e-8

# Privacy settings
privacy:
  enable_dp: true
  target_epsilon: 1.0
  target_delta: 1e-5
  noise_multiplier: 1.1
  max_grad_norm: 1.0

# Byzantine detection
byzantine:
  enable: true
  algorithm: "krum"  # krum | trimmed_mean | median
  threshold_std: 3.0

# Compression
compression:
  enable: true
  mode: "quantization"  # quantization | sparsification | mixed
  quantization_bits: 8  # 4 | 8 | 16
  sparsification_ratio: 0.1  # 0.01 | 0.05 | 0.1

# Monitoring
monitoring:
  enable_prometheus: true
  prometheus_port: 9090
  enable_tensorboard: true
  tensorboard_dir: "./runs"
  log_level: "INFO"  # DEBUG | INFO | WARNING | ERROR

# Model registry
model_registry:
  checkpoint_dir: "./checkpoints/federated"
  max_versions: 10
  save_interval: 1  # Save every N rounds

# Database
database:
  type: "sqlite"  # sqlite | postgresql
  path: "./data/federated.db"
  # For PostgreSQL:
  # host: "localhost"
  # port: 5432
  # database: "federated"
  # user: "federated"
  # password: "secure_password"
```

### Advanced Settings

```yaml
# Fault tolerance
fault_tolerance:
  enable_checkpoints: true
  checkpoint_interval: 5  # rounds
  enable_partition_detection: true
  partition_threshold_seconds: 60
  reconnection_strategy: "exponential_backoff"  # immediate | exponential_backoff | linear_backoff
  max_reconnection_attempts: 10

# Async training
async_training:
  enable_staleness_weighting: true
  staleness_alpha: 0.5
  max_staleness: 10
  enable_dynamic_timeout: true
  timeout_multiplier: 2.0

# Security
security:
  enable_audit_logging: true
  audit_log_dir: "./logs/audit"
  log_retention_days: 2555  # 7 years for HIPAA
  enable_tamper_detection: true
```

## Client Configuration

### Basic Config (`configs/federated/client.yaml`)

```yaml
# Client identity
client:
  id: "hospital-001"
  name: "General Hospital"
  site_code: "GH001"

# Coordinator connection
coordinator:
  url: "https://coordinator.example.com:8080"
  tls_cert: "./certs/coordinator-cert.pem"
  client_cert: "./certs/client-cert.pem"
  client_key: "./certs/client-key.pem"
  connection_timeout: 30
  heartbeat_interval: 60

# Local training
training:
  batch_size: 32
  local_epochs: 5
  learning_rate: 0.001
  optimizer: "adam"  # adam | sgd | adamw
  gradient_accumulation_steps: 1
  
  # Device settings
  device: "cuda"  # cuda | cpu
  mixed_precision: true  # FP16 training
  num_workers: 4

# PACS integration
pacs:
  enable: true
  host: "pacs.hospital.local"
  port: 11112
  aet: "HISTOCORE"
  calling_aet: "FL_CLIENT"
  
  # Data discovery
  modality: "SM"  # Slide Microscopy
  date_range_days: 30
  max_studies: 1000
  
  # Caching
  cache_dir: "./data/pacs_cache"
  cache_size_gb: 50

# Resource limits
resources:
  max_gpu_memory_gb: 8
  max_cpu_cores: 4
  max_disk_space_gb: 100
  
  # Scheduled training windows
  enable_scheduling: false
  training_windows:
    - start: "22:00"  # 10 PM
      end: "06:00"    # 6 AM
      days: ["monday", "tuesday", "wednesday", "thursday", "friday"]

# Privacy
privacy:
  enable_dp: true
  noise_multiplier: 1.1
  max_grad_norm: 1.0
  privacy_budget_epsilon: 1.0
  privacy_budget_delta: 1e-5

# Compression
compression:
  enable: true
  mode: "quantization"
  quantization_bits: 8

# Fault tolerance
fault_tolerance:
  enable_checkpoints: true
  checkpoint_dir: "./checkpoints/client"
  max_checkpoints: 5
  
  # Network monitoring
  enable_network_monitoring: true
  heartbeat_interval: 60
  failure_threshold: 3

# Logging
logging:
  level: "INFO"
  log_dir: "./logs/client"
  enable_audit: true
```

## Environment-Specific Configs

### Development

```yaml
# configs/federated/coordinator.dev.yaml
training:
  num_rounds: 10  # Fewer rounds for testing
  min_clients: 2

monitoring:
  log_level: "DEBUG"

database:
  type: "sqlite"
  path: "./data/dev.db"
```

### Staging

```yaml
# configs/federated/coordinator.staging.yaml
training:
  num_rounds: 50
  min_clients: 3

security:
  enable_audit_logging: true

database:
  type: "postgresql"
  host: "staging-db.internal"
```

### Production

```yaml
# configs/federated/coordinator.prod.yaml
training:
  num_rounds: 100
  min_clients: 5
  client_timeout_seconds: 1800  # 30 minutes

security:
  enable_audit_logging: true
  enable_tamper_detection: true

database:
  type: "postgresql"
  host: "prod-db.internal"
  backup_enabled: true

monitoring:
  enable_prometheus: true
  enable_tensorboard: true
  alert_email: "ops@hospital.org"
```

## Configuration Validation

```bash
# Validate coordinator config
python -m src.federated.production.config validate \
    --config configs/federated/coordinator.yaml \
    --type coordinator

# Validate client config
python -m src.federated.production.config validate \
    --config configs/federated/client.yaml \
    --type client
```

## Environment Variables

Override config via env vars:

```bash
# Coordinator
export FL_COORDINATOR_HOST="0.0.0.0"
export FL_COORDINATOR_PORT="8080"
export FL_NUM_ROUNDS="100"
export FL_MIN_CLIENTS="5"

# Client
export FL_CLIENT_ID="hospital-001"
export FL_COORDINATOR_URL="https://coordinator.example.com:8080"
export FL_PACS_HOST="pacs.hospital.local"
export FL_PACS_PORT="11112"
```

## Configuration Best Practices

1. **Never commit secrets** - Use env vars or secret management
2. **Use separate configs** - Dev/staging/prod environments
3. **Validate before deploy** - Run validation tool
4. **Document changes** - Track config changes in version control
5. **Test locally first** - Use dev config for testing
6. **Monitor resource usage** - Adjust limits based on actual usage
7. **Enable audit logging** - Required for HIPAA compliance

## Next Steps

- [API Reference](API_REFERENCE.md) - Detailed API docs
- [Troubleshooting](TROUBLESHOOTING.md) - Common issues
- [Security Guide](SECURITY.md) - Security best practices
