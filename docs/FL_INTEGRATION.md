# Federated Learning Integration Guide

## Overview

This guide shows how to integrate HistoCore's federated learning system into your digital pathology workflow. The FL system enables privacy-preserving multi-site training across hospitals without sharing patient data.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FL Coordinator (Central)                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Orchestrator │  │  Aggregator  │  │   Registry   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Privacy    │  │  Byzantine   │  │  Monitoring  │     │
│  │    Engine    │  │   Detector   │  │    System    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                              │
                    TLS 1.3 Encrypted
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼────────┐   ┌────────▼───────┐   ┌────────▼───────┐
│  Hospital A    │   │  Hospital B    │   │  Hospital C    │
│  ┌──────────┐  │   │  ┌──────────┐  │   │  ┌──────────┐  │
│  │   PACS   │  │   │  │   PACS   │  │   │  │   PACS   │  │
│  └────┬─────┘  │   │  └────┬─────┘  │   │  └────┬─────┘  │
│       │        │   │       │        │   │       │        │
│  ┌────▼─────┐  │   │  ┌────▼─────┐  │   │  ┌────▼─────┐  │
│  │ FL Client│  │   │  │ FL Client│  │   │  │ FL Client│  │
│  └──────────┘  │   │  └──────────┘  │   │  └──────────┘  │
└────────────────┘   └────────────────┘   └────────────────┘
```

## Prerequisites

### Coordinator Requirements

- **Hardware**: 4+ CPU cores, 8GB+ RAM, 50GB disk
- **Network**: Static IP, open ports 8080-8081
- **Software**: Python 3.9+, PyTorch 2.0+, Docker (optional)

### Client Requirements (Per Hospital)

- **Hardware**: 4+ CPU cores, 16GB+ RAM, 100GB disk, NVIDIA GPU (8GB+ VRAM recommended)
- **Network**: Outbound HTTPS access to coordinator
- **Software**: Python 3.9+, PyTorch 2.0+, CUDA 11.8+ (for GPU)
- **PACS**: Existing DICOM PACS system (optional but recommended)

## Installation

### 1. Install HistoCore

```bash
# Clone repository
git clone https://github.com/yourusername/histocore.git
cd histocore

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install with federated learning support
pip install -r requirements.txt
pip install -r requirements-federated.txt
pip install -e .
```

### 2. Generate TLS Certificates

```bash
# Generate coordinator certificates
python -m src.federated.communication.tls_utils generate \
    --output-dir ./certs \
    --coordinator-host coordinator.example.com \
    --validity-days 365

# Generate client certificates (per hospital)
python -m src.federated.communication.tls_utils generate_client_cert \
    --output-dir ./certs \
    --client-id hospital-001 \
    --ca-cert ./certs/coordinator-cert.pem \
    --ca-key ./certs/coordinator-key.pem
```

### 3. Configure Coordinator

Create `configs/federated/coordinator.yaml`:

```yaml
server:
  host: "0.0.0.0"
  port: 8080
  tls_cert: "./certs/coordinator-cert.pem"
  tls_key: "./certs/coordinator-key.pem"

training:
  num_rounds: 100
  min_clients: 3
  client_timeout_seconds: 600

aggregation:
  algorithm: "fedavg"  # fedavg | fedprox | fedadam

privacy:
  enable_dp: true
  target_epsilon: 1.0
  target_delta: 1e-5

monitoring:
  enable_prometheus: true
  enable_tensorboard: true
```

### 4. Configure Clients

Create `configs/federated/client.yaml` (per hospital):

```yaml
client:
  id: "hospital-001"
  name: "General Hospital"

coordinator:
  url: "https://coordinator.example.com:8080"
  tls_cert: "./certs/coordinator-cert.pem"
  client_cert: "./certs/client-cert.pem"
  client_key: "./certs/client-key.pem"

training:
  batch_size: 32
  local_epochs: 5
  learning_rate: 0.001
  device: "cuda"

pacs:
  enable: true
  host: "pacs.hospital.local"
  port: 11112
  aet: "HISTOCORE"
```

## Usage

### Start Coordinator

```bash
# Start coordinator server
python -m src.federated.production.coordinator_server \
    --config configs/federated/coordinator.yaml

# Verify coordinator is running
curl https://coordinator.example.com:8080/health
```

### Start Clients (Per Hospital)

```bash
# Start client
python -m src.federated.production.client_server \
    --config configs/federated/client.yaml \
    --coordinator-url https://coordinator.example.com:8080

# Verify client connection
python -m src.federated.client.test_connection \
    --coordinator-url https://coordinator.example.com:8080 \
    --cert ./certs/coordinator-cert.pem
```

### Monitor Training

```bash
# View coordinator logs
tail -f logs/coordinator/coordinator.log

# View client logs
tail -f logs/client/client.log

# Access Prometheus metrics
open http://localhost:9090

# Access TensorBoard
tensorboard --logdir=./runs --port=6006
open http://localhost:6006
```

## Integration with Existing Workflows

### PACS Integration

The FL client automatically discovers WSI data from your existing PACS:

```python
from src.federated.client.pacs_connector import PACSConnector

# Configure PACS connection
pacs = PACSConnector(
    host="pacs.hospital.local",
    port=11112,
    aet="HISTOCORE",
    calling_aet="FL_CLIENT"
)

# Discover WSI studies
studies = pacs.discover_wsi_studies(
    modality="SM",  # Slide Microscopy
    date_range_days=30,
    max_studies=1000
)

# Retrieve studies for training
for study in studies:
    pacs.retrieve_study(study.study_uid, cache_dir="./data/pacs_cache")
```

### Custom Model Integration

Integrate your existing PyTorch models:

```python
from src.federated import FederatedCoordinator
import torch.nn as nn

# Define your model
class MyPathologyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.ReLU(),
            # ... your architecture
        )
        self.classifier = nn.Linear(512, 2)
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Use in federated training
coordinator = FederatedCoordinator(
    config_path="configs/federated/coordinator.yaml",
    model_architecture=MyPathologyModel(),
    device="cuda"
)

coordinator.start_training(num_rounds=100, min_clients=3)
```

### Custom Data Loaders

Use your existing data preprocessing:

```python
from src.federated.client import FederatedClient
from torch.utils.data import DataLoader

# Your existing dataset
class HospitalDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        # Your data loading logic
        pass
    
    def __getitem__(self, idx):
        # Your preprocessing
        pass

# Integrate with FL client
client = FederatedClient(
    config_path="configs/federated/client.yaml",
    coordinator_url="https://coordinator.example.com:8080"
)

# Use custom dataloader
train_loader = DataLoader(
    HospitalDataset(data_dir="./data/hospital"),
    batch_size=32,
    shuffle=True
)

client.set_dataloader(train_loader)
client.start_training_loop()
```

## Docker Deployment

### Using Docker Compose

```bash
# Start coordinator + 3 clients
cd docker/federated
docker-compose up -d

# View logs
docker-compose logs -f coordinator
docker-compose logs -f client-1

# Stop all services
docker-compose down
```

### Kubernetes Deployment

```bash
# Create namespace
kubectl apply -f k8s/federated/namespace.yaml

# Deploy coordinator
kubectl apply -f k8s/federated/coordinator-deployment.yaml

# Deploy clients
kubectl apply -f k8s/federated/client-deployment.yaml

# Check status
kubectl get pods -n federated-learning

# View logs
kubectl logs -f deployment/fl-coordinator -n federated-learning
```

## Security Best Practices

### TLS Configuration

1. **Use strong certificates**: 2048-bit RSA or 256-bit ECDSA
2. **Enable mutual authentication**: Verify both coordinator and client identities
3. **Rotate certificates regularly**: Every 90 days recommended
4. **Use certificate pinning**: Prevent man-in-the-middle attacks

### Privacy Configuration

1. **Set appropriate epsilon**: Balance privacy (lower ε) vs utility (higher ε)
   - High privacy: ε = 0.1-1.0
   - Moderate privacy: ε = 1.0-5.0
   - Low privacy: ε = 5.0-10.0

2. **Adjust noise multiplier**: Higher noise = more privacy, slower convergence
   - Conservative: 1.5-2.0
   - Balanced: 1.0-1.5
   - Aggressive: 0.5-1.0

3. **Monitor privacy budget**: Track cumulative epsilon across rounds

### Network Security

1. **Firewall configuration**: Only allow necessary ports (8080, 9090)
2. **VPN recommended**: Use hospital VPN for additional security
3. **Rate limiting**: Prevent DoS attacks on coordinator
4. **Audit logging**: Enable comprehensive logging for compliance

## Troubleshooting

### Connection Issues

**Problem**: Client cannot connect to coordinator

**Solutions**:
1. Verify coordinator is running: `curl https://coordinator:8080/health`
2. Check firewall rules: `telnet coordinator.example.com 8080`
3. Verify TLS certificates: `openssl x509 -in certs/coordinator-cert.pem -text`
4. Check client config: Ensure correct coordinator URL

### Training Issues

**Problem**: Training not converging

**Solutions**:
1. Reduce learning rate: Try 0.0001 instead of 0.001
2. Increase local epochs: Try 10 instead of 5
3. Use FedProx for heterogeneous data: Set `algorithm: fedprox`
4. Reduce privacy noise: Lower `noise_multiplier` from 1.1 to 0.8

### Performance Issues

**Problem**: Slow training rounds

**Solutions**:
1. Enable gradient compression: Set `compression.enable: true`
2. Use async training: Set `sync_mode: semi_synchronous`
3. Reduce batch size: Try 16 instead of 32
4. Enable mixed precision: Set `mixed_precision: true`

## Advanced Topics

### Custom Aggregation Algorithms

```python
from src.federated.aggregation import BaseAggregator

class CustomAggregator(BaseAggregator):
    def aggregate(self, updates, weights):
        # Your custom aggregation logic
        weighted_sum = sum(u * w for u, w in zip(updates, weights))
        return weighted_sum / sum(weights)

# Use custom aggregator
coordinator = FederatedCoordinator(
    config_path="configs/federated/coordinator.yaml",
    model_architecture=MyModel(),
    aggregator=CustomAggregator()
)
```

### Continual Learning

```python
# Enable continual learning
coordinator = FederatedCoordinator(
    config_path="configs/federated/coordinator.yaml",
    model_architecture=MyModel(),
    continual_learning=True
)

# Schedule periodic training
coordinator.schedule_training(
    interval="weekly",  # weekly | monthly
    day_of_week="sunday",
    time="02:00"
)
```

### Cross-Institutional Benchmarking

```python
from src.federated.evaluation import FederatedEvaluator

evaluator = FederatedEvaluator(coordinator=coordinator)

# Evaluate on all client test sets
results = evaluator.federated_evaluation(
    global_model=coordinator.get_global_model(),
    compute_per_site_metrics=True
)

# Generate benchmarking report
evaluator.generate_report(
    results=results,
    output_path="results/federated_benchmark.pdf"
)
```

## Support

- **Documentation**: [docs/federated_learning/](docs/federated_learning/)
- **GitHub Issues**: https://github.com/yourusername/histocore/issues
- **Email**: support@histocore.org

## References

1. McMahan et al. (2017). "Communication-Efficient Learning of Deep Networks from Decentralized Data"
2. Abadi et al. (2016). "Deep Learning with Differential Privacy"
3. Bonawitz et al. (2017). "Practical Secure Aggregation for Privacy-Preserving Machine Learning"
4. Blanchard et al. (2017). "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent"

