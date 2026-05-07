# Design Document: Federated Learning System for HistoCore

## Overview

This document specifies the technical design for a privacy-preserving federated learning system that enables multi-site training across hospitals without centralizing patient data. The system integrates with HistoCore's existing PACS infrastructure and provides differential privacy, secure aggregation, and Byzantine robustness.

**Key Innovation**: First open-source federated learning framework specifically designed for digital pathology with PACS integration, property-based testing, and HIPAA compliance.

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     FL Coordinator (Central)                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Training   │  │  Aggregator  │  │   Model      │         │
│  │ Orchestrator │──│  (FedAvg/    │──│  Registry    │         │
│  │              │  │   FedProx)   │  │              │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│         │                  │                  │                  │
│         │          ┌───────┴────────┐        │                  │
│         │          │   Byzantine    │        │                  │
│         │          │   Detector     │        │                  │
│         │          └────────────────┘        │                  │
│         │                                     │                  │
│  ┌──────┴──────────────────────────────────┴──────┐           │
│  │         Monitoring & Audit System               │           │
│  └─────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
                              │
                    TLS 1.3 Secure Channel
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼────────┐   ┌────────▼────────┐   ┌──────▼──────────┐
│  FL Client     │   │  FL Client      │   │  FL Client      │
│  (Hospital A)  │   │  (Hospital B)   │   │  (Hospital C)   │
│                │   │                 │   │                 │
│ ┌────────────┐ │   │ ┌────────────┐  │   │ ┌────────────┐  │
│ │  Privacy   │ │   │ │  Privacy   │  │   │ │  Privacy   │  │
│ │  Engine    │ │   │ │  Engine    │  │   │ │  Engine    │  │
│ │  (DP-SGD)  │ │   │ │  (DP-SGD)  │  │   │ │  (DP-SGD)  │  │
│ └────────────┘ │   │ └────────────┘  │   │ └────────────┘  │
│ ┌────────────┐ │   │ ┌────────────┐  │   │ ┌────────────┐  │
│ │   PACS     │ │   │ │   PACS     │  │   │ │   PACS     │  │
│ │ Connector  │ │   │ │ Connector  │  │   │ │ Connector  │  │
│ └────────────┘ │   │ └────────────┘  │   │ └────────────┘  │
│ ┌────────────┐ │   │ ┌────────────┐  │   │ ┌────────────┐  │
│ │   Local    │ │   │ │   Local    │  │   │ │   Local    │  │
│ │  Trainer   │ │   │ │  Trainer   │  │   │ │  Trainer   │  │
│ └────────────┘ │   │ └────────────┘  │   │ └────────────┘  │
└────────────────┘   └─────────────────┘   └─────────────────┘
```

### Component Descriptions

#### FL Coordinator (Central Server)

**Purpose**: Orchestrates federated training rounds, aggregates client updates, manages global model.

**Subcomponents**:
- **Training Orchestrator**: Manages training rounds, broadcasts models, collects updates
- **Aggregator**: Combines client gradients using FedAvg/FedProx/FedAdam
- **Byzantine Detector**: Identifies and excludes malicious updates
- **Model Registry**: Versions and stores global models with provenance
- **Monitoring System**: Tracks metrics, convergence, client health
- **Audit Logger**: HIPAA-compliant logging of all operations

**Technology Stack**:
- Python 3.10+
- PyTorch 2.0+ (model handling)
- gRPC (communication)
- PostgreSQL (metadata storage)
- Redis (caching, job queue)
- Prometheus + Grafana (monitoring)

#### FL Client (Hospital-Side)

**Purpose**: Trains models on local data, applies privacy mechanisms, sends encrypted updates.

**Subcomponents**:
- **Privacy Engine**: Applies DP-SGD (gradient clipping + noise)
- **PACS Connector**: Discovers and loads WSI data from existing PACS
- **Local Trainer**: Trains model on local data
- **Secure Communicator**: Encrypts updates, handles TLS
- **Resource Manager**: Enforces GPU/CPU/disk limits

**Technology Stack**:
- Python 3.10+
- PyTorch 2.0+ (training)
- Opacus (differential privacy)
- TenSEAL (homomorphic encryption)
- gRPC (communication)

### Data Flow

#### Training Round Sequence

```
1. Coordinator: Broadcast global_model_v{N} to all clients
   ↓
2. Client: Download global_model_v{N}
   ↓
3. Client: Load local data from PACS
   ↓
4. Client: Train model for E local epochs
   ↓
5. Client: Compute gradients Δw
   ↓
6. Client: Apply DP-SGD (clip + noise) → Δw_private
   ↓
7. Client: Encrypt Δw_private → Δw_encrypted
   ↓
8. Client: Send Δw_encrypted to coordinator
   ↓
9. Coordinator: Collect updates from all clients
   ↓
10. Coordinator: Byzantine detection (flag outliers)
   ↓
11. Coordinator: Aggregate valid updates → Δw_global
   ↓
12. Coordinator: Update global_model_v{N+1} = global_model_v{N} + Δw_global
   ↓
13. Coordinator: Save global_model_v{N+1} to registry
   ↓
14. Repeat from step 1
```

## Core Algorithms

### 1. FedAvg (Federated Averaging)

**Purpose**: Weighted averaging of client model updates.

**Algorithm**:
```python
def fedavg_aggregate(client_updates, client_weights):
    """
    Aggregate client updates using weighted averaging.
    
    Args:
        client_updates: List of gradient dicts from clients
        client_weights: List of weights (typically dataset sizes)
    
    Returns:
        aggregated_update: Weighted average of client updates
    """
    total_weight = sum(client_weights)
    aggregated_update = {}
    
    for param_name in client_updates[0].keys():
        weighted_sum = sum(
            w * client_updates[i][param_name] 
            for i, w in enumerate(client_weights)
        )
        aggregated_update[param_name] = weighted_sum / total_weight
    
    return aggregated_update
```

**Correctness Property**: 
- Invariant: `aggregated_update = Σ(w_i * Δw_i) / Σ(w_i)`
- Metamorphic: Order of client updates doesn't affect result

### 2. FedProx (Federated Proximal)

**Purpose**: Handle heterogeneous data distributions with proximal term.

**Algorithm**:
```python
def fedprox_local_train(model, data_loader, global_model, mu=0.01, epochs=5):
    """
    Train local model with proximal term to global model.
    
    Args:
        model: Local model (initialized from global)
        data_loader: Local training data
        global_model: Global model (frozen)
        mu: Proximal term coefficient
        epochs: Number of local epochs
    
    Returns:
        trained_model: Updated local model
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    for epoch in range(epochs):
        for batch_x, batch_y in data_loader:
            optimizer.zero_grad()
            
            # Standard loss
            loss = F.cross_entropy(model(batch_x), batch_y)
            
            # Proximal term: ||w - w_global||^2
            proximal_term = 0.0
            for param, global_param in zip(model.parameters(), global_model.parameters()):
                proximal_term += ((param - global_param) ** 2).sum()
            
            total_loss = loss + (mu / 2) * proximal_term
            total_loss.backward()
            optimizer.step()
    
    return model
```

**Correctness Property**:
- Invariant: `mu = 0` reduces to standard FedAvg
- Metamorphic: Higher `mu` keeps local model closer to global

### 3. DP-SGD (Differentially Private SGD)

**Purpose**: Add calibrated noise to gradients for differential privacy.

**Algorithm**:
```python
def dp_sgd_step(gradients, clipping_bound=1.0, noise_multiplier=1.0):
    """
    Apply differential privacy to gradients.
    
    Args:
        gradients: Raw gradients from backprop
        clipping_bound: L2 norm clipping threshold (C)
        noise_multiplier: Noise scale relative to clipping (σ)
    
    Returns:
        private_gradients: DP-protected gradients
    """
    # Step 1: Clip gradients to bound sensitivity
    grad_norm = torch.norm(gradients)
    clipping_factor = min(1.0, clipping_bound / (grad_norm + 1e-8))
    clipped_gradients = gradients * clipping_factor
    
    # Step 2: Add calibrated Gaussian noise
    noise_std = clipping_bound * noise_multiplier
    noise = torch.randn_like(clipped_gradients) * noise_std
    private_gradients = clipped_gradients + noise
    
    return private_gradients
```

**Privacy Accounting**:
```python
def compute_epsilon(steps, noise_multiplier, delta=1e-5, batch_size=32, dataset_size=10000):
    """
    Compute privacy budget (epsilon) using RDP accounting.
    
    Uses Renyi Differential Privacy for tight composition bounds.
    """
    from opacus.accountants import RDPAccountant
    
    accountant = RDPAccountant()
    sampling_rate = batch_size / dataset_size
    
    accountant.step(noise_multiplier=noise_multiplier, sample_rate=sampling_rate)
    epsilon = accountant.get_epsilon(delta=delta, steps=steps)
    
    return epsilon
```

**Correctness Properties**:
- Invariant: `||clipped_grad|| ≤ clipping_bound`
- Invariant: `epsilon` increases monotonically with steps
- Round-trip: `clip(clip(g)) = clip(g)` (idempotence)

### 4. Secure Aggregation (Homomorphic Encryption)

**Purpose**: Aggregate encrypted gradients without decrypting individual updates.

**Algorithm**:
```python
import tenseal as ts

def secure_aggregate(encrypted_updates, context):
    """
    Aggregate encrypted gradients using homomorphic encryption.
    
    Args:
        encrypted_updates: List of encrypted gradient vectors
        context: TenSEAL encryption context
    
    Returns:
        encrypted_aggregate: Sum of encrypted updates
    """
    # Initialize with first update
    encrypted_aggregate = encrypted_updates[0]
    
    # Homomorphic addition
    for encrypted_update in encrypted_updates[1:]:
        encrypted_aggregate += encrypted_update
    
    return encrypted_aggregate

def client_encrypt_gradients(gradients, context):
    """Encrypt gradients before sending to coordinator."""
    flat_gradients = torch.cat([g.flatten() for g in gradients])
    encrypted = ts.ckks_vector(context, flat_gradients.tolist())
    return encrypted

def coordinator_decrypt_aggregate(encrypted_aggregate, context):
    """Decrypt only the final aggregated result."""
    decrypted = encrypted_aggregate.decrypt()
    return torch.tensor(decrypted)
```

**Correctness Property**:
- Homomorphic: `decrypt(sum(encrypt(g_i))) = sum(g_i)`
- Round-trip: `decrypt(encrypt(g)) = g` (within precision)

### 5. Byzantine Detection (Krum)

**Purpose**: Detect and exclude malicious client updates.

**Algorithm**:
```python
def krum_aggregate(client_updates, f=1):
    """
    Select most representative update using Krum algorithm.
    
    Args:
        client_updates: List of gradient vectors
        f: Maximum number of Byzantine clients to tolerate
    
    Returns:
        selected_update: Most representative (non-Byzantine) update
    """
    n = len(client_updates)
    n_select = n - f - 2  # Number of closest neighbors to consider
    
    scores = []
    for i, update_i in enumerate(client_updates):
        # Compute distances to all other updates
        distances = []
        for j, update_j in enumerate(client_updates):
            if i != j:
                dist = torch.norm(update_i - update_j)
                distances.append(dist)
        
        # Score = sum of distances to n_select closest neighbors
        distances.sort()
        score = sum(distances[:n_select])
        scores.append(score)
    
    # Select update with minimum score (most representative)
    selected_idx = scores.index(min(scores))
    return client_updates[selected_idx]
```

**Correctness Properties**:
- Invariant: Selected update has minimum distance to neighbors
- Error condition: Extreme outliers (10x magnitude) detected

### 6. Gradient Compression (Quantization)

**Purpose**: Reduce bandwidth by quantizing gradients to lower precision.

**Algorithm**:
```python
def quantize_gradients(gradients, num_bits=8):
    """
    Quantize gradients to reduce transmission size.
    
    Args:
        gradients: Full-precision gradients
        num_bits: Target bit-width (4, 8, or 16)
    
    Returns:
        quantized: Quantized gradients
        scale: Quantization scale factor
        zero_point: Quantization zero point
    """
    # Compute quantization parameters
    min_val = gradients.min()
    max_val = gradients.max()
    
    qmin = 0
    qmax = 2 ** num_bits - 1
    
    scale = (max_val - min_val) / (qmax - qmin)
    zero_point = qmin - min_val / scale
    
    # Quantize
    quantized = torch.clamp(
        torch.round(gradients / scale + zero_point),
        qmin, qmax
    ).to(torch.uint8)
    
    return quantized, scale, zero_point

def dequantize_gradients(quantized, scale, zero_point):
    """Dequantize gradients for aggregation."""
    return (quantized.float() - zero_point) * scale
```

**Correctness Properties**:
- Round-trip: `||dequantize(quantize(g)) - g|| ≤ scale/2` (bounded error)
- Invariant: Quantized size < original size

## Data Models

### Training Round Metadata

```python
@dataclass
class TrainingRound:
    round_id: int
    global_model_version: int
    start_time: datetime
    end_time: Optional[datetime]
    participants: List[str]  # Client IDs
    aggregation_algorithm: str  # "fedavg", "fedprox", "fedadam"
    convergence_metrics: Dict[str, float]  # loss, accuracy, grad_norm
    status: str  # "in_progress", "completed", "failed"
```

### Client Update

```python
@dataclass
class ClientUpdate:
    client_id: str
    round_id: int
    model_version: int
    gradients: Dict[str, torch.Tensor]  # param_name -> gradient
    dataset_size: int
    training_time_seconds: float
    privacy_epsilon: float
    is_encrypted: bool
    compression_method: Optional[str]  # "quantize_8bit", "sparsify_10pct"
```

### Global Model Checkpoint

```python
@dataclass
class ModelCheckpoint:
    version: int
    round_id: int
    timestamp: datetime
    model_state_dict: Dict[str, torch.Tensor]
    optimizer_state_dict: Dict[str, Any]
    contributors: List[str]  # Client IDs that contributed
    metrics: Dict[str, float]  # validation loss, accuracy
    provenance: Dict[str, Any]  # training config, hyperparams
```

### Privacy Budget Tracker

```python
@dataclass
class PrivacyBudget:
    client_id: str
    total_epsilon: float
    total_delta: float
    epsilon_per_round: List[float]
    budget_limit: float  # Maximum allowed epsilon
    is_exhausted: bool
```

### Audit Log Entry

```python
@dataclass
class AuditLogEntry:
    timestamp: datetime
    event_type: str  # "round_start", "update_received", "aggregation_complete"
    client_id: Optional[str]
    round_id: int
    details: Dict[str, Any]
    hash: str  # SHA-256 for tamper detection
```

## Communication Protocol

### gRPC Service Definition

```protobuf
service FederatedLearning {
    // Client requests global model
    rpc GetGlobalModel(ModelRequest) returns (ModelResponse);
    
    // Client submits encrypted update
    rpc SubmitUpdate(ClientUpdate) returns (SubmitResponse);
    
    // Client queries training status
    rpc GetTrainingStatus(StatusRequest) returns (StatusResponse);
    
    // Coordinator broadcasts round start
    rpc BroadcastRoundStart(RoundStartMessage) returns (Ack);
}

message ModelRequest {
    string client_id = 1;
    int32 current_version = 2;
}

message ModelResponse {
    int32 version = 1;
    bytes model_state = 2;  // Serialized state_dict
    TrainingConfig config = 3;
}

message ClientUpdate {
    string client_id = 1;
    int32 round_id = 2;
    bytes encrypted_gradients = 3;
    int32 dataset_size = 4;
    float privacy_epsilon = 5;
}
```

### TLS Configuration

```python
# Coordinator TLS setup
def create_secure_server():
    server_credentials = grpc.ssl_server_credentials(
        [(server_key, server_cert)],
        root_certificates=ca_cert,
        require_client_auth=True  # Mutual TLS
    )
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    server.add_secure_port('[::]:50051', server_credentials)
    return server

# Client TLS setup
def create_secure_channel(coordinator_address):
    channel_credentials = grpc.ssl_channel_credentials(
        root_certificates=ca_cert,
        private_key=client_key,
        certificate_chain=client_cert
    )
    channel = grpc.secure_channel(coordinator_address, channel_credentials)
    return channel
```

## Integration with Existing Systems

### PACS Integration

```python
from src.clinical.pacs import PACSService

class FLPACSConnector:
    """Reuse existing PACS infrastructure for data discovery."""
    
    def __init__(self, pacs_config_path: str):
        self.pacs_service = PACSService(config_path=pacs_config_path)
    
    def discover_wsi_studies(self, start_date: str, end_date: str) -> List[str]:
        """Query PACS for WSI studies in date range."""
        studies = self.pacs_service.query_studies(
            study_date_range=(start_date, end_date),
            modality="SM"  # Slide Microscopy
        )
        return [s.study_instance_uid for s in studies]
    
    def load_wsi_data(self, study_uid: str) -> torch.Tensor:
        """Retrieve and preprocess WSI for training."""
        result = self.pacs_service.retrieve_study(study_uid)
        # Convert DICOM to tensor, apply preprocessing
        return preprocess_wsi(result.file_paths[0])
```

### Model Registry Integration

```python
class FLModelRegistry:
    """Version and store global models with provenance."""
    
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(self, checkpoint: ModelCheckpoint):
        """Save model checkpoint with metadata."""
        checkpoint_path = self.storage_path / f"model_v{checkpoint.version}.pt"
        torch.save({
            'model_state_dict': checkpoint.model_state_dict,
            'optimizer_state_dict': checkpoint.optimizer_state_dict,
            'metadata': asdict(checkpoint)
        }, checkpoint_path)
        
        # Update version index
        self._update_version_index(checkpoint)
    
    def load_checkpoint(self, version: int) -> ModelCheckpoint:
        """Load specific model version."""
        checkpoint_path = self.storage_path / f"model_v{version}.pt"
        data = torch.load(checkpoint_path)
        return ModelCheckpoint(**data['metadata'])
```

## Deployment Architecture

### Coordinator Deployment (Docker)

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/

# Expose gRPC port
EXPOSE 50051

# Run coordinator
CMD ["python", "-m", "src.federated.coordinator", "--config", "configs/production.yaml"]
```

### Client Deployment (Hospital-Side)

```yaml
# docker-compose.yml for hospital deployment
version: '3.8'

services:
  fl-client:
    image: histocore-fl-client:latest
    environment:
      - COORDINATOR_ADDRESS=coordinator.example.com:50051
      - CLIENT_ID=hospital_a
      - PACS_CONFIG=/configs/pacs.yaml
      - GPU_MEMORY_LIMIT=8GB
      - CPU_CORES=4
    volumes:
      - ./configs:/configs
      - ./data:/data
      - ./certs:/certs
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 16G
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Correctness Properties Summary

### Property-Based Testing Targets

1. **Training Orchestration**
   - Invariant: Model version increments by 1 per round
   - Invariant: Aggregated updates ≤ active clients

2. **Privacy Engine**
   - Invariant: `||clipped_grad|| ≤ clipping_bound`
   - Invariant: `epsilon` monotonically increases
   - Round-trip: `clip(clip(g)) = clip(g)`

3. **Secure Aggregation**
   - Homomorphic: `decrypt(sum(encrypt(g_i))) = sum(g_i)`
   - Round-trip: `decrypt(encrypt(g)) ≈ g` (within precision)

4. **Byzantine Detection**
   - Invariant: Excluded updates have distance > threshold
   - Error condition: 10x magnitude outliers detected

5. **Gradient Compression**
   - Round-trip: `||dequantize(quantize(g)) - g|| ≤ ε`
   - Invariant: Compressed size < original size

6. **Fault Tolerance**
   - Invariant: Training continues when `active_clients ≥ min_threshold`
   - Error condition: 20% dropout doesn't prevent aggregation

## Performance Considerations

### Scalability Targets

- **Clients**: Support 10-50 hospitals simultaneously
- **Model Size**: Up to 100M parameters (ResNet-50 scale)
- **Round Time**: <10 minutes per round (including aggregation)
- **Bandwidth**: <100 MB per client per round (with compression)
- **Privacy Budget**: ε ≤ 1.0 for strong privacy

### Optimization Strategies

1. **Gradient Compression**: 8-bit quantization reduces bandwidth by 75%
2. **Asynchronous Updates**: Don't wait for slow clients
3. **Model Compression**: Prune 50% of weights for deployment
4. **Caching**: Redis for model checkpoints, avoid repeated downloads
5. **Batching**: Aggregate updates in batches of 10 clients

## Security Considerations

### Threat Model

**Trusted**: Coordinator infrastructure, TLS certificates
**Untrusted**: Client updates (may be malicious)
**Protected**: Individual patient data (never leaves hospital)

### Mitigations

1. **Byzantine Attacks**: Krum/Trimmed Mean detection
2. **Model Inversion**: Differential privacy (ε ≤ 1.0)
3. **Eavesdropping**: TLS 1.3 + homomorphic encryption
4. **Replay Attacks**: Nonce-based message authentication
5. **Denial of Service**: Rate limiting, client quotas

## Testing Strategy

### Unit Tests

- Test each algorithm in isolation (FedAvg, DP-SGD, Krum)
- Mock client updates, verify aggregation correctness
- Test privacy budget accounting

### Property-Based Tests

- Use Hypothesis to generate random client updates
- Verify invariants hold across 100+ scenarios
- Test Byzantine robustness with injected malicious updates

### Integration Tests

- Simulate 5 virtual clients with synthetic data
- Run 10 training rounds, verify convergence
- Test fault tolerance (client dropout, network failures)

### End-to-End Tests

- Deploy coordinator + 3 real clients
- Train on PCam dataset (distributed across clients)
- Verify federated accuracy within 2% of centralized

---

**Document Version**: 1.0  
**Last Updated**: 2026-04-25  
**Status**: Ready for Implementation
