# Federated Learning API Reference

## Core Components

### Coordinator

#### `FederatedCoordinator`

Main orchestrator for FL training.

```python
from src.federated.coordinator import FederatedCoordinator

coordinator = FederatedCoordinator(
    config_path="configs/federated/coordinator.yaml",
    model_architecture=MyModel(),
    device="cuda"
)

# Start training
coordinator.start_training(
    num_rounds=100,
    min_clients=3
)

# Get training status
status = coordinator.get_status()
# Returns: {"round": 42, "clients": 5, "loss": 0.234, "accuracy": 0.891}
```

**Methods:**

- `start_training(num_rounds, min_clients)` - Start FL training
- `stop_training()` - Stop training gracefully
- `get_status()` - Get current training status
- `get_global_model()` - Get latest global model
- `rollback_to_version(version)` - Rollback to previous model version

#### `TrainingOrchestrator`

Low-level training orchestration.

```python
from src.federated.coordinator.orchestrator import TrainingOrchestrator

orchestrator = TrainingOrchestrator(
    aggregator=aggregator,
    model_registry=registry,
    monitoring=monitor
)

# Initialize round
round_id = orchestrator.initialize_round(
    global_model=model,
    config=training_config
)

# Broadcast model
orchestrator.broadcast_model(round_id, client_ids)

# Collect updates
updates = orchestrator.collect_updates(round_id, timeout=600)

# Aggregate
new_model = orchestrator.aggregate_updates(updates)
```

### Client

#### `FederatedClient`

Client-side training component.

```python
from src.federated.client import FederatedClient

client = FederatedClient(
    config_path="configs/federated/client.yaml",
    coordinator_url="https://coordinator.example.com:8080"
)

# Connect to coordinator
client.connect()

# Start training loop
client.start_training_loop()

# Manual training round
update = client.train_round(
    global_model=model,
    local_epochs=5
)
```

**Methods:**

- `connect()` - Connect to coordinator
- `disconnect()` - Disconnect gracefully
- `start_training_loop()` - Start automatic training loop
- `train_round(global_model, local_epochs)` - Execute single training round
- `get_local_metrics()` - Get local training metrics

#### `LocalTrainer`

Local model training.

```python
from src.federated.client.trainer import LocalTrainer

trainer = LocalTrainer(
    model=model,
    optimizer=optimizer,
    privacy_engine=privacy_engine,
    device="cuda"
)

# Train locally
metrics = trainer.train(
    dataloader=train_loader,
    epochs=5
)
# Returns: {"loss": 0.234, "accuracy": 0.891, "gradient_norm": 1.23}

# Get model update
update = trainer.get_update()
```

### Aggregation

#### `FedAvgAggregator`

Federated averaging aggregation.

```python
from src.federated.aggregation import FedAvgAggregator

aggregator = FedAvgAggregator()

# Aggregate updates
global_update = aggregator.aggregate(
    updates=[update1, update2, update3],
    weights=[100, 150, 200]  # Dataset sizes
)
```

#### `FedProxAggregator`

FedProx with proximal term.

```python
from src.federated.aggregation import FedProxAggregator

aggregator = FedProxAggregator(mu=0.01)

global_update = aggregator.aggregate(
    updates=updates,
    weights=weights,
    global_model=current_global_model
)
```

#### `FedAdamAggregator`

FedAdam with adaptive learning.

```python
from src.federated.aggregation import FedAdamAggregator

aggregator = FedAdamAggregator(
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8
)

global_update = aggregator.aggregate(
    updates=updates,
    weights=weights
)
```

### Privacy

#### `PrivacyEngine`

Differential privacy (DP-SGD).

```python
from src.federated.privacy import PrivacyEngine

privacy_engine = PrivacyEngine(
    target_epsilon=1.0,
    target_delta=1e-5,
    noise_multiplier=1.1,
    max_grad_norm=1.0
)

# Apply DP to gradients
private_gradients = privacy_engine.privatize_gradients(
    gradients=raw_gradients,
    dataset_size=1000
)

# Get privacy spent
epsilon, delta = privacy_engine.get_privacy_spent()
```

**Methods:**

- `privatize_gradients(gradients, dataset_size)` - Apply DP-SGD
- `get_privacy_spent()` - Get cumulative privacy loss
- `reset_privacy_budget()` - Reset for new training phase
- `check_budget_exhausted()` - Check if budget exceeded

#### `SecureAggregator`

Homomorphic encryption for secure aggregation.

```python
from src.federated.privacy import SecureAggregator

aggregator = SecureAggregator(
    context=tenseal_context
)

# Client-side encryption
encrypted_update = aggregator.encrypt_update(update)

# Coordinator-side aggregation
encrypted_result = aggregator.aggregate_encrypted(
    encrypted_updates=[enc1, enc2, enc3],
    weights=[100, 150, 200]
)

# Decrypt final result
final_update = aggregator.decrypt_result(encrypted_result)
```

### Byzantine Detection

#### `ByzantineDetector`

Detect malicious updates.

```python
from src.federated.byzantine import ByzantineDetector

detector = ByzantineDetector(
    algorithm="krum",  # krum | trimmed_mean | median
    threshold_std=3.0
)

# Detect outliers
filtered_updates, excluded_indices = detector.filter_updates(
    updates=all_updates,
    weights=weights
)

# Get detection stats
stats = detector.get_detection_stats()
# Returns: {"total": 10, "excluded": 2, "exclusion_rate": 0.2}
```

### Compression

#### `GradientCompressor`

Gradient compression for bandwidth reduction.

```python
from src.federated.compression import GradientCompressor

# Quantization
compressor = GradientCompressor(
    mode="quantization",
    quantization_bits=8
)

compressed = compressor.compress(gradients)
decompressed = compressor.decompress(compressed)

# Sparsification
compressor = GradientCompressor(
    mode="sparsification",
    sparsification_ratio=0.1  # Top 10%
)

compressed = compressor.compress(gradients)
decompressed = compressor.decompress(compressed)

# Get compression stats
stats = compressor.get_stats()
# Returns: {"original_size": 1000000, "compressed_size": 125000, "ratio": 8.0}
```

### Fault Tolerance

#### `CheckpointManager`

Checkpoint-based recovery.

```python
from src.federated.fault_tolerance import CheckpointManager

manager = CheckpointManager(
    checkpoint_dir="./checkpoints",
    max_checkpoints=5
)

# Save checkpoint
manager.save_checkpoint(
    round_id=42,
    global_model=model,
    optimizer_state=optimizer.state_dict(),
    metadata={"loss": 0.234}
)

# Load latest checkpoint
checkpoint = manager.load_latest_checkpoint()
# Returns: {"round_id": 42, "model": ..., "optimizer_state": ..., "metadata": ...}

# List checkpoints
checkpoints = manager.list_checkpoints()
```

#### `NetworkMonitor`

Network health monitoring.

```python
from src.federated.fault_tolerance import NetworkMonitor

monitor = NetworkMonitor(
    heartbeat_interval=60,
    failure_threshold=3
)

# Start monitoring
monitor.start()

# Check client health
health = monitor.get_client_health(client_id)
# Returns: {"status": "healthy", "latency_ms": 45, "last_seen": "2026-05-02T10:30:00"}

# Get all clients
clients = monitor.get_all_clients()
```

#### `ReconnectionHandler`

Automatic reconnection.

```python
from src.federated.fault_tolerance import ReconnectionHandler

handler = ReconnectionHandler(
    strategy="exponential_backoff",
    max_attempts=10,
    base_delay=1.0
)

# Attempt reconnection
success = handler.reconnect(
    client_id=client_id,
    connect_fn=lambda: client.connect()
)

# Get reconnection stats
stats = handler.get_stats(client_id)
# Returns: {"attempts": 3, "last_attempt": "2026-05-02T10:30:00", "success": True}
```

### Async Training

#### `AsyncCoordinator`

Asynchronous training coordination.

```python
from src.federated.async_training import AsyncCoordinator, SyncMode

coordinator = AsyncCoordinator(
    sync_mode=SyncMode.SEMI_SYNCHRONOUS,
    min_client_percentage=0.8,
    staleness_alpha=0.5
)

# Process async update
coordinator.process_update(
    client_id=client_id,
    update=update,
    model_version=42
)

# Check if ready to aggregate
if coordinator.should_aggregate():
    updates = coordinator.get_pending_updates()
    # Aggregate...
```

#### `StalenessWeighting`

Staleness-aware weighting.

```python
from src.federated.async_training import StalenessWeighting

weighting = StalenessWeighting(alpha=0.5)

# Compute staleness weight
weight = weighting.compute_weight(
    staleness=5,  # 5 versions behind
    base_weight=100  # Dataset size
)
# Returns: 3.125 (exponential decay)
```

### Monitoring

#### `MonitoringSystem`

Real-time metrics tracking.

```python
from src.federated.monitoring import MonitoringSystem

monitor = MonitoringSystem(
    enable_prometheus=True,
    enable_tensorboard=True,
    tensorboard_dir="./runs"
)

# Log metrics
monitor.log_round_metrics(
    round_id=42,
    metrics={
        "global_loss": 0.234,
        "global_accuracy": 0.891,
        "num_clients": 5,
        "aggregation_time": 12.3
    }
)

# Log client metrics
monitor.log_client_metrics(
    client_id=client_id,
    round_id=42,
    metrics={
        "local_loss": 0.245,
        "local_accuracy": 0.885,
        "training_time": 45.6
    }
)

# Get convergence status
status = monitor.check_convergence(window=5)
# Returns: {"converged": False, "stalled": False, "rounds_without_improvement": 2}
```

### Model Registry

#### `ModelRegistry`

Model versioning and provenance.

```python
from src.federated.model_registry import ModelRegistry

registry = ModelRegistry(
    checkpoint_dir="./checkpoints/federated",
    max_versions=10
)

# Save model version
version_id = registry.save_model(
    model=global_model,
    round_id=42,
    metadata={
        "loss": 0.234,
        "accuracy": 0.891,
        "participants": ["client1", "client2", "client3"]
    }
)

# Load model version
model, metadata = registry.load_model(version_id)

# List versions
versions = registry.list_versions()

# Rollback
registry.rollback_to_version(version_id=38)
```

## Data Models

### `TrainingRound`

```python
from src.federated.models import TrainingRound

round = TrainingRound(
    round_id=42,
    global_model_version=41,
    participants=["client1", "client2"],
    start_time=datetime.now(),
    end_time=None,
    status="in_progress"
)
```

### `ClientUpdate`

```python
from src.federated.models import ClientUpdate

update = ClientUpdate(
    client_id="hospital-001",
    round_id=42,
    model_update=gradient_dict,
    dataset_size=1000,
    training_time=45.6,
    local_loss=0.245,
    local_accuracy=0.885
)
```

### `PrivacyBudget`

```python
from src.federated.models import PrivacyBudget

budget = PrivacyBudget(
    target_epsilon=1.0,
    target_delta=1e-5,
    spent_epsilon=0.42,
    spent_delta=4.2e-6,
    rounds_completed=42
)
```

## Configuration

### Load Config

```python
from src.federated.config import load_config

config = load_config("configs/federated/coordinator.yaml")

# Access settings
num_rounds = config.training.num_rounds
privacy_epsilon = config.privacy.target_epsilon
```

### Validate Config

```python
from src.federated.config import validate_config

errors = validate_config(config, config_type="coordinator")

if errors:
    print(f"Config errors: {errors}")
else:
    print("Config valid")
```

## Utilities

### TLS Setup

```python
from src.federated.communication.tls_utils import generate_certificates

# Generate coordinator certificates
generate_certificates(
    output_dir="./certs",
    coordinator_host="coordinator.example.com",
    validity_days=365
)
```

### Metrics Export

```python
from src.federated.monitoring import export_metrics

# Export to Prometheus
export_metrics(
    metrics=training_metrics,
    format="prometheus",
    output_file="metrics.txt"
)

# Export to TensorBoard
export_metrics(
    metrics=training_metrics,
    format="tensorboard",
    output_dir="./runs"
)
```

## Error Handling

All API methods raise specific exceptions:

```python
from src.federated.exceptions import (
    PrivacyBudgetExhausted,
    ByzantineAttackDetected,
    ClientDropoutError,
    AggregationError,
    CheckpointNotFoundError
)

try:
    coordinator.start_training(num_rounds=100)
except PrivacyBudgetExhausted as e:
    print(f"Privacy budget exhausted: {e}")
except ByzantineAttackDetected as e:
    print(f"Byzantine attack detected: {e}")
except ClientDropoutError as e:
    print(f"Client dropout: {e}")
```

## Examples

### Complete Training Pipeline

```python
from src.federated import FederatedCoordinator, FederatedClient

# Coordinator
coordinator = FederatedCoordinator(
    config_path="configs/federated/coordinator.yaml",
    model_architecture=MyModel()
)

coordinator.start_training(num_rounds=100, min_clients=3)

# Client
client = FederatedClient(
    config_path="configs/federated/client.yaml",
    coordinator_url="https://coordinator.example.com:8080"
)

client.connect()
client.start_training_loop()
```

### Custom Aggregation

```python
from src.federated.aggregation import BaseAggregator

class CustomAggregator(BaseAggregator):
    def aggregate(self, updates, weights):
        # Custom aggregation logic
        weighted_sum = sum(u * w for u, w in zip(updates, weights))
        return weighted_sum / sum(weights)

coordinator = FederatedCoordinator(
    config_path="configs/federated/coordinator.yaml",
    model_architecture=MyModel(),
    aggregator=CustomAggregator()
)
```

### Privacy-Preserving Training

```python
from src.federated.privacy import PrivacyEngine

privacy_engine = PrivacyEngine(
    target_epsilon=1.0,
    target_delta=1e-5,
    noise_multiplier=1.1,
    max_grad_norm=1.0
)

trainer = LocalTrainer(
    model=model,
    optimizer=optimizer,
    privacy_engine=privacy_engine
)

metrics = trainer.train(dataloader, epochs=5)

epsilon, delta = privacy_engine.get_privacy_spent()
print(f"Privacy spent: ε={epsilon:.3f}, δ={delta:.2e}")
```

## Next Steps

- [Installation Guide](INSTALLATION.md) - Setup instructions
- [Configuration Guide](CONFIGURATION.md) - Config reference
- [Troubleshooting](TROUBLESHOOTING.md) - Common issues
- [Examples](../examples/federated/) - Code examples

