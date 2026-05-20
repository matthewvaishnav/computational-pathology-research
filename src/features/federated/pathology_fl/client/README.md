# Hospital Client for Federated Learning

The Hospital Client system enables hospitals to participate in federated learning while keeping patient data completely local and private.

## Overview

The `HospitalClient` class manages:
- **Local model training** on hospital data
- **Secure communication** with central coordinator
- **Privacy preservation** through differential privacy
- **Model update computation** (only updates sent, never raw data)

## Key Features

### 1. Data Privacy
- Patient data **never leaves the hospital**
- Only model updates (gradients/weights) are sent to coordinator
- Differential privacy protects individual patients
- Configurable privacy budget (epsilon/delta)

### 2. Secure Communication
- TLS/mTLS encryption for all communications
- Certificate-based authentication
- Secure model update transmission

### 3. Local Training
- Full control over local training process
- Configurable epochs, batch size, learning rate
- Local validation and evaluation

### 4. Privacy Accounting
- Track privacy budget usage
- Monitor epsilon consumption
- Ensure privacy guarantees

## Quick Start

```python
from src.federated.client.hospital_client import HospitalClient
import torch.nn as nn
from torch.utils.data import DataLoader

# 1. Initialize your model
model = YourPathologyModel()

# 2. Create hospital client
client = HospitalClient(
    hospital_id="hospital_001",
    model=model,
    coordinator_host="coordinator.example.com",
    coordinator_port=50051,
    use_privacy=True,
    privacy_epsilon=1.0,
)

# 3. Load local hospital data
client.load_local_data(train_loader, val_loader)

# 4. Connect to coordinator
client.connect_to_coordinator()
client.register_with_coordinator()

# 5. Participate in federated learning
client.run_federated_training(
    num_rounds=10,
    num_local_epochs=5,
    batch_size=32,
    learning_rate=0.001,
)
```

## Architecture

```
┌─────────────────────────────────────────┐
│         Hospital Client                 │
├─────────────────────────────────────────┤
│                                         │
│  ┌──────────────────────────────────┐  │
│  │   Local Patient Data             │  │
│  │   (NEVER leaves hospital)        │  │
│  └──────────────────────────────────┘  │
│              ↓                          │
│  ┌──────────────────────────────────┐  │
│  │   Local Model Training           │  │
│  │   - Forward/backward pass        │  │
│  │   - Gradient computation         │  │
│  └──────────────────────────────────┘  │
│              ↓                          │
│  ┌──────────────────────────────────┐  │
│  │   Privacy Engine (Optional)      │  │
│  │   - Gradient clipping            │  │
│  │   - Noise addition               │  │
│  │   - Privacy accounting           │  │
│  └──────────────────────────────────┘  │
│              ↓                          │
│  ┌──────────────────────────────────┐  │
│  │   Model Update Computation       │  │
│  │   (Δw = w_new - w_old)          │  │
│  └──────────────────────────────────┘  │
│              ↓                          │
│  ┌──────────────────────────────────┐  │
│  │   Secure Communication           │  │
│  │   - TLS encryption               │  │
│  │   - Send updates only            │  │
│  └──────────────────────────────────┘  │
│              ↓                          │
└─────────────────────────────────────────┘
              ↓
    Central Coordinator
```

## API Reference

### HospitalClient

#### Initialization

```python
HospitalClient(
    hospital_id: str,
    model: nn.Module,
    coordinator_host: str = "localhost",
    coordinator_port: int = 50051,
    cert_dir: str = "./certs",
    use_privacy: bool = True,
    privacy_epsilon: float = 1.0,
    privacy_delta: float = 1e-5,
    max_grad_norm: float = 1.0,
)
```

**Parameters:**
- `hospital_id`: Unique identifier for the hospital
- `model`: PyTorch model for training
- `coordinator_host`: Central coordinator hostname
- `coordinator_port`: Central coordinator port
- `cert_dir`: Directory for TLS certificates
- `use_privacy`: Enable differential privacy
- `privacy_epsilon`: Privacy budget (lower = more private)
- `privacy_delta`: Privacy delta parameter
- `max_grad_norm`: Maximum gradient norm for clipping

#### Methods

##### load_local_data
```python
client.load_local_data(
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None
)
```
Load local hospital data for training. Data never leaves the hospital.

##### connect_to_coordinator
```python
success = client.connect_to_coordinator() -> bool
```
Establish secure connection to central coordinator.

##### register_with_coordinator
```python
success = client.register_with_coordinator(
    memory_gb: int = 16,
    num_gpus: int = 1,
    supported_algorithms: Optional[List[str]] = None,
) -> bool
```
Register hospital with central coordinator.

##### train_local_model
```python
metrics = client.train_local_model(
    num_epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 0.001,
) -> Dict[str, Any]
```
Train model on local hospital data.

**Returns:** Training metrics (loss, accuracy, time, privacy metrics)

##### participate_in_round
```python
success = client.participate_in_round(
    round_id: int,
    num_local_epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 0.001,
) -> bool
```
Participate in one federated learning round:
1. Receive global model
2. Train locally
3. Compute update
4. Send update to coordinator

##### run_federated_training
```python
success = client.run_federated_training(
    num_rounds: int = 10,
    num_local_epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 0.001,
) -> bool
```
Run complete federated training process.

##### evaluate_local_model
```python
metrics = client.evaluate_local_model() -> Dict[str, float]
```
Evaluate model on local validation data.

##### get_privacy_budget_status
```python
status = client.get_privacy_budget_status() -> Dict[str, float]
```
Get current privacy budget status.

**Returns:**
- `privacy_enabled`: Whether privacy is enabled
- `epsilon_budget`: Total epsilon budget
- `epsilon_used`: Epsilon consumed so far
- `epsilon_remaining`: Remaining epsilon budget
- `delta`: Privacy delta parameter

##### get_client_info
```python
info = client.get_client_info() -> Dict[str, Any]
```
Get comprehensive client information.

## Privacy Guarantees

The hospital client implements **differential privacy** to protect individual patients:

### What is Differential Privacy?

Differential privacy provides a mathematical guarantee that the presence or absence of any single patient in the dataset has minimal impact on the model updates.

### Privacy Parameters

- **Epsilon (ε)**: Privacy budget
  - Lower values = stronger privacy
  - Typical values: 0.1 - 10.0
  - Recommended: 1.0 for medical data

- **Delta (δ)**: Failure probability
  - Typical value: 1e-5
  - Should be much smaller than 1/dataset_size

### How It Works

1. **Gradient Clipping**: Limits sensitivity of individual samples
   ```python
   max_grad_norm = 1.0  # Maximum gradient norm
   ```

2. **Noise Addition**: Adds calibrated Gaussian noise to gradients
   ```python
   noise_scale = noise_multiplier * max_grad_norm / batch_size
   ```

3. **Privacy Accounting**: Tracks cumulative privacy loss
   ```python
   epsilon_used, delta = client.get_privacy_spent()
   ```

### Privacy Budget Management

```python
# Check privacy budget before training
status = client.get_privacy_budget_status()
print(f"Epsilon remaining: {status['epsilon_remaining']}")

# Train with privacy
client.train_local_model(num_epochs=5)

# Check budget after training
status = client.get_privacy_budget_status()
if status['epsilon_remaining'] <= 0:
    print("Privacy budget exhausted!")
```

## Security

### TLS/mTLS Encryption

All communication with the coordinator uses TLS encryption:

```python
# Certificates are automatically managed
client = HospitalClient(
    hospital_id="hospital_001",
    model=model,
    cert_dir="./certs",  # Certificate directory
)
```

### Certificate Management

1. **CA Certificate**: Root certificate authority
2. **Client Certificate**: Hospital-specific certificate
3. **Client Key**: Private key for authentication

Certificates are automatically generated on first connection.

## Examples

### Example 1: Basic Usage

```python
from src.federated.client.hospital_client import HospitalClient

# Initialize client
client = HospitalClient(
    hospital_id="hospital_001",
    model=model,
    use_privacy=False,  # Disable privacy for testing
)

# Load data
client.load_local_data(train_loader, val_loader)

# Train locally
metrics = client.train_local_model(num_epochs=5)
print(f"Loss: {metrics['loss']:.4f}")
```

### Example 2: With Privacy

```python
# Initialize with privacy
client = HospitalClient(
    hospital_id="hospital_002",
    model=model,
    use_privacy=True,
    privacy_epsilon=1.0,
)

# Load data
client.load_local_data(train_loader, val_loader)

# Train with privacy
metrics = client.train_local_model(num_epochs=5)

# Check privacy budget
status = client.get_privacy_budget_status()
print(f"Privacy epsilon used: {status['epsilon_used']:.4f}")
```

### Example 3: Federated Learning

```python
# Initialize client
client = HospitalClient(
    hospital_id="hospital_003",
    model=model,
    coordinator_host="coordinator.example.com",
    use_privacy=True,
)

# Load data
client.load_local_data(train_loader, val_loader)

# Run federated training
client.run_federated_training(
    num_rounds=10,
    num_local_epochs=5,
    batch_size=32,
    learning_rate=0.001,
)
```

## Testing

Run the test suite:

```bash
pytest tests/federated/test_hospital_client.py -v
```

Run the demo:

```bash
python examples/federated_hospital_client_demo.py
```

## Requirements

- Python 3.8+
- PyTorch 1.9+
- grpcio (for secure communication)
- cryptography (for TLS)

Optional:
- opacus (for advanced privacy accounting)

## Compliance

The hospital client is designed to support:

- **HIPAA**: Patient data never leaves hospital
- **GDPR**: Data minimization and privacy by design
- **FDA**: Audit trails and validation support

## Troubleshooting

### Connection Issues

```python
# Check connection
if not client.connect_to_coordinator():
    print("Connection failed - check coordinator address")
```

### Privacy Budget Exhausted

```python
# Monitor privacy budget
status = client.get_privacy_budget_status()
if status['epsilon_remaining'] <= 0:
    # Stop training or increase budget
    client.privacy_epsilon = 2.0  # Increase budget
```

### Certificate Errors

```bash
# Regenerate certificates
rm -rf ./certs
# Certificates will be regenerated on next connection
```

## Contributing

See the main project CONTRIBUTING.md for guidelines.

## License

See the main project LICENSE file.

## References

1. McMahan et al. (2017). "Communication-Efficient Learning of Deep Networks from Decentralized Data"
2. Abadi et al. (2016). "Deep Learning with Differential Privacy"
3. Kairouz et al. (2021). "Advances and Open Problems in Federated Learning"
