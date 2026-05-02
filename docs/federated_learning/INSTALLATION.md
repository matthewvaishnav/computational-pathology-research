# Federated Learning Installation Guide

## Prerequisites

### System Requirements

**Coordinator Node:**
- OS: Linux (Ubuntu 20.04+) or Windows 10/11
- CPU: 4+ cores
- RAM: 8GB minimum, 16GB recommended
- Disk: 50GB available space
- Network: Static IP, open ports 8080-8081

**Client Nodes (Hospital Sites):**
- OS: Linux (Ubuntu 20.04+) or Windows 10/11
- CPU: 4+ cores
- RAM: 16GB minimum, 32GB recommended
- GPU: NVIDIA GPU with 8GB+ VRAM (optional but recommended)
- Disk: 100GB available space
- Network: Outbound HTTPS access to coordinator

### Software Dependencies

- Python 3.9+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU support)
- Docker 20.10+ (optional, for containerized deployment)

## Installation Methods

### Method 1: pip Install (Recommended)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install federated learning module
pip install histocore-federated

# Verify installation
python -c "from src.federated import __version__; print(__version__)"
```

### Method 2: From Source

```bash
# Clone repository
git clone https://github.com/yourusername/computational-pathology-research.git
cd computational-pathology-research

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# Verify installation
pytest tests/federated/ -v
```

### Method 3: Docker (Production)

```bash
# Pull coordinator image
docker pull histocore/fl-coordinator:latest

# Pull client image
docker pull histocore/fl-client:latest

# Verify images
docker images | grep histocore
```

## Component Installation

### Coordinator Setup

```bash
# Install coordinator dependencies
pip install histocore-federated[coordinator]

# Generate TLS certificates
python -m src.federated.communication.tls_utils generate \
    --output-dir ./certs \
    --coordinator-host coordinator.example.com

# Initialize coordinator database
python -m src.federated.production.database init \
    --config configs/federated/coordinator.yaml

# Start coordinator
python -m src.federated.production.coordinator_server \
    --config configs/federated/coordinator.yaml
```

### Client Setup

```bash
# Install client dependencies
pip install histocore-federated[client]

# Configure PACS connection
cp configs/federated/client.example.yaml configs/federated/client.yaml
# Edit client.yaml with your PACS settings

# Copy coordinator TLS certificate
cp coordinator-cert.pem ./certs/

# Start client
python -m src.federated.production.client_server \
    --config configs/federated/client.yaml \
    --coordinator-url https://coordinator.example.com:8080
```

## Post-Installation Verification

### Test Coordinator

```bash
# Check coordinator health
curl https://coordinator.example.com:8080/health

# Expected output:
# {"status": "healthy", "version": "1.0.0", "clients": 0}
```

### Test Client Connection

```bash
# Run client connectivity test
python -m src.federated.client.test_connection \
    --coordinator-url https://coordinator.example.com:8080 \
    --cert ./certs/coordinator-cert.pem

# Expected output:
# ✓ TLS connection successful
# ✓ Authentication successful
# ✓ Client registered: client-abc123
```

### Run Simulation

```bash
# Run 3-client simulation
python -m src.federated.production.simulate \
    --num-clients 3 \
    --num-rounds 5 \
    --dataset synthetic

# Expected output:
# Round 1/5: loss=0.693, accuracy=0.502
# Round 2/5: loss=0.612, accuracy=0.651
# ...
# ✓ Simulation complete
```

## Troubleshooting

### Common Issues

**Issue: TLS certificate verification failed**
```bash
# Solution: Regenerate certificates
python -m src.federated.communication.tls_utils generate \
    --output-dir ./certs \
    --coordinator-host <actual-coordinator-ip>
```

**Issue: CUDA out of memory**
```bash
# Solution: Reduce batch size in client config
# Edit configs/federated/client.yaml:
training:
  batch_size: 16  # Reduce from 32
  gradient_accumulation_steps: 2  # Increase to compensate
```

**Issue: PACS connection timeout**
```bash
# Solution: Test PACS connectivity
python -m src.clinical.pacs.pacs_service test \
    --host <pacs-host> \
    --port <pacs-port> \
    --aet <your-aet>
```

## Next Steps

- [Configuration Guide](CONFIGURATION.md) - Configure coordinator and clients
- [Quick Start](QUICKSTART.md) - Run your first federated training
- [API Reference](API_REFERENCE.md) - Detailed API documentation
- [Troubleshooting](TROUBLESHOOTING.md) - Common issues and solutions

## Support

- GitHub Issues: https://github.com/yourusername/computational-pathology-research/issues
- Documentation: https://histocore.readthedocs.io
- Email: support@histocore.org
