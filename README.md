computational-pathology-research
=================================

PyTorch framework for computational pathology. Multiple Instance Learning (MIL)
models, whole slide image processing, federated learning integration, and distributed medical intelligence.

5,000+ tests across 310 test modules.


WHAT IT DOES
------------

Core models:
  - Attention-based MIL (nnMIL, AttentionMIL, CLAM, TransMIL)
  - WSI processing pipeline (OpenSlide: .svs, .tiff, .ndpi, DICOM)
  - Model interpretability (Grad-CAM, attention heatmaps)
  - Training optimizations (torch.compile, AMP, channels_last)

Distributed Medical Intelligence (DMI):
  - Expertise-weighted collaboration between medical centers
  - Knowledge synthesis from multiple institutions
  - Specialization matching and contribution weighting

Federated Learning Integration:
  - PathologyFL with expertise-weighted aggregation (hospital types, cancer-specific strategies)
  - Production security (DP-SGD, secure aggregation with TenSEAL, Byzantine robustness)
  - Multi-institutional AI training without sharing patient data
  - HIPAA-compliant federated learning with audit logging

Clinical integration:
  - PACS connectivity (DICOM C-FIND/C-MOVE/C-STORE)
  - FHIR adapter for patient metadata
  - Longitudinal analysis and patient context

Production infrastructure:
  - FastAPI with JWT authentication
  - Input validation, SQL parameterization
  - WebSocket streaming for real-time inference
  - Docker/K8s deployment configs


FEDERATED LEARNING INTEGRATION
------------------------------

HistoCore integrates advanced federated learning capabilities using the Flower (flwr) framework
with pathology-specific extensions:

PathologyFL Features:
  - Expertise-weighted aggregation (cancer centers vs community hospitals)
  - Cancer-type specific strategies (breast, lung, prostate, colorectal)
  - Slide quality assessment and automatic weighting
  - Attention-aware aggregation for MIL models

Production Security:
  - Differential Privacy (DP-SGD) with gradient clipping and noise calibration
  - Secure aggregation using TenSEAL homomorphic encryption
  - Byzantine robustness with Krum algorithm and coordinate-wise median
  - HIPAA-compliant audit logging and privacy budget tracking

Multi-institutional deployment:
  - Direct integration with hospital PACS systems
  - Preserves institutional autonomy while enabling collaboration
  - No patient data sharing - only encrypted model updates
  - Regulatory compliance for clinical environments


SECURITY
--------

Recent hardening (25 commits):
  - Command injection fixes (removed shell=True)
  - Path traversal protection (4 modules)
  - Input validation (DICOM size limits, slide ID format, array bounds)
  - Authentication (WebSocket tokens, origin validation)
  - Network security (HTTPS enforcement, STARTTLS, connection pooling)
  - SQL parameterization, graceful shutdown

PathologyFL security (14 commits):
  - Removed fake differential privacy implementation
  - TenSEAL and Opacus now required (no silent degradation)
  - Rate limiting on coordinator endpoints (100 req/min)
  - Pydantic validation for all API requests
  - Fixed model weight serialization
  - Removed legacy unhardened exports

DIRECTORY STRUCTURE
-------------------

src/
  api/              FastAPI server, JWT auth, input validation
  models/           MIL implementations (nnMIL, AttentionMIL, CLAM, TransMIL)
  training/         Training loops, optimizers, distributed training
  data/             Data loaders, WSI pipeline, preprocessing
  federated/        Federated learning integration (PathologyFL, secure aggregation)
  dmi/              Distributed Medical Intelligence (expertise weighting)
  clinical/         PACS integration, FHIR adapter, patient context
  streaming/        Real-time WSI processing, WebSocket server
  inference/        Model serving, batch inference
  database/         Connection pooling, parameterized queries
  utils/            Logging, metrics, visualization

tests/              5,071 test modules
docs/               Technical documentation
scripts/            Deployment, benchmarking, data preparation
experiments/        Experiment configs and results
k8s/                Kubernetes manifests
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/matthewvaishnav/computational-pathology-research.git
cd computational-pathology-research

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Basic Usage

```python


INSTALLATION
------------

    git clone https://github.com/matthewvaishnav/computational-pathology-research.git
    cd computational-pathology-research
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    pip install -e .

For federated learning with privacy guarantees:

    pip install tenseal opacus


USAGE
-----

Train a model:

    python -m src.training.train --dataset pcam --model nnmil --epochs 20

Run inference:

    python -m src.inference.predict --checkpoint model.pth --slide path/to/slide.svs

Start API server:

    python -m src.api.main

Run tests:

    pytest tests/ -v


SECURITY FEATURES
-----------------

Authentication:
  - JWT tokens with proper validation
  - Input sanitization (username, email, password)
  - WebSocket authentication with origin validation

Input validation:
  - File size limits (DICOM uploads)
  - Path traversal protection
  - Array bounds checking
  - Slide ID format validation

Network security:
  - HTTPS enforcement
  - SMTP STARTTLS
  - Connection pooling with retry limits
  - Timeout enforcement

Database:
  - Parameterized SQL queries (SQLAlchemy)
  - Connection pooling (pool_size=10, max_overflow=20)
  - Graceful shutdown


TESTING
-------

Run all tests:

    pytest tests/ -v

With coverage:

    pytest tests/ --cov=src --cov-report=html

Test categories:

    pytest tests/api/ -v          # API tests
    pytest tests/security/ -v     # Security tests
    pytest tests/clinical/ -v     # Clinical workflow tests

5,071 test files with property-based testing (Hypothesis).


DOCUMENTATION
-------------

See docs/ for technical documentation:

  - FRAMEWORK_OVERVIEW.md           System architecture
  - DOCS_INDEX.md                   Documentation index
  - PCAM_REAL_RESULTS.md            Benchmark results
  - CLINICAL_WORKFLOW_INTEGRATION.md Clinical deployment

```bash
# Start production API server
python -m src.api.main

# With custom configuration
python -m src.api.main --config config/production.yaml

# Health check
curl http://localhost:8000/health

# Metrics endpoint


PRODUCTION DEPLOYMENT
---------------------

API server:

    python -m src.api.main
    curl http://localhost:8000/health

Database setup:

    from src.database.connection import DatabaseManager
    db = DatabaseManager(
        database_url="postgresql://user:pass@localhost/db",
        pool_size=10,
        max_overflow=20
    )

DMI deployment (multi-center):

    python -m src.dmi.coordinator --config configs/dmi/coordinator.yaml
    python -m src.dmi.client --config configs/dmi/client.yaml

Federated learning deployment:

    # Start FL coordinator
    python -m src.federated.coordinator --config configs/fl/coordinator.yaml
    
    # Start hospital clients
    python -m src.federated.client --config configs/fl/hospital_client.yaml


CONFIGURATION
-------------

config/security.yaml:

    authentication:
      jwt_secret: ${JWT_SECRET}
      token_expiration: 3600
    
    input_validation:
      max_file_size: 104857600
      allowed_extensions: ['.dcm', '.svs', '.tiff']
    
    network:
      enforce_https: true
      connection_timeout: 30


LICENSE
-------

MIT License. See LICENSE file.


NOTES
-----

Research framework with production hardening. Independent security audit
recommended before clinical deployment.

No trained models included. Train on your own datasets.

For research purposes only. Not FDA-approved or CE-marked for clinical use.

Security improvements: 39 commits (25 general + 14 PathologyFL-specific)
covering authentication, input validation, privacy guarantees, and error handling.


CITATION
--------

    @software{computational_pathology_research,
      title={Computational Pathology Research Framework},
      author={Vaishnav, Matthew},
      year={2026},
      url={https://github.com/matthewvaishnav/computational-pathology-research}
    }
