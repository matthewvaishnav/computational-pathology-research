computational-pathology-research
=================================

PyTorch framework for computational pathology. Multiple Instance Learning (MIL)
models, whole slide image processing, distributed medical intelligence.

~195k LOC, 544 Python modules, 310 tests.


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

Clinical integration:
  - PACS connectivity (DICOM C-FIND/C-MOVE/C-STORE)
  - FHIR adapter for patient metadata
  - Longitudinal analysis and patient context

Production infrastructure:
  - FastAPI with JWT authentication
  - Input validation, SQL parameterization
  - WebSocket streaming for real-time inference
  - Docker/K8s deployment configs


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
  dmi/              Distributed Medical Intelligence (expertise weighting)
  clinical/         PACS integration, FHIR adapter, patient context
  streaming/        Real-time WSI processing, WebSocket server
  inference/        Model serving, batch inference
  database/         Connection pooling, parameterized queries
  utils/            Logging, metrics, visualization

tests/              310 test modules
docs/               Technical documentation
scripts/            Deployment, benchmarking, data preparation
experiments/        Experiment configs and results
k8s/                Kubernetes manifests
│   └── model_manager.py    # Version parsing validation
├── inference/               # Inference engines
├── database/                # Database connection management
│   ├── connection.py       # Connection pooling configured
│   └── operations.py       # Parameterized queries
└── utils/                   # Utilities and helpers

tests/                       # Test suite (310 test files)
docs/                        # Documentation (219 markdown files)
scripts/                     # Utility scripts
experiments/                 # Experiment configurations
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

310 test files with property-based testing (Hypothesis).


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

25 production improvements: security, input validation, error handling.


CITATION
--------

    @software{computational_pathology_research,
      title={Computational Pathology Research Framework},
      author={Vaishnav, Matthew},
      year={2026},
      url={https://github.com/matthewvaishnav/computational-pathology-research}
    }
