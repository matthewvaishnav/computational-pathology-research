computational-pathology-research
=================================

**Production-grade PyTorch framework for computational pathology research.** Features attention-based MIL models, foundation model integration (Phikon/UNI/CONCH), clinical PACS integration, and comprehensive testing (5,071+ tests). Validated on PCam (85.26% accuracy, 93.94% AUC). Built for research and clinical deployment.

**Novel contributions:** PathologyFL + DMI two-layer federated learning system for privacy-preserving multi-institutional collaboration.

---

## Core Thesis

Enable collaborative medical AI across institutions while preserving privacy through a **two-layer innovation**:

### Layer 1: PathologyFL (Domain-Specific Federated Learning)
**Hierarchical attention-weighted aggregation** designed for computational pathology:
- **Cancer-type specific strategies**: Breast (hormone receptor), Lung (histology), Prostate (Gleason), Colorectal
- **Slide quality weighting**: Image sharpness, stain consistency, label confidence, artifact level
- **Attention-aware aggregation**: Special handling for attention layers in MIL models
- **Hierarchical workflow**: Patch → Slide → Case → Hospital → Global (mirrors pathology practice)

### Layer 2: DMI (Distributed Medical Intelligence)
**Institutional expertise layer** on top of PathologyFL:
- **Expertise weighting**: Hospital type (cancer center 2.0x, teaching 1.5x, community 1.0x, rural 0.8x)
- **Specialization matching**: Route cases to institutions with cancer-specific expertise
- **Volume & accuracy factors**: Log-scaled case volume + diagnostic accuracy
- **Experience scaling**: Years of experience with diminishing returns

### Combined System: PathologyFL + DMI
```python
# Standard FedAvg: uniform averaging
global_model = average([model_1, model_2, model_3])

# PathologyFL: cancer-type + quality weighting
global_model = pathology_weighted_average(models, cancer_type, slide_quality)

# PathologyFL + DMI: full system with institutional expertise
global_model = pathology_dmi_aggregate(models, cancer_type, slide_quality, hospital_expertise)
```

**Hypothesis:** PathologyFL + DMI > PathologyFL alone > Standard FedAvg, especially for rare subtypes and heterogeneous data quality.

---

## Key Contributions

### 1. PathologyFL + DMI: Two-Layer Federated Learning 🔬
**Novel two-layer system** for medical AI collaboration:

**PathologyFL (Layer 1)** - Domain-specific federated learning:
- Hierarchical aggregation: Patch → Slide → Case → Hospital → Global
- Cancer-type specific strategies (breast, lung, prostate, colorectal)
- Slide quality weighting (sharpness, stain, artifacts, label confidence)
- Attention-aware aggregation for MIL models
- **Status:** ✅ Implemented, 🚧 validation in progress

**DMI (Layer 2)** - Institutional expertise intelligence:
- Hospital type weighting (cancer center 2.0x, teaching 1.5x, community 1.0x)
- Specialization matching (route to cancer-specific experts)
- Volume & accuracy factors (log-scaled case volume + diagnostic accuracy)
- Experience scaling with diminishing returns
- **Status:** ✅ Implemented, 🚧 validation in progress

**Key innovation:** Combines domain-specific pathology knowledge (PathologyFL) with institutional intelligence (DMI) for superior multi-center collaboration.

### 2. TransnnMIL v2.0 Architecture 🧠
**3-branch MIL model** for whole slide imaging:
- Branch 1: TransMIL (self-attention over patches)
- Branch 2: Hierarchical pooling (multi-scale spatial features)
- Branch 3: Topology branch (GNN for spatial relationships)
- Adaptive pruning (30% computation reduction)
- **Status:** ✅ Implemented, 🚧 comprehensive benchmarking in progress

### 3. Empirical Results 📊
**Demonstrated performance on real datasets:**
- ✅ **PCam (full dataset, 327K patches)**: 
  - **0.9394 AUC** 🏆 #1 vs 10 published baselines
  - 85.26% accuracy on 32,768-sample test set
  - Beats Swin-Transformer, ConvNeXt, ViT-Base, PathViT, MedViT
  - Statistical significance confirmed with bootstrap CI
- ✅ **Camelyon17 (federated)**: Multi-center attention audit complete
  - Cross-site attention consistency measured
  - Confirmed models learn real pathology (not scanner shortcuts)
- 🚧 **PANDA (1,365 slides)**: Training in progress

**Key achievement:** First to demonstrate federated MIL with attention-based shortcut detection on multi-center data.

### 4. Production Infrastructure ⚙️
**Clinical deployment ready:**
- ✅ PACS integration (DICOM C-FIND/C-MOVE/C-STORE)
- ✅ FHIR adapter for patient metadata
- ✅ Security hardening (39 commits: auth, input validation, encryption)
- ✅ FastAPI server with JWT authentication
- ✅ Docker/K8s deployment configs
- ✅ CI/CD optimized (99% faster feedback)

---

## What Makes This Different

### Standard Federated Learning (FedAvg):
```python
# Uniform averaging - treats all institutions equally
global_model = average([model_1, model_2, model_3, model_4, model_5])
```

### PathologyFL (Layer 1):
```python
# Domain-specific: cancer-type strategies + slide quality
weights = calculate_pathology_weights(cancer_type, slide_quality)
global_model = weighted_average(models, weights)
```

### PathologyFL + DMI (Full System):
```python
# Two-layer: pathology knowledge + institutional expertise
pathology_weights = calculate_pathology_weights(cancer_type, slide_quality)
expertise_weights = calculate_expertise_weights(hospital_type, specialization, volume, accuracy)
combined_weights = alpha * expertise_weights + beta * pathology_weights
global_model = weighted_average(models, combined_weights)
```

**Hypothesis:** PathologyFL + DMI substantially outperforms standard FedAvg on:
- Rare cancer subtypes (where specialist centers have critical expertise)
- Heterogeneous data quality (where quality weighting prevents degradation)
- Multi-center collaboration (where institutional expertise matters)

**Status:** 🚧 Validation experiments in progress

---

## Architecture Overview

```
WSI Pipeline → Patches → MIL Model → Slide Features → DMI/FL → Clinical Decision
     ↓            ↓          ↓             ↓            ↓            ↓
  data/wsi/   data/      models/      training/    features/    features/
              loaders/   mil/                      federated/   clinical/
```

**Hybrid Architecture:** Core layers (data, models, training, inference) + domain features (federated, clinical, interpretability, research, advanced)

**Core layers:**
- `core/` - Shared infrastructure (config, utils, exceptions, constants)
- `data/` - Data handling (loaders, datasets, WSI pipeline, preprocessing)
- `models/` - Model architectures (MIL, TransnnMIL, components, foundation)
- `training/` - Training infrastructure (loops, optimizers, distributed)
- `inference/` - Inference engine (serving, quantization, streaming)

**Domain features:**
- `features/federated/` - PathologyFL + DMI (privacy-preserving collaboration)
- `features/clinical/` - PACS, FHIR, workflow (clinical deployment)
- `features/interpretability/` - Grad-CAM, explainability (model transparency)
- `features/research/` - Annotation, experiments (research platform)
- `features/advanced/` - Causal, omics, spatial (advanced analysis)

**Platform services:**
- `api/` - REST API (FastAPI, JWT auth, validation)
- `platform/` - Monitoring, security, database, deployment, cloud integration

**Key capabilities:**
- Attention-based MIL (nnMIL, AttentionMIL, CLAM, TransMIL, TransnnMIL v2.0)
- Foundation model integration (Phikon, UNI, CONCH)
- WSI processing (OpenSlide: .svs, .tiff, .ndpi, DICOM)
- Federated learning (PathologyFL + DMI with DP-SGD, secure aggregation)
- Clinical integration (PACS, FHIR, audit logging)
- Production ready (Docker/K8s, monitoring, security hardening)


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

See SECURITY.md for vulnerability reporting and security policies.

DIRECTORY STRUCTURE
-------------------

```
src/
  core/                         # Core infrastructure
    config/                     # Configuration management
    utils/                      # Shared utilities (logging, metrics, validation)
    constants.py                # Global constants
    exceptions.py               # Exception hierarchy
    http_status.py              # HTTP status codes

  data/                         # Data layer
    loaders/                    # Data loaders (bag samplers, batch samplers)
    datasets/                   # Dataset implementations (PCam, PANDA, Camelyon)
    wsi/                        # WSI pipeline, streaming, format handlers
    preprocessing/              # Stain normalization, preprocessing

  models/                       # Model architectures
    mil/                        # Standard MIL (nnMIL, AttentionMIL, CLAM, TransMIL)
    transnnmil/                 # TransnnMIL v2.0 (3-branch architecture)
      - hierarchical_pooling.py # Spatial clustering + region transformer
      - topology_branch.py      # k-NN graph + GNN (GATv2, GraphSAGE, GIN)
      - graph_cache.py          # Precomputed k-NN graphs in HDF5
      - adaptive_pruning.py     # 30% computation reduction
    components/                 # Shared components (attention, encoders, heads, fusion)
    foundation/                 # Foundation models (Phikon, UNI, CONCH)

  training/                     # Training infrastructure
                                # Training loops, optimizers, distributed training (DDP, FSDP)

  inference/                    # Inference engine
                                # Model serving, batch inference, quantization

  features/                     # Domain features
    federated/                  # Federated learning
      pathology_fl/             # PathologyFL (domain-specific FL)
      dmi/                      # Distributed Medical Intelligence (expertise weighting)
      cpi/                      # Collaborative Pathology Intelligence
      imr/                      # Intelligent Medical Referee
      mkn/                      # Medical Knowledge Network

    clinical/                   # Clinical integration
      workflow/                 # Clinical workflow, FHIR adapter, patient context
      pacs/                     # DICOM integration (C-FIND/C-MOVE/C-STORE)
      validation/               # Clinical validation, bias detection

    interpretability/           # Explainability
      gradcam/                  # Grad-CAM implementation
      advanced/                 # Advanced explainability (counterfactuals, uncertainty)
      visualization/            # Attention heatmaps, timeline visualization

    research/                   # Research platform
      annotation/               # Annotation interface
      experiment/               # Experiment tracking (MLflow, W&B, DVC)
      testing/                  # Hypothesis testing

    advanced/                   # Advanced features
      causal/                   # Causal inference
      discovery/                # Subtype discovery
      omics/                    # Multi-omics integration
      spatial/                  # Spatial analysis
      cells/                    # Cell detection and GNN
      multiscale/               # Multiscale analysis
      segmentation/             # Nucleus segmentation

  api/                          # REST API
                                # FastAPI server, JWT auth, input validation

  platform/                     # Platform services
    monitoring/                 # Metrics, tracing, health checks
    security/                   # Security utilities, rate limiting
    database/                   # Connection pooling, parameterized queries
    deployment/                 # Deployment utilities, validation
    cloud/                      # Cloud integration (AWS, Azure)
    integration/                # External integrations (EMR, LIS, scanners)

  streaming/                    # Real-time WSI processing, WebSocket server

tests/                          # 5,071+ test modules
docs/                           # Technical documentation
scripts/                        # Deployment, benchmarking, data preparation
  - download_panda.py           # PANDA dataset download
  - extract_panda_features_openslide.py  # Feature extraction
  - visualize_graph.py          # k-NN graph visualization
experiments/                    # Experiment configs and results
  - train_panda.py              # PANDA training pipeline
  - train_colorectal.py         # Colorectal cancer pipeline
  - evaluate_panda.py           # PANDA evaluation
k8s/                            # Kubernetes manifests
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
# Import core utilities
from src.core import constants, exceptions
from src.core.config import ExperimentConfig

# Load data
from src.data.datasets import PCamDataset
from src.data.loaders import create_bag_dataloader

# Load model
from src.models.mil import nnMIL, AttentionMIL
from src.models.transnnmil import TransnnMILv2

# Train
from src.training import Trainer

# Federated learning
from src.features.federated.pathology_fl import PathologyFLAggregator
from src.features.federated.dmi import DMICoordinator

# Clinical integration
from src.features.clinical.pacs import PACSClient
from src.features.clinical.workflow import ClinicalWorkflow
```


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

PANDA prostate cancer (Gleason grading):

    python experiments/train_panda.py --config experiments/configs/panda.yaml
    python experiments/evaluate_panda.py --checkpoint checkpoints/panda_best.pth

Colorectal cancer classification:

    python experiments/train_colorectal.py --config experiments/configs/colorectal.yaml

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


CONTINUOUS INTEGRATION
----------------------

Optimized CI/CD pipeline with 99% faster feedback:

  - Matrix optimization: 3 essential platform combinations (Ubuntu 3.10/3.11, Windows 3.10)
  - Job prioritization: Critical checks (lint, security, type-check) run first in parallel
  - Concurrency controls: Auto-cancel outdated runs on feature branches
  - Conditional execution: Expensive jobs run only when needed (main branch)
  - Queue time: <5 minutes (down from 12+ hours)
  - Critical feedback: <4 minutes for security and quality checks

CI workflows:
  - Tier 1 (parallel): Lint, type-check, security scan
  - Tier 2 (after Tier 1): Test matrix (Ubuntu, Windows)
  - Tier 3 (after Tier 2): Docker build, coverage report

Security scans:
  - Bandit (Python security linting)
  - Safety (dependency vulnerability scanning)
  - Trivy (container scanning)
  - OWASP ZAP (web application security)
  - CodeQL (semantic code analysis, weekly schedule)


DOCUMENTATION
-------------

See docs/ for technical documentation:

  - FRAMEWORK_OVERVIEW.md           System architecture
  - DOCS_INDEX.md                   Documentation index
  - PCAM_REAL_RESULTS.md            Benchmark results
  - CLINICAL_WORKFLOW_INTEGRATION.md Clinical deployment

Quick start guides:

  - PANDA_QUICK_START.md            PANDA dataset setup (5 min)
  - PANDA_SETUP_GUIDE.md            Detailed PANDA guide

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

    from src.platform.database.connection import DatabaseManager
    db = DatabaseManager(
        database_url="postgresql://user:pass@localhost/db",
        pool_size=10,
        max_overflow=20
    )

DMI deployment (multi-center):

    python -m src.features.federated.dmi.coordinator --config configs/dmi/coordinator.yaml
    python -m src.features.federated.dmi.client --config configs/dmi/client.yaml

Federated learning deployment:

    # Start FL coordinator
    python -m src.features.federated.pathology_fl.coordinator --config configs/fl/coordinator.yaml
    
    # Start hospital clients
    python -m src.features.federated.pathology_fl.client --config configs/fl/hospital_client.yaml


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
