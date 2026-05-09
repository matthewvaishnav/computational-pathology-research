# HistoCore Agent Registry

Specialized agents for efficient codebase navigation and updates.

## Agent Roster

### 1. Core Models Agent
**Scope:** `src/models/`, `src/training/`
**Responsibilities:**
- MIL architectures (nnMIL, AttentionMIL, CLAM, TransMIL)
- Training pipelines and optimizations
- Model factory and configuration
**Key Files:** ~15 files, ~8K LOC

### 2. Data Pipeline Agent
**Scope:** `src/data/`, `src/preprocessing/`
**Responsibilities:**
- WSI processing and patch extraction
- Data loaders and augmentation
- Feature extraction pipelines
**Key Files:** ~20 files, ~12K LOC

### 3. Clinical Integration Agent
**Scope:** `src/clinical/`, `src/pacs/`, `src/integration/`
**Responsibilities:**
- PACS/FHIR/LIS integration
- Clinical workflows and patient context
- Multi-class disease classifier
**Key Files:** ~25 files, ~15K LOC

### 4. Federated Learning Agent
**Scope:** `src/federated/`, `src/dmi/`
**Responsibilities:**
- Federated training infrastructure
- Privacy mechanisms (DP, secure aggregation)
- DMI expertise weighting
**Key Files:** ~60 files, ~35K LOC

### 5. Production API Agent
**Scope:** `src/api/`, `src/streaming/`
**Responsibilities:**
- FastAPI endpoints and security
- Real-time WSI streaming
- Authentication and rate limiting
**Key Files:** ~30 files, ~20K LOC

### 6. Inference & Deployment Agent
**Scope:** `src/inference/`, `src/mobile_edge/`, `k8s/`, `docker/`
**Responsibilities:**
- Model serving and optimization
- Quantization and mobile deployment
- Kubernetes and Docker configs
**Key Files:** ~25 files, ~18K LOC

### 7. Testing & Validation Agent
**Scope:** `tests/`, `src/clinical_validation/`
**Responsibilities:**
- Test infrastructure (5000+ tests)
- Clinical validation and benchmarking
- Property-based testing
**Key Files:** ~310 files, ~40K LOC

### 8. Documentation Agent
**Scope:** `docs/`, `README.md`, `SECURITY.md`
**Responsibilities:**
- Technical documentation
- API references and guides
- Security and compliance docs
**Key Files:** ~105 files, ~50K LOC

## Usage Pattern

When you need information:
1. Identify the domain (models, data, clinical, etc.)
2. Query the specific agent
3. Agent scans only their scope (~10-35K LOC vs 195K)
4. Get focused, accurate answers

## Agent Query Examples

**"What MIL models are implemented?"**
→ Core Models Agent scans `src/models/` only

**"How does PACS integration work?"**
→ Clinical Integration Agent scans `src/clinical/pacs/` only

**"What security fixes were made?"**
→ Production API Agent scans `src/api/` + `SECURITY.md`

**"Are there multi-class results?"**
→ Testing & Validation Agent scans `tests/` + `docs/` for benchmarks

## Benefits

- **90% token reduction** per query (scan 10-50K vs 195K LOC)
- **Faster responses** (focused scope)
- **Better accuracy** (domain expertise)
- **Incremental updates** (only affected agents need refresh)
