# Architecture Migration Plan: Hybrid Structure

**Status:** Phase 1 - In Progress  
**Started:** May 19, 2026  
**Target:** Option 3 (Hybrid: Core layers + domain features)

---

## Target Structure

```
src/
├── core/                    # Core infrastructure
│   ├── config/              # Configuration management
│   ├── constants.py         # Global constants
│   ├── exceptions.py        # Exception hierarchy
│   ├── http_status.py       # HTTP status codes
│   └── utils/               # Shared utilities
│
├── data/                    # Data layer (shared)
│   ├── loaders/             # Data loaders
│   ├── preprocessing/       # Preprocessing
│   ├── wsi/                 # WSI-specific (pipeline, streaming)
│   └── datasets/            # Dataset implementations
│
├── models/                  # Model layer (shared)
│   ├── mil/                 # Standard MIL (nnMIL, CLAM, TransMIL)
│   ├── transnnmil/          # TransnnMIL v2.0
│   ├── foundation/          # Foundation models (Phikon, UNI, CONCH)
│   └── components/          # Shared components (attention, GNN, encoders)
│
├── training/                # Training layer (shared)
│   ├── trainers/            # Training loops
│   ├── distributed/         # DDP, FSDP
│   └── optimizers/          # Optimizers, schedulers
│
├── inference/               # Inference layer (shared)
│   ├── engine/              # Inference engine
│   ├── quantization/        # Model quantization
│   └── streaming/           # Streaming inference
│
├── features/                # Complex domain features
│   ├── federated/           # Federated learning (PathologyFL + DMI)
│   │   ├── pathology_fl/    # PathologyFL implementation
│   │   ├── dmi/             # DMI system
│   │   ├── privacy/         # DP-SGD, secure aggregation
│   │   ├── aggregation/     # Aggregation strategies
│   │   └── client_server/   # Client/server infrastructure
│   │
│   ├── clinical/            # Clinical integration
│   │   ├── pacs/            # DICOM integration
│   │   ├── fhir/            # FHIR adapter
│   │   ├── workflow/        # Clinical workflow
│   │   ├── audit/           # Audit logging
│   │   └── validation/      # Clinical validation
│   │
│   ├── interpretability/    # Explainability
│   │   ├── gradcam/         # Grad-CAM
│   │   ├── attention/       # Attention visualization
│   │   └── feature_importance/  # Feature analysis
│   │
│   ├── research/            # Research platform
│   │   ├── annotation/      # Annotation interface
│   │   ├── experiment/      # Experiment tracking
│   │   └── dataset_mgmt/    # Dataset management
│   │
│   └── advanced/            # Advanced features
│       ├── causal/          # Causal inference
│       ├── discovery/       # Subtype discovery
│       ├── omics/           # Multi-omics integration
│       └── spatial/         # Spatial analysis
│
├── api/                     # API server
│   ├── routers/             # API routes
│   ├── security/            # Authentication, authorization
│   └── main.py              # FastAPI app
│
└── platform/                # Platform services
    ├── monitoring/          # Metrics, tracing, health
    ├── security/            # Security utilities
    ├── database/            # Database connection
    └── deployment/          # Deployment utilities
```

---

## Migration Phases

### Phase 1: Core Infrastructure ✅ IN PROGRESS
**Timeline:** Week 1  
**Goal:** Establish core/ and consolidate shared utilities

**Actions:**
1. Create `src/core/` structure
2. Move `src/constants.py` → `src/core/constants.py`
3. Move `src/exceptions.py` → `src/core/exceptions.py`
4. Move `src/http_status.py` → `src/core/http_status.py`
5. Move `src/config/` → `src/core/config/`
6. Move `src/utils/` → `src/core/utils/`
7. Update imports across codebase
8. Run tests

**Files affected:**
- `src/constants.py`
- `src/exceptions.py`
- `src/http_status.py`
- `src/config/*`
- `src/utils/*`

### Phase 2: Data Layer
**Timeline:** Week 1-2  
**Goal:** Consolidate data handling

**Actions:**
1. Create `src/data/` structure (already exists, reorganize)
2. Move `src/preprocessing/` → `src/data/preprocessing/`
3. Move `src/streaming/wsi_stream_reader.py` → `src/data/wsi/streaming.py`
4. Move `src/data/wsi_pipeline.py` → `src/data/wsi/pipeline.py`
5. Organize datasets: `pcam_dataset.py`, `panda_dataset.py`, `camelyon_dataset.py`
6. Update imports
7. Run tests

**Directories affected:**
- `src/data/`
- `src/preprocessing/`
- `src/streaming/` (partial)

### Phase 3: Models Layer
**Timeline:** Week 2  
**Goal:** Organize model architectures

**Actions:**
1. Create `src/models/` structure (already exists, reorganize)
2. Create `src/models/mil/` for standard MIL models
   - Move `nnmil.py`, `attention_mil.py`, `clam.py`, `transmil.py`
3. Create `src/models/transnnmil/` for TransnnMIL v2.0
   - Move `transnnmil_v2.py`, `hierarchical_pooling.py`, `topology_branch.py`, `adaptive_pruning.py`
4. Create `src/models/components/` for shared components
   - Move `attention_mechanisms.py`, `encoders.py`, `heads.py`
5. Keep `src/models/foundation/` as-is
6. Update imports
7. Run tests

**Directories affected:**
- `src/models/`

### Phase 4: Training & Inference
**Timeline:** Week 2-3  
**Goal:** Consolidate training and inference

**Actions:**
1. Keep `src/training/` as-is (already well-organized)
2. Reorganize `src/inference/`:
   - Create `src/inference/engine/` (inference_engine.py, model_loader.py)
   - Create `src/inference/quantization/` (quantization.py)
   - Create `src/inference/streaming/` (streaming.py)
3. Update imports
4. Run tests

**Directories affected:**
- `src/training/`
- `src/inference/`

### Phase 5: Features - Federated
**Timeline:** Week 3  
**Goal:** Consolidate federated learning (PathologyFL + DMI)

**Actions:**
1. Create `src/features/federated/` structure
2. Move `src/federated/` → `src/features/federated/pathology_fl/`
3. Move `src/dmi/` → `src/features/federated/dmi/`
4. Move `src/cpi/` → `src/features/federated/cpi/`
5. Move `src/imr/` → `src/features/federated/imr/`
6. Move `src/mkn/` → `src/features/federated/mkn/`
7. Organize subdirectories:
   - `privacy/` (DP-SGD, secure aggregation)
   - `aggregation/` (aggregation strategies)
   - `client_server/` (client, server, coordinator)
8. Update imports
9. Run tests

**Directories affected:**
- `src/federated/`
- `src/dmi/`
- `src/cpi/`
- `src/imr/`
- `src/mkn/`

### Phase 6: Features - Clinical
**Timeline:** Week 3-4  
**Goal:** Consolidate clinical integration

**Actions:**
1. Create `src/features/clinical/` structure
2. Move `src/clinical/` → `src/features/clinical/workflow/`
3. Move `src/pacs/` → `src/features/clinical/pacs/`
4. Move `src/clinical_validation/` → `src/features/clinical/validation/`
5. Extract FHIR components → `src/features/clinical/fhir/`
6. Extract audit components → `src/features/clinical/audit/`
7. Update imports
8. Run tests

**Directories affected:**
- `src/clinical/`
- `src/pacs/`
- `src/clinical_validation/`

### Phase 7: Features - Interpretability
**Timeline:** Week 4  
**Goal:** Consolidate explainability

**Actions:**
1. Create `src/features/interpretability/` structure
2. Move `src/interpretability/` → `src/features/interpretability/gradcam/`
3. Move `src/explainability/` → `src/features/interpretability/advanced/`
4. Move `src/visualization/` → `src/features/interpretability/visualization/`
5. Update imports
6. Run tests

**Directories affected:**
- `src/interpretability/`
- `src/explainability/`
- `src/visualization/`

### Phase 8: Features - Research & Advanced
**Timeline:** Week 4  
**Goal:** Organize research and advanced features

**Actions:**
1. Create `src/features/research/` structure
   - Move `src/annotation_interface/` → `src/features/research/annotation/`
   - Move `src/research_platform/` → `src/features/research/experiment/`
   - Move `src/hypothesis/` → `src/features/research/testing/`
2. Create `src/features/advanced/` structure
   - Move `src/causal/` → `src/features/advanced/causal/`
   - Move `src/discovery/` → `src/features/advanced/discovery/`
   - Move `src/omics/` → `src/features/advanced/omics/`
   - Move `src/spatial/` → `src/features/advanced/spatial/`
   - Move `src/cells/` → `src/features/advanced/cells/`
   - Move `src/multiscale/` → `src/features/advanced/multiscale/`
   - Move `src/segmentation/` → `src/features/advanced/segmentation/`
3. Update imports
4. Run tests

**Directories affected:**
- `src/annotation_interface/`
- `src/research_platform/`
- `src/hypothesis/`
- `src/causal/`
- `src/discovery/`
- `src/omics/`
- `src/spatial/`
- `src/cells/`
- `src/multiscale/`
- `src/segmentation/`

### Phase 9: API & Platform
**Timeline:** Week 4-5  
**Goal:** Finalize API and platform services

**Actions:**
1. Keep `src/api/` as-is (already well-organized)
2. Create `src/platform/` structure
   - Move `src/monitoring/` → `src/platform/monitoring/`
   - Move `src/security/` → `src/platform/security/`
   - Move `src/database/` → `src/platform/database/`
   - Move `src/deployment/` → `src/platform/deployment/`
   - Move `src/cloud/` → `src/platform/cloud/`
   - Move `src/integration/` → `src/platform/integration/`
3. Update imports
4. Run tests

**Directories affected:**
- `src/api/`
- `src/monitoring/`
- `src/security/`
- `src/database/`
- `src/deployment/`
- `src/cloud/`
- `src/integration/`

### Phase 10: Cleanup & Documentation
**Timeline:** Week 5  
**Goal:** Remove old structure, update docs

**Actions:**
1. Remove empty old directories
2. Update all documentation:
   - `README.md`
   - `REPOSITORY_OVERVIEW.md`
   - `docs/FRAMEWORK_OVERVIEW.md`
   - `docs/ARCHITECTURE.md`
3. Update import paths in:
   - `tests/`
   - `scripts/`
   - `experiments/`
   - `notebooks/`
4. Update CI/CD paths in `.github/workflows/`
5. Update `setup.py` / `pyproject.toml`
6. Final test run (all 5,071+ tests)
7. Git commit with detailed message

---

## Import Update Strategy

### Automated Find-Replace:
```python
# Example: constants.py
OLD: from src.constants import
NEW: from src.core.constants import

OLD: from src import constants
NEW: from src.core import constants
```

### Tools:
- Use `grep_search` to find all imports
- Use `str_replace` for bulk updates
- Run `pytest` after each phase

---

## Rollback Plan

**If issues arise:**
1. Git branch for each phase: `migration/phase-1`, `migration/phase-2`, etc.
2. Each phase is a separate commit
3. Can revert individual phases if needed
4. Keep old structure until all tests pass

---

## Success Criteria

**Per Phase:**
- ✅ All files moved to new locations
- ✅ All imports updated
- ✅ All tests passing
- ✅ No broken references

**Final:**
- ✅ All 5,071+ tests passing
- ✅ Documentation updated
- ✅ CI/CD working
- ✅ Clear structure (easy to navigate)
- ✅ No old directories remaining

---

## Risk Mitigation

**Risks:**
1. **Import hell** - Many files to update
   - Mitigation: Automated search-replace, phase-by-phase
2. **Test failures** - Broken references
   - Mitigation: Run tests after each phase
3. **Circular dependencies** - New structure exposes issues
   - Mitigation: Identify and fix during migration
4. **Time overrun** - 5 weeks is ambitious
   - Mitigation: Can pause between phases, not blocking

---

## Current Status

**Phase 1:** ✅ COMPLETE (May 19, 2026)  
**Phase 2:** ✅ COMPLETE (May 19, 2026)  
**Phase 3:** ✅ COMPLETE (May 19, 2026)  
**Next:** Phase 4 - Training & Inference

**Phase 3 Results:**
- ✅ Created `src/models/mil/`, `src/models/transnnmil/`, `src/models/components/` structure
- ✅ Moved MIL models (nnMIL, AttentionMIL, CLAM, TransMIL) to `models/mil/`
- ✅ Moved TransnnMIL v2.0 components to `models/transnnmil/`
- ✅ Moved shared components (attention, encoders, heads, fusion) to `models/components/`
- ✅ Updated 100 imports across 52 files
- ✅ Fixed 10+ relative imports in moved files
- ✅ Models layer imports verified working

---

**Last Updated:** May 19, 2026
