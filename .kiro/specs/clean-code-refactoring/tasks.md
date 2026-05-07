# Clean Code Refactoring - Tasks

**Feature**: clean-code-refactoring  
**Date**: 2026-05-03  
**Status**: Ready for Implementation

---

## Task 1: Refactor API Routes (Priority 1)

**Goal**: Split `src/api/main.py` (1308 lines) → 8 focused files (<200 lines each)

### Task 1.1: Extract Dependencies Module
- [ ] Create `src/api/dependencies.py`
- [ ] Extract `get_db_session()` from main.py
- [ ] Extract `get_inference_engine()` from main.py
- [ ] Extract `get_current_user()` from main.py
- [ ] Update imports in main.py to use new module
- [ ] Run tests: `pytest tests/api/test_dependencies.py -v`
- [ ] Verify all API tests still pass

**Acceptance**: dependencies.py exists, all dependency functions work, tests pass

### Task 1.2: Extract Validators Module
- [ ] Create `src/api/validators.py`
- [ ] Extract `validate_email()` logic
- [ ] Extract `validate_password()` logic
- [ ] Extract `validate_file_upload()` logic
- [ ] Add unit tests for each validator
- [ ] Run tests: `pytest tests/api/test_validators.py -v`

**Acceptance**: validators.py exists, all validation logic extracted, tests pass

### Task 1.3: Extract Error Handlers Module
- [ ] Create `src/api/errors.py`
- [ ] Extract `not_found_handler()`
- [ ] Extract `internal_error_handler()`
- [ ] Extract validation error handlers
- [ ] Register handlers in main.py
- [ ] Test error responses unchanged

**Acceptance**: errors.py exists, error handling works, responses identical

### Task 1.4: Create Auth Router
- [ ] Create `src/api/routers/auth.py`
- [ ] Move `/api/register` endpoint
- [ ] Move `/api/login` endpoint
- [ ] Move `/api/user` endpoint
- [ ] Move `/oauth/login` endpoint
- [ ] Move `/oauth/callback` endpoint
- [ ] Include router in main.py: `app.include_router(auth.router)`
- [ ] Run tests: `pytest tests/api/test_auth.py -v`
- [ ] Verify auth endpoints work

**Acceptance**: auth.py exists, 5 auth endpoints moved, tests pass

### Task 1.5: Create Analysis Router
- [ ] Create `src/api/routers/analysis.py`
- [ ] Move `/api/upload` endpoint
- [ ] Move `/api/analysis/{id}` endpoint
- [ ] Move `/api/dicom` endpoint
- [ ] Move `/api/dicom/study/{id}` endpoint
- [ ] Move `/api/cases` endpoints (GET, POST, PATCH)
- [ ] Include router in main.py
- [ ] Run tests: `pytest tests/api/test_analysis.py -v`

**Acceptance**: analysis.py exists, 7 analysis endpoints moved, tests pass

### Task 1.6: Create Admin Router
- [ ] Create `src/api/routers/admin.py`
- [ ] Move `/api/users` endpoint
- [ ] Move `/api/config` endpoint
- [ ] Move `/api/audit-logs` endpoint
- [ ] Move `/api/reports` endpoints
- [ ] Include router in main.py
- [ ] Run tests: `pytest tests/api/test_admin.py -v`

**Acceptance**: admin.py exists, admin endpoints moved, tests pass

### Task 1.7: Create Mobile Router
- [ ] Create `src/api/routers/mobile.py`
- [ ] Move `/api/mobile/register` endpoint
- [ ] Move `/api/mobile/sync` endpoint
- [ ] Move `/api/mobile/offline-cases` endpoint
- [ ] Move `/api/mobile/model` endpoint
- [ ] Include router in main.py
- [ ] Run tests: `pytest tests/api/test_mobile.py -v`

**Acceptance**: mobile.py exists, mobile endpoints moved, tests pass

### Task 1.8: Create Monitoring Router
- [ ] Create `src/api/routers/monitoring.py`
- [ ] Move `/health` endpoint
- [ ] Move `/ready` endpoint
- [ ] Move `/metrics` endpoint
- [ ] Move `/api/ids/alerts` endpoint
- [ ] Move `/api/siem/incidents` endpoint
- [ ] Include router in main.py
- [ ] Run tests: `pytest tests/api/test_monitoring.py -v`

**Acceptance**: monitoring.py exists, monitoring endpoints moved, tests pass

### Task 1.9: Slim Down Main Module
- [ ] Remove all moved route handlers from main.py
- [ ] Keep only app setup, router includes, startup/shutdown
- [ ] Verify main.py <150 lines
- [ ] Run full API test suite: `pytest tests/api/ -v`
- [ ] Benchmark API response times (should be ±5%)

**Acceptance**: main.py <150 lines, all tests pass, performance maintained

### Task 1.10: Property Test API Equivalence
- [ ] Record 100 sample API requests/responses before refactor
- [ ] Create property test: `test_api_refactor_equivalence()`
- [ ] Run property test with recorded samples
- [ ] Verify all responses identical
- [ ] Commit: "refactor(api): split routes into focused routers"

**Acceptance**: Property test passes, API behavior unchanged

---

## Task 2: Refactor MIL Models (Priority 1)

**Goal**: Split `src/models/attention_mil.py` (1527 lines) → 7 focused files

### Task 2.1: Extract Fusion Strategies
- [ ] Create `src/models/fusion_strategies.py`
- [ ] Implement `FusionStrategy` base class
- [ ] Implement `EarlyFusion` class
- [ ] Implement `LateFusion` class
- [ ] Add unit tests for each fusion strategy
- [ ] Run tests: `pytest tests/models/test_fusion.py -v`

**Acceptance**: fusion_strategies.py exists, 2 fusion classes work, tests pass

### Task 2.2: Extract Attention Mechanisms
- [ ] Create `src/models/attention_mechanisms.py`
- [ ] Implement `AttentionMechanism` base class
- [ ] Implement `GatedAttention` class
- [ ] Implement `TransformerAttention` class
- [ ] Add unit tests for each attention mechanism
- [ ] Run tests: `pytest tests/models/test_attention.py -v`

**Acceptance**: attention_mechanisms.py exists, attention classes work, tests pass

### Task 2.3: Create MIL Base Class
- [ ] Create `src/models/mil_base.py`
- [ ] Implement `MILBase` class with common methods
- [ ] Add `compute_attention()` method
- [ ] Add `aggregate_features()` method
- [ ] Add `apply_fusion()` method
- [ ] Add unit tests for base class
- [ ] Run tests: `pytest tests/models/test_mil_base.py -v`

**Acceptance**: mil_base.py exists, base class works, tests pass

### Task 2.4: Refactor AttentionMIL Class
- [ ] Backup original attention_mil.py
- [ ] Refactor `AttentionMIL` to inherit from `MILBase`
- [ ] Replace fusion code with `FusionStrategy` instances
- [ ] Replace attention code with `GatedAttention`
- [ ] Remove duplicated code
- [ ] Verify class <300 lines
- [ ] Run tests: `pytest tests/models/test_attention_mil.py -v`

**Acceptance**: AttentionMIL refactored, <300 lines, tests pass

### Task 2.5: Property Test AttentionMIL Equivalence
- [ ] Create property test: `test_attention_mil_equivalence()`
- [ ] Generate 1000 random feature tensors
- [ ] Compare old vs new model outputs
- [ ] Verify outputs identical (atol=1e-6)
- [ ] Run: `pytest tests/models/test_attention_mil_properties.py -v`

**Acceptance**: Property test passes, outputs identical

### Task 2.6: Extract CLAM to Separate File
- [ ] Create `src/models/clam.py`
- [ ] Move `CLAM` class from attention_mil.py
- [ ] Refactor to inherit from `MILBase`
- [ ] Replace fusion code with `FusionStrategy`
- [ ] Replace attention code with `GatedAttention`
- [ ] Verify class <350 lines
- [ ] Run tests: `pytest tests/models/test_clam.py -v`

**Acceptance**: clam.py exists, CLAM refactored, <350 lines, tests pass

### Task 2.7: Property Test CLAM Equivalence
- [ ] Create property test: `test_clam_equivalence()`
- [ ] Generate 1000 random feature tensors
- [ ] Compare old vs new CLAM outputs
- [ ] Verify outputs identical (atol=1e-6)
- [ ] Run: `pytest tests/models/test_clam_properties.py -v`

**Acceptance**: Property test passes, CLAM outputs identical

### Task 2.8: Extract TransMIL to Separate File
- [ ] Create `src/models/transmil.py`
- [ ] Move `TransMIL` class from attention_mil.py
- [ ] Refactor to inherit from `MILBase`
- [ ] Replace fusion code with `FusionStrategy`
- [ ] Replace attention code with `TransformerAttention`
- [ ] Verify class <350 lines
- [ ] Run tests: `pytest tests/models/test_transmil.py -v`

**Acceptance**: transmil.py exists, TransMIL refactored, <350 lines, tests pass

### Task 2.9: Property Test TransMIL Equivalence
- [ ] Create property test: `test_transmil_equivalence()`
- [ ] Generate 1000 random feature tensors
- [ ] Compare old vs new TransMIL outputs
- [ ] Verify outputs identical (atol=1e-6)
- [ ] Run: `pytest tests/models/test_transmil_properties.py -v`

**Acceptance**: Property test passes, TransMIL outputs identical

### Task 2.10: Create Model Factory
- [ ] Create `src/models/factory.py`
- [ ] Move `create_attention_model()` function
- [ ] Update function to import from new modules
- [ ] Add tests for factory function
- [ ] Run tests: `pytest tests/models/test_factory.py -v`

**Acceptance**: factory.py exists, model creation works, tests pass

### Task 2.11: Clean Up Original File
- [ ] Remove CLAM class from attention_mil.py
- [ ] Remove TransMIL class from attention_mil.py
- [ ] Remove duplicated fusion code
- [ ] Verify attention_mil.py <300 lines
- [ ] Update all imports across codebase
- [ ] Run full model test suite: `pytest tests/models/ -v`

**Acceptance**: attention_mil.py <300 lines, all tests pass

### Task 2.12: Benchmark Model Performance
- [ ] Benchmark AttentionMIL inference (1000 iterations)
- [ ] Benchmark CLAM inference (1000 iterations)
- [ ] Benchmark TransMIL inference (1000 iterations)
- [ ] Verify performance within ±5% of original
- [ ] Commit: "refactor(models): extract fusion and attention components"

**Acceptance**: Performance maintained, all benchmarks pass

---

## Task 3: Refactor Memory Optimizer (Priority 2)

**Goal**: Split `src/streaming/memory_optimizer.py` (1097 lines) → 6 focused files

### Task 3.1: Create Memory Module Structure
- [ ] Create `src/streaming/memory/` directory
- [ ] Create `src/streaming/memory/__init__.py`
- [ ] Add module docstring

**Acceptance**: memory/ directory exists

### Task 3.2: Extract Memory Profiler
- [ ] Create `src/streaming/memory/profiler.py`
- [ ] Implement `MemorySnapshot` dataclass
- [ ] Implement `MemoryProfiler` class
- [ ] Extract profiling logic from MemoryOptimizer
- [ ] Add unit tests
- [ ] Run tests: `pytest tests/streaming/memory/test_profiler.py -v`

**Acceptance**: profiler.py exists, profiling works, tests pass

### Task 3.3: Extract Cache Manager
- [ ] Create `src/streaming/memory/cache_manager.py`
- [ ] Implement `CacheManager` class with LRU eviction
- [ ] Extract cache logic from MemoryOptimizer
- [ ] Add unit tests for get/put/evict
- [ ] Run tests: `pytest tests/streaming/memory/test_cache.py -v`

**Acceptance**: cache_manager.py exists, caching works, tests pass

### Task 3.4: Extract Batch Optimizer
- [ ] Create `src/streaming/memory/batch_optimizer.py`
- [ ] Implement `BatchOptimizer` class
- [ ] Extract batch size optimization logic
- [ ] Extract tile size optimization logic
- [ ] Add unit tests
- [ ] Run tests: `pytest tests/streaming/memory/test_optimizer.py -v`

**Acceptance**: batch_optimizer.py exists, optimization works, tests pass

### Task 3.5: Extract Memory Monitor
- [ ] Create `src/streaming/memory/monitor.py`
- [ ] Implement `MemoryMonitor` class
- [ ] Extract monitoring logic
- [ ] Extract alert logic
- [ ] Add unit tests
- [ ] Run tests: `pytest tests/streaming/memory/test_monitor.py -v`

**Acceptance**: monitor.py exists, monitoring works, tests pass

### Task 3.6: Extract Config Module
- [ ] Create `src/streaming/memory/config.py`
- [ ] Implement `OptimizerConfig` dataclass
- [ ] Extract config loading logic
- [ ] Extract config validation logic
- [ ] Add unit tests
- [ ] Run tests: `pytest tests/streaming/memory/test_config.py -v`

**Acceptance**: config.py exists, config handling works, tests pass

### Task 3.7: Create Coordinator Facade
- [ ] Create `src/streaming/memory/coordinator.py`
- [ ] Implement `MemoryCoordinator` class
- [ ] Wire together profiler, cache, optimizer, monitor
- [ ] Provide backward-compatible API
- [ ] Add integration tests
- [ ] Run tests: `pytest tests/streaming/memory/test_coordinator.py -v`

**Acceptance**: coordinator.py exists, facade works, tests pass

### Task 3.8: Update Streaming Code
- [x] Update imports in streaming modules
- [ ] Replace `MemoryOptimizer` with `MemoryCoordinator`
- [ ] Verify all streaming code works
- [ ] Run tests: `pytest tests/streaming/ -v`

**Acceptance**: All streaming code uses new memory module, tests pass

### Task 3.9: Delete Old Memory Optimizer
- [x] Verify no code references old `memory_optimizer.py`
- [ ] Delete `src/streaming/memory_optimizer.py`
- [ ] Run full test suite: `pytest tests/ -v`
- [ ] Verify all tests pass

**Acceptance**: Old file deleted, all tests pass

### Task 3.10: Benchmark Memory Operations
- [x] Benchmark profiling overhead
- [x] Benchmark cache operations
- [x] Benchmark optimization computation
- [x] Verify performance within ±5%
- [ ] Commit: "refactor(streaming): split memory optimizer into focused classes"

**Acceptance**: Performance maintained, benchmarks pass

---

## Task 4: Refactor Treatment Response (Priority 2)

**Goal**: Split `src/clinical/treatment_response.py` (1442 lines) → 4 focused files

### Task 4.1: Extract Response Calculator
- [ ] Create `src/clinical/response_calculator.py`
- [ ] Extract response calculation logic
- [ ] Extract RECIST criteria implementation
- [ ] Add unit tests
- [ ] Run tests: `pytest tests/clinical/test_response_calculator.py -v`

**Acceptance**: response_calculator.py exists, calculations work, tests pass

### Task 4.2: Extract Progression Analyzer
- [ ] Create `src/clinical/progression_analyzer.py`
- [ ] Extract progression detection logic
- [ ] Extract time-to-progression calculation
- [ ] Add unit tests
- [ ] Run tests: `pytest tests/clinical/test_progression.py -v`

**Acceptance**: progression_analyzer.py exists, analysis works, tests pass

### Task 4.3: Extract Outcome Predictor
- [ ] Create `src/clinical/outcome_predictor.py`
- [ ] Extract survival prediction logic
- [ ] Extract outcome modeling
- [ ] Add unit tests
- [ ] Run tests: `pytest tests/clinical/test_outcome.py -v`

**Acceptance**: outcome_predictor.py exists, prediction works, tests pass

### Task 4.4: Create Treatment Response Facade
- [ ] Create `src/clinical/treatment_facade.py`
- [ ] Wire together calculator, analyzer, predictor
- [ ] Provide backward-compatible API
- [ ] Add integration tests
- [ ] Run tests: `pytest tests/clinical/test_treatment_facade.py -v`

**Acceptance**: Facade works, backward compatible, tests pass

### Task 4.5: Update Clinical Code
- [ ] Update imports in clinical modules
- [ ] Replace old treatment_response with facade
- [ ] Run tests: `pytest tests/clinical/ -v`
- [ ] Delete old treatment_response.py
- [ ] Commit: "refactor(clinical): split treatment response into focused modules"

**Acceptance**: Old file deleted, all tests pass

---

## Task 5: Refactor Audit Module (Priority 3)

**Goal**: Split `src/clinical/audit.py` (1026 lines) → 3 focused files

### Task 5.1: Extract Audit Logger
- [ ] Create `src/clinical/audit_logger.py`
- [ ] Extract log writing logic
- [ ] Extract log formatting
- [ ] Add unit tests
- [ ] Run tests: `pytest tests/clinical/test_audit_logger.py -v`

**Acceptance**: audit_logger.py exists, logging works, tests pass

### Task 5.2: Extract Audit Query
- [ ] Create `src/clinical/audit_query.py`
- [ ] Extract log querying logic
- [ ] Extract filtering/search
- [ ] Add unit tests
- [ ] Run tests: `pytest tests/clinical/test_audit_query.py -v`

**Acceptance**: audit_query.py exists, queries work, tests pass

### Task 5.3: Extract Audit Reports
- [ ] Create `src/clinical/audit_reports.py`
- [ ] Extract report generation
- [ ] Extract compliance checks
- [ ] Add unit tests
- [ ] Run tests: `pytest tests/clinical/test_audit_reports.py -v`

**Acceptance**: audit_reports.py exists, reports work, tests pass

### Task 5.4: Clean Up Audit Module
- [ ] Update audit.py to use extracted modules
- [ ] Verify audit.py <300 lines
- [ ] Run tests: `pytest tests/clinical/test_audit.py -v`
- [ ] Commit: "refactor(clinical): split audit into focused modules"

**Acceptance**: audit.py <300 lines, all tests pass

---

## Task 6: Refactor Regulatory Module (Priority 3)

**Goal**: Split `src/clinical/regulatory.py` (976 lines) → 4 focused files

### Task 6.1: Extract DMR Manager
- [ ] Create `src/clinical/dmr_manager.py`
- [ ] Extract Device Master Record logic
- [ ] Add unit tests
- [ ] Run tests: `pytest tests/clinical/test_dmr.py -v`

**Acceptance**: dmr_manager.py exists, DMR works, tests pass

### Task 6.2: Extract Risk Manager
- [ ] Create `src/clinical/risk_manager.py`
- [ ] Extract ISO 14971 risk management
- [ ] Add unit tests
- [ ] Run tests: `pytest tests/clinical/test_risk.py -v`

**Acceptance**: risk_manager.py exists, risk mgmt works, tests pass

### Task 6.3: Extract V&V Manager
- [ ] Create `src/clinical/vv_manager.py`
- [ ] Extract verification/validation logic
- [ ] Add unit tests
- [ ] Run tests: `pytest tests/clinical/test_vv.py -v`

**Acceptance**: vv_manager.py exists, V&V works, tests pass

### Task 6.4: Extract Submission Generator
- [ ] Create `src/clinical/submission_generator.py`
- [ ] Extract 510(k)/CE submission logic
- [ ] Add unit tests
- [ ] Run tests: `pytest tests/clinical/test_submission.py -v`

**Acceptance**: submission_generator.py exists, generation works, tests pass

### Task 6.5: Clean Up Regulatory Module
- [ ] Update regulatory.py to use extracted modules
- [ ] Verify regulatory.py <300 lines
- [ ] Run tests: `pytest tests/clinical/test_regulatory.py -v`
- [ ] Commit: "refactor(clinical): split regulatory into focused modules"

**Acceptance**: regulatory.py <300 lines, all tests pass

---

## Task 7: Refactor Security Module (Priority 3)

**Goal**: Split `src/streaming/security.py` (852 lines) → 3 focused files

### Task 7.1: Extract Authentication
- [ ] Create `src/streaming/authentication.py`
- [ ] Extract auth logic
- [ ] Add unit tests
- [ ] Run tests: `pytest tests/streaming/test_authentication.py -v`

**Acceptance**: authentication.py exists, auth works, tests pass

### Task 7.2: Extract Authorization
- [ ] Create `src/streaming/authorization.py`
- [ ] Extract authz logic
- [ ] Add unit tests
- [ ] Run tests: `pytest tests/streaming/test_authorization.py -v`

**Acceptance**: authorization.py exists, authz works, tests pass

### Task 7.3: Extract Encryption
- [ ] Create `src/streaming/encryption.py`
- [ ] Extract encryption logic
- [ ] Add unit tests
- [ ] Run tests: `pytest tests/streaming/test_encryption.py -v`

**Acceptance**: encryption.py exists, encryption works, tests pass

### Task 7.4: Clean Up Security Module
- [ ] Update security.py to use extracted modules
- [ ] Verify security.py <300 lines
- [ ] Run tests: `pytest tests/streaming/test_security.py -v`
- [ ] Commit: "refactor(streaming): split security into focused modules"

**Acceptance**: security.py <300 lines, all tests pass

---

## Task 8: Final Validation

### Task 8.1: Run Full Test Suite
- [ ] Run all unit tests: `pytest tests/ -v`
- [ ] Run all integration tests: `pytest tests/integration/ -v`
- [ ] Run all property tests: `pytest tests/properties/ -v`
- [ ] Verify test coverage >80%: `pytest --cov=src --cov-report=html`

**Acceptance**: All tests pass, coverage maintained

### Task 8.2: Benchmark Performance
- [ ] Benchmark API response times
- [ ] Benchmark model inference
- [ ] Benchmark memory operations
- [ ] Verify all within ±5% of baseline
- [ ] Document results in `REFACTORING_RESULTS.md`

**Acceptance**: Performance maintained, results documented

### Task 8.3: Compute Code Quality Metrics
- [ ] Run metrics script: `python scripts/compute_metrics.py`
- [ ] Verify avg file size <400 lines
- [ ] Verify max file size <500 lines
- [ ] Verify no functions >50 lines
- [ ] Verify code duplication <5%
- [ ] Document metrics in `CODE_QUALITY_METRICS.md`

**Acceptance**: All metrics meet targets, documented

### Task 8.4: Update Documentation
- [ ] Update README with new module structure
- [ ] Update API docs with router organization
- [ ] Update model docs with new architecture
- [ ] Update developer guide with refactoring notes

**Acceptance**: Documentation updated, accurate

### Task 8.5: Final Commit
- [ ] Review all changes
- [ ] Verify no broken imports
- [ ] Verify no dead code
- [ ] Create final commit: "refactor: complete clean code refactoring initiative"
- [ ] Push to remote
- [ ] Create PR for review

**Acceptance**: PR created, ready for review

---

## Definition of Done

Refactoring complete when:

1. ✅ All 7 target files refactored
2. ✅ All files <500 lines
3. ✅ All functions <50 lines
4. ✅ No code duplication >10 lines
5. ✅ All tests pass (unit, integration, property)
6. ✅ Test coverage >80%
7. ✅ Performance within ±5%
8. ✅ Code quality metrics meet targets
9. ✅ Documentation updated
10. ✅ PR approved and merged

---

**Estimated Effort**: 120 hours (6 weeks)

**Status**: Ready for implementation

**Next**: Begin Task 1.1 - Extract Dependencies Module
