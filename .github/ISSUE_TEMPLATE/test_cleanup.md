---
name: Clean up optional and legacy federated test failures
about: Address non-blocking test failures in federated learning test suite
title: "Clean up optional and legacy federated test failures"
labels: ["testing", "tech-debt", "optional-dependencies"]
assignees: ""
---

## Overview

The federated learning test suite has 48 failed tests and 35 errors that are **non-blocking** for core functionality. These failures fall into two categories:

1. **Optional dependency failures** - Tests requiring TenSEAL or torchcsprng
2. **Legacy test failures** - Tests using old import paths or deprecated APIs

**Note:** Core federated learning functionality is working correctly (269 tests passing, including all 5 integration tests and 27 FAIR-WEIGHTS-H tests).

## Optional Dependency Failures (48 tests)

### Secure Aggregation Tests (37 failures)

**Issue:** Tests require TenSEAL library for homomorphic encryption

**Affected files:**

- `tests/federated/test_secure_aggregation.py` (12 tests)
- `tests/federated/test_secure_aggregator_integration.py` (13 tests)
- `tests/federated/test_fl_properties.py` (2 tests)

**Options:**

1. **Skip tests when TenSEAL not installed** (recommended)
   ```python
   @pytest.mark.skipif(not TENSEAL_AVAILABLE, reason="TenSEAL not installed")
   ```
2. **Add TenSEAL to optional dependencies**
   ```toml
   [project.optional-dependencies]
   secure = ["tenseal>=0.3.0"]
   ```
3. **Mock TenSEAL for basic functionality tests**

**Priority:** Low (secure aggregation is optional feature)

### Secure RNG Tests (11 errors)

**Issue:** Tests require torchcsprng library for cryptographically secure RNG

**Affected files:**

- `tests/federated/test_hospital_client.py` (privacy tests)
- `tests/federated/test_local_trainer.py` (privacy engine tests)

**Options:**

1. **Skip tests when torchcsprng not installed** (recommended)
2. **Use `secure_rng=False` in test fixtures**
3. **Add torchcsprng to optional dependencies**

**Priority:** Low (secure RNG is optional feature)

## Legacy Test Failures (35 tests)

### Import Path Errors (24 errors)

**Issue:** Tests use old import structure `src.federated` instead of `src.features.federated`

**Affected files:**

- `tests/federated/test_pacs_connector.py` (18 errors)
- `tests/federated/test_pacs_connector_unit.py` (15 errors)
- `tests/federated/test_hospital_client.py` (3 errors)

**Fix:**

```python
# Old
from src.federated.pathology_fl.client.pacs_connector import PACSConnector

# New
from src.features.federated.pathology_fl.client.pacs_connector import PACSConnector
```

**Priority:** Medium (straightforward find-replace)

### API Mismatch Errors (11 failures)

**Issue:** Tests use deprecated LocalTrainer API methods

**Examples:**

1. `train_epoch()` → should use `train_local_epochs()`
2. `batch_size` parameter in DPSGDEngine → removed
3. Old checkpoint naming → use `model_v{version}.pt`

**Affected files:**

- `tests/federated/test_hospital_client.py`
- `tests/federated/test_fl_properties.py`
- `tests/federated/test_grpc_communication.py`

**Fix:** Update tests to match current LocalTrainer API (see `test_fl_integration.py` for examples)

**Priority:** Medium (tests need refactoring)

## Implementation Plan

### Phase 1: Quick Wins (1-2 hours)

1. Add skip decorators for optional dependency tests
2. Fix import path errors (find-replace)

### Phase 2: API Updates (2-4 hours)

1. Update `test_hospital_client.py` to use new LocalTrainer API
2. Update `test_fl_properties.py` DPSGDEngine tests
3. Update `test_grpc_communication.py` tests

### Phase 3: Documentation (1 hour)

1. Update test README with optional dependency instructions
2. Document which tests require which optional dependencies
3. Add CI configuration to skip optional tests

## Success Criteria

- [ ] All tests either pass or are properly skipped with clear reasons
- [ ] No import errors
- [ ] No API mismatch errors
- [ ] Optional dependency tests clearly marked
- [ ] CI runs cleanly (passing or skipped tests only)

## Related Issues

- #XXX - Camelyon17 FAIR-WEIGHTS-H Validation (blocked by: none, this is cleanup only)

## Notes

- This cleanup is **not blocking** for Camelyon17 validation
- Core federated learning functionality is already working
- These failures do not affect the validity of FAIR-WEIGHTS-H validation
