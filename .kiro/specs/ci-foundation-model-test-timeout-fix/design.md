# CI Foundation Model Test Timeout Fix - Bugfix Design

## Overview

CI tests are timing out after 17+ minutes on Windows Python 3.11 because foundation model tests attempt to download the ~350MB Phikon model from HuggingFace during test execution. While these tests have `@pytest.mark.skipif(not torch.cuda.is_available())` decorators, they still execute in CI environments without GPUs and attempt large model downloads before the skip condition is evaluated. The fix involves marking these tests with the existing `@pytest.mark.slow` decorator and updating the CI pytest command to exclude slow tests using `-m "not property and not slow"`. This is a minimal, targeted fix that leverages existing pytest infrastructure without changing test logic or CI job structure.

## Glossary

- **Bug_Condition (C)**: The condition that triggers the bug - when CI runs pytest without excluding slow tests, causing foundation model tests to execute and download large models
- **Property (P)**: The desired behavior when CI runs pytest - foundation model tests should be skipped to avoid timeouts from model downloads
- **Preservation**: Existing test execution behavior for local development, property-based test exclusion, and all other CI jobs must remain unchanged
- **Foundation Model Tests**: Tests in `tests/test_foundation_models.py` that instantiate `PhikonEncoder`, `UNIEncoder`, or `CONCHEncoder` classes
- **Phikon Model Download**: The `ViTModel.from_pretrained("owkin/phikon")` call in `PhikonEncoder._build_model()` that downloads ~350MB from HuggingFace
- **Slow Marker**: The existing `@pytest.mark.slow` decorator defined in `pyproject.toml` for marking tests that take significant time or resources

## Bug Details

### Bug Condition

The bug manifests when CI runs `pytest tests/ -v -m "not property"` in the test job. The pytest command excludes property-based tests but does not exclude slow tests. Foundation model tests in `tests/test_foundation_models.py` lack the `@pytest.mark.slow` decorator, so they execute in CI. When these tests instantiate encoder classes (e.g., `PhikonEncoder()`), the `_build_model()` method calls `ViTModel.from_pretrained("owkin/phikon")`, triggering a ~350MB download from HuggingFace. This download takes 17+ minutes on Windows Python 3.11 CI runners, causing timeout failures.

**Formal Specification:**
```
FUNCTION isBugCondition(input)
  INPUT: input of type CITestExecution
  OUTPUT: boolean
  
  RETURN input.pytestCommand == "pytest tests/ -v -m \"not property\""
         AND input.testFile == "tests/test_foundation_models.py"
         AND NOT hasSlowMarker(input.testFunction)
         AND testInstantiatesFoundationModel(input.testFunction)
END FUNCTION
```

### Examples

- **Test: `test_phikon_load`** - Instantiates `PhikonEncoder(freeze=True)`, triggering `ViTModel.from_pretrained("owkin/phikon")` download (~350MB), causing 17+ minute timeout on Windows Python 3.11
- **Test: `test_phikon_forward`** - Instantiates `PhikonEncoder(freeze=True)`, same download behavior, timeout failure
- **Test: `test_load_phikon`** - Calls `load_foundation_model("phikon", freeze=True)`, which instantiates `PhikonEncoder`, same download behavior, timeout failure
- **Test: `test_encoder_projector_pipeline`** - Calls `load_foundation_model("phikon", freeze=True)`, same download behavior, timeout failure
- **Edge case: `test_projector_init`** - Does NOT instantiate foundation models, only tests `FeatureProjector` class, should continue running in CI (not marked slow)

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- Local test execution with `pytest tests/` must continue to run all tests including slow foundation model tests
- Developers running `pytest tests/ -m "not slow"` must continue to skip slow tests as configured
- Foundation model tests with GPU available must continue to execute and download models as needed
- Property-based test exclusion with `-m "not property"` must continue to work
- All other CI jobs (lint, type-check, security, docker, docs, quick-demo, coverage-report) must remain unchanged
- Tests that do NOT instantiate foundation models (e.g., `test_projector_init`, `test_projector_forward`) must continue running in CI

**Scope:**
All test execution contexts that do NOT involve CI pytest commands should be completely unaffected by this fix. This includes:
- Local development test runs (`pytest tests/`)
- Manual test runs with custom markers
- IDE test runners
- Pre-commit hooks (if any)
- Other CI workflow jobs that don't run the main test suite

## Hypothesized Root Cause

Based on the bug description and code analysis, the root causes are:

1. **Missing Slow Markers**: Foundation model tests in `tests/test_foundation_models.py` are not decorated with `@pytest.mark.slow`, despite requiring large model downloads that take 17+ minutes
   - The `slow` marker exists in `pyproject.toml` configuration
   - Tests like `test_phikon_load`, `test_phikon_forward`, `test_load_phikon` instantiate encoders but lack the marker
   - The `@pytest.mark.skipif(not torch.cuda.is_available())` decorator is present but doesn't prevent model download attempts

2. **Incomplete CI Pytest Command**: The CI workflow in `.github/workflows/ci.yml` uses `pytest tests/ -v -m "not property"` which only excludes property-based tests
   - The command should be `pytest tests/ -v -m "not property and not slow"` to exclude both markers
   - This is a one-line change in the test job's "Run tests" step

3. **Model Download Timing**: The `PhikonEncoder._build_model()` method calls `ViTModel.from_pretrained("owkin/phikon")` during `__init__`, before pytest can evaluate skip conditions
   - The download happens when the encoder is instantiated, not when the test function runs
   - The `@pytest.mark.skipif` decorator only prevents test execution, not class instantiation in test setup

4. **CI Environment Characteristics**: Windows Python 3.11 CI runners have slower network or disk I/O, causing the 350MB download to take 17+ minutes instead of completing quickly
   - Other OS/Python combinations may also timeout but Windows 3.11 is the reported failure case

## Correctness Properties

Property 1: Bug Condition - CI Skips Foundation Model Tests

_For any_ CI test execution where pytest is run with the marker expression "not property and not slow", and foundation model tests are marked with `@pytest.mark.slow`, the CI test job SHALL skip all tests that instantiate foundation model encoders (PhikonEncoder, UNIEncoder, CONCHEncoder), preventing model downloads and completing within reasonable time limits (< 10 minutes).

**Validates: Requirements 2.1, 2.2, 2.3, 2.4**

Property 2: Preservation - Local and Non-CI Test Execution

_For any_ test execution context that is NOT the CI pytest command (local development, manual runs, IDE runners), the test suite SHALL produce exactly the same behavior as before the fix, executing all tests including slow foundation model tests when no marker exclusions are specified, and respecting marker exclusions when explicitly provided (e.g., `-m "not slow"`).

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6**

## Fix Implementation

### Changes Required

Assuming our root cause analysis is correct:

**File 1**: `tests/test_foundation_models.py`

**Changes**: Add `@pytest.mark.slow` decorator to all test methods that instantiate foundation model encoders

**Specific Changes**:
1. **Import slow marker**: Add `import pytest` at the top (already present)

2. **Mark TestPhikonEncoder tests**: Add `@pytest.mark.slow` to:
   - `test_phikon_load` - instantiates `PhikonEncoder(freeze=True)`
   - `test_phikon_forward` - instantiates `PhikonEncoder(freeze=True)`
   - `test_phikon_extract_features` - instantiates `PhikonEncoder(freeze=True)`

3. **Mark TestLoadFoundationModel tests**: Add `@pytest.mark.slow` to:
   - `test_load_phikon` - calls `load_foundation_model("phikon", freeze=True)`
   - `test_load_with_freeze_false` - calls `load_foundation_model("phikon", freeze=False)`

4. **Mark TestFoundationModelIntegration tests**: Add `@pytest.mark.slow` to:
   - `test_encoder_projector_pipeline` - calls `load_foundation_model("phikon", freeze=True)`
   - `test_batch_processing` - calls `load_foundation_model("phikon", freeze=True)`

5. **Do NOT mark TestFeatureProjector tests**: These tests only instantiate `FeatureProjector` class, which does not download models, so they should continue running in CI

**File 2**: `.github/workflows/ci.yml`

**Changes**: Update pytest command in the test job to exclude slow tests

**Specific Changes**:
1. **Update "Run tests" step**: Change line ~48 from:
   ```yaml
   pytest tests/ -v -m "not property" --cov=src --cov-report=xml --cov-report=term
   ```
   to:
   ```yaml
   pytest tests/ -v -m "not property and not slow" --cov=src --cov-report=xml --cov-report=term
   ```

2. **Update "Generate coverage report" step in coverage-report job**: Change line ~238 from:
   ```yaml
   pytest tests/ -m "not property" --cov=src --cov-report=html --cov-report=term
   ```
   to:
   ```yaml
   pytest tests/ -m "not property and not slow" --cov=src --cov-report=html --cov-report=term
   ```

## Testing Strategy

### Validation Approach

The testing strategy follows a two-phase approach: first, surface counterexamples that demonstrate the bug on unfixed code by observing CI timeout failures, then verify the fix works correctly by confirming tests are skipped in CI and preserving existing behavior for local test execution.

### Exploratory Bug Condition Checking

**Goal**: Surface counterexamples that demonstrate the bug BEFORE implementing the fix. Confirm that foundation model tests execute in CI and cause timeouts due to model downloads.

**Test Plan**: Observe the current CI behavior on the unfixed code. The CI workflow should show:
- Foundation model tests executing in the test job
- Timeout failures after 17+ minutes on Windows Python 3.11
- Logs showing HuggingFace model download attempts

**Test Cases**:
1. **CI Execution Test**: Run CI workflow on unfixed code, observe that `test_phikon_load` executes and attempts to download Phikon model (will timeout on unfixed code)
2. **Marker Check Test**: Run `pytest tests/test_foundation_models.py --collect-only -m "not property"` locally, observe that foundation model tests are collected (will show tests on unfixed code)
3. **Download Trigger Test**: Run `pytest tests/test_foundation_models.py::TestPhikonEncoder::test_phikon_load -v` locally without GPU, observe model download attempt (will download on unfixed code)
4. **Projector Test Check**: Run `pytest tests/test_foundation_models.py::TestFeatureProjector -v` locally, observe that projector tests run quickly without downloads (will pass on unfixed code)

**Expected Counterexamples**:
- CI test job times out after 17+ minutes on Windows Python 3.11
- Foundation model tests are collected and executed when running `pytest tests/ -m "not property"`
- Logs show `Downloading owkin/phikon` or similar HuggingFace download messages
- Possible causes: missing `@pytest.mark.slow` decorators, incomplete pytest marker expression in CI command

### Fix Checking

**Goal**: Verify that for all inputs where the bug condition holds (CI pytest execution), the fixed configuration produces the expected behavior (foundation model tests are skipped).

**Pseudocode:**
```
FOR ALL ciExecution WHERE isBugCondition(ciExecution) DO
  result := runCITests_fixed(ciExecution)
  ASSERT result.foundationModelTestsSkipped == True
  ASSERT result.executionTime < 10 minutes
  ASSERT result.status == "success"
END FOR
```

**Test Plan**: After applying the fix, verify that:
1. Foundation model tests are marked with `@pytest.mark.slow`
2. CI pytest command includes `-m "not property and not slow"`
3. CI test job completes successfully without timeout
4. CI logs show foundation model tests are skipped (e.g., "7 skipped" in pytest summary)

**Test Cases**:
1. **Marker Verification**: Run `pytest tests/test_foundation_models.py --collect-only -m "slow"` locally, verify that foundation model tests are collected
2. **CI Marker Exclusion**: Run `pytest tests/test_foundation_models.py -m "not property and not slow" -v` locally, verify that foundation model tests are skipped
3. **Projector Tests Still Run**: Run `pytest tests/test_foundation_models.py::TestFeatureProjector -m "not property and not slow" -v` locally, verify that projector tests execute (not skipped)
4. **CI Workflow Success**: Trigger CI workflow on fixed code, verify that test job completes in < 10 minutes without timeout

### Preservation Checking

**Goal**: Verify that for all inputs where the bug condition does NOT hold (local test execution, other CI jobs), the fixed configuration produces the same result as the original configuration.

**Pseudocode:**
```
FOR ALL testExecution WHERE NOT isBugCondition(testExecution) DO
  ASSERT runTests_original(testExecution) = runTests_fixed(testExecution)
END FOR
```

**Testing Approach**: Property-based testing is recommended for preservation checking because:
- It generates many test execution scenarios automatically across different contexts
- It catches edge cases that manual unit tests might miss (e.g., different marker combinations)
- It provides strong guarantees that behavior is unchanged for all non-CI execution contexts

**Test Plan**: Observe behavior on UNFIXED code first for local test execution and other CI jobs, then verify the same behavior continues after the fix.

**Test Cases**:
1. **Local Full Test Run**: Run `pytest tests/test_foundation_models.py -v` locally on unfixed code, observe all tests execute including foundation model tests. After fix, verify same behavior (all tests execute).
2. **Local Slow Exclusion**: Run `pytest tests/test_foundation_models.py -m "not slow" -v` locally on unfixed code, observe that no tests are skipped (slow marker not applied). After fix, verify foundation model tests are skipped but projector tests run.
3. **Property Marker Exclusion**: Run `pytest tests/ -m "not property" -v` locally on unfixed code, observe which tests are skipped. After fix, verify same tests are skipped (property-based tests only, not foundation model tests unless also marked slow).
4. **Other CI Jobs**: Observe lint, type-check, security, docker, docs, quick-demo, coverage-report jobs on unfixed code. After fix, verify these jobs execute identically (no changes to their workflow definitions).
5. **GPU Available Scenario**: If GPU is available, run `pytest tests/test_foundation_models.py -v` on unfixed code, observe foundation model tests execute and download models. After fix, verify same behavior when running without marker exclusions.

### Unit Tests

- Test that `@pytest.mark.slow` decorator is present on all foundation model test methods that instantiate encoders
- Test that `@pytest.mark.slow` decorator is NOT present on projector test methods
- Test that CI pytest command includes `-m "not property and not slow"` marker expression
- Test that local pytest execution without markers runs all tests including slow ones

### Property-Based Tests

- Generate random pytest marker expressions and verify that `-m "not property and not slow"` correctly excludes both property-based and slow tests
- Generate random test execution contexts (local, CI, different OS/Python versions) and verify preservation of behavior for non-CI contexts
- Test that combining multiple markers (slow, property, skipif) works correctly across different pytest invocations

### Integration Tests

- Test full CI workflow execution with fixed code, verify test job completes successfully in < 10 minutes
- Test local development workflow with fixed code, verify all tests run when no markers are specified
- Test that developers can still run foundation model tests explicitly with `pytest tests/test_foundation_models.py -v`
- Test that coverage reports are generated correctly after excluding slow tests in CI
