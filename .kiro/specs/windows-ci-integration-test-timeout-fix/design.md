# Windows CI Integration Test Timeout Fix - Bugfix Design

## Overview

This fix addresses Windows CI job timeouts caused by integration tests in `tests/test_camelyon_training_integration.py` executing during CI runs. The tests run actual training and evaluation scripts via subprocess with cumulative timeouts of 390 seconds (6.5 minutes) plus overhead, causing Windows CI jobs to timeout after 17-21 minutes. The fix will mark all 5 integration tests with the `@pytest.mark.slow` decorator to exclude them from CI execution using the existing marker expression `"not property and not slow"`, while preserving their availability for local testing and explicit slow test runs.

## Glossary

- **Bug_Condition (C)**: Integration tests in `test_camelyon_training_integration.py` lacking the `@pytest.mark.slow` decorator, causing them to execute in CI
- **Property (P)**: Integration tests marked with `@pytest.mark.slow` are skipped during CI execution using the marker expression `"not property and not slow"`
- **Preservation**: Local test execution, explicit slow test runs, and all other CI behaviors remain unchanged
- **Integration Test**: A test that runs actual training/evaluation scripts via subprocess with timeouts ranging from 30-120 seconds
- **Marker Expression**: The pytest marker filter `"not property and not slow"` used in CI to exclude property-based tests and slow integration tests
- **test_camelyon_training_integration.py**: The file at `tests/test_camelyon_training_integration.py` containing 5 integration tests for CAMELYON slide-level training and evaluation

## Bug Details

### Bug Condition

The bug manifests when CI runs `pytest tests/ -v -m "not property and not slow"` on Windows and the 5 integration tests in `tests/test_camelyon_training_integration.py` execute because they lack the `@pytest.mark.slow` decorator. These tests run subprocess calls to training and evaluation scripts with cumulative timeouts of 390 seconds (6.5 minutes) plus test overhead, causing Windows CI jobs to timeout after 17-21 minutes.

**Formal Specification:**
```
FUNCTION isBugCondition(test_function)
  INPUT: test_function of type PytestTestFunction
  OUTPUT: boolean
  
  RETURN test_function.file_path = "tests/test_camelyon_training_integration.py"
         AND test_function.name IN [
           "test_end_to_end_training",
           "test_end_to_end_evaluation",
           "test_training_with_max_pooling",
           "test_evaluation_generates_plots",
           "test_training_validates_config"
         ]
         AND NOT has_marker(test_function, "slow")
END FUNCTION
```

### Examples

- **test_end_to_end_training**: Runs `experiments/train_camelyon.py` with 120-second timeout, executes in CI (expected: skipped)
- **test_end_to_end_evaluation**: Runs `experiments/evaluate_camelyon.py` with 60-second timeout, executes in CI (expected: skipped)
- **test_training_with_max_pooling**: Runs `experiments/train_camelyon.py` with 120-second timeout, executes in CI (expected: skipped)
- **test_evaluation_generates_plots**: Runs `experiments/evaluate_camelyon.py` with 60-second timeout, executes in CI (expected: skipped)
- **test_training_validates_config**: Runs `experiments/train_camelyon.py` with 30-second timeout, executes in CI (expected: skipped)

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- Developers running `pytest tests/test_camelyon_training_integration.py -v` locally must continue to execute all 5 integration tests
- Developers running `pytest tests/ -v -m slow` must continue to execute all integration tests marked with `@pytest.mark.slow`
- Integration tests must continue to run actual training and evaluation scripts via subprocess with the same timeout values (30-120 seconds per test)
- Integration tests must continue to validate training, evaluation, configuration, and plotting functionality
- CI runs on Ubuntu and macOS must continue to skip integration tests marked with `@pytest.mark.slow`
- Other CI test jobs (lint, type-check, security, docker, docs, quick-demo, coverage-report) must continue to execute unchanged
- Foundation model tests marked with `@pytest.mark.slow` must continue to be excluded from CI (from previous fix)
- Property-based tests marked with `@pytest.mark.property` must continue to be excluded from CI
- Unit tests and fast integration tests lacking slow/property markers must continue to execute in CI

**Scope:**
All test execution contexts that do NOT involve the CI marker expression `"not property and not slow"` should be completely unaffected by this fix. This includes:
- Local test runs without marker filters
- Explicit slow test runs with `-m slow`
- Individual test file execution
- Test runs on other platforms (Ubuntu, macOS) already using the marker expression

## Hypothesized Root Cause

Based on the bug description, the root cause is clear and straightforward:

1. **Missing Decorator**: The 5 integration tests in `tests/test_camelyon_training_integration.py` do not have the `@pytest.mark.slow` decorator applied to their function definitions

2. **Marker Expression Behavior**: The CI workflow uses the marker expression `"not property and not slow"` to exclude slow tests, but this only works if tests are actually marked with `@pytest.mark.slow`

3. **Subprocess Execution Time**: Each integration test runs subprocess calls with timeouts ranging from 30-120 seconds, and the cumulative execution time (390 seconds + overhead) causes Windows CI jobs to timeout

4. **Platform-Specific Timeout**: Windows CI jobs are timing out after 17-21 minutes, while Ubuntu and macOS jobs complete successfully (likely due to performance differences or other test execution patterns)

## Correctness Properties

Property 1: Bug Condition - Integration Tests Marked as Slow

_For any_ test function in `tests/test_camelyon_training_integration.py` where the test is one of the 5 integration tests (`test_end_to_end_training`, `test_end_to_end_evaluation`, `test_training_with_max_pooling`, `test_evaluation_generates_plots`, `test_training_validates_config`), the fixed code SHALL have the `@pytest.mark.slow` decorator applied, causing the test to be skipped when CI runs `pytest tests/ -v -m "not property and not slow"`, and Windows CI jobs SHALL complete within reasonable time limits (< 10 minutes).

**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8**

Property 2: Preservation - Local and Explicit Slow Test Execution

_For any_ test execution context that does NOT use the CI marker expression `"not property and not slow"` (local runs without markers, explicit `-m slow` runs, individual file execution), the fixed code SHALL produce exactly the same behavior as the original code, preserving the ability to execute all 5 integration tests with their subprocess calls, timeout values, and validation functionality unchanged.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10**

## Fix Implementation

### Changes Required

The root cause is confirmed: the 5 integration tests lack the `@pytest.mark.slow` decorator.

**File**: `tests/test_camelyon_training_integration.py`

**Function**: All 5 integration test functions

**Specific Changes**:
1. **Add @pytest.mark.slow to test_end_to_end_training**: Add the decorator immediately before the function definition (after docstring if present, or before `def` statement)
   - Location: Line ~110 (before `def test_end_to_end_training(training_config, tmp_path):`)
   - Change: Add `@pytest.mark.slow` decorator

2. **Add @pytest.mark.slow to test_end_to_end_evaluation**: Add the decorator immediately before the function definition
   - Location: Line ~140 (before `def test_end_to_end_evaluation(tmp_path, synthetic_camelyon_data):`)
   - Change: Add `@pytest.mark.slow` decorator

3. **Add @pytest.mark.slow to test_training_with_max_pooling**: Add the decorator immediately before the function definition
   - Location: Line ~220 (before `def test_training_with_max_pooling(tmp_path, synthetic_camelyon_data):`)
   - Change: Add `@pytest.mark.slow` decorator

4. **Add @pytest.mark.slow to test_evaluation_generates_plots**: Add the decorator immediately before the function definition
   - Location: Line ~280 (before `def test_evaluation_generates_plots(tmp_path, synthetic_camelyon_data):`)
   - Change: Add `@pytest.mark.slow` decorator

5. **Add @pytest.mark.slow to test_training_validates_config**: Add the decorator immediately before the function definition
   - Location: Line ~350 (before `def test_training_validates_config(tmp_path, synthetic_camelyon_data):`)
   - Change: Add `@pytest.mark.slow` decorator

**Example Change Pattern**:
```python
# Before (buggy):
def test_end_to_end_training(training_config, tmp_path):
    """Test end-to-end training for 2 epochs on synthetic data."""
    # ... test implementation

# After (fixed):
@pytest.mark.slow
def test_end_to_end_training(training_config, tmp_path):
    """Test end-to-end training for 2 epochs on synthetic data."""
    # ... test implementation
```

## Testing Strategy

### Validation Approach

The testing strategy follows a two-phase approach: first, surface counterexamples that demonstrate the bug on unfixed code (tests execute in CI), then verify the fix works correctly (tests are skipped in CI) and preserves existing behavior (tests still run locally and with explicit slow marker).

### Exploratory Bug Condition Checking

**Goal**: Surface counterexamples that demonstrate the bug BEFORE implementing the fix. Confirm that the 5 integration tests execute when CI runs `pytest tests/ -v -m "not property and not slow"` on the unfixed code.

**Test Plan**: Run the CI test command locally on the UNFIXED code and observe that the 5 integration tests execute (not skipped). Measure execution time to confirm it contributes to timeout issues.

**Test Cases**:
1. **CI Marker Expression Test**: Run `pytest tests/test_camelyon_training_integration.py -v -m "not property and not slow"` on unfixed code (will execute all 5 tests - demonstrates bug)
2. **Individual Test Execution**: Run `pytest tests/test_camelyon_training_integration.py::test_end_to_end_training -v` on unfixed code (will execute - baseline behavior)
3. **Slow Marker Test**: Run `pytest tests/test_camelyon_training_integration.py -v -m slow` on unfixed code (will skip all tests - demonstrates missing marker)
4. **Execution Time Measurement**: Measure total execution time of all 5 tests to confirm cumulative timeout contribution (expected: 390+ seconds)

**Expected Counterexamples**:
- All 5 integration tests execute when CI marker expression is used (not skipped)
- Tests are not collected when `-m slow` is used (confirming they lack the slow marker)
- Cumulative execution time exceeds 6 minutes, contributing to Windows CI timeout

### Fix Checking

**Goal**: Verify that for all integration tests where the bug condition holds (lacking `@pytest.mark.slow`), the fixed code has the decorator applied and tests are skipped in CI.

**Pseudocode:**
```
FOR ALL test_function WHERE isBugCondition(test_function) DO
  fixed_test := add_slow_marker(test_function)
  ASSERT has_marker(fixed_test, "slow")
  ASSERT is_skipped_when_marker_expression_used(fixed_test, "not property and not slow")
  ASSERT ci_execution_time < 10_minutes
END FOR
```

### Preservation Checking

**Goal**: Verify that for all test execution contexts where the bug condition does NOT hold (local runs, explicit slow marker runs), the fixed code produces the same result as the original code.

**Pseudocode:**
```
FOR ALL execution_context WHERE NOT uses_ci_marker_expression(execution_context) DO
  ASSERT test_behavior_original(execution_context) = test_behavior_fixed(execution_context)
END FOR
```

**Testing Approach**: Property-based testing is NOT recommended for preservation checking in this case because the changes are deterministic and the test execution contexts are well-defined. Manual verification with specific test commands is more appropriate.

**Test Plan**: Observe behavior on UNFIXED code first for local runs and explicit slow marker runs, then verify the same behavior occurs after the fix.

**Test Cases**:
1. **Local Execution Preservation**: Run `pytest tests/test_camelyon_training_integration.py -v` on unfixed code (all 5 tests execute), then verify same behavior on fixed code
2. **Explicit Slow Marker Preservation**: Run `pytest tests/ -v -m slow` on unfixed code (no integration tests collected), then verify fixed code collects and executes all 5 integration tests
3. **Individual Test Preservation**: Run `pytest tests/test_camelyon_training_integration.py::test_end_to_end_training -v` on both unfixed and fixed code (should execute in both cases)
4. **Other Platform CI Preservation**: Verify Ubuntu and macOS CI jobs continue to skip slow tests as before

### Unit Tests

- Test that each of the 5 integration tests has the `@pytest.mark.slow` decorator applied (can be verified via pytest marker inspection)
- Test that CI marker expression `"not property and not slow"` skips all 5 integration tests
- Test that local execution without marker filters runs all 5 integration tests
- Test that explicit `-m slow` marker collects and runs all 5 integration tests

### Property-Based Tests

Property-based testing is not applicable for this fix because:
- The changes are deterministic (adding decorators to specific functions)
- The input domain is finite and well-defined (5 specific test functions)
- The behavior is binary (test is skipped or not skipped based on marker)
- Manual verification with specific test commands provides complete coverage

### Integration Tests

- Run full CI test suite on Windows with fixed code and verify completion time < 10 minutes
- Run full CI test suite on Ubuntu and macOS with fixed code and verify no behavior change
- Run local test suite with fixed code and verify all integration tests execute
- Run explicit slow test suite with fixed code and verify all integration tests execute
- Verify that foundation model tests (from previous fix) continue to be excluded from CI
- Verify that property-based tests continue to be excluded from CI
