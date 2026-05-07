# Bugfix Requirements Document

## Introduction

Windows CI jobs are timing out after 17-21 minutes due to integration tests in `tests/test_camelyon_training_integration.py` executing during CI runs. These tests run actual training and evaluation scripts via subprocess with timeouts ranging from 30-120 seconds each. The tests are not marked with `@pytest.mark.slow`, causing them to execute when CI runs `pytest tests/ -v -m "not property and not slow"`. This fix will mark all integration tests in the file with the `@pytest.mark.slow` decorator to exclude them from CI execution while preserving their availability for local testing.

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN CI runs `pytest tests/ -v -m "not property and not slow"` on Windows THEN the system executes all 5 integration tests in `tests/test_camelyon_training_integration.py` because they lack the `@pytest.mark.slow` decorator

1.2 WHEN integration tests execute in CI THEN the system runs subprocess calls to training/evaluation scripts with cumulative timeouts of 390 seconds (6.5 minutes) plus test overhead

1.3 WHEN multiple integration tests accumulate execution time THEN the system causes Windows CI jobs to timeout after 17-21 minutes

1.4 WHEN `test_end_to_end_training` executes THEN the system runs `experiments/train_camelyon.py` with a 120-second timeout without the slow marker

1.5 WHEN `test_end_to_end_evaluation` executes THEN the system runs `experiments/evaluate_camelyon.py` with a 60-second timeout without the slow marker

1.6 WHEN `test_training_with_max_pooling` executes THEN the system runs `experiments/train_camelyon.py` with a 120-second timeout without the slow marker

1.7 WHEN `test_evaluation_generates_plots` executes THEN the system runs `experiments/evaluate_camelyon.py` with a 60-second timeout without the slow marker

1.8 WHEN `test_training_validates_config` executes THEN the system runs `experiments/train_camelyon.py` with a 30-second timeout without the slow marker

### Expected Behavior (Correct)

2.1 WHEN CI runs `pytest tests/ -v -m "not property and not slow"` on Windows THEN the system SHALL skip all 5 integration tests in `tests/test_camelyon_training_integration.py` because they are marked with `@pytest.mark.slow`

2.2 WHEN integration tests are marked with `@pytest.mark.slow` THEN the system SHALL exclude them from CI execution using the existing marker expression

2.3 WHEN integration tests are excluded from CI THEN the system SHALL complete Windows CI jobs within reasonable time limits (< 10 minutes)

2.4 WHEN `test_end_to_end_training` is marked with `@pytest.mark.slow` THEN the system SHALL skip it during CI execution

2.5 WHEN `test_end_to_end_evaluation` is marked with `@pytest.mark.slow` THEN the system SHALL skip it during CI execution

2.6 WHEN `test_training_with_max_pooling` is marked with `@pytest.mark.slow` THEN the system SHALL skip it during CI execution

2.7 WHEN `test_evaluation_generates_plots` is marked with `@pytest.mark.slow` THEN the system SHALL skip it during CI execution

2.8 WHEN `test_training_validates_config` is marked with `@pytest.mark.slow` THEN the system SHALL skip it during CI execution

### Unchanged Behavior (Regression Prevention)

3.1 WHEN developers run `pytest tests/test_camelyon_training_integration.py -v` locally THEN the system SHALL CONTINUE TO execute all 5 integration tests

3.2 WHEN developers run `pytest tests/ -v -m slow` THEN the system SHALL CONTINUE TO execute all integration tests marked with `@pytest.mark.slow`

3.3 WHEN integration tests execute locally or with explicit slow marker THEN the system SHALL CONTINUE TO run actual training and evaluation scripts via subprocess

3.4 WHEN integration tests execute THEN the system SHALL CONTINUE TO use the same timeout values (30-120 seconds per test)

3.5 WHEN integration tests execute THEN the system SHALL CONTINUE TO validate training, evaluation, configuration, and plotting functionality

3.6 WHEN CI runs tests on Ubuntu and macOS THEN the system SHALL CONTINUE TO skip integration tests marked with `@pytest.mark.slow`

3.7 WHEN CI runs other test jobs (lint, type-check, security, docker, docs, quick-demo, coverage-report) THEN the system SHALL CONTINUE TO execute unchanged

3.8 WHEN foundation model tests are marked with `@pytest.mark.slow` THEN the system SHALL CONTINUE TO be excluded from CI (from previous fix)

3.9 WHEN property-based tests are marked with `@pytest.mark.property` THEN the system SHALL CONTINUE TO be excluded from CI using the marker expression

3.10 WHEN unit tests and fast integration tests lack slow/property markers THEN the system SHALL CONTINUE TO execute in CI

## Bug Condition and Property

### Bug Condition Function

```pascal
FUNCTION isBugCondition(test_function)
  INPUT: test_function of type PytestTestFunction
  OUTPUT: boolean
  
  // Returns true when the test is an integration test in 
  // test_camelyon_training_integration.py without @pytest.mark.slow
  RETURN (
    test_function.file_path = "tests/test_camelyon_training_integration.py" AND
    test_function.name IN [
      "test_end_to_end_training",
      "test_end_to_end_evaluation", 
      "test_training_with_max_pooling",
      "test_evaluation_generates_plots",
      "test_training_validates_config"
    ] AND
    NOT has_marker(test_function, "slow")
  )
END FUNCTION
```

### Property Specification

```pascal
// Property: Fix Checking - Integration Tests Marked as Slow
FOR ALL test_function WHERE isBugCondition(test_function) DO
  marked_test ← add_slow_marker(test_function)
  ASSERT has_marker(marked_test, "slow") AND
         is_skipped_in_ci(marked_test) AND
         ci_execution_time < 10_minutes
END FOR
```

### Preservation Goal

```pascal
// Property: Preservation Checking - Local Test Execution Unchanged
FOR ALL test_function WHERE NOT isBugCondition(test_function) DO
  ASSERT test_function.behavior = test_function.behavior_after_fix
END FOR

// Specifically for integration tests:
FOR ALL test_function IN integration_tests DO
  ASSERT can_run_locally(test_function) AND
         can_run_with_slow_marker(test_function) AND
         subprocess_behavior_unchanged(test_function)
END FOR
```

## Counterexample

**Concrete Example Demonstrating the Bug:**

```python
# Current state (buggy):
def test_end_to_end_training(training_config, tmp_path):
    """Test end-to-end training for 2 epochs on synthetic data."""
    result = subprocess.run(
        [sys.executable, "experiments/train_camelyon.py", ...],
        timeout=120,
    )
    # This test executes in CI, contributing to timeout

# CI command:
# pytest tests/ -v -m "not property and not slow"
# Result: test_end_to_end_training RUNS (not skipped)
# Windows CI: TIMEOUT after 17-21 minutes
```

**Expected State (fixed):**

```python
# Fixed state:
@pytest.mark.slow
def test_end_to_end_training(training_config, tmp_path):
    """Test end-to-end training for 2 epochs on synthetic data."""
    result = subprocess.run(
        [sys.executable, "experiments/train_camelyon.py", ...],
        timeout=120,
    )
    # This test is skipped in CI

# CI command:
# pytest tests/ -v -m "not property and not slow"
# Result: test_end_to_end_training SKIPPED (marked as slow)
# Windows CI: COMPLETES in < 10 minutes
```
