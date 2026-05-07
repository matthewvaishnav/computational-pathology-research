# Bugfix Requirements Document

## Introduction

CI tests are failing on Windows Python 3.11 after 17 minutes due to foundation model tests attempting to download the ~350MB Phikon model from HuggingFace during test execution, causing timeout failures. While the tests have `@pytest.mark.skipif(not torch.cuda.is_available())` decorators to skip when GPU is unavailable, they still run in CI and attempt large model downloads. The CI workflow runs `pytest tests/ -v -m "not property"` which doesn't exclude these slow download tests. The `slow` marker already exists in `pyproject.toml` but is not applied to foundation model tests or excluded in the CI pytest command.

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN CI runs `pytest tests/ -v -m "not property"` THEN the system executes foundation model tests that attempt to download ~350MB Phikon model from HuggingFace

1.2 WHEN foundation model tests execute in CI without GPU THEN the system times out after 17+ minutes on Windows Python 3.11

1.3 WHEN foundation model tests in `tests/test_foundation_models.py` are defined THEN the system does not mark them with `@pytest.mark.slow` decorator

1.4 WHEN CI pytest command is configured THEN the system does not exclude slow tests using `-m "not property and not slow"`

### Expected Behavior (Correct)

2.1 WHEN CI runs pytest THEN the system SHALL skip foundation model tests that require model downloads using the `slow` marker

2.2 WHEN foundation model tests are defined in `tests/test_foundation_models.py` THEN the system SHALL mark all tests that download models with `@pytest.mark.slow` decorator

2.3 WHEN CI pytest command is configured in `.github/workflows/ci.yml` THEN the system SHALL exclude both property-based tests AND slow tests using `-m "not property and not slow"`

2.4 WHEN CI completes test execution THEN the system SHALL finish within reasonable time limits without attempting large model downloads

### Unchanged Behavior (Regression Prevention)

3.1 WHEN developers run tests locally with `pytest tests/` THEN the system SHALL CONTINUE TO execute all tests including slow foundation model tests

3.2 WHEN developers run tests with `-m "not slow"` THEN the system SHALL CONTINUE TO skip slow tests as configured

3.3 WHEN foundation model tests are run with GPU available THEN the system SHALL CONTINUE TO execute and download models as needed

3.4 WHEN property-based tests are excluded with `-m "not property"` THEN the system SHALL CONTINUE TO skip property-based tests

3.5 WHEN other CI jobs (lint, type-check, security, docker, docs, quick-demo, coverage-report) run THEN the system SHALL CONTINUE TO execute without modification
