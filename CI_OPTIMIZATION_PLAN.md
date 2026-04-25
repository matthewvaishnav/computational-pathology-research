# CI/CD Optimization Plan

## Current Status

**CI Pipeline**: Comprehensive with 11 jobs (test, lint, type-check, security, docker, docs, quick-demo, coverage-report, all-checks-passed)
**Test Matrix**: 3 OS × 3 Python versions = 9 configurations
**Average Runtime**: ~15-20 minutes
**Test Coverage**: 55% (1,448 tests)

## Completed Optimizations ✅

1. **Foundation Model Test Timeout Fix** ✅
   - All foundation model tests marked with `@pytest.mark.slow`
   - CI excludes slow tests: `-m "not property and not slow"`
   - Prevents ~350MB Phikon model downloads in CI

2. **Integration Test Timeout Fix** ✅
   - All CAMELYON integration tests marked with `@pytest.mark.slow`
   - Prevents subprocess training/evaluation in CI
   - Reduces Windows CI timeout from 17-21 minutes to <10 minutes

3. **Property-Based Test Exclusion** ✅
   - Property tests marked with `@pytest.mark.property`
   - CI excludes property tests to avoid long-running Hypothesis tests
   - Maintains fast CI feedback loop

## Proposed Optimizations

### Priority 1: High-Impact, Low-Effort

#### 1.1 Add PACS Integration Tests to CI ⭐
**Impact**: Validates production-critical PACS system (40/48 properties tested)
**Effort**: Low (add job to ci.yml)
**Benefit**: Catches PACS regressions early

```yaml
pacs-tests:
  name: PACS Integration Tests
  runs-on: ubuntu-latest
  steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.10'
        cache: 'pip'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -e .
    
    - name: Run PACS property tests
      run: |
        pytest tests/test_pacs_*.py -v --hypothesis-show-statistics
      timeout-minutes: 15
    
    - name: Upload PACS test results
      uses: actions/upload-artifact@v4
      if: always()
      with:
        name: pacs-test-results
        path: |
          .hypothesis/
          pytest-report.xml
```

#### 1.2 Optimize Dependency Installation ⭐
**Impact**: Reduces setup time by ~2-3 minutes per job
**Effort**: Low (modify pip install commands)
**Benefit**: Faster CI feedback

```yaml
# Current (slow):
- name: Install dependencies
  run: |
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    pip install -e ".[foundation]"

# Optimized (fast):
- name: Install dependencies
  run: |
    python -m pip install --upgrade pip
    pip install -r requirements.txt --no-deps  # Skip dependency resolution
    pip install -e . --no-deps
    pip install -r requirements.txt  # Resolve dependencies once
```

#### 1.3 Add Test Result Caching ⭐
**Impact**: Skip unchanged tests on subsequent runs
**Effort**: Medium (requires pytest-cache setup)
**Benefit**: 30-50% faster CI on incremental changes

```yaml
- name: Cache pytest results
  uses: actions/cache@v4
  with:
    path: .pytest_cache
    key: pytest-${{ runner.os }}-${{ hashFiles('tests/**/*.py', 'src/**/*.py') }}
    restore-keys: |
      pytest-${{ runner.os }}-

- name: Run tests with cache
  run: |
    pytest tests/ -v --lf --ff -m "not property and not slow"
    # --lf: run last failed tests first
    # --ff: run failed tests first, then others
```

#### 1.4 Parallelize Test Execution ⭐
**Impact**: Reduces test runtime by 40-60%
**Effort**: Medium (requires pytest-xdist)
**Benefit**: Faster CI feedback

```yaml
- name: Install test dependencies
  run: |
    pip install pytest-xdist

- name: Run tests in parallel
  run: |
    pytest tests/ -v -n auto -m "not property and not slow"
    # -n auto: use all available CPU cores
```

### Priority 2: Medium-Impact, Medium-Effort

#### 2.1 Add Nightly Full Test Suite
**Impact**: Comprehensive testing without blocking PRs
**Effort**: Medium (new workflow file)
**Benefit**: Catches edge cases in slow/property tests

```yaml
# .github/workflows/nightly.yml
name: Nightly Full Test Suite

on:
  schedule:
    - cron: '0 2 * * *'  # 2 AM UTC daily
  workflow_dispatch:

jobs:
  full-tests:
    name: Full Test Suite (including slow/property tests)
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install -e .
      
      - name: Run all tests (including slow and property)
        run: |
          pytest tests/ -v --hypothesis-show-statistics
        timeout-minutes: 120
      
      - name: Upload full test results
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: nightly-test-results
          path: |
            htmlcov/
            .hypothesis/
```

#### 2.2 Add Performance Regression Detection
**Impact**: Catches performance degradation early
**Effort**: Medium (requires pytest-benchmark)
**Benefit**: Maintains framework performance

```yaml
performance-tests:
  name: Performance Regression Tests
  runs-on: ubuntu-latest
  steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.10'
        cache: 'pip'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest-benchmark
    
    - name: Run performance benchmarks
      run: |
        pytest tests/performance/ --benchmark-only --benchmark-json=benchmark.json
    
    - name: Compare with baseline
      run: |
        python scripts/compare_benchmarks.py benchmark.json baseline.json
```

#### 2.3 Add Multi-GPU Test Job
**Impact**: Validates GPU-specific code paths
**Effort**: High (requires GPU runner)
**Benefit**: Catches GPU-specific bugs

```yaml
gpu-tests:
  name: GPU Tests
  runs-on: [self-hosted, gpu]  # Requires self-hosted GPU runner
  steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -e .
    
    - name: Run GPU tests
      run: |
        pytest tests/ -v -m "gpu" --cov=src
```

### Priority 3: Low-Impact, High-Effort

#### 3.1 Add Integration Test Environment
**Impact**: Tests against real PACS/DICOM servers
**Effort**: High (requires infrastructure setup)
**Benefit**: Validates production deployment

#### 3.2 Add Load Testing
**Impact**: Validates system under load
**Effort**: High (requires load testing infrastructure)
**Benefit**: Ensures production scalability

#### 3.3 Add Security Scanning with Snyk
**Impact**: Additional security layer
**Effort**: Medium (requires Snyk account)
**Benefit**: Catches more vulnerabilities

## Implementation Priority

### Phase 1 (Immediate - This Week)
1. ✅ Add PACS integration tests to CI
2. ✅ Optimize dependency installation
3. ✅ Add test result caching
4. ✅ Parallelize test execution

### Phase 2 (Short-term - Next 2 Weeks)
1. Add nightly full test suite
2. Add performance regression detection
3. Improve Docker build caching

### Phase 3 (Long-term - Next Month)
1. Add multi-GPU test job (if GPU runner available)
2. Add integration test environment
3. Add load testing infrastructure

## Success Metrics

**Current State**:
- CI runtime: ~15-20 minutes
- Test coverage: 55%
- PACS tests: Not in CI
- Parallel execution: No
- Caching: Minimal (pip only)

**Target State (Phase 1)**:
- CI runtime: <10 minutes (50% reduction)
- Test coverage: 55% (maintained)
- PACS tests: In CI (40/48 properties)
- Parallel execution: Yes (pytest-xdist)
- Caching: Tests + dependencies

**Target State (Phase 2)**:
- Nightly full tests: Yes (all slow/property tests)
- Performance regression: Detected automatically
- Docker build time: <5 minutes

## Risks and Mitigation

**Risk 1: Parallel tests cause flaky failures**
- Mitigation: Use pytest-xdist with proper isolation
- Mitigation: Mark non-thread-safe tests with `@pytest.mark.serial`

**Risk 2: Caching causes stale test results**
- Mitigation: Cache key includes test file hashes
- Mitigation: Clear cache on dependency changes

**Risk 3: PACS tests increase CI time**
- Mitigation: Run PACS tests in parallel job
- Mitigation: Set 15-minute timeout
- Mitigation: Use Hypothesis profiles for faster property tests

**Risk 4: Nightly tests fail without notification**
- Mitigation: Configure GitHub Actions notifications
- Mitigation: Send Slack/email alerts on failure
- Mitigation: Create GitHub issue automatically on failure

## Cost Analysis

**Current Cost**: $0 (public repo, free GitHub Actions)

**Phase 1 Cost**: $0 (no additional resources)
- Uses existing GitHub Actions minutes
- Optimizations reduce overall usage

**Phase 2 Cost**: $0 (nightly tests within free tier)
- 2,000 minutes/month free for private repos
- Public repos: unlimited

**Phase 3 Cost**: Variable
- GPU runner: $0.08-0.16/minute (if self-hosted: hardware cost)
- Integration environment: $50-100/month (cloud infrastructure)
- Load testing: $0-50/month (depends on scale)

## Monitoring and Alerts

### GitHub Actions Notifications
```yaml
# Add to workflow
- name: Notify on failure
  if: failure()
  uses: 8398a7/action-slack@v3
  with:
    status: ${{ job.status }}
    text: 'CI failed on ${{ github.ref }}'
    webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

### Status Badges
```markdown
![CI](https://github.com/matthewvaishnav/histocore/workflows/CI/badge.svg)
![PACS Tests](https://github.com/matthewvaishnav/histocore/workflows/PACS%20Tests/badge.svg)
![Nightly](https://github.com/matthewvaishnav/histocore/workflows/Nightly/badge.svg)
```

## Next Steps

1. Review and approve optimization plan
2. Implement Phase 1 optimizations
3. Monitor CI performance improvements
4. Iterate based on results
5. Proceed to Phase 2 when Phase 1 is stable

---

**Status**: Draft - Ready for Review
**Last Updated**: April 25, 2026
**Owner**: Matthew Vaishnav
