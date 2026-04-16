# Dataset Testing Suite

Comprehensive testing framework for computational pathology dataset implementations.

## Overview

This testing suite provides comprehensive validation for:
- **PCam Dataset**: Binary patch classification
- **CAMELYON Dataset**: Slide-level classification with attention
- **Multimodal Datasets**: WSI + genomic + clinical text
- **OpenSlide Integration**: WSI file reading and processing
- **Data Preprocessing**: Normalization, augmentation, batch processing
- **Performance & Scalability**: Memory usage, loading times, caching
- **Error Handling**: Corruption detection, recovery, constraints

## Test Structure

```
tests/dataset_testing/
├── unit/                    # Unit tests (55 tests)
│   ├── test_preprocessing.py           # Core preprocessing (28 tests)
│   ├── test_batch_preprocessing.py     # Batch processing (27 tests)
│   ├── test_error_handling.py          # Error detection (32 tests)
│   ├── test_network_storage_constraints.py  # Constraints (24 tests)
│   ├── test_openslide_properties.py    # OpenSlide properties (13 tests)
│   ├── test_openslide_tissue_detection.py  # Tissue detection (11 tests)
│   └── test_openslide_error_handling.py     # OpenSlide errors (11 tests)
├── integration/             # Integration tests (32 tests)
│   ├── test_pipeline_integration.py    # API compatibility (19 tests)
│   └── test_training_loop_integration.py    # Training loops (13 tests)
├── performance/             # Performance tests (28 tests)
│   ├── test_performance_benchmarks.py # Benchmarking (14 tests)
│   └── test_caching_optimization.py   # Caching (15 tests)
├── pcam/                    # PCam-specific tests
│   └── test_pcam_download.py          # Download validation (10 tests)
├── camelyon/                # CAMELYON-specific tests
├── multimodal/              # Multimodal tests
├── synthetic/               # Synthetic data generators
└── base_interfaces.py       # Shared test utilities
```

## Quick Start

### Run All Tests

```bash
# Complete test suite (171 tests, ~27s)
python scripts/run_all_dataset_tests.py

# Fast mode (stop on first failure)
python scripts/run_all_dataset_tests.py --fast

# Specific categories
python scripts/run_all_dataset_tests.py --categories unit integration
```

### Run by Category

```bash
# Unit tests only (111 tests, ~8s)
pytest tests/dataset_testing/unit/ -v

# Integration tests (32 tests, ~4s)
pytest tests/dataset_testing/integration/ -v

# Performance tests (28 tests, ~15s)
pytest tests/dataset_testing/performance/ -v
```

### Run with Coverage

```bash
# Generate coverage report
python scripts/generate_coverage_report.py

# View HTML report
open coverage_reports/html/index.html
```

## Test Categories

### Unit Tests (111 tests)

**Preprocessing Tests** (`test_preprocessing.py` - 28 tests):
- WSI feature normalization (minmax, standardize, L2, zscore, quantile)
- Genomic preprocessing (variance filtering, imputation)
- Clinical text processing (tokenization, vocab, padding)
- Edge cases (empty input, NaN handling, extreme values)

**Batch Processing** (`test_batch_preprocessing.py` - 27 tests):
- Batch consistency across transforms
- Configuration drift detection
- HDF5 batch operations
- End-to-end preprocessing pipelines

**Error Handling** (`test_error_handling.py` - 32 tests):
- Missing file scenarios with recovery suggestions
- Corrupted data detection (NaN/inf, dtype/shape issues)
- Memory constraint handling (chunking, generators)
- Invalid configuration validation

**Network/Storage Constraints** (`test_network_storage_constraints.py` - 24 tests):
- Network failure handling (timeout, retry, backoff)
- Disk space constraints (compression, cleanup)
- Resource cleanup (file handles, temp files)

**OpenSlide Integration** (35 tests across 3 files):
- WSI properties and metadata validation
- Tissue detection algorithms
- Error handling for corrupted files

### Integration Tests (32 tests)

**Pipeline Integration** (`test_pipeline_integration.py` - 19 tests):
- Dataset API backward compatibility
- Preprocessing pipeline integration
- Reproducibility (deterministic loading, config hashing)
- End-to-end dataset→preprocessing→DataLoader→model

**Training Loop Integration** (`test_training_loop_integration.py` - 13 tests):
- Basic training loops with validation
- Early stopping and gradient accumulation
- Dataset versioning and migration
- Failure isolation and component identification

### Performance Tests (28 tests)

**Performance Benchmarks** (`test_performance_benchmarks.py` - 14 tests):
- Loading time validation against thresholds
- Memory usage monitoring and leak detection
- Parallel loading thread safety
- Performance regression detection

**Caching Optimization** (`test_caching_optimization.py` - 15 tests):
- Cache hit rate validation
- Memory usage limits for large datasets
- Bottleneck identification (I/O, preprocessing)
- LRU eviction and cache tuning

## Coverage Analysis

Current coverage metrics:
- **preprocessing.py**: 72% (main target module)
- **openslide_utils.py**: 54% (WSI processing)
- **Overall**: 55% across all modules

### Coverage Reports

```bash
# Generate detailed coverage
python scripts/generate_coverage_report.py \
  --test-dir tests/dataset_testing/unit \
  --source-paths src/data/preprocessing.py src/data/openslide_utils.py

# View reports
open coverage_reports/html/index.html           # Interactive HTML
cat coverage_reports/coverage_report.md         # Markdown summary
```

## Test Execution Logging

Detailed execution logs with failure analysis:

```bash
# Run with detailed logging
python scripts/test_execution_logger.py tests/dataset_testing/unit/

# View execution report
cat test_logs/report_YYYYMMDD_HHMMSS.md
```

**Log Contents**:
- Environment information (OS, Python, memory, disk)
- Test execution timeline
- Failure analysis with reproduction steps
- Suggested fixes for common errors

## Synthetic Data Generation

Test suite includes synthetic data generators for isolated testing:

```bash
# Generate synthetic PCam data
python tests/dataset_testing/synthetic/generate_pcam.py

# Generate synthetic CAMELYON data  
python tests/dataset_testing/synthetic/generate_camelyon.py

# Generate multimodal synthetic data
python tests/dataset_testing/synthetic/generate_multimodal.py
```

**Benefits**:
- No dependency on large real datasets
- Controlled corruption scenarios
- Faster test execution
- Reproducible test conditions

## CI Integration

### GitHub Actions

Add to `.github/workflows/dataset-tests.yml`:

```yaml
name: Dataset Tests
on: [push, pull_request]

jobs:
  dataset-tests:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -e .
    
    - name: Run dataset tests
      run: python scripts/run_all_dataset_tests.py --ci
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage_reports/coverage.xml
```

### Local CI Simulation

```bash
# Simulate CI environment
python scripts/run_all_dataset_tests.py --ci --fast

# Check dependencies only
python scripts/run_all_dataset_tests.py --check-deps
```

## Performance Benchmarks

### Expected Performance

| Test Category | Tests | Duration | Memory |
|---------------|-------|----------|--------|
| Unit | 111 | ~8s | <500MB |
| Integration | 32 | ~4s | <1GB |
| Performance | 28 | ~15s | <2GB |
| **Total** | **171** | **~27s** | **<2GB** |

### Optimization Tips

**Faster Execution**:
```bash
# Parallel execution (pytest-xdist)
pip install pytest-xdist
pytest tests/dataset_testing/ -n auto

# Stop on first failure
pytest tests/dataset_testing/ -x

# Minimal output
pytest tests/dataset_testing/ -q
```

**Memory Optimization**:
```bash
# Chunked loading tests only
pytest tests/dataset_testing/ -k "chunk"

# Skip memory-intensive tests
pytest tests/dataset_testing/ -m "not memory_intensive"
```

## Troubleshooting

### Common Issues

**Import Errors**:
```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Install in development mode
pip install -e .
```

**Missing Dependencies**:
```bash
# Check all dependencies
python scripts/run_all_dataset_tests.py --check-deps

# Install missing packages
pip install hypothesis h5py psutil
```

**File Not Found**:
```bash
# Check test data exists
ls -la tests/dataset_testing/synthetic/

# Generate synthetic data
python tests/dataset_testing/synthetic/generate_all.py
```

**Memory Issues**:
```bash
# Run unit tests only (lower memory)
pytest tests/dataset_testing/unit/

# Skip performance tests
pytest tests/dataset_testing/ --ignore=tests/dataset_testing/performance/
```

### Debug Mode

```bash
# Verbose output with full tracebacks
pytest tests/dataset_testing/ -vvv --tb=long

# Drop into debugger on failure
pytest tests/dataset_testing/ --pdb

# Print statements (capture disabled)
pytest tests/dataset_testing/ -s
```

## Contributing

### Adding New Tests

1. **Choose appropriate category**: unit/integration/performance
2. **Follow naming convention**: `test_[module]_[feature].py`
3. **Use shared fixtures**: Import from `conftest.py`
4. **Add docstrings**: Describe test purpose and requirements
5. **Update coverage**: Ensure new code is tested

### Test Structure Template

```python
import pytest
from tests.dataset_testing.base_interfaces import DatasetGenerator

class TestNewFeature:
    """Test suite for new feature functionality."""
    
    def test_basic_functionality(self):
        """Test basic feature operation."""
        # Arrange
        input_data = ...
        
        # Act
        result = feature_function(input_data)
        
        # Assert
        assert result is not None
        assert len(result) > 0
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        with pytest.raises(ValueError):
            feature_function(invalid_input)
    
    @pytest.mark.parametrize("param", [1, 2, 3])
    def test_parameterized(self, param):
        """Test with multiple parameter values."""
        result = feature_function(param)
        assert result > 0
```

### Running Specific Tests

```bash
# Single test file
pytest tests/dataset_testing/unit/test_preprocessing.py -v

# Single test class
pytest tests/dataset_testing/unit/test_preprocessing.py::TestNormalization -v

# Single test method
pytest tests/dataset_testing/unit/test_preprocessing.py::TestNormalization::test_wsi_feature_norm_basic -v

# Tests matching pattern
pytest tests/dataset_testing/ -k "normalization" -v
```

## References

- **Requirements**: `.kiro/specs/comprehensive-dataset-testing/requirements.md`
- **Design**: `.kiro/specs/comprehensive-dataset-testing/design.md`
- **Tasks**: `.kiro/specs/comprehensive-dataset-testing/tasks.md`
- **Test Summary**: `tests/dataset_testing/TEST_SUMMARY.md`
- **Coverage Reports**: `coverage_reports/`
- **Execution Logs**: `test_logs/`