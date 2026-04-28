# Integration Testing Suite

Comprehensive integration testing suite for the Medical AI Revolution platform. This suite validates the complete system functionality from API endpoints to model inference, ensuring production readiness.

## Overview

The integration testing suite consists of multiple test modules that validate different aspects of the platform:

- **Full Workflow Tests** - End-to-end testing of complete analysis workflows
- **API Endpoint Tests** - Comprehensive API validation including authentication and mobile app endpoints
- **Performance Regression Tests** - Performance benchmarking and regression detection
- **Test Data Fixtures** - Synthetic data generation for consistent testing

## Quick Start

### Prerequisites

1. **Python Environment**
   ```bash
   pip install -r requirements.txt
   ```

2. **Docker Services** (optional but recommended)
   ```bash
   docker-compose up -d
   ```

3. **API Server Running**
   - Ensure the API server is accessible at `http://localhost:8000`
   - Health check endpoint should return 200 OK

### Running Tests

#### Run All Tests
```bash
python tests/integration/run_integration_tests.py
```

#### Run Specific Test Suite
```bash
# Workflow tests only
python tests/integration/run_integration_tests.py --suite workflow

# API endpoint tests only
python tests/integration/run_integration_tests.py --suite api

# Performance tests only
python tests/integration/run_integration_tests.py --suite performance
```

#### Run with Docker Auto-Start
```bash
python tests/integration/run_integration_tests.py --docker
```

#### Custom Configuration
```bash
python tests/integration/run_integration_tests.py --config config.json --base-url http://api.example.com
```

## Test Modules

### 1. Full Workflow Tests (`test_full_workflow.py`)

Tests the complete medical AI workflow from image upload to results delivery.

**Test Coverage:**
- Health check and API documentation
- Image upload and analysis workflow
- DICOM integration and processing
- Performance benchmarks with concurrent requests
- Error handling and edge cases
- Database operations and connectivity
- Monitoring endpoints and metrics

**Key Features:**
- Synthetic pathology image generation
- DICOM file creation and validation
- Concurrent load testing
- Resource usage monitoring

**Usage:**
```python
from test_full_workflow import IntegrationTestSuite

suite = IntegrationTestSuite(base_url="http://localhost:8000")
results = suite.run_all_tests()
```

### 2. API Endpoint Tests (`test_api_endpoints.py`)

Comprehensive testing of all API endpoints including authentication, mobile app integration, and administrative functions.

**Test Coverage:**
- Authentication and authorization
- Mobile app specific endpoints
- Case management workflows
- Reporting and analytics
- Administrative functions
- WebSocket and real-time features

**Key Features:**
- User registration and login testing
- Mobile device registration
- Case creation and status management
- Report generation and retrieval
- Real-time notification testing

**Usage:**
```python
from test_api_endpoints import APIEndpointTests

suite = APIEndpointTests(base_url="http://localhost:8000")
results = suite.run_all_endpoint_tests()
```

### 3. Performance Regression Tests (`test_performance_regression.py`)

Performance benchmarking and regression detection to ensure system performance meets production requirements.

**Test Coverage:**
- API response time measurement
- Model inference performance
- Concurrent load testing
- Database query performance
- Memory leak detection
- System resource monitoring

**Performance Thresholds:**
- API Response Time: < 2.0 seconds
- Inference Time: < 30.0 seconds
- Database Query Time: < 1.0 seconds
- Memory Usage: < 2048 MB
- CPU Usage: < 80%
- Throughput: > 5.0 requests/second

**Usage:**
```python
from test_performance_regression import PerformanceRegressionTests

suite = PerformanceRegressionTests(base_url="http://localhost:8000")
results = suite.run_all_performance_tests()
```

### 4. Test Data Fixtures (`test_data_fixtures.py`)

Synthetic data generation for consistent and reproducible testing across all test suites.

**Generated Data:**
- Synthetic pathology images (H&E, IHC, Trichrome, PAS stains)
- DICOM files with realistic metadata
- Test case data with annotations
- User account data for authentication testing
- Performance test configurations

**Key Features:**
- Realistic pathology image synthesis
- Cancer vs. normal tissue simulation
- DICOM standard compliance
- Configurable data generation
- Automatic cleanup capabilities

**Usage:**
```python
from test_data_fixtures import TestDataFixtures

fixtures = TestDataFixtures()
fixtures.save_fixtures_to_files()
fixtures.create_sample_images_and_dicoms(20)
```

## Configuration

### Default Configuration

The test runner uses the following default configuration:

```json
{
  "base_url": "http://localhost:8000",
  "timeout": 300,
  "retry_attempts": 3,
  "parallel_execution": false,
  "generate_fixtures": true,
  "cleanup_fixtures": true,
  "save_results": true,
  "results_dir": "tests/integration/results",
  "test_suites": {
    "workflow": true,
    "api_endpoints": true,
    "performance": true,
    "fixtures": true
  },
  "performance_thresholds": {
    "api_response_time": 2.0,
    "inference_time": 30.0,
    "throughput_rps": 5.0
  },
  "docker": {
    "auto_start": false,
    "compose_file": "docker-compose.yml",
    "services": ["api", "postgres", "redis"]
  }
}
```

### Custom Configuration

Create a custom configuration file:

```json
{
  "base_url": "https://api.production.com",
  "timeout": 600,
  "performance_thresholds": {
    "api_response_time": 1.0,
    "inference_time": 20.0,
    "throughput_rps": 10.0
  },
  "test_suites": {
    "workflow": true,
    "api_endpoints": false,
    "performance": true
  }
}
```

## Test Results

### Result Structure

Test results are saved in JSON format with the following structure:

```json
{
  "summary": {
    "total_tests": 25,
    "passed_tests": 23,
    "failed_tests": 2,
    "success_rate": 92.0,
    "duration_seconds": 180.5,
    "start_time": "2026-04-27T10:30:00",
    "end_time": "2026-04-27T10:33:00"
  },
  "configuration": { ... },
  "results": {
    "workflow": { ... },
    "api_endpoints": { ... },
    "performance": { ... }
  },
  "environment": { ... }
}
```

### Result Files

Results are automatically saved to:
- `tests/integration/results/integration_test_results_YYYYMMDD_HHMMSS.json`
- `tests/integration/performance_results.json` (performance-specific)

## CI/CD Integration

### GitHub Actions

Example workflow for CI/CD integration:

```yaml
name: Integration Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  integration-tests:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Start API server
      run: |
        python src/api/main.py &
        sleep 30
    
    - name: Run integration tests
      run: |
        python tests/integration/run_integration_tests.py --suite all
    
    - name: Upload test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: integration-test-results
        path: tests/integration/results/
```

### Docker Integration

Run tests in Docker environment:

```bash
# Build test image
docker build -t medical-ai-tests -f docker/Dockerfile.tests .

# Run tests
docker run --rm \
  --network medical-ai_default \
  -v $(pwd)/tests/integration/results:/app/tests/integration/results \
  medical-ai-tests \
  python tests/integration/run_integration_tests.py --base-url http://api:8000
```

## Troubleshooting

### Common Issues

1. **API Server Not Accessible**
   ```
   Error: API accessibility check failed
   ```
   - Ensure API server is running on the specified URL
   - Check firewall and network connectivity
   - Verify health endpoint returns 200 OK

2. **Docker Services Not Starting**
   ```
   Error: Failed to start Docker services
   ```
   - Check Docker daemon is running
   - Verify docker-compose.yml file exists
   - Ensure ports are not already in use

3. **Test Fixtures Generation Failed**
   ```
   Error: Failed to generate fixtures
   ```
   - Check write permissions in test directory
   - Ensure required Python packages are installed
   - Verify sufficient disk space

4. **Performance Tests Failing**
   ```
   Error: Performance thresholds exceeded
   ```
   - Check system resources (CPU, memory)
   - Adjust performance thresholds in configuration
   - Ensure no other heavy processes are running

### Debug Mode

Enable verbose logging:

```bash
export LOG_LEVEL=DEBUG
python tests/integration/run_integration_tests.py --suite workflow
```

### Manual Test Execution

Run individual test modules:

```bash
# Full workflow tests
python tests/integration/test_full_workflow.py

# API endpoint tests
python tests/integration/test_api_endpoints.py

# Performance tests
python tests/integration/test_performance_regression.py

# Generate fixtures only
python tests/integration/test_data_fixtures.py
```

## Best Practices

### Test Development

1. **Isolation** - Each test should be independent and not rely on other tests
2. **Cleanup** - Always clean up resources after tests complete
3. **Timeouts** - Set appropriate timeouts for all network operations
4. **Error Handling** - Gracefully handle and report all errors
5. **Documentation** - Document test purpose and expected behavior

### Performance Testing

1. **Baseline** - Establish performance baselines before making changes
2. **Consistency** - Run tests in consistent environments
3. **Monitoring** - Monitor system resources during tests
4. **Thresholds** - Set realistic performance thresholds
5. **Regression** - Detect and report performance regressions

### CI/CD Integration

1. **Automation** - Fully automate test execution
2. **Reporting** - Generate comprehensive test reports
3. **Artifacts** - Save test results and logs as artifacts
4. **Notifications** - Notify team of test failures
5. **Gating** - Use tests as deployment gates

## Contributing

### Adding New Tests

1. Create test module in `tests/integration/`
2. Follow existing naming conventions
3. Implement comprehensive error handling
4. Add configuration options
5. Update test runner integration
6. Document test purpose and usage

### Modifying Existing Tests

1. Maintain backward compatibility
2. Update documentation
3. Test changes thoroughly
4. Consider performance impact
5. Update configuration if needed

## Support

For issues and questions:

- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: Check inline code documentation
- **Examples**: Review existing test implementations
- **Configuration**: Refer to configuration examples

---

*Last Updated: April 27, 2026*
*Version: 1.0.0*