# Implementation Plan: Competitor Benchmark System

## Overview

This implementation plan breaks down the Competitor Benchmark System into discrete coding tasks. The system automates fair performance comparisons between HistoCore and competing frameworks (PathML, CLAM, baseline PyTorch) by executing identical training tasks on RTX 4070 GPU hardware. The implementation handles framework installation with dependency conflict resolution, long-running GPU workloads (20-40+ hours), checkpoint recovery, comprehensive metrics collection, and automated PERFORMANCE_COMPARISON.md updates.

## Tasks

- [x] 1. Set up project structure and core data models
  - Create directory structure: `experiments/benchmark_system/`
  - Define core data models in `experiments/benchmark_system/models.py`:
    - `TaskSpecification` dataclass with dataset, model, hyperparameter configuration
    - `FrameworkEnvironment` dataclass for isolated framework installations
    - `TrainingResult` dataclass for training outcomes and metrics
    - `BenchmarkConfig` dataclass for benchmark suite configuration
    - `BenchmarkSuiteResult` dataclass for aggregated results
    - `SignificanceTest` dataclass for statistical comparisons
  - Add type hints and validation for all data models
  - _Requirements: 2.1, 4.1-4.10, 9.1-9.4_

- [x] 1.1 Write property test for serialization round-trip
  - **Property 2: Serialization Round-Trip Preservation**
  - **Validates: Requirements 4.9, 9.8**
  - Use Hypothesis to generate random `BenchmarkConfig` and `TrainingResult` instances
  - Serialize to JSON, deserialize, verify equivalence
  - Test file: `tests/benchmark_system/test_serialization_properties.py`

- [x] 2. Implement Framework Manager component
  - [x] 2.1 Create `FrameworkManager` class in `experiments/benchmark_system/framework_manager.py`
    - Implement `install_framework()` method to create isolated venv per framework
    - Implement `validate_installation()` method to verify framework imports
    - Implement `apply_compatibility_patches()` for PathML numpy/pandas Python 3.14 issues
    - Implement `get_framework_version()` to extract version information
    - Add Python version detection logic
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8_

  - [x] 2.2 Write unit tests for Framework Manager
    - Test framework installation in isolated venv
    - Test Python 3.14 detection and patch application
    - Test installation validation and error logging
    - Test version extraction
    - Test file: `tests/benchmark_system/test_framework_manager.py`
    - _Requirements: 1.1, 1.2, 1.5, 1.7_

- [x] 3. Implement Training Task Executor component
  - [x] 3.1 Create `TrainingTaskExecutor` class in `experiments/benchmark_system/task_executor.py`
    - Implement `configure_task()` to translate TaskSpecification to framework-specific config
    - Implement `execute_training()` to run training with metrics collection
    - Implement `validate_equivalence()` to verify identical configurations across frameworks
    - Add random seed enforcement logic
    - Add data split consistency verification
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7_

  - [x] 3.2 Write property test for configuration equivalence
    - **Property 1: Configuration Equivalence Across Frameworks**
    - **Validates: Requirements 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 3.7**
    - Use Hypothesis to generate random TaskSpecification instances
    - Configure for multiple frameworks, verify all receive equivalent settings
    - Test file: `tests/benchmark_system/test_configuration_properties.py`

  - [x] 3.3 Write unit tests for Training Task Executor
    - Test task specification validation
    - Test configuration translation for each framework
    - Test configuration difference logging
    - Test file: `tests/benchmark_system/test_task_executor.py`
    - _Requirements: 2.1, 2.8_

- [x] 4. Checkpoint - Ensure all tests pass
  - Run pytest on all tests created so far
  - Verify data models serialize/deserialize correctly
  - Ensure all tests pass, ask the user if questions arise

- [x] 5. Implement Resource Manager component
  - [x] 5.1 Create `ResourceManager` class in `experiments/benchmark_system/resource_manager.py`
    - Implement `verify_gpu_availability()` to detect RTX 4070 GPU
    - Implement `allocate_gpu()` for exclusive framework GPU access
    - Implement `clear_gpu_memory()` to cleanup between framework executions
    - Implement `monitor_resources()` to track GPU memory, temperature, utilization
    - Implement `enforce_limits()` for memory limits and temperature throttling
    - Add GPU memory warning at 90% threshold
    - Add temperature throttling at 85°C
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8_

  - [x] 5.2 Write unit tests for Resource Manager
    - Test GPU detection (mock nvidia-smi)
    - Test exclusive execution enforcement
    - Test GPU memory cleanup
    - Test memory warning threshold
    - Test temperature throttling
    - Test OOM error handling
    - Test file: `tests/benchmark_system/test_resource_manager.py`
    - _Requirements: 3.1, 3.2, 3.3, 3.5, 3.8_

- [x] 6. Implement Metrics Collector component
  - [x] 6.1 Create `MetricsCollector` class in `experiments/benchmark_system/metrics_collector.py`
    - Implement `start_collection()` to begin metrics collection session
    - Implement `record_epoch_metrics()` to capture per-epoch training metrics
    - Implement `record_system_metrics()` to capture GPU memory, temperature, utilization
    - Implement `finalize_collection()` to aggregate and save metrics to JSON
    - Implement `compute_confidence_intervals()` using bootstrap method
    - Add timestamp synchronization for all metrics
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 4.10_

  - [x] 6.2 Write unit tests for Metrics Collector
    - Test metric recording for all metric types
    - Test JSON serialization of metrics
    - Test confidence interval computation
    - Test file: `tests/benchmark_system/test_metrics_collector.py`
    - _Requirements: 4.1-4.10_

- [x] 7. Implement Checkpoint Manager component
  - [x] 7.1 Create `CheckpointManager` class in `experiments/benchmark_system/checkpoint_manager.py`
    - Implement `save_checkpoint()` to serialize benchmark state every 30 minutes
    - Implement `load_checkpoint()` to restore benchmark state from file
    - Implement `resume_from_checkpoint()` to continue interrupted benchmark
    - Add checkpoint interval configuration (default 30 minutes)
    - Add checkpoint validation to detect corruption
    - _Requirements: 5.3, 5.4_

  - [x] 7.2 Write unit tests for Checkpoint Manager
    - Test checkpoint save/load round-trip
    - Test checkpoint interval enforcement
    - Test crash recovery simulation
    - Test file: `tests/benchmark_system/test_checkpoint_manager.py`
    - _Requirements: 5.3, 5.4_

- [x] 8. Checkpoint - Ensure all tests pass
  - Run pytest on all tests created so far
  - Verify resource management and checkpointing work correctly
  - Ensure all tests pass, ask the user if questions arise

- [x] 9. Implement Error Handler and Result Validator
  - [x] 9.1 Create `ErrorHandler` class in `experiments/benchmark_system/error_handler.py`
    - Implement `handle_error()` to determine recovery action based on error type
    - Implement retry logic with exponential backoff for transient failures
    - Implement error classification (recoverable vs fatal)
    - Add timeout enforcement for hanging tasks
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.7_

  - [x] 9.2 Create `ResultValidator` class in `experiments/benchmark_system/result_validator.py`
    - Implement `validate_training_result()` to check metric ranges and sanity
    - Implement anomaly detection (accuracy below random chance, NaN losses)
    - Implement training progress verification (loss decreasing)
    - Implement resource usage sanity checks
    - _Requirements: 8.5, 8.6, 10.1, 10.2, 10.4, 10.5, 10.6_

  - [x] 9.3 Write property test for result validation
    - **Property 3: Result Validation Sanity Checks**
    - **Validates: Requirements 8.5, 10.1, 10.6, 10.4**
    - Use Hypothesis to generate random TrainingResult instances with edge cases
    - Verify validation catches all issues (NaN, out-of-range, anomalies)
    - Test file: `tests/benchmark_system/test_validation_properties.py`

  - [x] 9.4 Write property test for exponential backoff
    - **Property 4: Exponential Backoff Retry Pattern**
    - **Validates: Requirement 8.2**
    - Use Hypothesis to generate random retry counts
    - Verify delay follows exponential pattern: delay(n) = base_delay * 2^n
    - Test file: `tests/benchmark_system/test_retry_properties.py`

  - [x] 9.5 Write unit tests for Error Handler and Result Validator
    - Test error isolation (one framework failure doesn't stop others)
    - Test error classification
    - Test fatal error handling
    - Test invalid output detection
    - Test anomaly detection
    - Test file: `tests/benchmark_system/test_error_handling.py`
    - _Requirements: 8.1, 8.3, 8.4, 8.5, 8.6, 10.2_

- [x] 10. Implement Report Generator component
  - [x] 10.1 Create `ReportGenerator` class in `experiments/benchmark_system/report_generator.py`
    - Implement `generate_comparison_table()` to create pandas DataFrame with all metrics
    - Implement `compute_statistical_significance()` using scipy.stats for t-tests and Cohen's d
    - Implement `generate_visualizations()` to create training curve plots and efficiency scatter plots
    - Implement `update_performance_comparison_md()` to update PERFORMANCE_COMPARISON.md with real data
    - Add reproducibility section with environment details
    - Add QA flags for suspicious results
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 7.10, 10.7_

  - [x] 10.2 Write unit tests for Report Generator
    - Test comparison table generation
    - Test statistical significance computation
    - Test visualization creation (mock matplotlib)
    - Test PERFORMANCE_COMPARISON.md update
    - Test QA flags inclusion
    - Test file: `tests/benchmark_system/test_report_generator.py`
    - _Requirements: 7.1-7.10, 10.7_

- [x] 11. Implement Benchmark Orchestrator component
  - [x] 11.1 Create `BenchmarkOrchestrator` class in `experiments/benchmark_system/orchestrator.py`
    - Implement `run_benchmark_suite()` to coordinate complete benchmark workflow
    - Implement `run_single_framework()` to execute benchmark for one framework
    - Implement `estimate_completion_time()` to calculate estimated duration
    - Add progress logging every 10 minutes
    - Add completion notification support
    - Add support for quick mode (reduced epochs/samples) and full mode
    - Add framework selection filtering
    - _Requirements: 5.1, 5.2, 5.5, 5.7, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7_

  - [x] 11.2 Write unit tests for Benchmark Orchestrator
    - Test quick mode configuration
    - Test full mode configuration
    - Test framework selection filtering
    - Test progress logging
    - Test completion notification
    - Test timeout enforcement
    - Test file: `tests/benchmark_system/test_orchestrator.py`
    - _Requirements: 5.5, 5.7, 5.8, 6.3, 6.4, 6.7_

- [x] 12. Checkpoint - Ensure all tests pass
  - Run pytest on all tests created so far
  - Verify error handling, reporting, and orchestration work correctly
  - Ensure all tests pass, ask the user if questions arise

- [x] 13. Implement framework-specific adapters
  - [x] 13.1 Create adapter for HistoCore in `experiments/benchmark_system/adapters/histocore_adapter.py`
    - Implement training loop using HistoCore APIs
    - Implement metrics extraction from HistoCore training results
    - _Requirements: 2.1, 2.4, 4.1-4.8_

  - [x] 13.2 Create adapter for PathML in `experiments/benchmark_system/adapters/pathml_adapter.py`
    - Implement training loop using PathML APIs
    - Implement metrics extraction from PathML training results
    - Handle PathML-specific configuration requirements
    - _Requirements: 2.1, 2.4, 4.1-4.8_

  - [x] 13.3 Create adapter for CLAM in `experiments/benchmark_system/adapters/clam_adapter.py`
    - Implement training loop using CLAM APIs
    - Implement metrics extraction from CLAM training results
    - Handle CLAM-specific configuration requirements
    - _Requirements: 2.1, 2.4, 4.1-4.8_

  - [x] 13.4 Create adapter for baseline PyTorch in `experiments/benchmark_system/adapters/pytorch_adapter.py`
    - Implement training loop using baseline PyTorch
    - Implement metrics extraction from PyTorch training results
    - Match HistoCore's PyTorch version
    - _Requirements: 1.5, 2.1, 2.4, 4.1-4.8_

  - [x] 13.5 Write unit tests for framework adapters
    - Test each adapter with mock training data
    - Test metrics extraction for each framework
    - Test configuration translation for each framework
    - Test file: `tests/benchmark_system/test_adapters.py`
    - _Requirements: 2.1, 2.4, 4.1-4.8_

- [x] 14. Implement versioning and reproducibility tracking
  - [x] 14.1 Create `VersionTracker` class in `experiments/benchmark_system/version_tracker.py`
    - Implement `record_environment()` to capture all version information
    - Record framework versions, PyTorch version, CUDA version, cuDNN version
    - Record hardware specifications (GPU model, driver version, memory)
    - Record OS version and Python version
    - Implement `generate_requirements_txt()` to create pinned dependencies file
    - Implement `export_config_yaml()` to save shareable configuration
    - Implement `validate_config()` to verify config matches current environment
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9_

  - [x] 14.2 Write unit tests for VersionTracker
    - Test version recording for all components
    - Test requirements.txt generation
    - Test config export/import
    - Test config validation
    - Test file: `tests/benchmark_system/test_version_tracker.py`
    - _Requirements: 9.1-9.9_

- [x] 15. Create command-line interface and main entry point
  - [x] 15.1 Create CLI in `experiments/benchmark_system/cli.py`
    - Add argparse configuration for mode selection (quick/full)
    - Add framework selection arguments
    - Add output directory configuration
    - Add checkpoint resume support
    - Add custom configuration file support
    - _Requirements: 6.5, 6.6, 6.7_

  - [x] 15.2 Create main entry point in `experiments/benchmark_system/run_benchmark.py`
    - Wire all components together
    - Initialize orchestrator with configuration
    - Handle command-line arguments
    - Add logging configuration
    - Add error summary generation
    - _Requirements: 5.1, 8.8_

  - [x] 15.3 Write integration tests
    - Test single framework benchmark with synthetic data
    - Test checkpoint recovery with simulated crash
    - Test multi-framework execution
    - Test error recovery with injected failures
    - Test report generation pipeline
    - Test file: `tests/benchmark_system/test_integration.py`
    - _Requirements: 5.4, 8.1, 7.1-7.10_

- [x] 16. Checkpoint - Ensure all tests pass
  - Run complete pytest suite
  - Verify all components integrate correctly
  - Ensure all tests pass, ask the user if questions arise

- [x] 17. Create setup validation script
  - [x] 17.1 Create `experiments/benchmark_system/validate_setup.py`
    - Verify GPU availability and specifications
    - Verify CUDA and cuDNN versions
    - Verify disk space availability
    - Verify Python version compatibility
    - Run smoke tests for framework imports
    - Generate setup validation report
    - _Requirements: 3.1, 9.2, 9.3, 9.4_

- [x] 18. Create documentation and examples
  - [x] 18.1 Create `experiments/benchmark_system/README.md`
    - Document installation instructions
    - Document execution commands for quick and full modes
    - Document checkpoint resume procedure
    - Document troubleshooting common issues
    - Document result validation procedure
    - _Requirements: 7.10, 10.8_

  - [x] 18.2 Create example configuration files
    - Create `experiments/benchmark_system/configs/quick_mode.yaml`
    - Create `experiments/benchmark_system/configs/full_mode.yaml`
    - Create `experiments/benchmark_system/configs/custom_example.yaml`
    - _Requirements: 6.5, 9.8_

- [x] 19. Implement historical comparison and manual approval
  - [x] 19.1 Add historical comparison in `experiments/benchmark_system/historical_comparison.py`
    - Implement `load_historical_results()` to load previous benchmark results
    - Implement `compare_to_historical()` to detect significant deviations
    - Add flagging for results that deviate from historical baselines
    - _Requirements: 10.3_

  - [x] 19.2 Add manual approval workflow in `experiments/benchmark_system/approval.py`
    - Implement `generate_approval_report()` with all QA flags and warnings
    - Implement `request_approval()` to prompt user for review
    - Implement `apply_approved_results()` to update PERFORMANCE_COMPARISON.md only after approval
    - _Requirements: 10.8_

  - [x] 19.3 Write unit tests for historical comparison and approval
    - Test historical result loading
    - Test deviation detection
    - Test approval workflow
    - Test file: `tests/benchmark_system/test_approval.py`
    - _Requirements: 10.3, 10.8_

- [x] 20. Final checkpoint - Run end-to-end smoke test
  - Run quick mode benchmark on real hardware (if available) or with mocked GPU
  - Verify all components work together correctly
  - Verify report generation and PERFORMANCE_COMPARISON.md update
  - Ensure all tests pass, ask the user if questions arise

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation throughout implementation
- Property tests validate universal correctness properties (Configuration Equivalence, Serialization Round-Trip, Result Validation, Exponential Backoff)
- Unit tests validate specific examples and edge cases
- Integration tests verify component interactions and end-to-end workflows
- The system is designed for Python 3.9+ with compatibility for Python 3.14
- Framework adapters isolate framework-specific logic from core system
- Manual approval workflow ensures quality control before updating documentation
- Historical comparison detects regressions and anomalies
