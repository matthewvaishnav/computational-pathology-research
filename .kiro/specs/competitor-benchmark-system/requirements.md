# Requirements Document: Competitor Benchmark System

## Introduction

The Competitor Benchmark System enables fair, reproducible performance comparisons between HistoCore and competing frameworks (PathML, CLAM, baseline PyTorch) by running identical training tasks on the same hardware (RTX 4070 GPU). The system addresses the current limitation where PERFORMANCE_COMPARISON.md contains estimated/unfair competitor numbers by executing actual benchmarks that take 20-40+ hours of GPU time for the full suite.

## Glossary

- **Benchmark_System**: The automated framework that installs, configures, and executes training tasks across multiple frameworks
- **Competitor_Framework**: External deep learning frameworks for pathology (PathML, CLAM, baseline PyTorch)
- **Training_Task**: An identical machine learning training workload executed across all frameworks for comparison
- **Hardware_Environment**: The RTX 4070 GPU system where all benchmarks execute
- **Benchmark_Suite**: A collection of training tasks ranging from quick baseline runs (3-4 hours) to full evaluations (20-40+ hours)
- **Performance_Metrics**: Quantitative measurements including training time, memory usage, accuracy, throughput, and AUC
- **Dependency_Resolver**: Component that handles framework installation and resolves conflicts (e.g., PathML numpy/pandas issues with Python 3.14)
- **Comparison_Report**: Generated documentation with real, fair performance numbers replacing current estimates
- **Long_Running_Workload**: GPU training tasks that execute for multiple hours without user intervention
- **Fair_Comparison**: Benchmarks where all frameworks use identical datasets, hyperparameters, and hardware

## Requirements

### Requirement 1: Framework Installation and Dependency Management

**User Story:** As a researcher, I want the system to automatically install PathML, CLAM, and baseline PyTorch with proper dependency resolution, so that I can run benchmarks without manual environment configuration.

#### Acceptance Criteria

1. WHEN the Benchmark_System initializes, THE Dependency_Resolver SHALL detect the Python version
2. IF Python 3.14 is detected, THEN THE Dependency_Resolver SHALL apply compatibility patches for PathML numpy/pandas compilation issues
3. THE Dependency_Resolver SHALL install PathML with all required dependencies in an isolated environment
4. THE Dependency_Resolver SHALL install CLAM with all required dependencies in an isolated environment
5. THE Dependency_Resolver SHALL verify baseline PyTorch installation matches HistoCore's PyTorch version
6. WHEN installation completes, THE Dependency_Resolver SHALL validate each framework by importing core modules
7. IF any framework installation fails, THEN THE Dependency_Resolver SHALL log detailed error messages and suggest remediation steps
8. THE Dependency_Resolver SHALL create separate virtual environments for each Competitor_Framework to prevent dependency conflicts

### Requirement 2: Identical Training Task Configuration

**User Story:** As a researcher, I want all frameworks to execute identical training tasks with the same datasets and hyperparameters, so that performance comparisons are fair and meaningful.

#### Acceptance Criteria

1. THE Benchmark_System SHALL define a standard Training_Task specification including dataset, batch size, epochs, learning rate, and model architecture
2. WHEN executing a Training_Task, THE Benchmark_System SHALL use identical random seeds across all frameworks
3. THE Benchmark_System SHALL ensure all frameworks load the same training and validation data splits
4. THE Benchmark_System SHALL configure equivalent model architectures across frameworks (matching layer counts, hidden dimensions, activation functions)
5. THE Benchmark_System SHALL apply identical data augmentation pipelines to all frameworks
6. THE Benchmark_System SHALL use the same optimizer type and hyperparameters for all frameworks
7. WHEN a Training_Task completes, THE Benchmark_System SHALL verify that all frameworks processed the same number of samples
8. THE Benchmark_System SHALL log any configuration differences that could affect Fair_Comparison validity

### Requirement 3: Hardware Resource Management

**User Story:** As a researcher, I want the system to manage GPU memory and ensure all frameworks run on identical hardware, so that performance measurements reflect framework efficiency rather than resource availability.

#### Acceptance Criteria

1. THE Benchmark_System SHALL verify RTX 4070 GPU availability before starting any Training_Task
2. WHEN executing a Training_Task, THE Benchmark_System SHALL run only one framework at a time on the Hardware_Environment
3. THE Benchmark_System SHALL clear GPU memory between framework executions
4. THE Benchmark_System SHALL monitor GPU memory usage during each Training_Task
5. IF GPU memory usage exceeds 90 percent, THEN THE Benchmark_System SHALL log a warning and continue execution
6. THE Benchmark_System SHALL record peak GPU memory usage as a Performance_Metric
7. THE Benchmark_System SHALL ensure CPU thread counts are identical across all framework executions
8. WHEN a Training_Task fails due to out-of-memory errors, THE Benchmark_System SHALL log the failure and continue with remaining benchmarks

### Requirement 4: Performance Metrics Collection

**User Story:** As a researcher, I want the system to collect comprehensive performance metrics during training, so that I can compare frameworks across multiple dimensions beyond just accuracy.

#### Acceptance Criteria

1. WHEN a Training_Task executes, THE Benchmark_System SHALL measure total training time in seconds
2. THE Benchmark_System SHALL record training time per epoch for each framework
3. THE Benchmark_System SHALL measure peak GPU memory usage in megabytes
4. THE Benchmark_System SHALL calculate training throughput in samples per second
5. THE Benchmark_System SHALL record final validation accuracy for each framework
6. THE Benchmark_System SHALL record final validation AUC for each framework
7. THE Benchmark_System SHALL record final validation F1 score for each framework
8. THE Benchmark_System SHALL measure inference time per sample in milliseconds
9. WHEN metrics collection completes, THE Benchmark_System SHALL save all Performance_Metrics to structured JSON files
10. THE Benchmark_System SHALL include confidence intervals for accuracy, AUC, and F1 metrics where applicable

### Requirement 5: Long-Running Workload Management

**User Story:** As a researcher, I want the system to handle long-running GPU workloads (20-40+ hours) reliably without requiring user intervention, so that I can run comprehensive benchmarks overnight or over weekends.

#### Acceptance Criteria

1. THE Benchmark_System SHALL support execution of Long_Running_Workload tasks without user interaction
2. WHEN a Long_Running_Workload starts, THE Benchmark_System SHALL log estimated completion time
3. THE Benchmark_System SHALL save checkpoint files every 30 minutes during Long_Running_Workload execution
4. IF the Benchmark_System crashes or is interrupted, THEN THE Benchmark_System SHALL resume from the most recent checkpoint
5. THE Benchmark_System SHALL log progress updates every 10 minutes during Long_Running_Workload execution
6. THE Benchmark_System SHALL monitor system temperature and throttle execution if GPU temperature exceeds 85 degrees Celsius
7. WHEN a Long_Running_Workload completes, THE Benchmark_System SHALL send a completion notification
8. THE Benchmark_System SHALL implement timeout limits for individual Training_Task executions to prevent infinite hangs

### Requirement 6: Benchmark Suite Execution Modes

**User Story:** As a researcher, I want to choose between quick baseline runs (3-4 hours) and full benchmark suites (20-40+ hours), so that I can validate the system quickly or run comprehensive evaluations based on available time.

#### Acceptance Criteria

1. THE Benchmark_System SHALL provide a quick mode that executes baseline Training_Task configurations in 3-4 hours
2. THE Benchmark_System SHALL provide a full mode that executes comprehensive Benchmark_Suite configurations in 20-40+ hours
3. WHEN quick mode is selected, THE Benchmark_System SHALL use reduced epoch counts and smaller validation sets
4. WHEN full mode is selected, THE Benchmark_System SHALL use production-level epoch counts and complete validation sets
5. THE Benchmark_System SHALL allow users to specify custom Benchmark_Suite configurations via command-line arguments
6. THE Benchmark_System SHALL display estimated execution time before starting any Benchmark_Suite
7. THE Benchmark_System SHALL allow users to select specific frameworks to benchmark (e.g., only PathML and HistoCore)
8. WHEN a Benchmark_Suite completes, THE Benchmark_System SHALL generate summary statistics comparing execution times across modes

### Requirement 7: Comparison Report Generation

**User Story:** As a researcher, I want the system to automatically generate updated PERFORMANCE_COMPARISON.md documentation with real benchmark numbers, so that I can replace estimated/unfair competitor numbers with actual measurements.

#### Acceptance Criteria

1. WHEN all Training_Task executions complete, THE Benchmark_System SHALL generate a Comparison_Report
2. THE Comparison_Report SHALL include a summary table with Performance_Metrics for all frameworks
3. THE Comparison_Report SHALL highlight HistoCore's performance advantages and disadvantages relative to each Competitor_Framework
4. THE Comparison_Report SHALL include statistical significance tests comparing HistoCore to each Competitor_Framework
5. THE Comparison_Report SHALL document the Hardware_Environment specifications used for benchmarking
6. THE Comparison_Report SHALL include timestamps and software versions for reproducibility
7. THE Comparison_Report SHALL generate visualization plots comparing training curves across frameworks
8. THE Comparison_Report SHALL generate efficiency scatter plots (accuracy vs training time, accuracy vs memory usage)
9. WHEN the Comparison_Report is generated, THE Benchmark_System SHALL update PERFORMANCE_COMPARISON.md with real numbers
10. THE Comparison_Report SHALL include a reproducibility section with exact commands to re-run benchmarks

### Requirement 8: Error Handling and Robustness

**User Story:** As a researcher, I want the system to handle framework-specific errors gracefully and continue benchmarking remaining frameworks, so that one framework's failure doesn't invalidate the entire benchmark suite.

#### Acceptance Criteria

1. IF a Competitor_Framework fails during Training_Task execution, THEN THE Benchmark_System SHALL log the error and continue with remaining frameworks
2. THE Benchmark_System SHALL implement retry logic with exponential backoff for transient failures
3. THE Benchmark_System SHALL distinguish between recoverable errors (e.g., temporary GPU unavailability) and fatal errors (e.g., framework installation failure)
4. WHEN a fatal error occurs, THE Benchmark_System SHALL mark the affected framework as unavailable and exclude it from the Comparison_Report
5. THE Benchmark_System SHALL validate Training_Task outputs to detect silent failures (e.g., NaN losses, zero accuracy)
6. IF invalid outputs are detected, THEN THE Benchmark_System SHALL mark the Training_Task as failed and log diagnostic information
7. THE Benchmark_System SHALL implement timeout mechanisms to prevent individual Training_Task executions from hanging indefinitely
8. WHEN the Benchmark_Suite completes, THE Benchmark_System SHALL generate an error summary report listing all failures and their causes

### Requirement 9: Reproducibility and Versioning

**User Story:** As a researcher, I want the system to record all configuration details and software versions, so that benchmark results are reproducible and can be verified by other researchers.

#### Acceptance Criteria

1. THE Benchmark_System SHALL record the exact version of each Competitor_Framework used in benchmarks
2. THE Benchmark_System SHALL record the PyTorch version, CUDA version, and cuDNN version
3. THE Benchmark_System SHALL record the Hardware_Environment specifications including GPU model, driver version, and memory capacity
4. THE Benchmark_System SHALL record the operating system version and Python version
5. THE Benchmark_System SHALL save all Training_Task configuration files alongside Performance_Metrics
6. THE Benchmark_System SHALL generate a requirements.txt file listing all Python dependencies with pinned versions
7. WHEN generating the Comparison_Report, THE Benchmark_System SHALL include a reproducibility section with environment details
8. THE Benchmark_System SHALL support exporting benchmark configurations to shareable YAML files
9. THE Benchmark_System SHALL validate that loaded configurations match the current environment before execution

### Requirement 10: Result Validation and Quality Assurance

**User Story:** As a researcher, I want the system to validate benchmark results for sanity and detect anomalies, so that I can trust the performance comparisons are accurate and meaningful.

#### Acceptance Criteria

1. WHEN a Training_Task completes, THE Benchmark_System SHALL validate that accuracy metrics are within expected ranges (0.0 to 1.0)
2. THE Benchmark_System SHALL detect anomalous results such as accuracy significantly below random chance
3. THE Benchmark_System SHALL compare results against historical benchmarks and flag significant deviations
4. IF training loss does not decrease over epochs, THEN THE Benchmark_System SHALL flag the Training_Task as potentially invalid
5. THE Benchmark_System SHALL verify that training and validation metrics follow expected patterns (e.g., validation loss not significantly lower than training loss)
6. THE Benchmark_System SHALL implement sanity checks for Performance_Metrics (e.g., throughput not exceeding theoretical hardware limits)
7. WHEN generating the Comparison_Report, THE Benchmark_System SHALL include quality assurance flags for any suspicious results
8. THE Benchmark_System SHALL allow manual review and approval of benchmark results before updating PERFORMANCE_COMPARISON.md
