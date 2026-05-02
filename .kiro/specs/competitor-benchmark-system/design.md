# Design Document: Competitor Benchmark System

## Overview

The Competitor Benchmark System is an automated framework for conducting fair, reproducible performance comparisons between HistoCore and competing computational pathology frameworks (PathML, CLAM, baseline PyTorch). The system addresses the critical need to replace estimated performance numbers in PERFORMANCE_COMPARISON.md with real benchmark data obtained from identical training tasks executed on the same hardware (RTX 4070 GPU).

### Problem Statement

Current performance comparisons rely on estimated or unfair competitor numbers, undermining the credibility of HistoCore's claimed advantages. Running comprehensive benchmarks manually is error-prone, time-consuming (20-40+ hours of GPU time), and difficult to reproduce. Framework-specific dependency conflicts (particularly PathML's numpy/pandas issues with Python 3.14) create additional barriers to fair comparison.

### Solution Approach

The system automates the entire benchmark pipeline:

1. **Automated Environment Setup**: Isolates each framework in separate virtual environments with dependency conflict resolution
2. **Standardized Training Tasks**: Ensures all frameworks execute identical workloads (same datasets, hyperparameters, random seeds)
3. **Resource Management**: Coordinates GPU access, memory management, and long-running workload execution
4. **Comprehensive Metrics**: Collects performance data across multiple dimensions (accuracy, speed, memory, throughput)
5. **Automated Reporting**: Generates updated PERFORMANCE_COMPARISON.md with real, statistically validated results

### Key Design Principles

- **Fairness**: All frameworks use identical datasets, hyperparameters, and hardware
- **Reproducibility**: Complete environment and configuration versioning for result verification
- **Robustness**: Graceful error handling ensures one framework's failure doesn't invalidate entire benchmark
- **Efficiency**: Supports both quick validation mode (3-4 hours) and comprehensive evaluation mode (20-40+ hours)
- **Automation**: Minimal user intervention required for long-running GPU workloads

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    Benchmark Orchestrator                        │
│  - Workflow coordination                                         │
│  - Progress tracking                                             │
│  - Error recovery                                                │
└────────────────┬────────────────────────────────────────────────┘
                 │
    ┌────────────┼────────────┬────────────────┐
    │            │            │                │
    ▼            ▼            ▼                ▼
┌─────────┐ ┌─────────┐ ┌──────────┐ ┌──────────────┐
│Framework│ │Training │ │ Resource │ │   Metrics    │
│ Manager │ │  Task   │ │ Manager  │ │  Collector   │
│         │ │Executor │ │          │ │              │
└────┬────┘ └────┬────┘ └────┬─────┘ └──────┬───────┘
     │           │           │               │
     │           │           │               │
     ▼           ▼           ▼               ▼
┌─────────────────────────────────────────────────────┐
│              Framework Environments                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │HistoCore │  │  PathML  │  │   CLAM   │          │
│  │  venv    │  │   venv   │  │   venv   │          │
│  └──────────┘  └──────────┘  └──────────┘          │
└─────────────────────────────────────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   Report Generator    │
         │ - Statistical analysis│
         │ - Visualization       │
         │ - Documentation       │
         └───────────────────────┘
```

### Data Flow

1. **Initialization Phase**
   - Load benchmark configuration (quick/full mode, framework selection)
   - Validate hardware availability (RTX 4070 GPU)
   - Initialize framework environments

2. **Setup Phase**
   - Install frameworks in isolated environments
   - Resolve dependency conflicts
   - Validate installations

3. **Execution Phase**
   - For each framework:
     - Configure identical training task
     - Execute training with metrics collection
     - Save checkpoints periodically
     - Handle errors gracefully
   - Clear GPU memory between frameworks

4. **Analysis Phase**
   - Aggregate performance metrics
   - Compute statistical significance
   - Generate comparison visualizations

5. **Reporting Phase**
   - Update PERFORMANCE_COMPARISON.md
   - Generate detailed benchmark report
   - Archive results for reproducibility

### Technology Stack

- **Core Language**: Python 3.9+ (with Python 3.14 compatibility patches)
- **Dependency Management**: venv + pip (isolated environments per framework)
- **GPU Management**: PyTorch CUDA utilities, nvidia-smi
- **Metrics Collection**: Custom instrumentation + psutil
- **Statistical Analysis**: scipy.stats, numpy
- **Visualization**: matplotlib, seaborn
- **Configuration**: YAML (Hydra-compatible)
- **Checkpointing**: PyTorch state_dict + JSON metadata
- **Logging**: Python logging module with structured output

## Components and Interfaces

### 1. Framework Manager

**Responsibility**: Manages installation, configuration, and validation of competitor frameworks.

**Interface**:
```python
class FrameworkManager:
    def install_framework(
        self, 
        framework_name: str, 
        python_version: str
    ) -> FrameworkEnvironment:
        """Install framework in isolated environment with dependency resolution."""
        
    def validate_installation(
        self, 
        env: FrameworkEnvironment
    ) -> ValidationResult:
        """Verify framework can be imported and basic operations work."""
        
    def apply_compatibility_patches(
        self, 
        framework_name: str, 
        python_version: str
    ) -> List[Patch]:
        """Apply version-specific compatibility fixes (e.g., PathML numpy issues)."""
        
    def get_framework_version(
        self, 
        env: FrameworkEnvironment
    ) -> VersionInfo:
        """Extract exact version information for reproducibility."""
```

**Key Features**:
- Separate virtual environment per framework
- Automatic detection of Python 3.14 numpy/pandas conflicts
- Compatibility patch application for PathML
- Installation validation with detailed error reporting
- Version pinning for reproducibility

### 2. Training Task Executor

**Responsibility**: Executes identical training tasks across all frameworks.

**Interface**:
```python
class TrainingTaskExecutor:
    def configure_task(
        self, 
        task_spec: TaskSpecification, 
        framework: str
    ) -> TrainingConfig:
        """Translate standard task spec to framework-specific configuration."""
        
    def execute_training(
        self, 
        config: TrainingConfig, 
        env: FrameworkEnvironment
    ) -> TrainingResult:
        """Run training task with metrics collection and checkpointing."""
        
    def validate_equivalence(
        self, 
        configs: List[TrainingConfig]
    ) -> EquivalenceReport:
        """Verify all framework configs represent identical training tasks."""
```

**Key Features**:
- Framework-agnostic task specification
- Automatic translation to framework-specific APIs
- Identical random seed enforcement
- Data split consistency verification
- Configuration equivalence validation

### 3. Resource Manager

**Responsibility**: Manages GPU resources and ensures fair hardware allocation.

**Interface**:
```python
class ResourceManager:
    def verify_gpu_availability(self) -> GPUInfo:
        """Check RTX 4070 GPU is available and ready."""
        
    def allocate_gpu(self, framework: str) -> GPUAllocation:
        """Reserve GPU for exclusive framework use."""
        
    def clear_gpu_memory(self) -> None:
        """Force GPU memory cleanup between framework executions."""
        
    def monitor_resources(self) -> ResourceMetrics:
        """Track GPU memory, temperature, utilization during training."""
        
    def enforce_limits(self, limits: ResourceLimits) -> None:
        """Apply memory limits and temperature throttling."""
```

**Key Features**:
- Exclusive GPU allocation per framework
- Memory cleanup between executions
- Temperature monitoring and throttling
- Resource usage tracking
- Out-of-memory error detection

### 4. Metrics Collector

**Responsibility**: Collects comprehensive performance metrics during training.

**Interface**:
```python
class MetricsCollector:
    def start_collection(self, framework: str) -> CollectionSession:
        """Begin metrics collection for a training run."""
        
    def record_epoch_metrics(
        self, 
        epoch: int, 
        metrics: Dict[str, float]
    ) -> None:
        """Record per-epoch training metrics."""
        
    def record_system_metrics(self) -> None:
        """Capture GPU memory, temperature, utilization."""
        
    def finalize_collection(self) -> MetricsReport:
        """Aggregate and save all collected metrics."""
        
    def compute_confidence_intervals(
        self, 
        metrics: MetricsReport
    ) -> ConfidenceIntervals:
        """Calculate bootstrap confidence intervals for key metrics."""
```

**Key Features**:
- Per-epoch metric tracking
- System resource monitoring
- Statistical confidence intervals
- Structured JSON output
- Timestamp synchronization

### 5. Checkpoint Manager

**Responsibility**: Manages checkpoints for long-running workloads and crash recovery.

**Interface**:
```python
class CheckpointManager:
    def save_checkpoint(
        self, 
        state: BenchmarkState, 
        interval_minutes: int = 30
    ) -> CheckpointPath:
        """Save benchmark state for recovery."""
        
    def load_checkpoint(
        self, 
        checkpoint_path: Path
    ) -> BenchmarkState:
        """Restore benchmark state from checkpoint."""
        
    def resume_from_checkpoint(
        self, 
        checkpoint_path: Path
    ) -> BenchmarkOrchestrator:
        """Resume interrupted benchmark from last checkpoint."""
```

**Key Features**:
- Periodic automatic checkpointing (every 30 minutes)
- Complete state serialization
- Crash recovery support
- Progress preservation

### 6. Report Generator

**Responsibility**: Generates comparison reports and updates documentation.

**Interface**:
```python
class ReportGenerator:
    def generate_comparison_table(
        self, 
        results: List[BenchmarkResult]
    ) -> pd.DataFrame:
        """Create comprehensive comparison table."""
        
    def compute_statistical_significance(
        self, 
        histocore_metrics: Metrics, 
        competitor_metrics: Metrics
    ) -> SignificanceTest:
        """Perform statistical tests comparing HistoCore to competitors."""
        
    def generate_visualizations(
        self, 
        results: List[BenchmarkResult], 
        output_dir: Path
    ) -> List[Path]:
        """Create performance comparison plots."""
        
    def update_performance_comparison_md(
        self, 
        results: List[BenchmarkResult]
    ) -> None:
        """Update PERFORMANCE_COMPARISON.md with real benchmark data."""
```

**Key Features**:
- Statistical significance testing
- Confidence interval visualization
- Efficiency scatter plots (accuracy vs parameters, accuracy vs time)
- Automated PERFORMANCE_COMPARISON.md updates
- Reproducibility documentation

### 7. Benchmark Orchestrator

**Responsibility**: Coordinates the entire benchmark workflow.

**Interface**:
```python
class BenchmarkOrchestrator:
    def run_benchmark_suite(
        self, 
        config: BenchmarkConfig
    ) -> BenchmarkSuiteResult:
        """Execute complete benchmark suite (quick or full mode)."""
        
    def run_single_framework(
        self, 
        framework: str, 
        task: TaskSpecification
    ) -> FrameworkResult:
        """Run benchmark for a single framework."""
        
    def handle_error(
        self, 
        error: Exception, 
        context: ExecutionContext
    ) -> RecoveryAction:
        """Determine recovery strategy for errors."""
        
    def estimate_completion_time(
        self, 
        config: BenchmarkConfig
    ) -> timedelta:
        """Estimate total benchmark execution time."""
```

**Key Features**:
- Multi-framework workflow coordination
- Error recovery and retry logic
- Progress tracking and logging
- Completion time estimation
- Checkpoint-based resumption

## Data Models

### TaskSpecification

Defines a framework-agnostic training task.

```python
@dataclass
class TaskSpecification:
    """Standard specification for a training task."""
    
    # Dataset configuration
    dataset_name: str  # e.g., "PatchCamelyon"
    data_root: Path
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Model architecture
    model_architecture: str  # e.g., "resnet18_transformer"
    feature_dim: int = 512
    num_classes: int = 2
    
    # Training hyperparameters
    num_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    optimizer: str = "AdamW"
    
    # Reproducibility
    random_seed: int = 42
    
    # Data augmentation
    augmentation_config: Dict[str, Any] = field(default_factory=dict)
    
    # Evaluation
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "auc", "f1"])
```

### FrameworkEnvironment

Represents an isolated framework installation.

```python
@dataclass
class FrameworkEnvironment:
    """Isolated environment for a competitor framework."""
    
    framework_name: str  # "HistoCore", "PathML", "CLAM", "PyTorch"
    venv_path: Path
    python_version: str
    framework_version: str
    dependencies: Dict[str, str]  # package_name -> version
    
    # Installation metadata
    installed_at: datetime
    patches_applied: List[str]
    validation_status: str  # "valid", "invalid", "not_validated"
    validation_errors: List[str] = field(default_factory=list)
```

### TrainingResult

Captures the outcome of a training run.

```python
@dataclass
class TrainingResult:
    """Results from a single framework training run."""
    
    framework_name: str
    task_spec: TaskSpecification
    
    # Training metrics
    training_time_seconds: float
    epochs_completed: int
    final_train_loss: float
    final_val_loss: float
    
    # Performance metrics
    test_accuracy: float
    test_auc: float
    test_f1: float
    test_precision: float
    test_recall: float
    
    # Confidence intervals (bootstrap)
    accuracy_ci: Tuple[float, float]
    auc_ci: Tuple[float, float]
    f1_ci: Tuple[float, float]
    
    # Resource usage
    peak_gpu_memory_mb: float
    avg_gpu_utilization: float
    peak_gpu_temperature: float
    
    # Throughput
    samples_per_second: float
    inference_time_ms: float
    
    # Model info
    model_parameters: int
    
    # Paths
    checkpoint_path: Path
    metrics_path: Path
    log_path: Path
    
    # Status
    status: str  # "success", "failed", "timeout"
    error_message: Optional[str] = None
```

### BenchmarkConfig

Configuration for a benchmark suite execution.

```python
@dataclass
class BenchmarkConfig:
    """Configuration for benchmark suite execution."""
    
    # Execution mode
    mode: str  # "quick" (3-4 hours) or "full" (20-40+ hours)
    
    # Framework selection
    frameworks: List[str] = field(
        default_factory=lambda: ["HistoCore", "PathML", "CLAM", "PyTorch"]
    )
    
    # Task configuration
    task_spec: TaskSpecification
    
    # Quick mode overrides
    quick_mode_epochs: int = 3
    quick_mode_samples: int = 1000
    
    # Resource limits
    max_gpu_memory_mb: int = 12000  # RTX 4070 has 12GB
    max_gpu_temperature: float = 85.0
    timeout_hours: float = 48.0
    
    # Checkpointing
    checkpoint_interval_minutes: int = 30
    
    # Output
    output_dir: Path = Path("results/competitor_benchmarks")
    
    # Reproducibility
    random_seed: int = 42
    
    # Statistical analysis
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95
```

### BenchmarkSuiteResult

Aggregated results from a complete benchmark suite.

```python
@dataclass
class BenchmarkSuiteResult:
    """Aggregated results from complete benchmark suite."""
    
    config: BenchmarkConfig
    
    # Framework results
    framework_results: Dict[str, TrainingResult]
    
    # Timing
    start_time: datetime
    end_time: datetime
    total_duration_hours: float
    
    # Statistical comparisons
    significance_tests: Dict[str, SignificanceTest]
    
    # Rankings
    accuracy_ranking: List[Tuple[str, float]]
    auc_ranking: List[Tuple[str, float]]
    efficiency_ranking: List[Tuple[str, float]]  # accuracy / parameters
    
    # Paths
    report_path: Path
    visualization_dir: Path
    
    # Status
    successful_frameworks: List[str]
    failed_frameworks: List[str]
    errors: Dict[str, str]
```

### SignificanceTest

Statistical comparison between HistoCore and a competitor.

```python
@dataclass
class SignificanceTest:
    """Statistical significance test results."""
    
    histocore_metric: float
    competitor_metric: float
    competitor_name: str
    metric_name: str  # "accuracy", "auc", "f1"
    
    # Test results
    improvement: float  # histocore - competitor
    improvement_pct: float
    
    # Statistical measures
    cohens_d: float  # effect size
    p_value: float
    ci_overlap: bool  # do confidence intervals overlap?
    
    # Interpretation
    significance_level: str  # "Large Effect", "Medium Effect", "Small Effect", "No Effect"
    statistically_significant: bool  # p < 0.05 and |d| > 0.2
```

## Error Handling

### Error Categories

1. **Installation Errors**
   - Dependency conflicts (PathML numpy/pandas with Python 3.14)
   - Missing system libraries (CUDA, cuDNN)
   - Network failures during package download
   - **Recovery**: Retry with exponential backoff, apply compatibility patches, skip framework if fatal

2. **Configuration Errors**
   - Invalid task specification
   - Incompatible hyperparameters for framework
   - Missing dataset files
   - **Recovery**: Validate configuration before execution, provide detailed error messages, halt benchmark

3. **Runtime Errors**
   - Out-of-memory errors
   - GPU unavailability
   - Training divergence (NaN losses)
   - Framework-specific crashes
   - **Recovery**: Log error, mark framework as failed, continue with remaining frameworks

4. **Timeout Errors**
   - Training exceeds time limit
   - Hung processes
   - **Recovery**: Terminate process, save partial results, mark as timeout

5. **Data Errors**
   - Corrupted checkpoints
   - Invalid metrics (NaN, Inf)
   - Mismatched data splits
   - **Recovery**: Validate data integrity, flag suspicious results, exclude from comparison

### Error Handling Strategy

```python
class ErrorHandler:
    def handle_error(
        self, 
        error: Exception, 
        context: ExecutionContext
    ) -> RecoveryAction:
        """Determine appropriate recovery action based on error type."""
        
        if isinstance(error, DependencyConflictError):
            if context.retry_count < 3:
                return RecoveryAction.RETRY_WITH_PATCH
            else:
                return RecoveryAction.SKIP_FRAMEWORK
                
        elif isinstance(error, OutOfMemoryError):
            return RecoveryAction.LOG_AND_CONTINUE
            
        elif isinstance(error, ConfigurationError):
            return RecoveryAction.HALT_BENCHMARK
            
        elif isinstance(error, TimeoutError):
            return RecoveryAction.SAVE_PARTIAL_AND_CONTINUE
            
        else:
            return RecoveryAction.LOG_AND_CONTINUE
```

### Retry Logic

- **Transient Errors**: Retry up to 3 times with exponential backoff (1s, 2s, 4s)
- **Dependency Conflicts**: Apply compatibility patches, then retry
- **GPU Unavailability**: Wait and retry (max 5 minutes)
- **Fatal Errors**: Log detailed diagnostics, skip framework, continue benchmark

### Validation and Sanity Checks

```python
class ResultValidator:
    def validate_training_result(
        self, 
        result: TrainingResult
    ) -> ValidationReport:
        """Validate training result for sanity and quality."""
        
        issues = []
        
        # Check metric ranges
        if not (0.0 <= result.test_accuracy <= 1.0):
            issues.append("Accuracy out of valid range [0, 1]")
            
        # Check for anomalies
        if result.test_accuracy < 0.5 and result.task_spec.num_classes == 2:
            issues.append("Accuracy below random chance for binary classification")
            
        # Check training progress
        if result.final_train_loss >= result.initial_train_loss:
            issues.append("Training loss did not decrease")
            
        # Check resource usage
        if result.samples_per_second > 10000:
            issues.append("Suspiciously high throughput (possible measurement error)")
            
        return ValidationReport(
            valid=len(issues) == 0,
            issues=issues,
            requires_manual_review=len(issues) > 0
        )
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Configuration Equivalence Across Frameworks

*For any* training task specification and set of frameworks, when the task is configured for each framework, all frameworks SHALL receive equivalent configurations including identical random seeds, data splits, model architectures, augmentation pipelines, optimizer settings, and thread counts.

**Validates: Requirements 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 3.7**

### Property 2: Serialization Round-Trip Preservation

*For any* benchmark configuration or metrics report, serializing to JSON/YAML and then deserializing SHALL produce an equivalent object with all data preserved.

**Validates: Requirements 4.9, 9.8**

### Property 3: Result Validation Sanity Checks

*For any* training result, the validation system SHALL verify that: (1) accuracy metrics are within [0.0, 1.0], (2) no metrics contain NaN or Inf values, (3) throughput does not exceed theoretical hardware limits, and (4) training loss shows decreasing trend over epochs.

**Validates: Requirements 8.5, 10.1, 10.6, 10.4**

### Property 4: Exponential Backoff Retry Pattern

*For any* transient failure requiring retry, the retry delays SHALL follow an exponential backoff pattern where delay(n) = base_delay * 2^n for retry attempt n.

**Validates: Requirement 8.2**

## Testing Strategy

### Dual Testing Approach

The Competitor Benchmark System uses a **dual testing approach** combining unit tests for specific examples and property-based tests for universal correctness guarantees:

- **Unit Tests**: Verify specific examples, edge cases, and error conditions
- **Property Tests**: Verify universal properties across all inputs (where applicable)
- Together: Comprehensive coverage (unit tests catch concrete bugs, property tests verify general correctness)

### Property-Based Tests

**Library**: Hypothesis (Python property-based testing library)

**Configuration**: Minimum 100 iterations per property test

**Property Tests to Implement**:

1. **test_configuration_equivalence_property()**
   - **Property**: Configuration Equivalence Across Frameworks
   - **Strategy**: Generate random TaskSpecification instances, configure for multiple frameworks, verify all receive equivalent settings
   - **Tag**: `# Feature: competitor-benchmark-system, Property 1: Configuration Equivalence Across Frameworks`
   - **Validates**: Requirements 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 3.7

2. **test_serialization_roundtrip_property()**
   - **Property**: Serialization Round-Trip Preservation
   - **Strategy**: Generate random BenchmarkConfig and MetricsReport instances, serialize to JSON/YAML, deserialize, verify equivalence
   - **Tag**: `# Feature: competitor-benchmark-system, Property 2: Serialization Round-Trip Preservation`
   - **Validates**: Requirements 4.9, 9.8

3. **test_result_validation_property()**
   - **Property**: Result Validation Sanity Checks
   - **Strategy**: Generate random TrainingResult instances (including edge cases with NaN, out-of-range values), verify validation catches all issues
   - **Tag**: `# Feature: competitor-benchmark-system, Property 3: Result Validation Sanity Checks`
   - **Validates**: Requirements 8.5, 10.1, 10.6, 10.4

4. **test_exponential_backoff_property()**
   - **Property**: Exponential Backoff Retry Pattern
   - **Strategy**: Generate random retry counts, verify delay follows exponential pattern
   - **Tag**: `# Feature: competitor-benchmark-system, Property 4: Exponential Backoff Retry Pattern`
   - **Validates**: Requirement 8.2

### Unit Tests

**Scope**: Individual components in isolation

**Examples**:
- `test_framework_manager_install()`: Verify framework installation in isolated venv
- `test_dependency_resolver_patches()`: Verify PathML numpy/pandas patches applied correctly
- `test_task_specification_validation()`: Verify invalid task specs are rejected
- `test_metrics_collector_recording()`: Verify metrics are recorded correctly
- `test_checkpoint_save_load()`: Verify checkpoint serialization round-trip
- `test_resource_manager_gpu_allocation()`: Verify GPU allocation and cleanup
- `test_report_generator_table_creation()`: Verify comparison table generation
- `test_python_version_detection()`: Verify Python version detection (Req 1.1)
- `test_python_314_patch_application()`: Verify patches applied for Python 3.14 (Req 1.2)
- `test_pytorch_version_matching()`: Verify PyTorch version comparison (Req 1.5)
- `test_installation_error_logging()`: Verify error messages on installation failure (Req 1.7)
- `test_configuration_difference_logging()`: Verify warnings logged for config differences (Req 2.8)
- `test_gpu_detection()`: Verify RTX 4070 GPU detection (Req 3.1)
- `test_exclusive_execution()`: Verify only one framework runs at a time (Req 3.2)
- `test_gpu_memory_cleanup()`: Verify GPU memory cleared between frameworks (Req 3.3)
- `test_memory_warning_threshold()`: Verify warning at 90% memory usage (Req 3.5)
- `test_oom_error_handling()`: Verify OOM errors are caught and logged (Req 3.8)
- `test_metric_recording()`: Verify all metrics are recorded correctly (Req 4.1-4.8, 4.10)
- `test_checkpoint_interval()`: Verify checkpoints created every 30 minutes (Req 5.3)
- `test_crash_recovery()`: Verify system resumes from checkpoint after crash (Req 5.4)
- `test_progress_logging()`: Verify progress logged every 10 minutes (Req 5.5)
- `test_temperature_throttling()`: Verify throttling at 85°C (Req 5.6)
- `test_completion_notification()`: Verify notification sent on completion (Req 5.7)
- `test_timeout_enforcement()`: Verify timeout terminates hanging tasks (Req 5.8)
- `test_quick_mode_configuration()`: Verify quick mode reduces epochs and samples (Req 6.3)
- `test_full_mode_configuration()`: Verify full mode uses complete configuration (Req 6.4)
- `test_framework_selection()`: Verify framework filtering works (Req 6.7)
- `test_report_generation()`: Verify comparison report is generated (Req 7.1-7.10)
- `test_error_isolation()`: Verify one framework failure doesn't stop others (Req 8.1)
- `test_error_classification()`: Verify recoverable vs fatal error distinction (Req 8.3)
- `test_fatal_error_handling()`: Verify fatal errors mark framework unavailable (Req 8.4)
- `test_invalid_output_detection()`: Verify invalid outputs are flagged (Req 8.6)
- `test_error_summary_generation()`: Verify error summary report is generated (Req 8.8)
- `test_version_recording()`: Verify all versions are recorded (Req 9.1-9.4)
- `test_config_file_saving()`: Verify config files saved with results (Req 9.5)
- `test_requirements_generation()`: Verify requirements.txt is generated (Req 9.6)
- `test_config_validation()`: Verify config validation against environment (Req 9.9)
- `test_anomaly_detection()`: Verify anomalous results are detected (Req 10.2)
- `test_historical_comparison()`: Verify comparison against historical benchmarks (Req 10.3)
- `test_qa_flags()`: Verify QA flags included in report (Req 10.7)
- `test_manual_approval()`: Verify manual approval required before update (Req 10.8)

**Testing Approach**:
- Mock framework installations to avoid actual environment creation
- Use synthetic training results for report generation tests
- Mock GPU operations for resource manager tests
- Test error handling with injected exceptions

### Integration Tests

**Scope**: Component interactions and end-to-end workflows

**Examples**:
- `test_single_framework_benchmark()`: Run complete benchmark for one framework (HistoCore) with minimal dataset
- `test_checkpoint_recovery()`: Simulate crash and verify benchmark resumes from checkpoint
- `test_multi_framework_execution()`: Run benchmark for 2 frameworks sequentially
- `test_error_recovery()`: Inject framework failure, verify benchmark continues with remaining frameworks
- `test_report_generation_pipeline()`: Execute benchmark and verify report is generated correctly

**Testing Approach**:
- Use small synthetic datasets (100 samples) for fast execution
- Reduce epochs to 2-3 for quick tests
- Test on CPU to avoid GPU availability issues in CI
- Verify file outputs (checkpoints, reports, metrics) are created correctly

### Property-Based Tests

**Note**: Property-based testing is NOT appropriate for this system because:

1. **Infrastructure Focus**: The system primarily manages external processes, environments, and GPU resources - not pure computational logic
2. **External Dependencies**: Behavior depends on framework installations, GPU availability, and file system state
3. **Side Effects**: Core operations involve creating environments, running processes, and writing files
4. **Non-Deterministic Timing**: Training times and resource usage vary based on system load
5. **Integration Nature**: Most valuable tests verify correct interaction with external systems

Instead, the testing strategy focuses on:
- **Example-based unit tests** for configuration validation and data structures
- **Integration tests** with mocked external dependencies
- **End-to-end smoke tests** with minimal workloads
- **Manual validation** of benchmark results against known baselines

### Smoke Tests

**Scope**: Quick validation that system is functional

**Examples**:
- `test_gpu_availability()`: Verify RTX 4070 GPU is detected
- `test_framework_imports()`: Verify all frameworks can be imported
- `test_quick_mode_execution()`: Run quick mode benchmark (3-4 hours) on real hardware
- `test_checkpoint_creation()`: Verify checkpoints are created during long-running workload

**Testing Approach**:
- Run before full benchmark suite to catch environment issues early
- Use minimal configurations to reduce execution time
- Verify critical paths work end-to-end

### Manual Testing

**Scope**: Full benchmark suite validation

**Procedure**:
1. Run quick mode benchmark (3-4 hours) on RTX 4070
2. Verify all frameworks complete successfully
3. Inspect generated comparison report for sanity
4. Compare results against historical benchmarks
5. Run full mode benchmark (20-40+ hours) for final validation
6. Review updated PERFORMANCE_COMPARISON.md for accuracy

**Validation Criteria**:
- All frameworks complete without fatal errors
- Metrics are within expected ranges
- Statistical significance tests produce reasonable results
- Visualizations render correctly
- PERFORMANCE_COMPARISON.md updates are accurate

## Deployment and Operations

### System Requirements

**Hardware**:
- NVIDIA RTX 4070 GPU (12GB VRAM)
- 32GB+ system RAM
- 100GB+ free disk space (for environments, checkpoints, results)
- Adequate cooling for 20-40+ hour GPU workloads

**Software**:
- Ubuntu 20.04+ or Windows 10/11
- Python 3.9-3.14
- CUDA 11.8+ and cuDNN 8.6+
- Git (for framework installation from source)

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/histocore.git
cd histocore

# Install benchmark system dependencies
pip install -r requirements_benchmark.txt

# Verify GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Run installation validation
python experiments/benchmark_system/validate_setup.py
```

### Execution

**Quick Mode (3-4 hours)**:
```bash
python experiments/benchmark_system/run_benchmark.py \
    --mode quick \
    --frameworks HistoCore PathML CLAM PyTorch \
    --output results/quick_benchmark
```

**Full Mode (20-40+ hours)**:
```bash
python experiments/benchmark_system/run_benchmark.py \
    --mode full \
    --frameworks HistoCore PathML CLAM PyTorch \
    --output results/full_benchmark \
    --checkpoint-interval 30
```

**Resume from Checkpoint**:
```bash
python experiments/benchmark_system/run_benchmark.py \
    --resume results/full_benchmark/checkpoint_latest.json
```

### Monitoring

**Progress Tracking**:
- Log files: `results/benchmark_name/benchmark.log`
- Progress updates every 10 minutes
- Estimated completion time displayed at start

**Resource Monitoring**:
- GPU utilization: `nvidia-smi` output logged every 5 minutes
- Temperature monitoring with automatic throttling at 85°C
- Memory usage tracking to detect leaks

**Notifications**:
- Completion notification (email/Slack webhook)
- Error alerts for framework failures
- Temperature warnings

### Result Validation

**Automated Checks**:
- Metric range validation (accuracy ∈ [0, 1])
- Anomaly detection (accuracy below random chance)
- Training progress verification (loss decreasing)
- Resource usage sanity checks

**Manual Review**:
- Compare against historical benchmarks
- Inspect training curves for anomalies
- Verify statistical significance makes sense
- Review error logs for any warnings

### Troubleshooting

**Common Issues**:

1. **PathML Installation Fails (Python 3.14)**
   - **Symptom**: numpy/pandas compilation errors
   - **Solution**: Compatibility patches applied automatically, check `logs/pathml_install.log`

2. **Out of Memory Errors**
   - **Symptom**: CUDA out of memory during training
   - **Solution**: Reduce batch size in task specification, clear GPU memory between runs

3. **Training Divergence (NaN Loss)**
   - **Symptom**: Loss becomes NaN during training
   - **Solution**: Reduce learning rate, check data normalization, inspect framework-specific logs

4. **Checkpoint Corruption**
   - **Symptom**: Cannot resume from checkpoint
   - **Solution**: Use previous checkpoint, re-run from last successful framework

5. **Framework Import Errors**
   - **Symptom**: Cannot import framework modules
   - **Solution**: Verify virtual environment activation, check installation logs, re-install framework

## Future Enhancements

### Phase 2 Features

1. **Additional Frameworks**
   - Add support for more competitors (HistoNet, PathViT, MedViT)
   - Support for custom framework plugins

2. **Advanced Metrics**
   - Calibration metrics (ECE, MCE)
   - Fairness metrics (demographic parity, equalized odds)
   - Interpretability metrics (attention map quality)

3. **Distributed Execution**
   - Multi-GPU support for parallel framework execution
   - Cloud execution (AWS, Azure, GCP)
   - Distributed training for large-scale benchmarks

4. **Enhanced Reporting**
   - Interactive web dashboard for results
   - Real-time progress visualization
   - Automated publication-ready figures

5. **Continuous Benchmarking**
   - Scheduled benchmark runs (weekly/monthly)
   - Regression detection for HistoCore updates
   - Historical trend analysis

### Scalability Considerations

- **Multi-GPU**: Extend resource manager to support multiple GPUs
- **Cloud Deployment**: Add cloud provider integrations for elastic compute
- **Parallel Execution**: Run multiple frameworks simultaneously on separate GPUs
- **Caching**: Cache framework installations to speed up repeated benchmarks
- **Incremental Updates**: Only re-run frameworks that have changed

---

**Document Version**: 1.0  
**Last Updated**: 2026-01-08  
**Status**: Ready for Implementation
