"""
Core data models for the Competitor Benchmark System.

This module defines the data structures used throughout the benchmark system,
including task specifications, framework environments, training results, and
benchmark configurations.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class TaskSpecification:
    """Standard specification for a training task."""
    
    # Required fields (no defaults)
    dataset_name: str  # e.g., "PatchCamelyon"
    data_root: Path
    model_architecture: str  # e.g., "resnet18_transformer"
    
    # Dataset configuration (with defaults)
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Model architecture (with defaults)
    feature_dim: int = 512
    num_classes: int = 2
    
    # Training hyperparameters (with defaults)
    num_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    optimizer: str = "AdamW"
    
    # Reproducibility (with defaults)
    random_seed: int = 42
    
    # Data augmentation (with defaults)
    augmentation_config: Dict[str, Any] = field(default_factory=dict)
    
    # Evaluation (with defaults)
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "auc", "f1"])
    
    def __post_init__(self):
        """Validate task specification."""
        if not isinstance(self.data_root, Path):
            self.data_root = Path(self.data_root)
        
        if not 0.0 <= self.train_split <= 1.0:
            raise ValueError(f"train_split must be in [0, 1], got {self.train_split}")
        if not 0.0 <= self.val_split <= 1.0:
            raise ValueError(f"val_split must be in [0, 1], got {self.val_split}")
        if not 0.0 <= self.test_split <= 1.0:
            raise ValueError(f"test_split must be in [0, 1], got {self.test_split}")
        
        total_split = self.train_split + self.val_split + self.test_split
        if not 0.99 <= total_split <= 1.01:  # Allow small floating point errors
            raise ValueError(f"Splits must sum to 1.0, got {total_split}")
        
        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {self.num_epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")


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
    
    def __post_init__(self):
        """Validate framework environment."""
        if not isinstance(self.venv_path, Path):
            self.venv_path = Path(self.venv_path)
        
        valid_statuses = ["valid", "invalid", "not_validated"]
        if self.validation_status not in valid_statuses:
            raise ValueError(
                f"validation_status must be one of {valid_statuses}, "
                f"got {self.validation_status}"
            )


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
    
    def __post_init__(self):
        """Validate training result."""
        if not isinstance(self.checkpoint_path, Path):
            self.checkpoint_path = Path(self.checkpoint_path)
        if not isinstance(self.metrics_path, Path):
            self.metrics_path = Path(self.metrics_path)
        if not isinstance(self.log_path, Path):
            self.log_path = Path(self.log_path)
        
        valid_statuses = ["success", "failed", "timeout"]
        if self.status not in valid_statuses:
            raise ValueError(
                f"status must be one of {valid_statuses}, got {self.status}"
            )


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
    task_spec: Optional[TaskSpecification] = None
    
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
    
    def __post_init__(self):
        """Validate benchmark configuration."""
        if not isinstance(self.output_dir, Path):
            self.output_dir = Path(self.output_dir)
        
        valid_modes = ["quick", "full"]
        if self.mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got {self.mode}")
        
        if self.quick_mode_epochs <= 0:
            raise ValueError(
                f"quick_mode_epochs must be positive, got {self.quick_mode_epochs}"
            )
        if self.quick_mode_samples <= 0:
            raise ValueError(
                f"quick_mode_samples must be positive, got {self.quick_mode_samples}"
            )
        if self.max_gpu_memory_mb <= 0:
            raise ValueError(
                f"max_gpu_memory_mb must be positive, got {self.max_gpu_memory_mb}"
            )
        if not 0.0 < self.max_gpu_temperature <= 100.0:
            raise ValueError(
                f"max_gpu_temperature must be in (0, 100], "
                f"got {self.max_gpu_temperature}"
            )
        if self.timeout_hours <= 0:
            raise ValueError(
                f"timeout_hours must be positive, got {self.timeout_hours}"
            )
        if self.checkpoint_interval_minutes <= 0:
            raise ValueError(
                f"checkpoint_interval_minutes must be positive, "
                f"got {self.checkpoint_interval_minutes}"
            )
        if self.bootstrap_samples <= 0:
            raise ValueError(
                f"bootstrap_samples must be positive, got {self.bootstrap_samples}"
            )
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError(
                f"confidence_level must be in (0, 1), got {self.confidence_level}"
            )


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
    significance_tests: Dict[str, 'SignificanceTest']
    
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
    
    def __post_init__(self):
        """Validate benchmark suite result."""
        if not isinstance(self.report_path, Path):
            self.report_path = Path(self.report_path)
        if not isinstance(self.visualization_dir, Path):
            self.visualization_dir = Path(self.visualization_dir)


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
    
    def __post_init__(self):
        """Validate significance test."""
        valid_levels = ["Large Effect", "Medium Effect", "Small Effect", "No Effect"]
        if self.significance_level not in valid_levels:
            raise ValueError(
                f"significance_level must be one of {valid_levels}, "
                f"got {self.significance_level}"
            )
