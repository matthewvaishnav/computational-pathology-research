"""
Metrics Collector for the Competitor Benchmark System.

This module collects comprehensive performance metrics during training,
including per-epoch training metrics, system resource metrics, timing metrics,
efficiency metrics, and statistical metrics with confidence intervals.

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 4.10
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from experiments.benchmark_system.resource_manager import ResourceManager, ResourceMetrics

logger = logging.getLogger(__name__)


@dataclass
class EpochMetrics:
    """Per-epoch training metrics."""
    
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    learning_rate: float
    epoch_duration_seconds: float
    samples_per_second: float
    timestamp: float
    
    # Optional metrics
    train_auc: Optional[float] = None
    val_auc: Optional[float] = None
    train_f1: Optional[float] = None
    val_f1: Optional[float] = None


@dataclass
class SystemMetrics:
    """System resource metrics at a point in time."""
    
    gpu_memory_used_mb: float
    gpu_memory_total_mb: float
    gpu_memory_percent: float
    gpu_temperature: float
    gpu_utilization: float
    timestamp: float


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across all epochs."""
    
    # Training metrics
    mean_train_loss: float
    std_train_loss: float
    mean_val_loss: float
    std_val_loss: float
    mean_train_accuracy: float
    std_train_accuracy: float
    mean_val_accuracy: float
    std_val_accuracy: float
    
    # Timing metrics
    total_training_time_seconds: float
    mean_epoch_duration_seconds: float
    std_epoch_duration_seconds: float
    
    # Efficiency metrics
    mean_samples_per_second: float
    std_samples_per_second: float
    mean_gpu_utilization: float
    std_gpu_utilization: float
    
    # Resource metrics
    peak_gpu_memory_mb: float
    mean_gpu_memory_mb: float
    peak_gpu_temperature: float
    mean_gpu_temperature: float
    
    # Confidence intervals (95% by default)
    train_accuracy_ci: Tuple[float, float]
    val_accuracy_ci: Tuple[float, float]
    train_loss_ci: Tuple[float, float]
    val_loss_ci: Tuple[float, float]


@dataclass
class CollectionSession:
    """Active metrics collection session."""
    
    framework_name: str
    session_id: str
    start_time: float
    epoch_metrics: List[EpochMetrics] = field(default_factory=list)
    system_metrics: List[SystemMetrics] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Collects comprehensive performance metrics during training."""
    
    def __init__(
        self,
        resource_manager: Optional[ResourceManager] = None,
        confidence_level: float = 0.95,
        bootstrap_samples: int = 1000,
    ):
        """
        Initialize Metrics Collector.
        
        Args:
            resource_manager: ResourceManager for system metrics collection
            confidence_level: Confidence level for bootstrap intervals (default 0.95)
            bootstrap_samples: Number of bootstrap samples (default 1000)
        """
        self.resource_manager = resource_manager or ResourceManager()
        self.confidence_level = confidence_level
        self.bootstrap_samples = bootstrap_samples
        self.current_session: Optional[CollectionSession] = None
        
    def start_collection(
        self,
        framework: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CollectionSession:
        """
        Begin metrics collection session for a training run.
        
        Args:
            framework: Name of framework being benchmarked
            metadata: Optional metadata about the training run
            
        Returns:
            CollectionSession with session details
            
        Requirements: 4.6 (timestamp synchronization)
        """
        if self.current_session is not None:
            logger.warning(
                f"Starting new collection session while session "
                f"{self.current_session.session_id} is active. "
                f"Previous session will be discarded."
            )
            # Just discard the previous session without finalizing
            # (it may not have any metrics yet)
            self.current_session = None
        
        session_id = f"{framework}_{int(time.time())}"
        start_time = time.time()
        
        session = CollectionSession(
            framework_name=framework,
            session_id=session_id,
            start_time=start_time,
            metadata=metadata or {},
        )
        
        self.current_session = session
        
        logger.info(
            f"Started metrics collection session: {session_id} "
            f"for framework: {framework}"
        )
        
        return session
    
    def record_epoch_metrics(
        self,
        epoch: int,
        train_loss: float,
        train_accuracy: float,
        val_loss: float,
        val_accuracy: float,
        learning_rate: float,
        epoch_duration_seconds: float,
        samples_per_second: float,
        train_auc: Optional[float] = None,
        val_auc: Optional[float] = None,
        train_f1: Optional[float] = None,
        val_f1: Optional[float] = None,
    ) -> None:
        """
        Record per-epoch training metrics.
        
        Args:
            epoch: Epoch number
            train_loss: Training loss for this epoch
            train_accuracy: Training accuracy for this epoch
            val_loss: Validation loss for this epoch
            val_accuracy: Validation accuracy for this epoch
            learning_rate: Learning rate for this epoch
            epoch_duration_seconds: Time taken for this epoch
            samples_per_second: Training throughput
            train_auc: Optional training AUC
            val_auc: Optional validation AUC
            train_f1: Optional training F1 score
            val_f1: Optional validation F1 score
            
        Requirements: 4.1 (per-epoch metrics), 4.3 (timing), 4.4 (efficiency),
                     4.6 (timestamp synchronization), 4.10 (validation)
        """
        if self.current_session is None:
            raise RuntimeError(
                "No active collection session. Call start_collection() first."
            )
        
        # Validate metrics (Requirement 4.10)
        self._validate_metrics(
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
            learning_rate=learning_rate,
            epoch_duration_seconds=epoch_duration_seconds,
            samples_per_second=samples_per_second,
            train_auc=train_auc,
            val_auc=val_auc,
            train_f1=train_f1,
            val_f1=val_f1,
        )
        
        # Create epoch metrics with synchronized timestamp (Requirement 4.6)
        epoch_metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
            learning_rate=learning_rate,
            epoch_duration_seconds=epoch_duration_seconds,
            samples_per_second=samples_per_second,
            timestamp=time.time(),
            train_auc=train_auc,
            val_auc=val_auc,
            train_f1=train_f1,
            val_f1=val_f1,
        )
        
        self.current_session.epoch_metrics.append(epoch_metrics)
        
        logger.debug(
            f"Recorded epoch {epoch} metrics: "
            f"train_loss={train_loss:.4f}, train_acc={train_accuracy:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_accuracy:.4f}, "
            f"lr={learning_rate:.6f}, duration={epoch_duration_seconds:.2f}s"
        )
    
    def record_system_metrics(self) -> SystemMetrics:
        """
        Capture GPU memory, temperature, and utilization.
        
        Returns:
            SystemMetrics with current resource usage
            
        Requirements: 4.2 (system metrics), 4.6 (timestamp synchronization)
        """
        if self.current_session is None:
            raise RuntimeError(
                "No active collection session. Call start_collection() first."
            )
        
        # Get resource metrics from ResourceManager
        resource_metrics = self.resource_manager.monitor_resources()
        
        # Convert to SystemMetrics with synchronized timestamp (Requirement 4.6)
        system_metrics = SystemMetrics(
            gpu_memory_used_mb=resource_metrics.gpu_memory_used_mb,
            gpu_memory_total_mb=resource_metrics.gpu_memory_total_mb,
            gpu_memory_percent=resource_metrics.gpu_memory_percent,
            gpu_temperature=resource_metrics.gpu_temperature,
            gpu_utilization=resource_metrics.gpu_utilization,
            timestamp=time.time(),
        )
        
        self.current_session.system_metrics.append(system_metrics)
        
        logger.debug(
            f"Recorded system metrics: "
            f"GPU memory={system_metrics.gpu_memory_used_mb:.0f}MB/"
            f"{system_metrics.gpu_memory_total_mb:.0f}MB "
            f"({system_metrics.gpu_memory_percent:.1f}%), "
            f"temp={system_metrics.gpu_temperature}°C, "
            f"util={system_metrics.gpu_utilization}%"
        )
        
        return system_metrics
    
    def finalize_collection(self, output_path: Optional[Path] = None) -> AggregatedMetrics:
        """
        Aggregate and save metrics to JSON.
        
        Args:
            output_path: Optional path to save metrics JSON file
            
        Returns:
            AggregatedMetrics with aggregated statistics
            
        Requirements: 4.5 (statistical metrics), 4.7 (JSON serialization),
                     4.8 (metrics aggregation), 4.9 (confidence intervals)
        """
        if self.current_session is None:
            raise RuntimeError(
                "No active collection session. Call start_collection() first."
            )
        
        logger.info(
            f"Finalizing metrics collection for session: "
            f"{self.current_session.session_id}"
        )
        
        # Compute aggregated metrics (Requirement 4.8)
        aggregated = self._compute_aggregated_metrics()
        
        # Prepare data for JSON serialization (Requirement 4.7)
        metrics_data = {
            "session_id": self.current_session.session_id,
            "framework_name": self.current_session.framework_name,
            "start_time": self.current_session.start_time,
            "end_time": time.time(),
            "metadata": self.current_session.metadata,
            "epoch_metrics": [asdict(em) for em in self.current_session.epoch_metrics],
            "system_metrics": [asdict(sm) for sm in self.current_session.system_metrics],
            "aggregated_metrics": asdict(aggregated),
        }
        
        # Save to JSON if path provided (Requirement 4.7)
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, "w") as f:
                json.dump(metrics_data, f, indent=2)
            
            logger.info(f"Saved metrics to: {output_path}")
        
        # Clear current session
        self.current_session = None
        
        return aggregated
    
    def compute_confidence_intervals(
        self,
        values: List[float],
        confidence_level: Optional[float] = None,
        n_bootstrap: Optional[int] = None,
    ) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence intervals for metrics.
        
        Uses bootstrap resampling to estimate confidence intervals for
        a given metric across epochs.
        
        Args:
            values: List of metric values (e.g., per-epoch accuracies)
            confidence_level: Confidence level (default: self.confidence_level)
            n_bootstrap: Number of bootstrap samples (default: self.bootstrap_samples)
            
        Returns:
            Tuple of (lower_bound, upper_bound) for confidence interval
            
        Requirements: 4.9 (bootstrap confidence intervals)
        """
        if len(values) == 0:
            raise ValueError("Cannot compute confidence interval for empty values")
        
        if len(values) == 1:
            # Single value - return it as both bounds
            return (values[0], values[0])
        
        confidence_level = confidence_level or self.confidence_level
        n_bootstrap = n_bootstrap or self.bootstrap_samples
        
        # Convert to numpy array
        values_array = np.array(values)
        
        # Bootstrap resampling
        bootstrap_means = []
        rng = np.random.RandomState(42)  # Fixed seed for reproducibility
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            sample = rng.choice(values_array, size=len(values_array), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        # Compute percentiles for confidence interval
        alpha = 1.0 - confidence_level
        lower_percentile = (alpha / 2.0) * 100
        upper_percentile = (1.0 - alpha / 2.0) * 100
        
        lower_bound = np.percentile(bootstrap_means, lower_percentile)
        upper_bound = np.percentile(bootstrap_means, upper_percentile)
        
        return (float(lower_bound), float(upper_bound))
    
    def _compute_aggregated_metrics(self) -> AggregatedMetrics:
        """
        Compute aggregated statistics across all epochs.
        
        Returns:
            AggregatedMetrics with mean, std, and confidence intervals
            
        Requirements: 4.5 (statistical metrics), 4.8 (aggregation),
                     4.9 (confidence intervals)
        """
        if not self.current_session.epoch_metrics:
            raise ValueError("No epoch metrics to aggregate")
        
        # Extract per-epoch values
        train_losses = [em.train_loss for em in self.current_session.epoch_metrics]
        val_losses = [em.val_loss for em in self.current_session.epoch_metrics]
        train_accuracies = [em.train_accuracy for em in self.current_session.epoch_metrics]
        val_accuracies = [em.val_accuracy for em in self.current_session.epoch_metrics]
        epoch_durations = [em.epoch_duration_seconds for em in self.current_session.epoch_metrics]
        samples_per_second = [em.samples_per_second for em in self.current_session.epoch_metrics]
        
        # Extract system metrics
        gpu_memory_used = [sm.gpu_memory_used_mb for sm in self.current_session.system_metrics]
        gpu_temperatures = [sm.gpu_temperature for sm in self.current_session.system_metrics]
        gpu_utilizations = [sm.gpu_utilization for sm in self.current_session.system_metrics]
        
        # Compute timing metrics (Requirement 4.3)
        total_training_time = sum(epoch_durations)
        
        # Compute confidence intervals (Requirement 4.9)
        train_accuracy_ci = self.compute_confidence_intervals(train_accuracies)
        val_accuracy_ci = self.compute_confidence_intervals(val_accuracies)
        train_loss_ci = self.compute_confidence_intervals(train_losses)
        val_loss_ci = self.compute_confidence_intervals(val_losses)
        
        # Create aggregated metrics (Requirements 4.5, 4.8)
        aggregated = AggregatedMetrics(
            # Training metrics
            mean_train_loss=float(np.mean(train_losses)),
            std_train_loss=float(np.std(train_losses)),
            mean_val_loss=float(np.mean(val_losses)),
            std_val_loss=float(np.std(val_losses)),
            mean_train_accuracy=float(np.mean(train_accuracies)),
            std_train_accuracy=float(np.std(train_accuracies)),
            mean_val_accuracy=float(np.mean(val_accuracies)),
            std_val_accuracy=float(np.std(val_accuracies)),
            # Timing metrics
            total_training_time_seconds=total_training_time,
            mean_epoch_duration_seconds=float(np.mean(epoch_durations)),
            std_epoch_duration_seconds=float(np.std(epoch_durations)),
            # Efficiency metrics
            mean_samples_per_second=float(np.mean(samples_per_second)),
            std_samples_per_second=float(np.std(samples_per_second)),
            mean_gpu_utilization=float(np.mean(gpu_utilizations)) if gpu_utilizations else 0.0,
            std_gpu_utilization=float(np.std(gpu_utilizations)) if gpu_utilizations else 0.0,
            # Resource metrics
            peak_gpu_memory_mb=float(np.max(gpu_memory_used)) if gpu_memory_used else 0.0,
            mean_gpu_memory_mb=float(np.mean(gpu_memory_used)) if gpu_memory_used else 0.0,
            peak_gpu_temperature=float(np.max(gpu_temperatures)) if gpu_temperatures else 0.0,
            mean_gpu_temperature=float(np.mean(gpu_temperatures)) if gpu_temperatures else 0.0,
            # Confidence intervals
            train_accuracy_ci=train_accuracy_ci,
            val_accuracy_ci=val_accuracy_ci,
            train_loss_ci=train_loss_ci,
            val_loss_ci=val_loss_ci,
        )
        
        logger.info(
            f"Aggregated metrics computed: "
            f"mean_val_acc={aggregated.mean_val_accuracy:.4f} "
            f"(CI: {aggregated.val_accuracy_ci[0]:.4f}-{aggregated.val_accuracy_ci[1]:.4f}), "
            f"total_time={aggregated.total_training_time_seconds:.2f}s, "
            f"peak_memory={aggregated.peak_gpu_memory_mb:.0f}MB"
        )
        
        return aggregated
    
    def _validate_metrics(
        self,
        train_loss: float,
        train_accuracy: float,
        val_loss: float,
        val_accuracy: float,
        learning_rate: float,
        epoch_duration_seconds: float,
        samples_per_second: float,
        train_auc: Optional[float] = None,
        val_auc: Optional[float] = None,
        train_f1: Optional[float] = None,
        val_f1: Optional[float] = None,
    ) -> None:
        """
        Validate metrics for sanity and correctness.
        
        Raises:
            ValueError: If any metric is invalid
            
        Requirements: 4.10 (metrics validation)
        """
        # Check for NaN or Inf
        metrics_to_check = {
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "learning_rate": learning_rate,
            "epoch_duration_seconds": epoch_duration_seconds,
            "samples_per_second": samples_per_second,
        }
        
        if train_auc is not None:
            metrics_to_check["train_auc"] = train_auc
        if val_auc is not None:
            metrics_to_check["val_auc"] = val_auc
        if train_f1 is not None:
            metrics_to_check["train_f1"] = train_f1
        if val_f1 is not None:
            metrics_to_check["val_f1"] = val_f1
        
        for name, value in metrics_to_check.items():
            if not np.isfinite(value):
                raise ValueError(
                    f"Invalid metric value: {name}={value} (must be finite)"
                )
        
        # Check accuracy ranges [0, 1]
        accuracy_metrics = {
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
        }
        
        if train_auc is not None:
            accuracy_metrics["train_auc"] = train_auc
        if val_auc is not None:
            accuracy_metrics["val_auc"] = val_auc
        if train_f1 is not None:
            accuracy_metrics["train_f1"] = train_f1
        if val_f1 is not None:
            accuracy_metrics["val_f1"] = val_f1
        
        for name, value in accuracy_metrics.items():
            if not (0.0 <= value <= 1.0):
                raise ValueError(
                    f"Invalid {name}: {value} (must be in [0.0, 1.0])"
                )
        
        # Check loss is non-negative
        if train_loss < 0.0:
            raise ValueError(f"Invalid train_loss: {train_loss} (must be >= 0)")
        if val_loss < 0.0:
            raise ValueError(f"Invalid val_loss: {val_loss} (must be >= 0)")
        
        # Check learning rate is positive
        if learning_rate <= 0.0:
            raise ValueError(
                f"Invalid learning_rate: {learning_rate} (must be > 0)"
            )
        
        # Check timing metrics are positive
        if epoch_duration_seconds <= 0.0:
            raise ValueError(
                f"Invalid epoch_duration_seconds: {epoch_duration_seconds} (must be > 0)"
            )
        if samples_per_second <= 0.0:
            raise ValueError(
                f"Invalid samples_per_second: {samples_per_second} (must be > 0)"
            )
