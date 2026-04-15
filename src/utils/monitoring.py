"""
Monitoring and Logging Utilities

This module provides comprehensive monitoring and logging utilities for:
- Structured logging with JSON support
- Performance metrics tracking
- Resource usage monitoring
- Training progress tracking
- Prometheus metrics export
- Custom metric aggregation

Usage:
    from src.utils.monitoring import get_logger, MetricsTracker

    logger = get_logger(__name__)
    logger.info("Training started", extra={"epoch": 1})

    tracker = MetricsTracker()
    tracker.log_metric("loss", 0.5, step=100)
"""

import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil
import torch

# ============================================================================
# Structured Logging
# ============================================================================


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields
        if hasattr(record, "extra"):
            log_data.update(record.extra)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def get_logger(
    name: str, level: int = logging.INFO, log_file: Optional[str] = None, json_format: bool = False
) -> logging.Logger:
    """
    Get configured logger.

    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path
        json_format: Whether to use JSON formatting

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ============================================================================
# Metrics Tracking
# ============================================================================


class MetricsTracker:
    """Track and aggregate metrics during training."""

    def __init__(self, log_dir: Optional[str] = None):
        """
        Initialize metrics tracker.

        Args:
            log_dir: Optional directory to save metrics
        """
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.steps: Dict[str, List[int]] = defaultdict(list)
        self.log_dir = Path(log_dir) if log_dir else None

        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """
        Log a metric value.

        Args:
            name: Metric name
            value: Metric value
            step: Optional step number
        """
        self.metrics[name].append(value)
        if step is not None:
            self.steps[name].append(step)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log multiple metrics.

        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number
        """
        for name, value in metrics.items():
            self.log_metric(name, value, step)

    def get_metric(self, name: str) -> List[float]:
        """Get all values for a metric."""
        return self.metrics.get(name, [])

    def get_latest(self, name: str) -> Optional[float]:
        """Get latest value for a metric."""
        values = self.metrics.get(name, [])
        return values[-1] if values else None

    def get_average(self, name: str, last_n: Optional[int] = None) -> Optional[float]:
        """
        Get average value for a metric.

        Args:
            name: Metric name
            last_n: Optional number of recent values to average

        Returns:
            Average value or None if no values
        """
        values = self.metrics.get(name, [])
        if not values:
            return None

        if last_n:
            values = values[-last_n:]

        return sum(values) / len(values)

    def reset(self, name: Optional[str] = None) -> None:
        """
        Reset metrics.

        Args:
            name: Optional metric name to reset (resets all if None)
        """
        if name:
            self.metrics[name].clear()
            self.steps[name].clear()
        else:
            self.metrics.clear()
            self.steps.clear()

    def save(self, filename: str = "metrics.json") -> None:
        """Save metrics to file."""
        if not self.log_dir:
            raise ValueError("log_dir not set")

        filepath = self.log_dir / filename

        data = {
            "metrics": {k: list(v) for k, v in self.metrics.items()},
            "steps": {k: list(v) for k, v in self.steps.items()},
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, filename: str = "metrics.json") -> None:
        """Load metrics from file."""
        if not self.log_dir:
            raise ValueError("log_dir not set")

        filepath = self.log_dir / filename

        with open(filepath) as f:
            data = json.load(f)

        self.metrics = defaultdict(list, data["metrics"])
        self.steps = defaultdict(list, data["steps"])


# ============================================================================
# Resource Monitoring
# ============================================================================


class ResourceMonitor:
    """Monitor system resource usage."""

    def __init__(self):
        """Initialize resource monitor."""
        self.process = psutil.Process()

    def get_cpu_usage(self) -> float:
        """Get CPU usage percentage."""
        return self.process.cpu_percent(interval=0.1)

    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage in MB."""
        mem_info = self.process.memory_info()
        return {
            "rss": mem_info.rss / 1024 / 1024,  # Resident Set Size
            "vms": mem_info.vms / 1024 / 1024,  # Virtual Memory Size
        }

    def get_gpu_usage(self) -> Optional[Dict[str, Any]]:
        """Get GPU usage if available."""
        if not torch.cuda.is_available():
            return None

        gpu_stats = {}
        for i in range(torch.cuda.device_count()):
            gpu_stats[f"gpu_{i}"] = {
                "memory_allocated": torch.cuda.memory_allocated(i) / 1024 / 1024,
                "memory_reserved": torch.cuda.memory_reserved(i) / 1024 / 1024,
                "utilization": (
                    torch.cuda.utilization(i) if hasattr(torch.cuda, "utilization") else None
                ),
            }

        return gpu_stats

    def get_all_stats(self) -> Dict[str, Any]:
        """Get all resource statistics."""
        stats = {
            "cpu_percent": self.get_cpu_usage(),
            "memory": self.get_memory_usage(),
            "timestamp": time.time(),
        }

        gpu_stats = self.get_gpu_usage()
        if gpu_stats:
            stats["gpu"] = gpu_stats

        return stats


# ============================================================================
# Training Progress Tracking
# ============================================================================


class ProgressTracker:
    """Track training progress with ETA estimation."""

    def __init__(self, total_steps: int):
        """
        Initialize progress tracker.

        Args:
            total_steps: Total number of steps
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        self.step_times: List[float] = []

    def update(self, step: Optional[int] = None) -> None:
        """
        Update progress.

        Args:
            step: Optional step number (increments by 1 if None)
        """
        current_time = time.time()

        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1

        # Track step time
        if len(self.step_times) > 0:
            step_time = current_time - self.last_update_time
            self.step_times.append(step_time)

            # Keep only last 100 step times
            if len(self.step_times) > 100:
                self.step_times.pop(0)

        self.last_update_time = current_time

    def get_progress(self) -> float:
        """Get progress percentage."""
        return (self.current_step / self.total_steps) * 100

    def get_eta(self) -> Optional[float]:
        """Get estimated time to completion in seconds."""
        if not self.step_times:
            return None

        avg_step_time = sum(self.step_times) / len(self.step_times)
        remaining_steps = self.total_steps - self.current_step

        return avg_step_time * remaining_steps

    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time

    def get_stats(self) -> Dict[str, Any]:
        """Get all progress statistics."""
        eta = self.get_eta()

        return {
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "progress_percent": self.get_progress(),
            "elapsed_time": self.get_elapsed_time(),
            "eta": eta,
            "eta_formatted": self._format_time(eta) if eta else None,
        }

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format time in human-readable format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"


# ============================================================================
# Prometheus Metrics (Optional)
# ============================================================================

try:
    from prometheus_client import Counter, Gauge

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


class PrometheusMetrics:
    """Prometheus metrics exporter."""

    def __init__(self):
        """Initialize Prometheus metrics."""
        if not PROMETHEUS_AVAILABLE:
            raise ImportError("prometheus_client not installed")

        # Training metrics
        self.train_loss = Gauge("train_loss", "Training loss")
        self.val_loss = Gauge("val_loss", "Validation loss")
        self.train_accuracy = Gauge("train_accuracy", "Training accuracy")
        self.val_accuracy = Gauge("val_accuracy", "Validation accuracy")

        # System metrics
        self.cpu_usage = Gauge("cpu_usage_percent", "CPU usage percentage")
        self.memory_usage = Gauge("memory_usage_mb", "Memory usage in MB")
        self.gpu_memory = Gauge("gpu_memory_mb", "GPU memory usage in MB", ["gpu_id"])

        # Training progress
        self.epoch = Gauge("current_epoch", "Current training epoch")
        self.batch = Gauge("current_batch", "Current training batch")

        # Counters
        self.total_batches = Counter("total_batches_processed", "Total batches processed")
        self.total_samples = Counter("total_samples_processed", "Total samples processed")

    def update_training_metrics(
        self,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        train_acc: Optional[float] = None,
        val_acc: Optional[float] = None,
    ) -> None:
        """Update training metrics."""
        if train_loss is not None:
            self.train_loss.set(train_loss)
        if val_loss is not None:
            self.val_loss.set(val_loss)
        if train_acc is not None:
            self.train_accuracy.set(train_acc)
        if val_acc is not None:
            self.val_accuracy.set(val_acc)

    def update_system_metrics(self, monitor: ResourceMonitor) -> None:
        """Update system metrics."""
        stats = monitor.get_all_stats()

        self.cpu_usage.set(stats["cpu_percent"])
        self.memory_usage.set(stats["memory"]["rss"])

        if "gpu" in stats:
            for gpu_id, gpu_stats in stats["gpu"].items():
                self.gpu_memory.labels(gpu_id=gpu_id).set(gpu_stats["memory_allocated"])


# ============================================================================
# Utility Functions
# ============================================================================


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> str:
    """
    Format metrics dictionary as string.

    Args:
        metrics: Dictionary of metrics
        precision: Number of decimal places

    Returns:
        Formatted string
    """
    return " | ".join([f"{k}: {v:.{precision}f}" for k, v in metrics.items()])


def log_system_info(logger: logging.Logger) -> None:
    """Log system information."""
    logger.info("System Information:")
    logger.info(f"  Python: {sys.version}")
    logger.info(f"  PyTorch: {torch.__version__}")
    logger.info(f"  CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        logger.info(f"  CUDA version: {torch.version.cuda}")
        logger.info(f"  GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    logger.info(f"  CPU count: {psutil.cpu_count()}")
    logger.info(f"  Total memory: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.2f} GB")
