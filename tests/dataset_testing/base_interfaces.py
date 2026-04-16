"""
Base interfaces and utilities for dataset testing.

This module provides abstract base classes and interfaces for
synthetic data generation, performance benchmarking, and error simulation.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, ContextManager
from pathlib import Path
import time
import psutil
from contextlib import contextmanager


class DatasetGenerator(ABC):
    """Abstract base class for synthetic dataset generators."""

    @abstractmethod
    def generate_samples(self, num_samples: int, **kwargs) -> Dict[str, Any]:
        """Generate synthetic samples matching real data statistics.

        Args:
            num_samples: Number of samples to generate
            **kwargs: Generator-specific parameters

        Returns:
            Dictionary containing generated samples and metadata
        """
        pass

    @abstractmethod
    def corrupt_samples(self, samples: Dict[str, Any], corruption_type: str) -> Dict[str, Any]:
        """Introduce controlled corruption for testing error handling.

        Args:
            samples: Original samples to corrupt
            corruption_type: Type of corruption to introduce

        Returns:
            Dictionary containing corrupted samples
        """
        pass

    @abstractmethod
    def validate_samples(self, samples: Dict[str, Any]) -> bool:
        """Validate that generated samples meet expected criteria.

        Args:
            samples: Samples to validate

        Returns:
            True if samples are valid, False otherwise
        """
        pass


class PerformanceBenchmark:
    """Performance benchmarking utilities for dataset operations."""

    def __init__(self, baseline_metrics: Dict[str, float]):
        """Initialize with baseline performance metrics.

        Args:
            baseline_metrics: Dictionary of baseline performance values
        """
        self.baselines = baseline_metrics
        self.results = {}
        self.process = psutil.Process()

    def benchmark_loading(self, dataset_loader_func, *args, **kwargs) -> Dict[str, float]:
        """Benchmark dataset loading performance.

        Args:
            dataset_loader_func: Function to benchmark
            *args, **kwargs: Arguments for the function

        Returns:
            Dictionary with performance metrics
        """
        # Record initial memory
        initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB

        # Benchmark loading time
        start_time = time.time()
        result = dataset_loader_func(*args, **kwargs)
        end_time = time.time()

        # Record peak memory
        peak_memory = self.process.memory_info().rss / 1024 / 1024  # MB

        metrics = {
            "loading_time_seconds": end_time - start_time,
            "memory_usage_mb": peak_memory - initial_memory,
            "peak_memory_mb": peak_memory,
        }

        # Calculate throughput if num_samples is available
        if hasattr(result, "__len__"):
            metrics["throughput_samples_per_second"] = len(result) / metrics["loading_time_seconds"]

        return metrics

    def benchmark_memory_usage(self, operation_func, *args, **kwargs) -> Dict[str, float]:
        """Monitor memory usage during dataset operations.

        Args:
            operation_func: Function to monitor
            *args, **kwargs: Arguments for the function

        Returns:
            Dictionary with memory usage metrics
        """
        initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB

        # Monitor memory during operation
        memory_samples = []

        def memory_monitor():
            while True:
                try:
                    current_memory = self.process.memory_info().rss / 1024 / 1024
                    memory_samples.append(current_memory)
                    time.sleep(0.1)  # Sample every 100ms
                except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                    break

        import threading

        monitor_thread = threading.Thread(target=memory_monitor, daemon=True)
        monitor_thread.start()

        try:
            operation_func(*args, **kwargs)
        finally:
            monitor_thread = None  # Stop monitoring

        final_memory = self.process.memory_info().rss / 1024 / 1024  # MB

        return {
            "initial_memory_mb": initial_memory,
            "final_memory_mb": final_memory,
            "peak_memory_mb": max(memory_samples) if memory_samples else final_memory,
            "memory_delta_mb": final_memory - initial_memory,
            "memory_samples": memory_samples,
        }

    def check_regression(self, current_metrics: Dict[str, float]) -> List[str]:
        """Check for performance regressions against baselines.

        Args:
            current_metrics: Current performance metrics

        Returns:
            List of regression warnings
        """
        regressions = []

        for metric_name, current_value in current_metrics.items():
            if metric_name in self.baselines:
                baseline_value = self.baselines[metric_name]

                # Check for significant regression (>20% worse)
                if metric_name.endswith("_time_seconds") or metric_name.endswith("_memory_mb"):
                    # Lower is better for time and memory
                    if current_value > baseline_value * 1.2:
                        regressions.append(
                            f"{metric_name}: {current_value:.2f} vs baseline {baseline_value:.2f} "
                            f"({((current_value / baseline_value - 1) * 100):.1f}% worse)"
                        )
                elif metric_name.endswith("_per_second"):
                    # Higher is better for throughput
                    if current_value < baseline_value * 0.8:
                        regressions.append(
                            f"{metric_name}: {current_value:.2f} vs baseline {baseline_value:.2f} "
                            f"({((1 - current_value / baseline_value) * 100):.1f}% worse)"
                        )

        return regressions


class ErrorSimulator:
    """Simulate various error conditions for robustness testing."""

    def __init__(self, temp_dir: Path):
        """Initialize error simulator.

        Args:
            temp_dir: Temporary directory for creating corrupted files
        """
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def corrupt_file(self, file_path: Path, corruption_type: str) -> Path:
        """Create corrupted version of a file.

        Args:
            file_path: Original file to corrupt
            corruption_type: Type of corruption to apply

        Returns:
            Path to corrupted file
        """
        corrupted_path = self.temp_dir / f"corrupted_{file_path.name}"

        if corruption_type == "file_truncation":
            # Truncate file to 50% of original size
            with open(file_path, "rb") as src, open(corrupted_path, "wb") as dst:
                data = src.read()
                dst.write(data[: len(data) // 2])

        elif corruption_type == "random_bytes":
            # Replace random bytes with garbage
            with open(file_path, "rb") as src:
                data = bytearray(src.read())

            import random

            for _ in range(len(data) // 100):  # Corrupt 1% of bytes
                idx = random.randint(0, len(data) - 1)
                data[idx] = random.randint(0, 255)

            with open(corrupted_path, "wb") as dst:
                dst.write(data)

        elif corruption_type == "header_corruption":
            # Corrupt first 100 bytes
            with open(file_path, "rb") as src:
                data = bytearray(src.read())

            for i in range(min(100, len(data))):
                data[i] = 0xFF

            with open(corrupted_path, "wb") as dst:
                dst.write(data)

        else:
            # Default: copy original file
            import shutil

            shutil.copy2(file_path, corrupted_path)

        return corrupted_path

    def simulate_network_failure(self, download_function):
        """Simulate network failures during downloads.

        Args:
            download_function: Function that performs network operations

        Returns:
            Context manager that simulates network failures
        """

        @contextmanager
        def network_failure_context():
            # Mock network failure by raising ConnectionError
            def failing_function(*args, **kwargs):
                raise ConnectionError("Simulated network failure")

            try:
                yield failing_function
            finally:
                pass  # Restore original function if needed

        return network_failure_context()

    @contextmanager
    def limit_memory(self, memory_limit_mb: int) -> ContextManager:
        """Context manager to limit available memory.

        Args:
            memory_limit_mb: Memory limit in megabytes

        Yields:
            Context with memory limit applied
        """
        try:
            import resource

            # Set memory limit (Linux/macOS only)
            old_limit = resource.getrlimit(resource.RLIMIT_AS)
            new_limit = (memory_limit_mb * 1024 * 1024, old_limit[1])
            resource.setrlimit(resource.RLIMIT_AS, new_limit)
            yield
        except (ImportError, OSError):
            # Windows or other systems - just yield without limit
            yield
        finally:
            try:
                resource.setrlimit(resource.RLIMIT_AS, old_limit)
            except (OSError, ValueError):
                pass


class TestResult:
    """Standardized test result format."""

    def __init__(
        self,
        test_name: str,
        test_type: str,
        status: str,
        execution_time_seconds: float,
        memory_usage_mb: Optional[float] = None,
        error_message: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        artifacts: Optional[List[Path]] = None,
    ):
        """Initialize test result.

        Args:
            test_name: Name of the test
            test_type: Type of test ('unit', 'integration', 'performance', 'property')
            status: Test status ('passed', 'failed', 'skipped')
            execution_time_seconds: Time taken to execute test
            memory_usage_mb: Memory usage during test
            error_message: Error message if test failed
            metrics: Additional test metrics
            artifacts: Generated files, plots, etc.
        """
        self.test_name = test_name
        self.test_type = test_type
        self.status = status
        self.execution_time_seconds = execution_time_seconds
        self.memory_usage_mb = memory_usage_mb
        self.error_message = error_message
        self.metrics = metrics or {}
        self.artifacts = artifacts or []

    def to_dict(self) -> Dict[str, Any]:
        """Convert test result to dictionary."""
        return {
            "test_name": self.test_name,
            "test_type": self.test_type,
            "status": self.status,
            "execution_time_seconds": self.execution_time_seconds,
            "memory_usage_mb": self.memory_usage_mb,
            "error_message": self.error_message,
            "metrics": self.metrics,
            "artifacts": [str(p) for p in self.artifacts],
        }
