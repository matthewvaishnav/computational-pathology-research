"""
Automated Performance Regression Testing for HistoCore Real-Time WSI Streaming.

Tracks performance baselines and detects regressions in processing time,
memory usage, and throughput.

Task 8.2.3: Automated performance regression testing
"""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.streaming.attention_aggregator import StreamingAttentionAggregator
from src.streaming.gpu_pipeline import GPUPipeline
from src.streaming.model_optimizer import ModelOptimizer, OptimizationConfig

logger = logging.getLogger(__name__)


# ============================================================================
# Performance Baseline Management
# ============================================================================


@dataclass
class PerformanceBaseline:
    """Performance baseline metrics."""

    test_name: str
    timestamp: str
    git_commit: Optional[str] = None

    # Processing metrics
    processing_time_ms: float = 0.0
    throughput_patches_per_sec: float = 0.0

    # Memory metrics
    peak_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0

    # GPU metrics
    gpu_utilization_pct: float = 0.0
    gpu_memory_mb: float = 0.0

    # Quality metrics
    accuracy: Optional[float] = None
    confidence: Optional[float] = None

    # System info
    gpu_name: str = "unknown"
    cuda_version: str = "unknown"
    pytorch_version: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerformanceBaseline":
        """Create from dictionary."""
        return cls(**data)


class BaselineManager:
    """Manager for performance baselines."""

    def __init__(self, baseline_dir: str = "./tests/baselines"):
        """Initialize baseline manager."""
        self.baseline_dir = Path(baseline_dir)
        self.baseline_dir.mkdir(parents=True, exist_ok=True)

        self.baseline_file = self.baseline_dir / "performance_baselines.json"
        self.baselines: Dict[str, List[PerformanceBaseline]] = {}

        self._load_baselines()

        logger.info(f"Baseline manager initialized: {len(self.baselines)} test baselines loaded")

    def _load_baselines(self):
        """Load baselines from disk."""
        if self.baseline_file.exists():
            try:
                with open(self.baseline_file, "r") as f:
                    data = json.load(f)

                for test_name, baseline_list in data.items():
                    self.baselines[test_name] = [
                        PerformanceBaseline.from_dict(b) for b in baseline_list
                    ]

                logger.info(f"Loaded baselines for {len(self.baselines)} tests")

            except Exception as e:
                logger.error(f"Failed to load baselines: {e}")
                self.baselines = {}
        else:
            logger.info("No existing baselines found")

    def _save_baselines(self):
        """Save baselines to disk."""
        try:
            data = {
                test_name: [b.to_dict() for b in baseline_list]
                for test_name, baseline_list in self.baselines.items()
            }

            with open(self.baseline_file, "w") as f:
                json.dump(data, f, indent=2)

            logger.debug("Baselines saved to disk")

        except Exception as e:
            logger.error(f"Failed to save baselines: {e}")

    def add_baseline(self, baseline: PerformanceBaseline):
        """Add a new baseline measurement."""
        test_name = baseline.test_name

        if test_name not in self.baselines:
            self.baselines[test_name] = []

        self.baselines[test_name].append(baseline)

        # Keep only last 100 baselines per test
        if len(self.baselines[test_name]) > 100:
            self.baselines[test_name] = self.baselines[test_name][-100:]

        self._save_baselines()

        logger.info(f"Added baseline for {test_name}")

    def get_baseline(self, test_name: str, percentile: int = 50) -> Optional[PerformanceBaseline]:
        """
        Get baseline for a test.

        Args:
            test_name: Name of the test
            percentile: Percentile to use (50 = median, 95 = p95)

        Returns:
            Baseline at specified percentile, or None if no baselines exist
        """
        if test_name not in self.baselines or not self.baselines[test_name]:
            return None

        baselines = self.baselines[test_name]

        # Calculate percentile for each metric
        processing_times = [b.processing_time_ms for b in baselines]
        throughputs = [b.throughput_patches_per_sec for b in baselines]
        peak_memories = [b.peak_memory_mb for b in baselines]

        baseline = PerformanceBaseline(
            test_name=test_name,
            timestamp=datetime.now().isoformat(),
            processing_time_ms=np.percentile(processing_times, percentile),
            throughput_patches_per_sec=np.percentile(throughputs, percentile),
            peak_memory_mb=np.percentile(peak_memories, percentile),
        )

        return baseline

    def compare_to_baseline(
        self, test_name: str, current: PerformanceBaseline, threshold_pct: float = 10.0
    ) -> Dict[str, Any]:
        """
        Compare current performance to baseline.

        Args:
            test_name: Name of the test
            current: Current performance measurement
            threshold_pct: Regression threshold percentage

        Returns:
            Comparison results with regression flags
        """
        baseline = self.get_baseline(test_name, percentile=50)

        if baseline is None:
            return {
                "has_baseline": False,
                "regression_detected": False,
                "message": "No baseline available for comparison",
            }

        # Calculate differences
        time_diff_pct = (
            (current.processing_time_ms - baseline.processing_time_ms)
            / baseline.processing_time_ms
            * 100
        )

        throughput_diff_pct = (
            (current.throughput_patches_per_sec - baseline.throughput_patches_per_sec)
            / baseline.throughput_patches_per_sec
            * 100
        )

        memory_diff_pct = (
            (current.peak_memory_mb - baseline.peak_memory_mb) / baseline.peak_memory_mb * 100
        )

        # Detect regressions
        regressions = []

        if time_diff_pct > threshold_pct:
            regressions.append(f"Processing time increased by {time_diff_pct:.1f}%")

        if throughput_diff_pct < -threshold_pct:
            regressions.append(f"Throughput decreased by {abs(throughput_diff_pct):.1f}%")

        if memory_diff_pct > threshold_pct:
            regressions.append(f"Memory usage increased by {memory_diff_pct:.1f}%")

        # Detect improvements
        improvements = []

        if time_diff_pct < -threshold_pct:
            improvements.append(f"Processing time decreased by {abs(time_diff_pct):.1f}%")

        if throughput_diff_pct > threshold_pct:
            improvements.append(f"Throughput increased by {throughput_diff_pct:.1f}%")

        if memory_diff_pct < -threshold_pct:
            improvements.append(f"Memory usage decreased by {abs(memory_diff_pct):.1f}%")

        return {
            "has_baseline": True,
            "regression_detected": len(regressions) > 0,
            "regressions": regressions,
            "improvements": improvements,
            "baseline": baseline.to_dict(),
            "current": current.to_dict(),
            "differences": {
                "processing_time_pct": time_diff_pct,
                "throughput_pct": throughput_diff_pct,
                "memory_pct": memory_diff_pct,
            },
        }


# ============================================================================
# Performance Test Fixtures
# ============================================================================


@pytest.fixture
def baseline_manager():
    """Create baseline manager."""
    return BaselineManager()


@pytest.fixture
def mock_cnn_model():
    """Create mock CNN model."""

    class MockCNN(nn.Module):
        def __init__(self, feature_dim=256):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(64, feature_dim)

        def forward(self, x):
            x = self.conv(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    return MockCNN()


@pytest.fixture
def mock_attention_model():
    """Create mock attention model."""

    class MockAttentionMIL(nn.Module):
        def __init__(self, feature_dim=256, num_classes=2):
            super().__init__()
            self.attention = nn.Sequential(
                nn.Linear(feature_dim, 128), nn.Tanh(), nn.Linear(128, 1)
            )
            self.classifier = nn.Sequential(
                nn.Linear(feature_dim, 128), nn.ReLU(), nn.Linear(128, num_classes)
            )

        def forward(self, features):
            attention_weights = self.attention(features)
            attention_weights = torch.softmax(attention_weights, dim=1)
            aggregated = torch.sum(features * attention_weights, dim=1)
            logits = self.classifier(aggregated)
            return logits, attention_weights.squeeze(-1)

    return MockAttentionMIL()


@pytest.fixture
def gpu_available():
    """Check if GPU is available."""
    return torch.cuda.is_available()


# ============================================================================
# Performance Regression Tests
# ============================================================================


class TestFeatureExtractionPerformance:
    """Test feature extraction performance regression."""

    def test_feature_extraction_baseline(self, mock_cnn_model, baseline_manager, gpu_available):
        """Test feature extraction and track baseline."""
        if not gpu_available:
            pytest.skip("GPU not available")

        test_name = "feature_extraction_1000_patches"

        device = torch.device("cuda:0")
        mock_cnn_model = mock_cnn_model.to(device)

        gpu_pipeline = GPUPipeline(
            model=mock_cnn_model, batch_size=64, gpu_ids=[0], enable_fp16=True
        )

        # Warmup
        for _ in range(10):
            patches = torch.randn(64, 3, 224, 224)
            _ = gpu_pipeline._process_batch_sync(patches)

        # Benchmark
        num_patches = 1000
        batch_size = 64
        num_batches = (num_patches + batch_size - 1) // batch_size

        torch.cuda.reset_peak_memory_stats(device)

        start_time = time.time()

        for _ in range(num_batches):
            patches = torch.randn(batch_size, 3, 224, 224)
            _ = gpu_pipeline._process_batch_sync(patches)

        torch.cuda.synchronize()
        processing_time = time.time() - start_time

        # Get metrics
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        throughput = num_patches / processing_time

        gpu_pipeline.cleanup()

        # Create baseline
        baseline = PerformanceBaseline(
            test_name=test_name,
            timestamp=datetime.now().isoformat(),
            processing_time_ms=processing_time * 1000,
            throughput_patches_per_sec=throughput,
            peak_memory_mb=peak_memory,
            gpu_name=torch.cuda.get_device_name(device),
            cuda_version=torch.version.cuda,
            pytorch_version=torch.__version__,
        )

        # Compare to baseline
        comparison = baseline_manager.compare_to_baseline(test_name, baseline, threshold_pct=10.0)

        # Add to baseline history
        baseline_manager.add_baseline(baseline)

        # Log results
        logger.info(f"Feature extraction performance:")
        logger.info(f"  Processing time: {processing_time*1000:.2f}ms")
        logger.info(f"  Throughput: {throughput:.1f} patches/sec")
        logger.info(f"  Peak memory: {peak_memory:.1f}MB")

        if comparison["has_baseline"]:
            logger.info(f"  Baseline comparison:")
            if comparison["regressions"]:
                logger.warning(f"    Regressions: {comparison['regressions']}")
            if comparison["improvements"]:
                logger.info(f"    Improvements: {comparison['improvements']}")

        # Assert no regressions
        if comparison["regression_detected"]:
            pytest.fail(f"Performance regression detected: {comparison['regressions']}")


class TestEndToEndPerformance:
    """Test end-to-end pipeline performance regression."""

    def test_e2e_processing_baseline(
        self, mock_cnn_model, mock_attention_model, baseline_manager, gpu_available
    ):
        """Test end-to-end processing and track baseline."""
        if not gpu_available:
            pytest.skip("GPU not available")

        test_name = "e2e_processing_5000_patches"

        device = torch.device("cuda:0")
        mock_cnn_model = mock_cnn_model.to(device)
        mock_attention_model = mock_attention_model.to(device)

        # Initialize pipeline
        gpu_pipeline = GPUPipeline(
            model=mock_cnn_model, batch_size=64, gpu_ids=[0], enable_fp16=True
        )

        aggregator = StreamingAttentionAggregator(
            attention_model=mock_attention_model, feature_dim=256, num_classes=2
        )

        # Warmup
        for _ in range(10):
            patches = torch.randn(64, 3, 224, 224)
            features = gpu_pipeline._process_batch_sync(patches)
            aggregator.update(features.to(device))

        # Reset aggregator
        aggregator = StreamingAttentionAggregator(
            attention_model=mock_attention_model, feature_dim=256, num_classes=2
        )

        # Benchmark
        num_patches = 5000
        batch_size = 64
        num_batches = (num_patches + batch_size - 1) // batch_size

        torch.cuda.reset_peak_memory_stats(device)

        start_time = time.time()

        for _ in range(num_batches):
            patches = torch.randn(batch_size, 3, 224, 224)
            features = gpu_pipeline._process_batch_sync(patches)
            aggregator.update(features.to(device))

        prediction, confidence = aggregator.get_prediction()

        torch.cuda.synchronize()
        processing_time = time.time() - start_time

        # Get metrics
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        throughput = num_patches / processing_time

        gpu_pipeline.cleanup()

        # Create baseline
        baseline = PerformanceBaseline(
            test_name=test_name,
            timestamp=datetime.now().isoformat(),
            processing_time_ms=processing_time * 1000,
            throughput_patches_per_sec=throughput,
            peak_memory_mb=peak_memory,
            confidence=confidence,
            gpu_name=torch.cuda.get_device_name(device),
            cuda_version=torch.version.cuda,
            pytorch_version=torch.__version__,
        )

        # Compare to baseline
        comparison = baseline_manager.compare_to_baseline(test_name, baseline, threshold_pct=10.0)

        # Add to baseline history
        baseline_manager.add_baseline(baseline)

        # Log results
        logger.info(f"End-to-end performance:")
        logger.info(f"  Processing time: {processing_time*1000:.2f}ms")
        logger.info(f"  Throughput: {throughput:.1f} patches/sec")
        logger.info(f"  Peak memory: {peak_memory:.1f}MB")
        logger.info(f"  Confidence: {confidence:.3f}")

        if comparison["has_baseline"]:
            logger.info(f"  Baseline comparison:")
            if comparison["regressions"]:
                logger.warning(f"    Regressions: {comparison['regressions']}")
            if comparison["improvements"]:
                logger.info(f"    Improvements: {comparison['improvements']}")

        # Assert no regressions
        if comparison["regression_detected"]:
            pytest.fail(f"Performance regression detected: {comparison['regressions']}")


class TestMemoryPerformance:
    """Test memory usage performance regression."""

    def test_memory_usage_baseline(self, mock_cnn_model, baseline_manager, gpu_available):
        """Test memory usage and track baseline."""
        if not gpu_available:
            pytest.skip("GPU not available")

        test_name = "memory_usage_sustained_load"

        device = torch.device("cuda:0")
        mock_cnn_model = mock_cnn_model.to(device)

        gpu_pipeline = GPUPipeline(
            model=mock_cnn_model, batch_size=64, gpu_ids=[0], enable_fp16=True
        )

        # Track memory over time
        memory_samples = []
        num_batches = 100

        torch.cuda.reset_peak_memory_stats(device)

        start_time = time.time()

        for i in range(num_batches):
            patches = torch.randn(64, 3, 224, 224)
            _ = gpu_pipeline._process_batch_sync(patches)

            if i % 10 == 0:
                memory_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
                memory_samples.append(memory_mb)

        processing_time = time.time() - start_time

        # Get metrics
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        avg_memory = np.mean(memory_samples)

        gpu_pipeline.cleanup()

        # Create baseline
        baseline = PerformanceBaseline(
            test_name=test_name,
            timestamp=datetime.now().isoformat(),
            processing_time_ms=processing_time * 1000,
            peak_memory_mb=peak_memory,
            avg_memory_mb=avg_memory,
            gpu_name=torch.cuda.get_device_name(device),
            cuda_version=torch.version.cuda,
            pytorch_version=torch.__version__,
        )

        # Compare to baseline
        comparison = baseline_manager.compare_to_baseline(test_name, baseline, threshold_pct=10.0)

        # Add to baseline history
        baseline_manager.add_baseline(baseline)

        # Log results
        logger.info(f"Memory usage performance:")
        logger.info(f"  Peak memory: {peak_memory:.1f}MB")
        logger.info(f"  Average memory: {avg_memory:.1f}MB")

        if comparison["has_baseline"]:
            logger.info(f"  Baseline comparison:")
            if comparison["regressions"]:
                logger.warning(f"    Regressions: {comparison['regressions']}")
            if comparison["improvements"]:
                logger.info(f"    Improvements: {comparison['improvements']}")

        # Assert no regressions
        if comparison["regression_detected"]:
            pytest.fail(f"Performance regression detected: {comparison['regressions']}")


# ============================================================================
# CI/CD Integration
# ============================================================================


def test_generate_performance_report(baseline_manager):
    """Generate performance report for CI/CD."""
    report = {"timestamp": datetime.now().isoformat(), "tests": {}}

    # Get all test baselines
    for test_name in baseline_manager.baselines.keys():
        baseline = baseline_manager.get_baseline(test_name, percentile=50)

        if baseline:
            report["tests"][test_name] = {
                "processing_time_ms": baseline.processing_time_ms,
                "throughput_patches_per_sec": baseline.throughput_patches_per_sec,
                "peak_memory_mb": baseline.peak_memory_mb,
            }

    # Save report
    report_path = Path("./tests/baselines/performance_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Performance report generated: {report_path}")

    assert report_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
