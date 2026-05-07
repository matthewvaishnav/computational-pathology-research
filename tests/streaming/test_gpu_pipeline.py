"""Unit tests for GPUPipeline - Memory management and batch processing.

Tests cover:
- Async batch processing
- Memory optimization and OOM recovery
- Dynamic batch size adjustment
- Multi-GPU support
- FP16 precision
- Throughput metrics
- Resource cleanup
"""

import asyncio
import importlib.util

# Import directly from gpu_pipeline module file to avoid OpenSlide dependency
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

# Load gpu_pipeline module directly without going through __init__.py
gpu_pipeline_path = Path(__file__).parent.parent.parent / "src" / "streaming" / "gpu_pipeline.py"
spec = importlib.util.spec_from_file_location("gpu_pipeline", gpu_pipeline_path)
gpu_pipeline_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpu_pipeline_module)

# Import classes from loaded module
GPUPipeline = gpu_pipeline_module.GPUPipeline
GPUMemoryManager = gpu_pipeline_module.GPUMemoryManager
BatchSizeOptimizer = gpu_pipeline_module.BatchSizeOptimizer
ThroughputMetrics = gpu_pipeline_module.ThroughputMetrics
get_optimal_batch_size = gpu_pipeline_module.get_optimal_batch_size


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_model():
    """Simple CNN model for testing."""

    class SimpleModel(nn.Module):
        def __init__(self, feature_dim=128):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(64, feature_dim)

        def forward(self, x):
            x = self.conv(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    return SimpleModel(feature_dim=128)


@pytest.fixture
def device():
    """Get test device (CPU for CI compatibility)."""
    return torch.device("cpu")


@pytest.fixture
def gpu_pipeline(simple_model, device):
    """Create GPUPipeline instance."""
    return GPUPipeline(
        model=simple_model, batch_size=32, gpu_ids=None, enable_fp16=False  # Auto-detect
    )


@pytest.fixture
def sample_patches():
    """Generate sample patch tensor."""
    return torch.randn(16, 3, 96, 96)


# ============================================================================
# GPUMemoryManager Tests
# ============================================================================


class TestGPUMemoryManager:
    """Test GPU memory management."""

    def test_initialization(self, device):
        """Test memory manager initialization."""
        manager = GPUMemoryManager(device, memory_limit_gb=8.0)

        assert manager.device == device
        assert manager.memory_limit_gb == 8.0
        assert manager.total_memory_gb >= 0.0

    def test_auto_memory_limit(self, device):
        """Test automatic memory limit (80% of total)."""
        manager = GPUMemoryManager(device, memory_limit_gb=None)

        if device.type == "cuda":
            assert manager.memory_limit_gb == manager.total_memory_gb * 0.8
        else:
            assert manager.memory_limit_gb == 0.0

    def test_get_memory_usage(self, device):
        """Test memory usage tracking."""
        manager = GPUMemoryManager(device)

        usage = manager.get_memory_usage()
        assert usage >= 0.0
        assert isinstance(usage, float)

    def test_get_memory_reserved(self, device):
        """Test reserved memory tracking."""
        manager = GPUMemoryManager(device)

        reserved = manager.get_memory_reserved()
        assert reserved >= 0.0
        assert isinstance(reserved, float)

    def test_is_memory_available(self, device):
        """Test memory availability check."""
        manager = GPUMemoryManager(device, memory_limit_gb=8.0)

        # Small allocation should be available
        assert manager.is_memory_available(0.1)

        # Huge allocation should not be available
        assert not manager.is_memory_available(100.0)

    def test_get_available_memory(self, device):
        """Test available memory calculation."""
        manager = GPUMemoryManager(device, memory_limit_gb=8.0)

        available = manager.get_available_memory()
        assert available >= 0.0
        assert available <= manager.memory_limit_gb

    def test_cleanup(self, device):
        """Test memory cleanup."""
        manager = GPUMemoryManager(device)

        # Should not raise
        manager.cleanup()


# ============================================================================
# BatchSizeOptimizer Tests
# ============================================================================


class TestBatchSizeOptimizer:
    """Test dynamic batch size optimization."""

    def test_initialization(self):
        """Test optimizer initialization."""
        optimizer = BatchSizeOptimizer(initial_batch_size=64, min_batch_size=1, max_batch_size=256)

        assert optimizer.current_batch_size == 64
        assert optimizer.min_batch_size == 1
        assert optimizer.max_batch_size == 256
        assert optimizer.oom_count == 0

    def test_record_batch(self):
        """Test batch metrics recording."""
        optimizer = BatchSizeOptimizer(initial_batch_size=64)

        optimizer.record_batch(batch_time=0.5, memory_used_gb=2.0)

        assert len(optimizer.batch_times) == 1
        assert len(optimizer.memory_usage) == 1
        assert optimizer.batch_times[0] == 0.5
        assert optimizer.memory_usage[0] == 2.0

    def test_handle_oom(self):
        """Test OOM handling."""
        optimizer = BatchSizeOptimizer(initial_batch_size=64)

        optimizer.handle_oom()

        assert optimizer.oom_count == 1
        assert optimizer.last_oom_batch_size == 64
        assert optimizer.current_batch_size == 16  # 64 // 4

    def test_optimize_critical_pressure(self):
        """Test optimization under critical memory pressure."""
        optimizer = BatchSizeOptimizer(initial_batch_size=64)

        # Critical pressure (>90%)
        new_size = optimizer.optimize(memory_pressure=0.95)

        assert new_size == 32  # 64 // 2
        assert optimizer.current_batch_size == 32

    def test_optimize_high_pressure(self):
        """Test optimization under high memory pressure."""
        optimizer = BatchSizeOptimizer(initial_batch_size=64)

        # High pressure (>80%)
        new_size = optimizer.optimize(memory_pressure=0.85)

        assert new_size == 48  # 64 * 0.75
        assert optimizer.current_batch_size == 48

    def test_optimize_low_pressure(self):
        """Test optimization under low memory pressure."""
        optimizer = BatchSizeOptimizer(initial_batch_size=64)

        # Record fast batches
        for _ in range(5):
            optimizer.record_batch(batch_time=0.3, memory_used_gb=1.0)

        # Low pressure (<40%)
        new_size = optimizer.optimize(memory_pressure=0.3)

        assert new_size == 76  # 64 * 1.2
        assert optimizer.current_batch_size == 76

    def test_optimize_respects_min_max(self):
        """Test that optimization respects min/max bounds."""
        optimizer = BatchSizeOptimizer(initial_batch_size=2, min_batch_size=1, max_batch_size=4)

        # Try to reduce below min
        optimizer.optimize(memory_pressure=0.95)
        assert optimizer.current_batch_size >= optimizer.min_batch_size

        # Try to increase above max
        for _ in range(5):
            optimizer.record_batch(batch_time=0.1, memory_used_gb=0.5)
        optimizer.optimize(memory_pressure=0.2)
        assert optimizer.current_batch_size <= optimizer.max_batch_size

    def test_get_batch_size(self):
        """Test batch size getter."""
        optimizer = BatchSizeOptimizer(initial_batch_size=64)

        assert optimizer.get_batch_size() == 64


# ============================================================================
# ThroughputMetrics Tests
# ============================================================================


class TestThroughputMetrics:
    """Test throughput metrics."""

    def test_initialization(self):
        """Test metrics initialization."""
        metrics = ThroughputMetrics(
            patches_per_second=1000.0,
            batches_per_second=50.0,
            avg_batch_time=0.02,
            gpu_utilization=85.0,
            gpu_memory_used_gb=4.5,
            gpu_memory_total_gb=8.0,
            current_batch_size=32,
            total_patches_processed=10000,
        )

        assert metrics.patches_per_second == 1000.0
        assert metrics.batches_per_second == 50.0
        assert metrics.avg_batch_time == 0.02
        assert metrics.gpu_utilization == 85.0
        assert metrics.gpu_memory_used_gb == 4.5
        assert metrics.gpu_memory_total_gb == 8.0
        assert metrics.current_batch_size == 32
        assert metrics.total_patches_processed == 10000

    def test_gpu_memory_percent(self):
        """Test GPU memory percentage calculation."""
        metrics = ThroughputMetrics(
            patches_per_second=1000.0,
            batches_per_second=50.0,
            avg_batch_time=0.02,
            gpu_utilization=85.0,
            gpu_memory_used_gb=4.0,
            gpu_memory_total_gb=8.0,
            current_batch_size=32,
            total_patches_processed=10000,
        )

        assert metrics.gpu_memory_percent == 50.0

    def test_gpu_memory_percent_zero_total(self):
        """Test memory percent with zero total (CPU case)."""
        metrics = ThroughputMetrics(
            patches_per_second=1000.0,
            batches_per_second=50.0,
            avg_batch_time=0.02,
            gpu_utilization=0.0,
            gpu_memory_used_gb=0.0,
            gpu_memory_total_gb=0.0,
            current_batch_size=32,
            total_patches_processed=10000,
        )

        assert metrics.gpu_memory_percent == 0.0


# ============================================================================
# GPUPipeline Tests
# ============================================================================


class TestGPUPipeline:
    """Test GPU pipeline functionality."""

    def test_initialization(self, simple_model, device):
        """Test pipeline initialization."""
        pipeline = GPUPipeline(model=simple_model, batch_size=32, gpu_ids=None, enable_fp16=False)

        assert pipeline.initial_batch_size == 32
        assert pipeline.enable_fp16 is False
        assert pipeline.primary_device.type in ["cuda", "cpu"]
        assert pipeline.total_patches_processed == 0
        assert pipeline.total_batches_processed == 0

    def test_initialization_with_fp16(self, simple_model):
        """Test initialization with FP16 enabled."""
        pipeline = GPUPipeline(model=simple_model, batch_size=32, enable_fp16=True)

        assert pipeline.enable_fp16 is True

        # Check if model is in half precision (only on CUDA)
        if pipeline.primary_device.type == "cuda":
            # Model should be in FP16
            for param in pipeline.model.parameters():
                assert param.dtype == torch.float16

    def test_process_batch_async(self, gpu_pipeline, sample_patches):
        """Test async batch processing."""
        # Run async function synchronously
        features = asyncio.run(gpu_pipeline.process_batch_async(sample_patches))

        assert features.shape[0] == sample_patches.shape[0]
        assert features.shape[1] == 128  # feature_dim
        assert features.device.type == "cpu"  # Results moved to CPU

    def test_process_batch_sync(self, gpu_pipeline, sample_patches):
        """Test synchronous batch processing."""
        features = gpu_pipeline._process_batch_sync(sample_patches)

        assert features.shape[0] == sample_patches.shape[0]
        assert features.shape[1] == 128
        assert features.device.type == "cpu"

    def test_process_batch_updates_metrics(self, gpu_pipeline, sample_patches):
        """Test that batch processing updates metrics."""
        initial_patches = gpu_pipeline.total_patches_processed
        initial_batches = gpu_pipeline.total_batches_processed

        gpu_pipeline._process_batch_sync(sample_patches)

        assert gpu_pipeline.total_patches_processed == initial_patches + sample_patches.shape[0]
        assert gpu_pipeline.total_batches_processed == initial_batches + 1
        assert len(gpu_pipeline.batch_times) > 0

    def test_process_in_subbatches(self, gpu_pipeline):
        """Test processing large batch in sub-batches."""
        large_batch = torch.randn(100, 3, 96, 96)

        features = gpu_pipeline._process_in_subbatches(large_batch, subbatch_size=32)

        assert features.shape[0] == 100
        assert features.shape[1] == 128

    @patch("torch.cuda.is_available", return_value=False)
    def test_oom_recovery(self, mock_cuda, gpu_pipeline):
        """Test OOM error recovery."""
        # Mock OOM error
        original_model = gpu_pipeline.model

        def mock_forward(*args, **kwargs):
            raise RuntimeError("CUDA out of memory")

        gpu_pipeline.model = Mock(side_effect=mock_forward)

        # Should handle OOM and retry
        with pytest.raises(RuntimeError):
            gpu_pipeline._process_batch_sync(torch.randn(64, 3, 96, 96))

        # Batch size should be reduced
        assert gpu_pipeline.batch_optimizer.oom_count > 0

        # Restore original model
        gpu_pipeline.model = original_model

    def test_optimize_batch_size(self, gpu_pipeline):
        """Test dynamic batch size optimization."""
        initial_size = gpu_pipeline.batch_optimizer.get_batch_size()

        # High memory usage should reduce batch size
        new_size = gpu_pipeline.optimize_batch_size(memory_usage=7.0)

        assert new_size <= initial_size

    def test_get_throughput_stats(self, gpu_pipeline, sample_patches):
        """Test throughput statistics."""
        # Process some batches
        gpu_pipeline._process_batch_sync(sample_patches)
        gpu_pipeline._process_batch_sync(sample_patches)

        stats = gpu_pipeline.get_throughput_stats()

        assert isinstance(stats, ThroughputMetrics)
        assert stats.patches_per_second > 0
        assert stats.batches_per_second > 0
        assert stats.avg_batch_time > 0
        assert stats.total_patches_processed == sample_patches.shape[0] * 2
        assert stats.current_batch_size > 0

    def test_reset_stats(self, gpu_pipeline, sample_patches):
        """Test statistics reset."""
        # Process batch
        gpu_pipeline._process_batch_sync(sample_patches)

        assert gpu_pipeline.total_patches_processed > 0

        # Reset
        gpu_pipeline.reset_stats()

        assert gpu_pipeline.total_patches_processed == 0
        assert gpu_pipeline.total_batches_processed == 0
        assert gpu_pipeline.total_processing_time == 0.0
        assert len(gpu_pipeline.batch_times) == 0

    def test_cleanup(self, gpu_pipeline):
        """Test resource cleanup."""
        # Should not raise
        gpu_pipeline.cleanup()

    def test_context_manager(self, simple_model):
        """Test context manager usage."""
        with GPUPipeline(model=simple_model, batch_size=32) as pipeline:
            assert pipeline is not None

        # Cleanup should have been called

    def test_periodic_cleanup(self, gpu_pipeline, sample_patches):
        """Test periodic memory cleanup."""
        # Process 10 batches to trigger cleanup
        for _ in range(10):
            gpu_pipeline._process_batch_sync(sample_patches)

        # Should have triggered cleanup at batch 10
        assert gpu_pipeline.total_batches_processed == 10

    def test_multi_gpu_detection(self, simple_model):
        """Test multi-GPU detection."""
        pipeline = GPUPipeline(model=simple_model, batch_size=32, gpu_ids=None)

        # Should detect available GPUs or fall back to CPU
        assert len(pipeline.devices) >= 1
        assert pipeline.primary_device.type in ["cuda", "cpu"]

    def test_explicit_gpu_ids(self, simple_model):
        """Test explicit GPU ID specification."""
        # Request GPU 0 (may not be available in CI)
        pipeline = GPUPipeline(model=simple_model, batch_size=32, gpu_ids=[0])

        # Should either use GPU 0 or fall back to CPU
        assert pipeline.primary_device.type in ["cuda", "cpu"]


# ============================================================================
# Utility Function Tests
# ============================================================================


class TestUtilityFunctions:
    """Test utility functions."""

    def test_get_optimal_batch_size(self, simple_model, device):
        """Test optimal batch size estimation."""
        input_shape = (3, 96, 96)

        optimal_size = get_optimal_batch_size(
            model=simple_model, input_shape=input_shape, device=device, memory_limit_gb=8.0
        )

        assert optimal_size >= 1
        assert optimal_size <= 256
        assert isinstance(optimal_size, int)

    def test_get_optimal_batch_size_small_memory(self, simple_model, device):
        """Test batch size with small memory limit."""
        input_shape = (3, 96, 96)

        optimal_size = get_optimal_batch_size(
            model=simple_model,
            input_shape=input_shape,
            device=device,
            memory_limit_gb=1.0,  # Small limit
        )

        # Should return reasonable batch size (may be larger on CPU)
        assert optimal_size >= 1
        assert optimal_size <= 256


# ============================================================================
# Integration Tests
# ============================================================================


class TestGPUPipelineIntegration:
    """Integration tests for complete pipeline."""

    def test_end_to_end_processing(self, gpu_pipeline):
        """Test end-to-end batch processing."""
        # Create multiple batches
        batches = [torch.randn(16, 3, 96, 96) for _ in range(5)]

        all_features = []
        for batch in batches:
            features = asyncio.run(gpu_pipeline.process_batch_async(batch))
            all_features.append(features)

        # Check results
        assert len(all_features) == 5
        for features in all_features:
            assert features.shape[0] == 16
            assert features.shape[1] == 128

        # Check metrics
        stats = gpu_pipeline.get_throughput_stats()
        assert stats.total_patches_processed == 80  # 16 * 5
        assert stats.patches_per_second > 0

    def test_memory_pressure_adaptation(self, gpu_pipeline):
        """Test adaptation to memory pressure."""
        initial_batch_size = gpu_pipeline.batch_optimizer.get_batch_size()

        # Simulate high memory pressure (only works on CUDA)
        if gpu_pipeline.primary_device.type == "cuda":
            for _ in range(3):
                gpu_pipeline.optimize_batch_size(memory_usage=7.5)

            # Batch size should be reduced
            final_batch_size = gpu_pipeline.batch_optimizer.get_batch_size()
            assert final_batch_size < initial_batch_size
        else:
            # On CPU, memory_limit_gb=0, so no adaptation occurs
            for _ in range(3):
                gpu_pipeline.optimize_batch_size(memory_usage=0.0)

            # Batch size should remain unchanged
            final_batch_size = gpu_pipeline.batch_optimizer.get_batch_size()
            assert final_batch_size == initial_batch_size

    def test_concurrent_processing(self, gpu_pipeline):
        """Test concurrent batch processing."""
        batches = [torch.randn(8, 3, 96, 96) for _ in range(3)]

        # Process concurrently using asyncio.gather
        async def process_all():
            tasks = [gpu_pipeline.process_batch_async(batch) for batch in batches]
            return await asyncio.gather(*tasks)

        results = asyncio.run(process_all())

        assert len(results) == 3
        for features in results:
            assert features.shape[0] == 8
            assert features.shape[1] == 128


# ============================================================================
# Performance Tests
# ============================================================================


class TestGPUPipelinePerformance:
    """Performance benchmarks."""

    def test_throughput_target(self, gpu_pipeline):
        """Test that throughput meets target (>100 patches/sec on CPU)."""
        # Process multiple batches
        for _ in range(10):
            batch = torch.randn(32, 3, 96, 96)
            gpu_pipeline._process_batch_sync(batch)

        stats = gpu_pipeline.get_throughput_stats()

        # Should achieve >100 patches/sec even on CPU
        assert stats.patches_per_second > 100

    def test_batch_processing_time(self, gpu_pipeline, sample_patches):
        """Test batch processing time is reasonable."""
        import time

        start = time.time()
        gpu_pipeline._process_batch_sync(sample_patches)
        elapsed = time.time() - start

        # Should process batch in <1 second on CPU
        assert elapsed < 1.0

    def test_memory_efficiency(self, gpu_pipeline, sample_patches):
        """Test memory usage stays within bounds."""
        # Process multiple batches
        for _ in range(20):
            gpu_pipeline._process_batch_sync(sample_patches)

        # Memory manager should keep usage reasonable
        memory_used = gpu_pipeline.memory_manager.get_memory_usage()
        memory_limit = gpu_pipeline.memory_manager.memory_limit_gb

        # Should stay under limit
        assert memory_used <= memory_limit


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
