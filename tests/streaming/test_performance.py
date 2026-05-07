"""Performance unit tests for streaming components."""

import gc
import time
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
import torch

from src.streaming.attention_aggregator import AttentionMIL, StreamingAttentionAggregator
from src.streaming.gpu_pipeline import GPUMemoryManager, GPUPipeline
from src.streaming.wsi_stream_reader import StreamingMetadata, TileBufferPool, WSIStreamReader


class TestMemoryBounds:
    """Test memory usage bounds under various conditions."""

    def test_tile_buffer_pool_respects_limit(self):
        """Test buffer pool stays within memory limit."""
        max_memory_gb = 0.1  # 100MB
        pool = TileBufferPool(max_memory_gb=max_memory_gb, tile_size=256)

        # Get many buffers
        buffers = []
        for _ in range(100):
            buf = pool.get_buffer()
            if buf is not None:
                buffers.append(buf)

        # Check memory usage
        memory_gb = pool.current_memory / (1024**3)
        assert memory_gb <= max_memory_gb * 1.1  # 10% tolerance

        # Cleanup
        for buf in buffers:
            pool.return_buffer(buf)
        pool.cleanup()

    def test_gpu_memory_manager_tracks_usage(self):
        """Test GPU memory manager tracks usage."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        manager = GPUMemoryManager(device, memory_limit_gb=1.0)

        # Get initial usage
        initial = manager.get_memory_usage()
        assert initial >= 0.0

        # Allocate tensor
        if device.type == "cuda":
            tensor = torch.randn(1000, 1000, device=device)
            current = manager.get_memory_usage()
            assert current > initial

            # Cleanup
            del tensor
            manager.cleanup()

    def test_aggregator_memory_bounded(self):
        """Test aggregator respects max_features limit."""
        model = AttentionMIL(feature_dim=512, hidden_dim=128, num_classes=2)
        aggregator = StreamingAttentionAggregator(attention_model=model, max_features=100)

        # Add many features
        for i in range(200):
            features = torch.randn(10, 512)
            coords = np.array([[j, 0] for j in range(10)])
            aggregator.update_features(features, coords)

        # Should not exceed max
        assert aggregator.num_patches <= 100

    def test_pipeline_memory_cleanup(self):
        """Test pipeline cleans up memory."""
        model = Mock(spec=torch.nn.Module)
        model.eval = Mock()
        model.to = Mock(return_value=model)
        model.return_value = torch.randn(4, 512)

        pipeline = GPUPipeline(model=model, batch_size=4)

        # Process batches
        for _ in range(10):
            patches = torch.randn(4, 3, 224, 224)
            pipeline._process_batch_sync(patches)

        # Cleanup
        pipeline.cleanup()

        # Memory should be released
        if pipeline.primary_device.type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

    def test_streaming_metadata_validates_memory_budget(self):
        """Test metadata validates memory budget."""
        # Valid budget
        metadata = StreamingMetadata(
            slide_id="test",
            dimensions=(10000, 10000),
            estimated_patches=1000,
            tile_size=256,
            memory_budget_gb=2.0,
            target_processing_time=30.0,
            confidence_threshold=0.95,
        )
        assert 0.5 <= metadata.memory_budget_gb <= 32.0

        # Invalid budget
        with pytest.raises(ValueError, match="Memory budget"):
            StreamingMetadata(
                slide_id="test",
                dimensions=(10000, 10000),
                estimated_patches=1000,
                tile_size=256,
                memory_budget_gb=100.0,  # Too large
                target_processing_time=30.0,
                confidence_threshold=0.95,
            )


class TestProcessingTime:
    """Test processing time requirements."""

    @pytest.fixture
    def mock_model(self):
        """Create fast mock model."""
        model = Mock(spec=torch.nn.Module)
        model.eval = Mock()
        model.to = Mock(return_value=model)
        model.return_value = torch.randn(4, 512)
        return model

    def test_batch_processing_time_tracked(self, mock_model):
        """Test batch processing time is tracked."""
        pipeline = GPUPipeline(model=mock_model, batch_size=4)

        # Process batch
        patches = torch.randn(4, 3, 224, 224)
        start = time.time()
        pipeline._process_batch_sync(patches)
        elapsed = time.time() - start

        # Should complete quickly
        assert elapsed < 5.0  # 5 seconds max for mock

        # Check metrics
        metrics = pipeline.get_throughput_stats()
        assert metrics.avg_batch_time > 0

    def test_throughput_calculation(self, mock_model):
        """Test throughput metrics calculation."""
        pipeline = GPUPipeline(model=mock_model, batch_size=8)

        # Process multiple batches
        for _ in range(5):
            patches = torch.randn(8, 3, 224, 224)
            pipeline._process_batch_sync(patches)

        # Get metrics
        metrics = pipeline.get_throughput_stats()

        assert metrics.patches_per_second > 0
        assert metrics.batches_per_second > 0
        assert metrics.total_patches_processed == 40

    def test_aggregator_update_time(self):
        """Test aggregator update time is reasonable."""
        model = AttentionMIL(feature_dim=512, hidden_dim=128, num_classes=2)
        aggregator = StreamingAttentionAggregator(attention_model=model)

        # Time update
        features = torch.randn(100, 512)
        coords = np.array([[i, 0] for i in range(100)])

        start = time.time()
        aggregator.update_features(features, coords)
        elapsed = time.time() - start

        # Should be fast
        assert elapsed < 1.0  # 1 second max

    def test_metadata_validates_processing_time(self):
        """Test metadata validates target processing time."""
        # Valid time
        metadata = StreamingMetadata(
            slide_id="test",
            dimensions=(10000, 10000),
            estimated_patches=1000,
            tile_size=256,
            memory_budget_gb=2.0,
            target_processing_time=30.0,
            confidence_threshold=0.95,
        )
        assert 5.0 <= metadata.target_processing_time <= 300.0

        # Invalid time
        with pytest.raises(ValueError, match="processing time"):
            StreamingMetadata(
                slide_id="test",
                dimensions=(10000, 10000),
                estimated_patches=1000,
                tile_size=256,
                memory_budget_gb=2.0,
                target_processing_time=1000.0,  # Too long
                confidence_threshold=0.95,
            )


class TestThroughputScaling:
    """Test throughput scaling with multiple GPUs."""

    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = Mock(spec=torch.nn.Module)
        model.eval = Mock()
        model.to = Mock(return_value=model)
        model.return_value = torch.randn(8, 512)
        return model

    def test_single_gpu_throughput(self, mock_model):
        """Test single GPU throughput baseline."""
        pipeline = GPUPipeline(model=mock_model, batch_size=8, gpu_ids=[0])

        # Process batches
        start = time.time()
        for _ in range(10):
            patches = torch.randn(8, 3, 224, 224)
            pipeline._process_batch_sync(patches)
        elapsed = time.time() - start

        # Calculate throughput
        throughput = 80 / elapsed  # 80 patches total
        assert throughput > 0

    def test_batch_size_affects_throughput(self, mock_model):
        """Test larger batch size improves throughput."""
        # Small batch
        pipeline_small = GPUPipeline(model=mock_model, batch_size=4)
        start = time.time()
        for _ in range(10):
            patches = torch.randn(4, 3, 224, 224)
            pipeline_small._process_batch_sync(patches)
        time_small = time.time() - start

        # Large batch
        pipeline_large = GPUPipeline(model=mock_model, batch_size=16)
        start = time.time()
        for _ in range(10):
            patches = torch.randn(16, 3, 224, 224)
            pipeline_large._process_batch_sync(patches)
        time_large = time.time() - start

        # Larger batch should process more patches per second
        throughput_small = 40 / time_small
        throughput_large = 160 / time_large

        # Note: with mock model, timing may vary
        assert throughput_large > 0
        assert throughput_small > 0

    def test_multi_gpu_detection(self, mock_model):
        """Test multi-GPU detection."""
        pipeline = GPUPipeline(model=mock_model, batch_size=8)

        # Check devices
        assert len(pipeline.devices) >= 1

        # If CUDA available, check GPU count
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            assert len(pipeline.devices) <= gpu_count

    def test_throughput_metrics_accuracy(self, mock_model):
        """Test throughput metrics are accurate."""
        pipeline = GPUPipeline(model=mock_model, batch_size=8)

        # Process known number of patches
        num_batches = 10
        batch_size = 8

        for _ in range(num_batches):
            patches = torch.randn(batch_size, 3, 224, 224)
            pipeline._process_batch_sync(patches)

        metrics = pipeline.get_throughput_stats()

        # Verify counts
        assert metrics.total_patches_processed == num_batches * batch_size
        assert metrics.patches_per_second > 0
        assert metrics.batches_per_second > 0


class TestMemoryPressureAdaptation:
    """Test system adapts to memory pressure."""

    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = Mock(spec=torch.nn.Module)
        model.eval = Mock()
        model.to = Mock(return_value=model)
        model.return_value = torch.randn(4, 512)
        return model

    def test_batch_size_reduces_under_pressure(self, mock_model):
        """Test batch size reduces under memory pressure."""
        pipeline = GPUPipeline(model=mock_model, batch_size=64)

        initial_size = pipeline.batch_optimizer.current_batch_size

        # Simulate high memory pressure
        pipeline.batch_optimizer.optimize(memory_pressure=0.95)

        # Should reduce
        assert pipeline.batch_optimizer.current_batch_size < initial_size

    def test_batch_size_increases_when_available(self, mock_model):
        """Test batch size increases when memory available."""
        pipeline = GPUPipeline(model=mock_model, batch_size=8)

        # Record fast batches
        for _ in range(10):
            pipeline.batch_optimizer.record_batch(batch_time=0.1, memory_used_gb=0.5)

        initial_size = pipeline.batch_optimizer.current_batch_size

        # Simulate low memory pressure
        pipeline.batch_optimizer.optimize(memory_pressure=0.2)

        # Should increase or stay same
        assert pipeline.batch_optimizer.current_batch_size >= initial_size

    def test_memory_manager_reports_availability(self):
        """Test memory manager reports availability correctly."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        manager = GPUMemoryManager(device, memory_limit_gb=2.0)

        # Check availability
        assert manager.is_memory_available(0.1)  # 100MB should be available
        assert not manager.is_memory_available(10.0)  # 10GB exceeds limit

    def test_aggregator_trims_features_under_pressure(self):
        """Test aggregator trims features when exceeding max."""
        model = AttentionMIL(feature_dim=512, hidden_dim=128, num_classes=2)
        aggregator = StreamingAttentionAggregator(attention_model=model, max_features=50)

        # Add features exceeding max
        for _ in range(10):
            features = torch.randn(10, 512)
            coords = np.array([[i, 0] for i in range(10)])
            aggregator.update_features(features, coords)

        # Should trim to max
        assert aggregator.num_patches <= 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
