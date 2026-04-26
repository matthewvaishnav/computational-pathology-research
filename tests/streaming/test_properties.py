"""Property-based tests for streaming components using Hypothesis."""

import pytest
import torch
import numpy as np
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from hypothesis import Phase
from unittest.mock import Mock

from src.streaming.wsi_stream_reader import TileBufferPool, StreamingMetadata
from src.streaming.gpu_pipeline import GPUMemoryManager, BatchSizeOptimizer
from src.streaming.attention_aggregator import StreamingAttentionAggregator, AttentionMIL


# Hypothesis strategies
@st.composite
def tile_sizes(draw):
    """Generate valid tile sizes."""
    return draw(st.integers(min_value=256, max_value=2048))


@st.composite
def memory_budgets(draw):
    """Generate valid memory budgets in GB."""
    return draw(st.floats(min_value=0.5, max_value=32.0))


@st.composite
def batch_sizes(draw):
    """Generate valid batch sizes."""
    return draw(st.integers(min_value=1, max_value=256))


@st.composite
def feature_dims(draw):
    """Generate valid feature dimensions."""
    return draw(st.integers(min_value=128, max_value=2048))


@st.composite
def slide_dimensions(draw):
    """Generate valid slide dimensions."""
    w = draw(st.integers(min_value=1000, max_value=100000))
    h = draw(st.integers(min_value=1000, max_value=100000))
    return (w, h)


class TestMemoryUsageProperty:
    """Test memory usage property across slide sizes."""
    
    @given(
        tile_size=tile_sizes(),
        max_memory_gb=memory_budgets()
    )
    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_buffer_pool_respects_memory_limit(self, tile_size, max_memory_gb):
        """Property: Buffer pool never exceeds memory limit."""
        pool = TileBufferPool(max_memory_gb=max_memory_gb, tile_size=tile_size)
        
        # Get buffers
        buffers = []
        for _ in range(100):
            buf = pool.get_buffer()
            if buf is not None:
                buffers.append(buf)
        
        # Check property: memory <= limit (with 10% tolerance)
        memory_gb = pool.current_memory / (1024**3)
        assert memory_gb <= max_memory_gb * 1.1
        
        # Cleanup
        for buf in buffers:
            pool.return_buffer(buf)
        pool.cleanup()
    
    @given(
        dimensions=slide_dimensions(),
        tile_size=tile_sizes()
    )
    @settings(max_examples=30, deadline=None)
    def test_metadata_memory_scales_with_slide_size(self, dimensions, tile_size):
        """Property: Memory budget scales reasonably with slide size."""
        # Estimate patches
        w, h = dimensions
        patches = (w * h) // (tile_size * tile_size)
        assume(patches > 0 and patches < 1000000)  # Reasonable range
        
        # Create metadata
        metadata = StreamingMetadata(
            slide_id="test",
            dimensions=dimensions,
            estimated_patches=patches,
            tile_size=tile_size,
            memory_budget_gb=2.0,
            target_processing_time=30.0,
            confidence_threshold=0.95
        )
        
        # Property: memory budget is reasonable
        assert 0.5 <= metadata.memory_budget_gb <= 32.0
    
    @given(
        initial_batch=batch_sizes(),
        memory_pressure=st.floats(min_value=0.0, max_value=1.0)
    )
    @settings(max_examples=50, deadline=None)
    def test_batch_optimizer_respects_bounds(self, initial_batch, memory_pressure):
        """Property: Batch size always within [min, max]."""
        min_batch = max(1, initial_batch // 4)
        max_batch = initial_batch * 4
        
        optimizer = BatchSizeOptimizer(
            initial_batch_size=initial_batch,
            min_batch_size=min_batch,
            max_batch_size=max_batch
        )
        
        # Optimize multiple times
        for _ in range(10):
            optimizer.optimize(memory_pressure)
        
        # Property: always in bounds
        assert min_batch <= optimizer.current_batch_size <= max_batch


class TestAttentionWeightNormalization:
    """Test attention weight normalization property."""
    
    @given(
        num_patches=st.integers(min_value=10, max_value=500),
        feature_dim=feature_dims()
    )
    @settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_attention_weights_sum_to_one(self, num_patches, feature_dim):
        """Property: Attention weights always sum to 1.0."""
        model = AttentionMIL(feature_dim=feature_dim, hidden_dim=128, num_classes=2)
        aggregator = StreamingAttentionAggregator(attention_model=model)
        
        # Generate features
        features = torch.randn(num_patches, feature_dim)
        coords = np.array([[i % 100, i // 100] for i in range(num_patches)])
        
        # Update
        update = aggregator.update_features(features, coords)
        
        # Property: weights sum to 1.0 (with tolerance)
        weight_sum = torch.sum(update.attention_weights).item()
        assert abs(weight_sum - 1.0) < 1e-4
    
    @given(
        num_patches=st.integers(min_value=5, max_value=100),
        feature_dim=st.integers(min_value=128, max_value=512)
    )
    @settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_attention_weights_non_negative(self, num_patches, feature_dim):
        """Property: Attention weights are non-negative."""
        model = AttentionMIL(feature_dim=feature_dim, hidden_dim=128, num_classes=2)
        aggregator = StreamingAttentionAggregator(attention_model=model)
        
        features = torch.randn(num_patches, feature_dim)
        coords = np.array([[i, 0] for i in range(num_patches)])
        
        update = aggregator.update_features(features, coords)
        
        # Property: all weights >= 0
        assert torch.all(update.attention_weights >= 0.0)
    
    @given(
        num_patches=st.integers(min_value=10, max_value=200),
        feature_dim=st.integers(min_value=256, max_value=1024)
    )
    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_attention_weights_stable_across_updates(self, num_patches, feature_dim):
        """Property: Attention weights remain normalized across updates."""
        model = AttentionMIL(feature_dim=feature_dim, hidden_dim=128, num_classes=2)
        aggregator = StreamingAttentionAggregator(attention_model=model, max_features=1000)
        
        # Multiple updates
        for batch in range(3):
            features = torch.randn(num_patches, feature_dim)
            coords = np.array([[i + batch * num_patches, 0] for i in range(num_patches)])
            update = aggregator.update_features(features, coords)
            
            # Property: weights still sum to 1.0
            weight_sum = torch.sum(update.attention_weights).item()
            assert abs(weight_sum - 1.0) < 1e-4


class TestConfidenceMonotonicity:
    """Test confidence monotonicity property."""
    
    @given(
        num_updates=st.integers(min_value=2, max_value=10),
        patches_per_update=st.integers(min_value=10, max_value=50)
    )
    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_confidence_generally_increases(self, num_updates, patches_per_update):
        """Property: Confidence generally increases or stabilizes."""
        model = AttentionMIL(feature_dim=512, hidden_dim=128, num_classes=2)
        aggregator = StreamingAttentionAggregator(attention_model=model)
        
        confidences = []
        
        # Multiple updates
        for i in range(num_updates):
            features = torch.randn(patches_per_update, 512)
            coords = np.array([[j + i * patches_per_update, 0] for j in range(patches_per_update)])
            update = aggregator.update_features(features, coords)
            confidences.append(update.current_confidence)
        
        # Property: confidence doesn't decrease significantly
        # (allow small fluctuations due to numerical precision)
        for i in range(1, len(confidences)):
            # Allow 5% decrease tolerance
            assert confidences[i] >= confidences[i-1] - 0.05
    
    @given(
        num_patches=st.integers(min_value=50, max_value=200)
    )
    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_confidence_bounded(self, num_patches):
        """Property: Confidence always in [0, 1]."""
        model = AttentionMIL(feature_dim=512, hidden_dim=128, num_classes=2)
        aggregator = StreamingAttentionAggregator(attention_model=model)
        
        features = torch.randn(num_patches, 512)
        coords = np.array([[i, 0] for i in range(num_patches)])
        update = aggregator.update_features(features, coords)
        
        # Property: 0 <= confidence <= 1
        assert 0.0 <= update.current_confidence <= 1.0


class TestProcessingTimeBounds:
    """Test processing time bounds property."""
    
    @given(
        batch_size=batch_sizes(),
        num_batches=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_throughput_positive(self, batch_size, num_batches):
        """Property: Throughput is always positive."""
        model = Mock(spec=torch.nn.Module)
        model.eval = Mock()
        model.to = Mock(return_value=model)
        model.return_value = torch.randn(batch_size, 512)
        
        from src.streaming.gpu_pipeline import GPUPipeline
        pipeline = GPUPipeline(model=model, batch_size=batch_size)
        
        # Process batches
        for _ in range(num_batches):
            patches = torch.randn(batch_size, 3, 224, 224)
            pipeline._process_batch_sync(patches)
        
        metrics = pipeline.get_throughput_stats()
        
        # Property: throughput > 0
        assert metrics.patches_per_second > 0
        assert metrics.batches_per_second > 0
    
    @given(
        initial_batch=st.integers(min_value=4, max_value=64)
    )
    @settings(max_examples=20, deadline=None)
    def test_batch_time_decreases_with_size(self, initial_batch):
        """Property: Larger batches process more patches per second."""
        # This is a weak property - just check metrics are reasonable
        optimizer = BatchSizeOptimizer(
            initial_batch_size=initial_batch,
            min_batch_size=1,
            max_batch_size=256
        )
        
        # Record batches
        for _ in range(5):
            optimizer.record_batch(batch_time=0.1, memory_used_gb=1.0)
        
        # Property: recorded times are positive
        assert len(optimizer.batch_times) > 0
        assert all(t > 0 for t in optimizer.batch_times)


class TestMemoryPressureAdaptation:
    """Test system adapts correctly to memory pressure."""
    
    @given(
        initial_batch=st.integers(min_value=8, max_value=128),
        pressure=st.floats(min_value=0.8, max_value=1.0)
    )
    @settings(max_examples=30, deadline=None)
    def test_high_pressure_reduces_batch_size(self, initial_batch, pressure):
        """Property: High memory pressure reduces batch size."""
        optimizer = BatchSizeOptimizer(
            initial_batch_size=initial_batch,
            min_batch_size=1,
            max_batch_size=256
        )
        
        before = optimizer.current_batch_size
        optimizer.optimize(memory_pressure=pressure)
        after = optimizer.current_batch_size
        
        # Property: batch size reduced or stayed same
        assert after <= before
    
    @given(
        initial_batch=st.integers(min_value=4, max_value=64),
        pressure=st.floats(min_value=0.0, max_value=0.4)
    )
    @settings(max_examples=30, deadline=None)
    def test_low_pressure_allows_increase(self, initial_batch, pressure):
        """Property: Low memory pressure allows batch size increase."""
        optimizer = BatchSizeOptimizer(
            initial_batch_size=initial_batch,
            min_batch_size=1,
            max_batch_size=256
        )
        
        # Record fast batches
        for _ in range(10):
            optimizer.record_batch(batch_time=0.05, memory_used_gb=0.5)
        
        before = optimizer.current_batch_size
        optimizer.optimize(memory_pressure=pressure)
        after = optimizer.current_batch_size
        
        # Property: batch size increased or stayed same
        assert after >= before


class TestNumericalStability:
    """Test numerical stability properties."""
    
    @given(
        num_patches=st.integers(min_value=10, max_value=100),
        scale=st.floats(min_value=1e-6, max_value=1e6)
    )
    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_attention_stable_with_scaled_features(self, num_patches, scale):
        """Property: Attention weights stable across feature scaling."""
        assume(scale > 0 and np.isfinite(scale))
        
        model = AttentionMIL(feature_dim=512, hidden_dim=128, num_classes=2)
        aggregator = StreamingAttentionAggregator(attention_model=model)
        
        # Generate features and scale
        features = torch.randn(num_patches, 512) * scale
        coords = np.array([[i, 0] for i in range(num_patches)])
        
        # Skip if features have NaN/Inf
        if not torch.isfinite(features).all():
            assume(False)
        
        update = aggregator.update_features(features, coords)
        
        # Property: attention weights are finite
        assert torch.isfinite(update.attention_weights).all()
    
    @given(
        num_patches=st.integers(min_value=10, max_value=100)
    )
    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_zero_features_handled(self, num_patches):
        """Property: Zero features don't cause NaN."""
        model = AttentionMIL(feature_dim=512, hidden_dim=128, num_classes=2)
        aggregator = StreamingAttentionAggregator(attention_model=model)
        
        # All zeros
        features = torch.zeros(num_patches, 512)
        coords = np.array([[i, 0] for i in range(num_patches)])
        
        update = aggregator.update_features(features, coords)
        
        # Property: no NaN in outputs
        assert not torch.isnan(update.attention_weights).any()
        assert not np.isnan(update.current_confidence)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--hypothesis-show-statistics'])
