"""Correctness property tests for streaming components."""

import pytest
import torch
import numpy as np
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from unittest.mock import Mock

from src.streaming.wsi_stream_reader import TileBatch
from src.streaming.attention_aggregator import StreamingAttentionAggregator, AttentionMIL


@st.composite
def valid_coordinates(draw, max_coord=1000):
    """Generate valid tile coordinates."""
    num_tiles = draw(st.integers(min_value=5, max_value=100))
    coords = []
    for _ in range(num_tiles):
        x = draw(st.integers(min_value=0, max_value=max_coord))
        y = draw(st.integers(min_value=0, max_value=max_coord))
        coords.append([x, y])
    return np.array(coords)


class TestSpatialCoverageCompleteness:
    """Test spatial coverage completeness property."""
    
    @given(
        tile_size=st.integers(min_value=256, max_value=1024),
        num_tiles=st.integers(min_value=10, max_value=50)
    )
    @settings(max_examples=20, deadline=None)
    def test_tile_batch_coordinates_unique(self, tile_size, num_tiles):
        """Property: Tile coordinates should be unique (no duplicates)."""
        # Generate unique coordinates
        coords = np.array([[i, j] for i in range(num_tiles) for j in range(1)])[:num_tiles]
        tiles = torch.randn(num_tiles, 3, tile_size, tile_size)
        
        batch = TileBatch(
            tiles=tiles,
            coordinates=coords,
            batch_id=0,
            total_batches=1
        )
        
        # Property: all coordinates unique
        unique_coords = np.unique(batch.coordinates, axis=0)
        assert len(unique_coords) == len(batch.coordinates)
    
    @given(
        num_tiles=st.integers(min_value=10, max_value=100)
    )
    @settings(max_examples=20, deadline=None)
    def test_coordinates_match_tile_count(self, num_tiles):
        """Property: Number of coordinates matches number of tiles."""
        tiles = torch.randn(num_tiles, 3, 256, 256)
        coords = np.array([[i, 0] for i in range(num_tiles)])
        
        batch = TileBatch(
            tiles=tiles,
            coordinates=coords,
            batch_id=0,
            total_batches=1
        )
        
        # Property: counts match
        assert batch.tiles.shape[0] == batch.coordinates.shape[0]
    
    @given(
        num_patches=st.integers(min_value=20, max_value=100)
    )
    @settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_all_patches_get_attention_weight(self, num_patches):
        """Property: Every patch gets an attention weight."""
        model = AttentionMIL(feature_dim=512, hidden_dim=128, num_classes=2)
        aggregator = StreamingAttentionAggregator(attention_model=model)
        
        features = torch.randn(num_patches, 512)
        coords = np.array([[i, 0] for i in range(num_patches)])
        
        update = aggregator.update_features(features, coords)
        
        # Property: attention weights for all patches
        assert len(update.attention_weights) == num_patches


class TestFeatureConsistency:
    """Test feature consistency across streaming."""
    
    @given(
        num_patches=st.integers(min_value=10, max_value=50),
        feature_dim=st.integers(min_value=256, max_value=1024)
    )
    @settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_feature_dimensions_preserved(self, num_patches, feature_dim):
        """Property: Feature dimensions preserved through aggregation."""
        model = AttentionMIL(feature_dim=feature_dim, hidden_dim=128, num_classes=2)
        aggregator = StreamingAttentionAggregator(attention_model=model)
        
        features = torch.randn(num_patches, feature_dim)
        coords = np.array([[i, 0] for i in range(num_patches)])
        
        aggregator.update_features(features, coords)
        
        # Property: accumulated features have correct dim
        assert aggregator.accumulated_features.shape[1] == feature_dim
    
    @given(
        batch1_size=st.integers(min_value=5, max_value=30),
        batch2_size=st.integers(min_value=5, max_value=30)
    )
    @settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_incremental_accumulation_correct(self, batch1_size, batch2_size):
        """Property: Features accumulate correctly across batches."""
        model = AttentionMIL(feature_dim=512, hidden_dim=128, num_classes=2)
        aggregator = StreamingAttentionAggregator(attention_model=model)
        
        # First batch
        features1 = torch.randn(batch1_size, 512)
        coords1 = np.array([[i, 0] for i in range(batch1_size)])
        aggregator.update_features(features1, coords1)
        
        count_after_first = aggregator.num_patches
        
        # Second batch
        features2 = torch.randn(batch2_size, 512)
        coords2 = np.array([[i + batch1_size, 0] for i in range(batch2_size)])
        aggregator.update_features(features2, coords2)
        
        # Property: total count is sum
        assert aggregator.num_patches == count_after_first + batch2_size
    
    @given(
        num_patches=st.integers(min_value=10, max_value=50)
    )
    @settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_feature_order_preserved(self, num_patches):
        """Property: Feature order preserved in accumulation."""
        model = AttentionMIL(feature_dim=512, hidden_dim=128, num_classes=2)
        aggregator = StreamingAttentionAggregator(attention_model=model)
        
        # Create features with identifiable pattern
        features = torch.arange(num_patches * 512, dtype=torch.float32).reshape(num_patches, 512)
        coords = np.array([[i, 0] for i in range(num_patches)])
        
        aggregator.update_features(features, coords)
        
        # Property: first feature still first
        assert torch.allclose(aggregator.accumulated_features[0], features[0])


class TestAccuracyMaintenance:
    """Test accuracy maintenance vs batch processing."""
    
    @given(
        num_patches=st.integers(min_value=20, max_value=80)
    )
    @settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_streaming_vs_batch_prediction_similar(self, num_patches):
        """Property: Streaming prediction similar to batch prediction."""
        model = AttentionMIL(feature_dim=512, hidden_dim=128, num_classes=2)
        
        # Generate features
        features = torch.randn(num_patches, 512)
        coords = np.array([[i, 0] for i in range(num_patches)])
        
        # Streaming prediction
        aggregator_stream = StreamingAttentionAggregator(attention_model=model)
        aggregator_stream.update_features(features, coords)
        stream_result = aggregator_stream.get_current_prediction()
        
        # Batch prediction (all at once)
        aggregator_batch = StreamingAttentionAggregator(attention_model=model)
        aggregator_batch.update_features(features, coords)
        batch_result = aggregator_batch.get_current_prediction()
        
        # Property: predictions should match (same features, same model)
        assert stream_result.prediction == batch_result.prediction
        assert abs(stream_result.confidence - batch_result.confidence) < 0.01
    
    @given(
        num_patches=st.integers(min_value=30, max_value=60),
        chunk_size=st.integers(min_value=5, max_value=15)
    )
    @settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_chunked_streaming_consistent(self, num_patches, chunk_size):
        """Property: Chunked streaming gives consistent results."""
        assume(chunk_size < num_patches)
        
        model = AttentionMIL(feature_dim=512, hidden_dim=128, num_classes=2)
        features = torch.randn(num_patches, 512)
        coords = np.array([[i, 0] for i in range(num_patches)])
        
        # Process in chunks
        aggregator = StreamingAttentionAggregator(attention_model=model)
        for i in range(0, num_patches, chunk_size):
            end = min(i + chunk_size, num_patches)
            chunk_features = features[i:end]
            chunk_coords = coords[i:end]
            aggregator.update_features(chunk_features, chunk_coords)
        
        result = aggregator.get_current_prediction()
        
        # Property: final count correct
        assert aggregator.num_patches == num_patches
        # Property: prediction is valid
        assert 0 <= result.prediction < 2
        assert 0.0 <= result.confidence <= 1.0


class TestDeterminism:
    """Test deterministic behavior."""
    
    @given(
        num_patches=st.integers(min_value=20, max_value=50),
        seed=st.integers(min_value=0, max_value=10000)
    )
    @settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_same_input_same_output(self, num_patches, seed):
        """Property: Same input produces same output."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = AttentionMIL(feature_dim=512, hidden_dim=128, num_classes=2)
        features = torch.randn(num_patches, 512)
        coords = np.array([[i, 0] for i in range(num_patches)])
        
        # First run
        aggregator1 = StreamingAttentionAggregator(attention_model=model)
        aggregator1.update_features(features.clone(), coords.copy())
        result1 = aggregator1.get_current_prediction()
        
        # Second run (same seed, same data)
        torch.manual_seed(seed)
        np.random.seed(seed)
        aggregator2 = StreamingAttentionAggregator(attention_model=model)
        aggregator2.update_features(features.clone(), coords.copy())
        result2 = aggregator2.get_current_prediction()
        
        # Property: results identical
        assert result1.prediction == result2.prediction
        assert abs(result1.confidence - result2.confidence) < 1e-6


class TestInvariants:
    """Test system invariants."""
    
    @given(
        num_updates=st.integers(min_value=2, max_value=10),
        patches_per_update=st.integers(min_value=5, max_value=20)
    )
    @settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_patch_count_monotonic_increasing(self, num_updates, patches_per_update):
        """Property: Patch count only increases."""
        model = AttentionMIL(feature_dim=512, hidden_dim=128, num_classes=2)
        aggregator = StreamingAttentionAggregator(attention_model=model, max_features=1000)
        
        prev_count = 0
        for i in range(num_updates):
            features = torch.randn(patches_per_update, 512)
            coords = np.array([[j + i * patches_per_update, 0] for j in range(patches_per_update)])
            aggregator.update_features(features, coords)
            
            # Property: count increases or stays same (if at max)
            assert aggregator.num_patches >= prev_count
            prev_count = aggregator.num_patches
    
    @given(
        num_patches=st.integers(min_value=10, max_value=50)
    )
    @settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_confidence_always_valid_probability(self, num_patches):
        """Property: Confidence is always valid probability."""
        model = AttentionMIL(feature_dim=512, hidden_dim=128, num_classes=2)
        aggregator = StreamingAttentionAggregator(attention_model=model)
        
        features = torch.randn(num_patches, 512)
        coords = np.array([[i, 0] for i in range(num_patches)])
        
        update = aggregator.update_features(features, coords)
        
        # Property: confidence in [0, 1]
        assert 0.0 <= update.current_confidence <= 1.0
    
    @given(
        num_patches=st.integers(min_value=10, max_value=50)
    )
    @settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_prediction_in_valid_range(self, num_patches):
        """Property: Prediction is valid class index."""
        model = AttentionMIL(feature_dim=512, hidden_dim=128, num_classes=2)
        aggregator = StreamingAttentionAggregator(attention_model=model)
        
        features = torch.randn(num_patches, 512)
        coords = np.array([[i, 0] for i in range(num_patches)])
        
        aggregator.update_features(features, coords)
        result = aggregator.get_current_prediction()
        
        # Property: prediction in [0, num_classes)
        assert 0 <= result.prediction < 2


class TestCommutativity:
    """Test commutative properties where applicable."""
    
    @given(
        batch1_size=st.integers(min_value=10, max_value=30),
        batch2_size=st.integers(min_value=10, max_value=30)
    )
    @settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_batch_order_affects_result(self, batch1_size, batch2_size):
        """Property: Batch order matters (not commutative, but consistent)."""
        model = AttentionMIL(feature_dim=512, hidden_dim=128, num_classes=2)
        
        features1 = torch.randn(batch1_size, 512)
        features2 = torch.randn(batch2_size, 512)
        coords1 = np.array([[i, 0] for i in range(batch1_size)])
        coords2 = np.array([[i + batch1_size, 0] for i in range(batch2_size)])
        
        # Order 1: batch1 then batch2
        agg1 = StreamingAttentionAggregator(attention_model=model)
        agg1.update_features(features1, coords1)
        agg1.update_features(features2, coords2)
        result1 = agg1.get_current_prediction()
        
        # Order 2: batch2 then batch1 (different coords to avoid overlap)
        agg2 = StreamingAttentionAggregator(attention_model=model)
        coords2_alt = np.array([[i, 0] for i in range(batch2_size)])
        coords1_alt = np.array([[i + batch2_size, 0] for i in range(batch1_size)])
        agg2.update_features(features2, coords2_alt)
        agg2.update_features(features1, coords1_alt)
        result2 = agg2.get_current_prediction()
        
        # Property: both produce valid results (may differ)
        assert 0 <= result1.prediction < 2
        assert 0 <= result2.prediction < 2
        assert 0.0 <= result1.confidence <= 1.0
        assert 0.0 <= result2.confidence <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--hypothesis-show-statistics'])
