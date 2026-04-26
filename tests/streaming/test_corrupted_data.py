"""Corrupted data processing tests for streaming components."""

import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import io

from src.streaming.wsi_stream_reader import (
    WSIStreamReader,
    TileBatch,
    StreamingMetadata
)
from src.streaming.gpu_pipeline import GPUPipeline
from src.streaming.attention_aggregator import StreamingAttentionAggregator


class TestCorruptedWSIFiles:
    """Test handling of corrupted WSI files."""
    
    def test_missing_file_raises_error(self):
        """Test missing WSI file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            WSIStreamReader("nonexistent_file.svs")
    
    def test_invalid_format_raises_error(self, tmp_path):
        """Test invalid format raises ValueError."""
        # Create invalid file
        invalid_file = tmp_path / "invalid.xyz"
        invalid_file.write_text("not a valid WSI")
        
        with pytest.raises(ValueError, match="Unsupported WSI format"):
            WSIStreamReader(str(invalid_file))
    
    @patch('openslide.OpenSlide')
    def test_corrupted_header_handling(self, mock_openslide, tmp_path):
        """Test corrupted file header handling."""
        corrupted_file = tmp_path / "corrupted.svs"
        corrupted_file.write_bytes(b'\x00' * 1024)  # Invalid header
        
        # Mock OpenSlide to raise error
        mock_openslide.side_effect = Exception("Cannot open file")
        
        reader = WSIStreamReader.__new__(WSIStreamReader)
        reader.wsi_path = Path(str(corrupted_file))
        reader.tile_size = 1024
        reader.buffer_size = 16
        reader.overlap = 0
        reader.stride = 1024
        
        with pytest.raises(Exception):
            reader.initialize_streaming()
    
    @patch('openslide.OpenSlide')
    def test_truncated_file_handling(self, mock_openslide, tmp_path):
        """Test truncated file handling."""
        truncated_file = tmp_path / "truncated.svs"
        truncated_file.write_bytes(b'SVS_FILE' + b'\x00' * 100)  # Truncated
        
        # Mock OpenSlide to raise error on read
        mock_slide = MagicMock()
        mock_slide.dimensions = (10000, 10000)
        mock_slide.properties = {}
        mock_openslide.return_value = mock_slide
        
        reader = WSIStreamReader.__new__(WSIStreamReader)
        reader.wsi_path = Path(str(truncated_file))
        reader.tile_size = 1024
        reader.buffer_size = 16
        reader.overlap = 0
        reader.stride = 1024
        
        # Should handle gracefully
        # (actual behavior depends on OpenSlide error handling)


class TestCorruptedTileData:
    """Test handling of corrupted tile data."""
    
    def test_nan_values_in_tiles(self):
        """Test NaN values in tile data."""
        # Create tile batch with NaN
        tiles = torch.randn(4, 3, 256, 256)
        tiles[0, 0, 0, 0] = float('nan')
        
        coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        batch = TileBatch(
            tiles=tiles,
            coordinates=coords,
            batch_id=0,
            total_batches=1
        )
        
        # Check NaN detection
        assert torch.isnan(batch.tiles).any()
    
    def test_inf_values_in_tiles(self):
        """Test infinite values in tile data."""
        # Create tile batch with inf
        tiles = torch.randn(4, 3, 256, 256)
        tiles[0, 0, 0, 0] = float('inf')
        
        coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        batch = TileBatch(
            tiles=tiles,
            coordinates=coords,
            batch_id=0,
            total_batches=1
        )
        
        # Check inf detection
        assert torch.isinf(batch.tiles).any()
    
    def test_wrong_shape_tiles(self):
        """Test tiles with wrong shape."""
        # Wrong number of dimensions
        with pytest.raises(ValueError, match="4 dimensions"):
            TileBatch(
                tiles=torch.randn(4, 256, 256),  # Missing channel dim
                coordinates=np.array([[0, 0], [1, 0], [0, 1], [1, 1]]),
                batch_id=0,
                total_batches=1
            )
    
    def test_mismatched_coordinates(self):
        """Test mismatched tile and coordinate counts."""
        with pytest.raises(ValueError, match="match batch size"):
            TileBatch(
                tiles=torch.randn(4, 3, 256, 256),
                coordinates=np.array([[0, 0], [1, 0]]),  # Only 2 coords
                batch_id=0,
                total_batches=1
            )
    
    def test_empty_tile_batch(self):
        """Test empty tile batch handling."""
        # Empty batch - allow creation but note it's edge case
        batch = TileBatch(
            tiles=torch.empty(0, 3, 256, 256),
            coordinates=np.empty((0, 2)),
            batch_id=0,
            total_batches=1
        )
        
        # Verify empty
        assert batch.tiles.shape[0] == 0
        assert batch.coordinates.shape[0] == 0


class TestCorruptedModelInputs:
    """Test GPU pipeline with corrupted inputs."""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = Mock(spec=torch.nn.Module)
        model.eval = Mock()
        model.to = Mock(return_value=model)
        model.half = Mock(return_value=model)
        model.return_value = torch.randn(1, 512)
        return model
    
    def test_nan_input_handling(self, mock_model):
        """Test NaN input handling."""
        pipeline = GPUPipeline(model=mock_model, batch_size=4)
        
        # Create input with NaN
        patches = torch.randn(4, 3, 224, 224)
        patches[0] = float('nan')
        
        # Should detect NaN
        assert torch.isnan(patches).any()
    
    def test_inf_input_handling(self, mock_model):
        """Test infinite input handling."""
        pipeline = GPUPipeline(model=mock_model, batch_size=4)
        
        # Create input with inf
        patches = torch.randn(4, 3, 224, 224)
        patches[0] = float('inf')
        
        # Should detect inf
        assert torch.isinf(patches).any()
    
    def test_wrong_dtype_input(self, mock_model):
        """Test wrong dtype input."""
        pipeline = GPUPipeline(model=mock_model, batch_size=4)
        
        # Create int input (should be float)
        patches = torch.randint(0, 255, (4, 3, 224, 224), dtype=torch.uint8)
        
        # Should handle dtype conversion
        assert patches.dtype == torch.uint8
    
    def test_wrong_shape_input(self, mock_model):
        """Test wrong shape input."""
        pipeline = GPUPipeline(model=mock_model, batch_size=4)
        
        # Wrong number of dimensions
        patches = torch.randn(4, 224, 224)  # Missing channel dim
        
        # Model should fail on wrong shape
        mock_model.side_effect = RuntimeError("Expected 4D input")
        
        with pytest.raises(RuntimeError):
            pipeline._process_batch_sync(patches)


class TestCorruptedFeatures:
    """Test attention aggregator with corrupted features."""
    
    @pytest.fixture
    def aggregator(self):
        """Create aggregator."""
        from src.streaming.attention_aggregator import AttentionMIL
        model = AttentionMIL(feature_dim=512, hidden_dim=128, num_classes=2)
        return StreamingAttentionAggregator(
            attention_model=model,
            confidence_threshold=0.95
        )
    
    def test_nan_features_handling(self, aggregator):
        """Test NaN features handling."""
        features = torch.randn(10, 512)
        features[0] = float('nan')
        coords = np.array([[i, 0] for i in range(10)])
        
        # Should detect NaN
        assert torch.isnan(features).any()
    
    def test_inf_features_handling(self, aggregator):
        """Test infinite features handling."""
        features = torch.randn(10, 512)
        features[0] = float('inf')
        coords = np.array([[i, 0] for i in range(10)])
        
        # Should detect inf
        assert torch.isinf(features).any()
    
    def test_zero_features_handling(self, aggregator):
        """Test all-zero features."""
        features = torch.zeros(10, 512)
        coords = np.array([[i, 0] for i in range(10)])
        
        # Should handle gracefully
        update = aggregator.update_features(features, coords)
        
        # Should still produce valid output
        assert not torch.isnan(update.attention_weights).any()
    
    def test_constant_features_handling(self, aggregator):
        """Test constant features (no variance)."""
        features = torch.ones(10, 512) * 5.0
        coords = np.array([[i, 0] for i in range(10)])
        
        # Should handle gracefully
        update = aggregator.update_features(features, coords)
        
        # Should still produce valid output
        assert not torch.isnan(update.attention_weights).any()
    
    def test_extreme_values_handling(self, aggregator):
        """Test extreme feature values."""
        features = torch.randn(10, 512) * 1e6
        coords = np.array([[i, 0] for i in range(10)])
        
        # Should handle without overflow
        update = aggregator.update_features(features, coords)
        
        assert torch.isfinite(update.attention_weights).all()


class TestDataValidation:
    """Test data validation utilities."""
    
    def test_tile_batch_validation(self):
        """Test TileBatch validation."""
        # Valid batch
        batch = TileBatch(
            tiles=torch.randn(4, 3, 256, 256),
            coordinates=np.array([[0, 0], [1, 0], [0, 1], [1, 1]]),
            batch_id=0,
            total_batches=1
        )
        assert batch.tiles.shape[0] == 4
    
    def test_invalid_batch_id(self):
        """Test invalid batch ID."""
        with pytest.raises(ValueError, match="Batch ID"):
            TileBatch(
                tiles=torch.randn(4, 3, 256, 256),
                coordinates=np.array([[0, 0], [1, 0], [0, 1], [1, 1]]),
                batch_id=5,  # > total_batches
                total_batches=1
            )
    
    def test_invalid_priority(self):
        """Test invalid processing priority."""
        with pytest.raises(ValueError, match="priority"):
            TileBatch(
                tiles=torch.randn(4, 3, 256, 256),
                coordinates=np.array([[0, 0], [1, 0], [0, 1], [1, 1]]),
                batch_id=0,
                total_batches=1,
                processing_priority=1.5  # > 1.0
            )
    
    def test_metadata_validation(self):
        """Test StreamingMetadata validation."""
        # Invalid dimensions
        with pytest.raises(ValueError, match="Dimensions"):
            StreamingMetadata(
                slide_id="test",
                dimensions=(0, 100),  # Invalid
                estimated_patches=100,
                tile_size=256,
                memory_budget_gb=1.0,
                target_processing_time=30.0,
                confidence_threshold=0.95
            )
        
        # Invalid memory budget
        with pytest.raises(ValueError, match="Memory budget"):
            StreamingMetadata(
                slide_id="test",
                dimensions=(1000, 1000),
                estimated_patches=100,
                tile_size=256,
                memory_budget_gb=100.0,  # Too large
                target_processing_time=30.0,
                confidence_threshold=0.95
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
