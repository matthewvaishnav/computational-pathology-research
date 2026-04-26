"""Tests for WSI Stream Reader."""

import pytest
import numpy as np
import torch
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from PIL import Image

from src.streaming.wsi_stream_reader import (
    WSIStreamReader, StreamingMetadata, StreamingProgress, TileBatch,
    TileBufferPool, get_supported_formats, validate_wsi_format_compat
)


class TestTileBufferPool:
    """Test tile buffer pool functionality."""
    
    def test_buffer_pool_initialization(self):
        """Test buffer pool initialization."""
        pool = TileBufferPool(max_memory_gb=0.1, tile_size=256)
        assert pool.max_memory_bytes == int(0.1 * 1024**3)
        assert pool.tile_size == 256
        assert pool.current_memory > 0  # Should have pre-allocated buffers
    
    def test_buffer_allocation_and_return(self):
        """Test buffer allocation and return."""
        pool = TileBufferPool(max_memory_gb=0.1, tile_size=256)
        
        # Get buffer
        buffer = pool.get_buffer()
        assert buffer is not None
        assert buffer.shape == (256, 256, 3)
        
        # Return buffer
        pool.return_buffer(buffer)
        
        # Should be able to get it back
        buffer2 = pool.get_buffer()
        assert buffer2 is not None
    
    def test_memory_limit_enforcement(self):
        """Test that memory limits are enforced."""
        # Very small memory limit
        pool = TileBufferPool(max_memory_gb=0.001, tile_size=1024)
        
        # Should eventually return None when memory limit reached
        buffers = []
        for _ in range(100):  # Try to allocate many buffers
            buffer = pool.get_buffer()
            if buffer is None:
                break
            buffers.append(buffer)
        
        # Should have hit memory limit
        assert pool.get_buffer() is None


class TestStreamingMetadata:
    """Test streaming metadata validation."""
    
    def test_valid_metadata(self):
        """Test valid metadata creation."""
        metadata = StreamingMetadata(
            slide_id="test_slide",
            dimensions=(10000, 8000),
            estimated_patches=1000,
            tile_size=1024,
            memory_budget_gb=2.0,
            target_processing_time=30.0,
            confidence_threshold=0.95
        )
        assert metadata.slide_id == "test_slide"
        assert metadata.dimensions == (10000, 8000)
    
    def test_invalid_dimensions(self):
        """Test validation of invalid dimensions."""
        with pytest.raises(ValueError, match="Dimensions must be positive"):
            StreamingMetadata(
                slide_id="test",
                dimensions=(0, 1000),
                estimated_patches=100,
                tile_size=1024,
                memory_budget_gb=1.0,
                target_processing_time=30.0,
                confidence_threshold=0.95
            )
    
    def test_invalid_memory_budget(self):
        """Test validation of invalid memory budget."""
        with pytest.raises(ValueError, match="Memory budget must be between"):
            StreamingMetadata(
                slide_id="test",
                dimensions=(1000, 1000),
                estimated_patches=100,
                tile_size=1024,
                memory_budget_gb=50.0,  # Too large
                target_processing_time=30.0,
                confidence_threshold=0.95
            )


class TestTileBatch:
    """Test tile batch validation."""
    
    def test_valid_tile_batch(self):
        """Test valid tile batch creation."""
        tiles = torch.randn(4, 3, 256, 256)
        coordinates = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        
        batch = TileBatch(
            tiles=tiles,
            coordinates=coordinates,
            batch_id=1,
            total_batches=10
        )
        
        assert batch.tiles.shape == (4, 3, 256, 256)
        assert batch.coordinates.shape == (4, 2)
    
    def test_mismatched_batch_size(self):
        """Test validation of mismatched batch sizes."""
        tiles = torch.randn(4, 3, 256, 256)
        coordinates = np.array([[0, 0], [1, 0]])  # Wrong size
        
        with pytest.raises(ValueError, match="Coordinates must match batch size"):
            TileBatch(
                tiles=tiles,
                coordinates=coordinates,
                batch_id=1,
                total_batches=10
            )
    
    def test_invalid_batch_id(self):
        """Test validation of invalid batch ID."""
        tiles = torch.randn(2, 3, 256, 256)
        coordinates = np.array([[0, 0], [1, 0]])
        
        with pytest.raises(ValueError, match="Batch ID must be <= total batches"):
            TileBatch(
                tiles=tiles,
                coordinates=coordinates,
                batch_id=15,
                total_batches=10
            )


class TestWSIStreamReader:
    """Test WSI Stream Reader functionality."""
    
    @patch('src.streaming.wsi_stream_reader.validate_wsi_format')
    @patch('pathlib.Path.exists')
    def test_initialization_valid_file(self, mock_exists, mock_validate):
        """Test initialization with valid file."""
        mock_exists.return_value = True
        mock_validate.return_value = True
        
        reader = WSIStreamReader("test.svs", tile_size=512, buffer_size=8)
        assert reader.tile_size == 512
        assert reader.buffer_size == 8
        assert reader.wsi_path.name == "test.svs"
    
    @patch('pathlib.Path.exists')
    def test_initialization_missing_file(self, mock_exists):
        """Test initialization with missing file."""
        mock_exists.return_value = False
        
        with pytest.raises(FileNotFoundError, match="WSI file not found"):
            WSIStreamReader("missing.svs")
    
    @patch('src.streaming.wsi_stream_reader.validate_wsi_format')
    @patch('pathlib.Path.exists')
    def test_initialization_unsupported_format(self, mock_exists, mock_validate):
        """Test initialization with unsupported format."""
        mock_exists.return_value = True
        mock_validate.return_value = False
        
        with pytest.raises(ValueError, match="Unsupported WSI format"):
            WSIStreamReader("test.xyz")
    
    @patch('src.streaming.wsi_stream_reader.get_wsi_handler')
    @patch('src.streaming.wsi_stream_reader.validate_wsi_format')
    @patch('pathlib.Path.exists')
    def test_initialize_streaming_success(self, mock_exists, mock_validate, mock_handler):
        """Test successful streaming initialization."""
        mock_exists.return_value = True
        mock_validate.return_value = True
        
        # Mock slide object
        mock_slide = Mock()
        mock_slide.dimensions = (20000, 15000)
        mock_slide.properties = {
            'openslide.objective-power': '20',
            'openslide.vendor': 'aperio'
        }
        
        # Mock handler
        mock_handler_obj = Mock()
        mock_handler_obj.open_slide.return_value = mock_slide
        mock_handler_obj.get_dimensions.return_value = (20000, 15000)
        mock_handler_obj.get_properties.return_value = {
            'openslide.objective-power': '20',
            'openslide.vendor': 'aperio'
        }
        mock_handler.return_value = mock_handler_obj
        
        # Mock DeepZoom
        with patch('src.streaming.wsi_stream_reader.DeepZoomGenerator') as mock_dz:
            mock_dz_obj = Mock()
            mock_dz_obj.level_count = 5
            mock_dz_obj.level_dimensions = [(1250, 938), (2500, 1875), (5000, 3750), (10000, 7500), (20000, 15000)]
            mock_dz_obj.level_tiles = [(1, 1), (2, 2), (4, 4), (8, 8), (16, 12)]
            mock_dz.return_value = mock_dz_obj
            
            reader = WSIStreamReader("test.svs")
            metadata = reader.initialize_streaming()
            
            assert metadata.slide_id == "test"
            assert metadata.dimensions == (20000, 15000)
            assert metadata.estimated_patches == 16 * 12  # 192
            assert metadata.magnification == 20.0
            assert metadata.vendor == "aperio"
    
    def test_estimate_total_patches_no_metadata(self):
        """Test patch estimation without metadata."""
        with patch('pathlib.Path.exists', return_value=True), \
             patch('src.streaming.wsi_stream_reader.validate_wsi_format', return_value=True):
            reader = WSIStreamReader("test.svs")
            
            # Should return default when no metadata
            patches = reader.estimate_total_patches()
            assert patches == 1000  # Default fallback
    
    @patch('src.streaming.wsi_stream_reader.OpenSlide')
    @patch('pathlib.Path.exists')
    def test_estimate_total_patches_with_slide(self, mock_exists, mock_openslide):
        """Test patch estimation with slide dimensions."""
        mock_exists.return_value = True
        
        # Mock slide
        mock_slide = Mock()
        mock_slide.dimensions = (10240, 10240)  # 10K x 10K
        mock_openslide.return_value.__enter__.return_value = mock_slide
        
        with patch('src.streaming.wsi_stream_reader.validate_wsi_format', return_value=True):
            reader = WSIStreamReader("test.svs", tile_size=1024)
            patches = reader.estimate_total_patches()
            
            # Should estimate based on slide area / tile area
            expected = (10240 * 10240) // (1024 * 1024)  # 100 patches
            assert patches == expected
    
    def test_progress_tracking(self):
        """Test progress tracking functionality."""
        with patch('pathlib.Path.exists', return_value=True), \
             patch('src.streaming.wsi_stream_reader.validate_wsi_format', return_value=True):
            reader = WSIStreamReader("test.svs")
            reader.total_patches = 1000
            reader.patches_processed = 250
            reader.start_time = time.time() - 10.0  # 10 seconds ago
            
            progress = reader.get_progress()
            
            assert progress.patches_processed == 250
            assert progress.total_patches == 1000
            assert progress.progress_percentage == 25.0
            assert progress.elapsed_time >= 10.0
            assert progress.throughput_patches_per_sec > 0
    
    def test_memory_pressure_detection(self):
        """Test memory pressure detection."""
        with patch('pathlib.Path.exists', return_value=True), \
             patch('src.streaming.wsi_stream_reader.validate_wsi_format', return_value=True):
            reader = WSIStreamReader("test.svs")
            
            # Mock psutil
            with patch('psutil.virtual_memory') as mock_memory:
                mock_memory.return_value.percent = 85.0
                
                pressure = reader._check_memory_pressure()
                assert pressure is True
                assert reader.memory_pressure is True
    
    def test_memory_pressure_adaptation(self):
        """Test adaptation to memory pressure."""
        with patch('pathlib.Path.exists', return_value=True), \
             patch('src.streaming.wsi_stream_reader.validate_wsi_format', return_value=True):
            reader = WSIStreamReader("test.svs", tile_size=1024, buffer_size=16)
            reader.memory_pressure = True
            
            original_tile_size = reader.current_tile_size
            original_buffer_size = reader.buffer_size
            
            reader._adapt_to_memory_pressure()
            
            # Should reduce both tile size and buffer size
            assert reader.current_tile_size < original_tile_size
            assert reader.buffer_size < original_buffer_size
    
    def test_background_tile_detection(self):
        """Test background tile detection."""
        with patch('pathlib.Path.exists', return_value=True), \
             patch('src.streaming.wsi_stream_reader.validate_wsi_format', return_value=True):
            reader = WSIStreamReader("test.svs")
            
            # Create white (background) tile
            white_tile = Image.new('RGB', (256, 256), color=(255, 255, 255))
            assert reader._is_background_tile(white_tile) is True
            
            # Create non-white tile
            colored_tile = Image.new('RGB', (256, 256), color=(100, 150, 200))
            assert reader._is_background_tile(colored_tile) is False
    
    def test_context_manager(self):
        """Test context manager functionality."""
        with patch('pathlib.Path.exists', return_value=True), \
             patch('src.streaming.wsi_stream_reader.validate_wsi_format', return_value=True):
            
            with WSIStreamReader("test.svs") as reader:
                assert reader is not None
            
            # Should have cleaned up (no exception)


class TestFormatSupport:
    """Test format support functions."""
    
    def test_get_supported_formats(self):
        """Test getting supported formats."""
        formats = get_supported_formats()
        assert isinstance(formats, list)
        assert len(formats) > 0
        assert '.svs' in formats
        assert '.tiff' in formats
    
    @patch('src.streaming.wsi_stream_reader.validate_wsi_format')
    def test_validate_wsi_format_compat(self, mock_validate):
        """Test format validation compatibility function."""
        mock_validate.return_value = True
        
        result = validate_wsi_format_compat("test.svs")
        assert result is True
        mock_validate.assert_called_once_with("test.svs")


if __name__ == "__main__":
    pytest.main([__file__])