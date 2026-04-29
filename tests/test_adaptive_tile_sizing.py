"""
Tests for adaptive tile sizing functionality in TileBufferPool and WSIStreamReader.

This module tests the adaptive tile sizing implementation that adjusts tile dimensions
based on available memory, GPU memory pressure, and system conditions.
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import psutil
import pytest
import torch

from src.data.wsi_pipeline.exceptions import ProcessingError, ResourceError
from src.data.wsi_pipeline.tile_buffer_pool import TileBufferConfig, TileBufferPool
from src.data.wsi_pipeline.wsi_stream_reader import (
    StreamingMetadata,
    StreamingProgress,
    WSIStreamReader,
)


class TestTileBufferConfigValidation:
    """Test TileBufferConfig validation with adaptive sizing parameters."""

    def test_valid_adaptive_sizing_config(self):
        """Test valid adaptive sizing configuration."""
        config = TileBufferConfig(
            adaptive_sizing_enabled=True,
            min_tile_size=64,
            max_tile_size=2048,
            tile_size=1024,
            memory_pressure_tile_reduction=0.75,
        )

        # Should not raise any exceptions
        config.validate()

        assert config.adaptive_sizing_enabled is True
        assert config.min_tile_size == 64
        assert config.max_tile_size == 2048
        assert config.tile_size == 1024
        assert config.memory_pressure_tile_reduction == 0.75

    def test_invalid_min_tile_size(self):
        """Test validation with invalid min_tile_size."""
        config = TileBufferConfig(min_tile_size=32)  # Below minimum

        with pytest.raises(ValueError, match="min_tile_size must be between 64 and 4096"):
            config.validate()

    def test_invalid_max_tile_size(self):
        """Test validation with invalid max_tile_size."""
        config = TileBufferConfig(max_tile_size=8192)  # Above maximum

        with pytest.raises(ValueError, match="max_tile_size must be between 64 and 4096"):
            config.validate()

    def test_min_greater_than_max_tile_size(self):
        """Test validation when min_tile_size > max_tile_size."""
        config = TileBufferConfig(min_tile_size=2048, max_tile_size=1024)

        with pytest.raises(ValueError, match="min_tile_size .* cannot exceed max_tile_size"):
            config.validate()

    def test_tile_size_out_of_bounds(self):
        """Test validation when tile_size is outside min/max bounds."""
        config = TileBufferConfig(tile_size=512, min_tile_size=1024, max_tile_size=2048)

        with pytest.raises(
            ValueError, match="tile_size .* must be between min_tile_size .* and max_tile_size"
        ):
            config.validate()

    def test_invalid_memory_pressure_reduction(self):
        """Test validation with invalid memory_pressure_tile_reduction."""
        config = TileBufferConfig(memory_pressure_tile_reduction=1.5)  # Above 1.0

        with pytest.raises(
            ValueError, match="memory_pressure_tile_reduction must be between 0.1 and 1.0"
        ):
            config.validate()


class TestAdaptiveTileSizing:
    """Test adaptive tile sizing functionality in TileBufferPool."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return TileBufferConfig(
            adaptive_sizing_enabled=True,
            min_tile_size=64,
            max_tile_size=2048,
            tile_size=1024,
            memory_pressure_tile_reduction=0.75,
            max_memory_gb=2.0,
        )

    @pytest.fixture
    def tile_pool(self, config):
        """Create test tile buffer pool."""
        return TileBufferPool(config)

    def test_adaptive_sizing_disabled(self):
        """Test that adaptive sizing can be disabled."""
        config = TileBufferConfig(adaptive_sizing_enabled=False)
        pool = TileBufferPool(config)

        # Should return default tile size
        adaptive_size = pool.calculate_adaptive_tile_size()
        assert adaptive_size == config.tile_size

        # Update should return False
        changed = pool.update_adaptive_tile_size()
        assert changed is False

    @patch("psutil.virtual_memory")
    def test_high_memory_increases_tile_size(self, mock_memory, tile_pool):
        """Test that high available memory increases tile size."""
        # Mock high memory availability (16GB available)
        mock_memory.return_value = Mock(available=16 * 1024**3, percent=30.0)

        adaptive_size = tile_pool.calculate_adaptive_tile_size()

        # Should increase tile size for high memory
        assert adaptive_size >= tile_pool.config.tile_size
        assert adaptive_size <= tile_pool.config.max_tile_size

    @patch("psutil.virtual_memory")
    def test_low_memory_decreases_tile_size(self, mock_memory, tile_pool):
        """Test that low available memory decreases tile size."""
        # Mock low memory availability (1GB available)
        mock_memory.return_value = Mock(available=1 * 1024**3, percent=85.0)

        adaptive_size = tile_pool.calculate_adaptive_tile_size()

        # Should decrease tile size for low memory
        assert adaptive_size <= tile_pool.config.tile_size
        assert adaptive_size >= tile_pool.config.min_tile_size

    @patch("psutil.virtual_memory")
    @patch("torch.cuda.is_available")
    @patch("torch.cuda.memory_allocated")
    @patch("torch.cuda.get_device_properties")
    def test_gpu_memory_pressure_reduces_tile_size(
        self, mock_props, mock_allocated, mock_cuda_available, mock_memory, tile_pool
    ):
        """Test that GPU memory pressure reduces tile size."""
        # Mock system memory (normal)
        mock_memory.return_value = Mock(available=8 * 1024**3, percent=50.0)

        # Mock GPU memory pressure
        mock_cuda_available.return_value = True
        mock_allocated.return_value = 7 * 1024**3  # 7GB used
        mock_props.return_value = Mock(total_memory=8 * 1024**3)  # 8GB total

        adaptive_size = tile_pool.calculate_adaptive_tile_size()

        # Should reduce tile size due to GPU memory pressure
        assert adaptive_size < tile_pool.config.tile_size

    def test_tile_size_power_of_two_alignment(self, tile_pool):
        """Test that adaptive tile size is aligned to power of 2."""
        # Test multiple scenarios
        with patch("psutil.virtual_memory") as mock_memory:
            mock_memory.return_value = Mock(available=4 * 1024**3, percent=60.0)

            adaptive_size = tile_pool.calculate_adaptive_tile_size()

            # Should be a power of 2
            assert adaptive_size & (adaptive_size - 1) == 0
            assert adaptive_size >= tile_pool.config.min_tile_size
            assert adaptive_size <= tile_pool.config.max_tile_size

    def test_update_adaptive_tile_size_tracking(self, tile_pool):
        """Test that tile size updates are properly tracked."""
        initial_size = tile_pool.get_current_tile_size()

        # Mock memory pressure to force size change
        with patch("psutil.virtual_memory") as mock_memory:
            mock_memory.return_value = Mock(available=1 * 1024**3, percent=90.0)  # Low memory

            changed = tile_pool.update_adaptive_tile_size()

            if changed:
                new_size = tile_pool.get_current_tile_size()
                assert new_size != initial_size

                # Check statistics
                stats = tile_pool.get_adaptive_sizing_stats()
                assert stats["total_adjustments"] > 0
                assert len(stats["recent_changes"]) > 0
                assert stats["current_tile_size"] == new_size

    def test_memory_estimation_functions(self, tile_pool):
        """Test memory estimation utility functions."""
        # Test tile memory estimation
        memory_bytes = tile_pool.estimate_tile_memory_usage(1024, channels=3, dtype_bytes=1)
        expected = 1024 * 1024 * 3 * 1  # 3MB for 1024x1024 RGB uint8
        assert memory_bytes == expected

        # Test optimal batch size calculation
        batch_size = tile_pool.get_optimal_batch_size_for_tile_size(1024, target_memory_gb=1.0)
        assert batch_size > 0
        assert isinstance(batch_size, int)

    def test_buffer_stats_include_adaptive_sizing(self, tile_pool):
        """Test that buffer statistics include adaptive sizing information."""
        stats = tile_pool.get_buffer_stats()

        assert "current_tile_size" in stats
        assert "adaptive_sizing_enabled" in stats
        assert "tile_size_adjustments" in stats

        assert stats["current_tile_size"] == tile_pool.get_current_tile_size()
        assert stats["adaptive_sizing_enabled"] == tile_pool.config.adaptive_sizing_enabled


class TestWSIStreamReaderAdaptiveSizing:
    """Test adaptive tile sizing in WSIStreamReader."""

    @pytest.fixture
    def mock_wsi_reader(self):
        """Create mock WSI reader."""
        mock_reader = Mock()
        mock_reader.dimensions = (10000, 10000)
        mock_reader.level_count = 3
        mock_reader.level_dimensions = [(10000, 10000), (5000, 5000), (2500, 2500)]
        mock_reader.get_magnification.return_value = 20.0
        mock_reader.get_mpp.return_value = (0.25, 0.25)
        mock_reader.read_region.return_value = np.random.randint(
            0, 255, (1024, 1024, 3), dtype=np.uint8
        )
        return mock_reader

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return TileBufferConfig(
            adaptive_sizing_enabled=True,
            min_tile_size=256,
            max_tile_size=2048,
            tile_size=1024,
            max_memory_gb=1.0,
        )

    def test_initialization_with_adaptive_sizing(self, config, mock_wsi_reader):
        """Test WSIStreamReader initialization with adaptive sizing."""
        with patch(
            "src.data.wsi_pipeline.wsi_stream_reader.WSIReader", return_value=mock_wsi_reader
        ):
            reader = WSIStreamReader("test.svs", config)

            assert reader.config.adaptive_sizing_enabled is True
            assert reader.tile_pool.config.adaptive_sizing_enabled is True

    def test_streaming_metadata_with_adaptive_sizing(self, config, mock_wsi_reader):
        """Test streaming metadata includes adaptive tile size information."""
        with patch(
            "src.data.wsi_pipeline.wsi_stream_reader.WSIReader", return_value=mock_wsi_reader
        ):
            reader = WSIStreamReader("test.svs", config)

            metadata = reader.initialize_streaming()

            assert isinstance(metadata, StreamingMetadata)
            assert metadata.tile_size == reader.tile_pool.get_current_tile_size()
            assert metadata.estimated_patches > 0

    @patch("psutil.virtual_memory")
    def test_memory_budget_enforcement(self, mock_memory, config, mock_wsi_reader):
        """Test that memory budget is enforced through adaptive sizing."""
        # Mock low memory to trigger size reduction
        mock_memory.return_value = Mock(available=0.5 * 1024**3, percent=90.0)  # 0.5GB available

        with patch(
            "src.data.wsi_pipeline.wsi_stream_reader.WSIReader", return_value=mock_wsi_reader
        ):
            reader = WSIStreamReader("test.svs", config)

            # Should reduce tile size to fit memory budget
            metadata = reader.initialize_streaming()

            # Tile size should be reduced from default
            assert metadata.tile_size <= config.tile_size
            assert metadata.tile_size >= config.min_tile_size

    def test_memory_budget_exceeded_raises_error(self, config, mock_wsi_reader):
        """Test that ResourceError is raised when memory budget cannot be met."""
        # Set very low memory budget (but still valid for config validation)
        config.max_memory_gb = 0.5  # Minimum valid value
        config.min_memory_gb = 0.5  # Set min equal to max to force tight constraints
        config.min_tile_size = 1024  # Force large minimum tile size
        config.max_tile_size = 1024  # Force large tile size that can't be reduced
        config.tile_size = 1024

        # Mock very large slide dimensions to exceed memory budget
        mock_wsi_reader.dimensions = (100000, 100000)  # Extremely large slide
        mock_wsi_reader.level_dimensions = [(100000, 100000)]

        with patch(
            "src.data.wsi_pipeline.wsi_stream_reader.WSIReader", return_value=mock_wsi_reader
        ):
            reader = WSIStreamReader("test.svs", config)

            with pytest.raises(
                ResourceError, match="Estimated memory usage .* exceeds maximum allowed"
            ):
                reader.initialize_streaming()

    def test_tile_size_adaptation_during_streaming(self, config, mock_wsi_reader):
        """Test that tile size can adapt during streaming."""
        with patch(
            "src.data.wsi_pipeline.wsi_stream_reader.WSIReader", return_value=mock_wsi_reader
        ):
            reader = WSIStreamReader("test.svs", config)
            metadata = reader.initialize_streaming()

            initial_tile_size = reader.tile_pool.get_current_tile_size()

            # Mock memory pressure during streaming
            with patch("psutil.virtual_memory") as mock_memory:
                mock_memory.return_value = Mock(available=0.5 * 1024**3, percent=95.0)  # Low memory

                # Process a few batches
                batch_count = 0
                for batch in reader.stream_tiles(batch_size=4):
                    batch_count += 1
                    if batch_count >= 2:  # Process enough batches to trigger adaptation
                        break

                # Tile size might have changed due to memory pressure
                final_tile_size = reader.tile_pool.get_current_tile_size()

                # Size should be within valid range
                assert config.min_tile_size <= final_tile_size <= config.max_tile_size

    def test_progress_tracking_with_adaptive_sizing(self, config, mock_wsi_reader):
        """Test that progress tracking works with adaptive tile sizing."""
        with patch(
            "src.data.wsi_pipeline.wsi_stream_reader.WSIReader", return_value=mock_wsi_reader
        ):
            reader = WSIStreamReader("test.svs", config)
            metadata = reader.initialize_streaming()

            # Process one batch
            for batch in reader.stream_tiles(batch_size=2):
                break

            progress = reader.get_progress()

            assert progress.tiles_processed > 0
            assert progress.total_tiles == metadata.estimated_patches
            assert 0.0 <= progress.progress_ratio <= 1.0
            assert progress.current_tile_size == reader.tile_pool.get_current_tile_size()
            assert progress.memory_usage_gb >= 0.0

    def test_adaptive_sizing_stats_access(self, config, mock_wsi_reader):
        """Test access to adaptive sizing statistics."""
        with patch(
            "src.data.wsi_pipeline.wsi_stream_reader.WSIReader", return_value=mock_wsi_reader
        ):
            reader = WSIStreamReader("test.svs", config)

            stats = reader.get_adaptive_sizing_stats()

            assert "adaptive_sizing_enabled" in stats
            assert "current_tile_size" in stats
            assert "recommended_tile_size" in stats
            assert stats["adaptive_sizing_enabled"] is True


class TestAdaptiveSizingIntegration:
    """Integration tests for adaptive tile sizing across components."""

    def test_end_to_end_adaptive_sizing_workflow(self):
        """Test complete workflow with adaptive sizing."""
        config = TileBufferConfig(
            adaptive_sizing_enabled=True,
            min_tile_size=256,
            max_tile_size=1024,
            tile_size=512,
            max_memory_gb=0.5,
        )

        # Create mock WSI reader
        mock_wsi_reader = Mock()
        mock_wsi_reader.dimensions = (2048, 2048)
        mock_wsi_reader.level_count = 1
        mock_wsi_reader.level_dimensions = [(2048, 2048)]
        mock_wsi_reader.get_magnification.return_value = 20.0
        mock_wsi_reader.get_mpp.return_value = (0.25, 0.25)

        def mock_read_region(location, level, size):
            return np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)

        mock_wsi_reader.read_region = mock_read_region

        with patch(
            "src.data.wsi_pipeline.wsi_stream_reader.WSIReader", return_value=mock_wsi_reader
        ):
            # Initialize reader
            reader = WSIStreamReader("test.svs", config)

            # Initialize streaming
            metadata = reader.initialize_streaming()

            # Verify adaptive sizing is working
            assert metadata.tile_size >= config.min_tile_size
            assert metadata.tile_size <= config.max_tile_size

            # Process tiles
            tiles_processed = 0
            for batch in reader.stream_tiles(batch_size=2):
                tiles_processed += len(batch.tiles)

                # Verify batch properties
                assert batch.tiles.shape[0] > 0  # Has tiles
                assert batch.tiles.shape[1] == batch.tiles.shape[2]  # Square tiles
                assert batch.coordinates.shape[0] == batch.tiles.shape[0]  # Matching coordinates

                if tiles_processed >= 10:  # Process enough for testing
                    break

            # Check final statistics
            stats = reader.get_adaptive_sizing_stats()
            progress = reader.get_progress()

            assert stats["current_tile_size"] >= config.min_tile_size
            assert stats["current_tile_size"] <= config.max_tile_size
            assert progress.tiles_processed == tiles_processed
            assert progress.memory_usage_gb <= config.max_memory_gb

            reader.close()

    def test_memory_pressure_response(self):
        """Test system response to memory pressure."""
        config = TileBufferConfig(
            adaptive_sizing_enabled=True,
            min_tile_size=128,
            max_tile_size=1024,
            tile_size=512,
            max_memory_gb=1.0,
            memory_pressure_threshold=0.7,
        )

        pool = TileBufferPool(config)

        # Simulate memory pressure by filling buffer
        large_tile = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

        # Store many tiles to create memory pressure
        for i in range(50):
            coord = (i * 512, 0)
            pool.store_tile(coord, 0, large_tile)

        # Check that memory optimization was triggered
        stats = pool.get_buffer_stats()

        # Memory usage should be managed
        assert stats["memory_usage_gb"] <= config.max_memory_gb

        # Adaptive sizing should have responded
        if stats["tile_size_adjustments"] > 0:
            assert stats["current_tile_size"] <= config.tile_size


if __name__ == "__main__":
    pytest.main([__file__])
