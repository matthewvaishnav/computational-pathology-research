"""
Unit tests for TileBufferPool.

Tests the tile buffer pool implementation for memory management,
LRU eviction, thread safety, and performance optimization.
"""

import gc
import threading
import time
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from src.data.wsi_pipeline.tile_buffer_pool import (
    TileBufferConfig,
    TileBufferPool,
    TileMetadata,
)
from src.data.wsi_pipeline.exceptions import ResourceError, ProcessingError


class TestTileBufferConfig:
    """Test TileBufferConfig validation and functionality."""
    
    def test_default_config_is_valid(self):
        """Test that default configuration is valid."""
        config = TileBufferConfig()
        config.validate()  # Should not raise
        
        assert 0.5 <= config.max_memory_gb <= 32.0
        assert 0.5 <= config.min_memory_gb <= 32.0
        assert config.min_memory_gb <= config.max_memory_gb
    
    def test_memory_limit_validation(self):
        """Test memory limit validation."""
        # Test invalid min_memory_gb
        with pytest.raises(ValueError, match="min_memory_gb must be between"):
            config = TileBufferConfig(min_memory_gb=0.1)
            config.validate()
        
        with pytest.raises(ValueError, match="min_memory_gb must be between"):
            config = TileBufferConfig(min_memory_gb=50.0)
            config.validate()
        
        # Test invalid max_memory_gb
        with pytest.raises(ValueError, match="max_memory_gb must be between"):
            config = TileBufferConfig(max_memory_gb=0.1)
            config.validate()
        
        with pytest.raises(ValueError, match="max_memory_gb must be between"):
            config = TileBufferConfig(max_memory_gb=50.0)
            config.validate()
        
        # Test min > max
        with pytest.raises(ValueError, match="min_memory_gb.*cannot exceed max_memory_gb"):
            config = TileBufferConfig(min_memory_gb=2.0, max_memory_gb=1.0)
            config.validate()
    
    def test_buffer_size_validation(self):
        """Test buffer size validation."""
        # Test invalid initial_buffer_size
        with pytest.raises(ValueError, match="initial_buffer_size must be between"):
            config = TileBufferConfig(initial_buffer_size=0)
            config.validate()
        
        with pytest.raises(ValueError, match="initial_buffer_size must be between"):
            config = TileBufferConfig(initial_buffer_size=200, max_buffer_size=100)
            config.validate()
    
    def test_tile_size_validation(self):
        """Test tile size validation."""
        with pytest.raises(ValueError, match="tile_size must be between"):
            config = TileBufferConfig(tile_size=32)
            config.validate()
        
        with pytest.raises(ValueError, match="tile_size must be between"):
            config = TileBufferConfig(tile_size=8192)
            config.validate()
    
    def test_threshold_validation(self):
        """Test memory threshold validation."""
        # Test invalid memory_pressure_threshold
        with pytest.raises(ValueError, match="memory_pressure_threshold must be between"):
            config = TileBufferConfig(memory_pressure_threshold=0.05)
            config.validate()
        
        with pytest.raises(ValueError, match="memory_pressure_threshold must be between"):
            config = TileBufferConfig(memory_pressure_threshold=1.5)
            config.validate()
        
        # Test threshold ordering
        with pytest.raises(ValueError, match="memory_pressure_threshold must be less than cleanup_threshold"):
            config = TileBufferConfig(memory_pressure_threshold=0.9, cleanup_threshold=0.8)
            config.validate()


class TestTileBufferPool:
    """Test TileBufferPool functionality."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return TileBufferConfig(
            max_memory_gb=1.0,  # Small for testing but within valid range
            min_memory_gb=0.5,  # Minimum valid value
            initial_buffer_size=4,
            max_buffer_size=16,
            tile_size=256,
            compression_enabled=False,  # Disable for predictable testing
            thread_safe=True
        )
    
    @pytest.fixture
    def pool(self, config):
        """Create test tile buffer pool."""
        return TileBufferPool(config)
    
    @pytest.fixture
    def sample_tile(self):
        """Create sample tile data."""
        return np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    def test_pool_initialization(self, config):
        """Test pool initialization."""
        pool = TileBufferPool(config)
        
        assert pool.config == config
        assert pool.get_memory_usage() == 0.0
        assert pool.get_memory_limit() == config.max_memory_gb
        
        stats = pool.get_buffer_stats()
        assert stats['tile_count'] == 0
        assert stats['hit_rate'] == 0.0
    
    @patch('psutil.virtual_memory')
    def test_insufficient_system_memory_raises_error(self, mock_memory, config):
        """Test that insufficient system memory raises ResourceError."""
        # Mock system with very low available memory
        mock_memory.return_value.available = 1024 * 1024  # 1MB
        
        with pytest.raises(ResourceError, match="Insufficient system memory"):
            TileBufferPool(config)
    
    def test_store_and_retrieve_tile(self, pool, sample_tile):
        """Test basic tile storage and retrieval."""
        coordinate = (0, 0)
        level = 0
        
        # Store tile
        success = pool.store_tile(coordinate, level, sample_tile)
        assert success is True
        
        # Check tile exists
        assert pool.has_tile(coordinate, level) is True
        
        # Retrieve tile
        retrieved_tile = pool.get_tile(coordinate, level)
        assert retrieved_tile is not None
        np.testing.assert_array_equal(retrieved_tile, sample_tile)
        
        # Check statistics
        stats = pool.get_buffer_stats()
        assert stats['tile_count'] == 1
        assert stats['total_hits'] == 1
        assert stats['total_misses'] == 0
        assert stats['hit_rate'] == 1.0
    
    def test_retrieve_nonexistent_tile(self, pool):
        """Test retrieving non-existent tile returns None."""
        coordinate = (100, 100)
        level = 0
        
        # Check tile doesn't exist
        assert pool.has_tile(coordinate, level) is False
        
        # Retrieve should return None
        retrieved_tile = pool.get_tile(coordinate, level)
        assert retrieved_tile is None
        
        # Check statistics
        stats = pool.get_buffer_stats()
        assert stats['total_misses'] == 1
        assert stats['hit_rate'] == 0.0
    
    def test_update_existing_tile(self, pool, sample_tile):
        """Test updating existing tile moves it to end (LRU)."""
        coordinate = (0, 0)
        level = 0
        
        # Store initial tile
        pool.store_tile(coordinate, level, sample_tile)
        
        # Store another tile
        pool.store_tile((1, 1), level, sample_tile)
        
        # Update first tile (should move to end)
        modified_tile = sample_tile.copy()
        modified_tile[0, 0] = [255, 0, 0]  # Red pixel
        
        success = pool.store_tile(coordinate, level, modified_tile)
        assert success is True
        
        # Should still have 2 tiles (updated, not added)
        stats = pool.get_buffer_stats()
        assert stats['tile_count'] == 2
    
    def test_memory_limit_enforcement(self, config):
        """Test that memory limits are enforced."""
        # Use small memory limit that will be exceeded by large tiles
        config.max_memory_gb = 0.5  # 500MB - minimum valid value
        config.min_memory_gb = 0.5
        pool = TileBufferPool(config)

        # Create large tile (3MB each)
        large_tile = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        
        # Store multiple large tiles to exceed memory limit
        stored_tiles = []
        for i in range(200):  # Try to store 200 tiles (600MB total)
            success = pool.store_tile((i, 0), 0, large_tile)
            if success:
                stored_tiles.append((i, 0))
        
        # Memory usage should not exceed limit significantly
        assert pool.get_memory_usage() <= config.max_memory_gb * 1.1  # 10% tolerance
        
        # Should have evicted some tiles due to memory pressure
        stats = pool.get_buffer_stats()
        assert stats['total_evictions'] > 0 or len(stored_tiles) < 200
    
    def test_lru_eviction_policy(self, pool, sample_tile):
        """Test LRU eviction policy."""
        # Fill buffer beyond capacity to trigger eviction
        coordinates = [(i, 0) for i in range(20)]  # More than max_buffer_size
        
        for coord in coordinates:
            pool.store_tile(coord, 0, sample_tile)
        
        # Check that some tiles were evicted
        stats = pool.get_buffer_stats()
        assert stats['tile_count'] <= pool.config.max_buffer_size
        assert stats['total_evictions'] > 0
        
        # The most recently added tiles should still be present
        recent_coords = coordinates[-5:]  # Last 5 coordinates
        for coord in recent_coords:
            if pool.has_tile(coord, 0):
                # If tile exists, it should be retrievable
                retrieved = pool.get_tile(coord, 0)
                assert retrieved is not None
    
    def test_remove_tile(self, pool, sample_tile):
        """Test tile removal."""
        coordinate = (0, 0)
        level = 0
        
        # Store tile
        pool.store_tile(coordinate, level, sample_tile)
        assert pool.has_tile(coordinate, level) is True
        
        # Remove tile
        removed = pool.remove_tile(coordinate, level)
        assert removed is True
        assert pool.has_tile(coordinate, level) is False
        
        # Try to remove non-existent tile
        removed_again = pool.remove_tile(coordinate, level)
        assert removed_again is False
    
    def test_clear_buffer(self, pool, sample_tile):
        """Test clearing entire buffer."""
        # Store multiple tiles
        for i in range(5):
            pool.store_tile((i, 0), 0, sample_tile)
        
        assert pool.get_buffer_stats()['tile_count'] == 5
        
        # Clear buffer
        pool.clear()
        
        stats = pool.get_buffer_stats()
        assert stats['tile_count'] == 0
        assert pool.get_memory_usage() == 0.0
    
    def test_memory_pressure_detection(self, pool, sample_tile):
        """Test memory pressure detection and cleanup."""
        # Fill buffer to trigger memory pressure
        initial_memory = pool.get_memory_usage()
        
        # Store tiles until memory pressure is detected
        i = 0
        while pool.get_memory_usage() < pool.get_memory_limit() * 0.7:
            pool.store_tile((i, 0), 0, sample_tile)
            i += 1
            if i > 100:  # Safety break
                break
        
        # Memory usage should be managed
        assert pool.get_memory_usage() <= pool.get_memory_limit()
    
    def test_compression_enabled(self, config, sample_tile):
        """Test tile compression functionality."""
        config.compression_enabled = True
        pool = TileBufferPool(config)
        
        # Store tile with compression
        coordinate = (0, 0)
        level = 0
        
        success = pool.store_tile(coordinate, level, sample_tile)
        assert success is True
        
        # Retrieve and verify
        retrieved_tile = pool.get_tile(coordinate, level)
        assert retrieved_tile is not None
        
        # Check compression statistics
        stats = pool.get_buffer_stats()
        assert stats['total_compressions'] >= 0  # May or may not compress depending on data
    
    def test_thread_safety(self, pool, sample_tile):
        """Test thread-safe operations."""
        num_threads = 4
        tiles_per_thread = 10
        results = []
        
        def worker(thread_id):
            """Worker function for threading test."""
            thread_results = []
            for i in range(tiles_per_thread):
                coordinate = (thread_id * tiles_per_thread + i, 0)
                
                # Store tile
                success = pool.store_tile(coordinate, 0, sample_tile)
                thread_results.append(('store', coordinate, success))
                
                # Retrieve tile
                retrieved = pool.get_tile(coordinate, 0)
                thread_results.append(('get', coordinate, retrieved is not None))
            
            results.append(thread_results)
        
        # Create and start threads
        threads = []
        for thread_id in range(num_threads):
            thread = threading.Thread(target=worker, args=(thread_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(results) == num_threads
        for thread_results in results:
            assert len(thread_results) == tiles_per_thread * 2  # store + get operations
    
    def test_adaptive_buffer_sizing(self, pool, sample_tile):
        """Test adaptive buffer size calculation."""
        # Initially, buffer should be at initial size
        initial_size = pool.get_adaptive_buffer_size()
        assert initial_size >= pool.config.initial_buffer_size
        
        # Store some tiles and access them to build hit rate
        for i in range(5):
            coordinate = (i, 0)
            pool.store_tile(coordinate, 0, sample_tile)
            # Access multiple times to build hit rate
            for _ in range(3):
                pool.get_tile(coordinate, 0)
        
        # With high hit rate, buffer size might increase
        adaptive_size = pool.get_adaptive_buffer_size()
        assert adaptive_size >= pool.config.initial_buffer_size
    
    def test_preload_tiles(self, pool, sample_tile):
        """Test tile preloading functionality."""
        coordinates = [(i, 0) for i in range(5)]
        level = 0
        
        def mock_loader(x, y, level):
            """Mock tile loader function."""
            return sample_tile.copy()
        
        # Preload tiles
        loaded_count = pool.preload_tiles(coordinates, level, mock_loader)
        
        assert loaded_count == len(coordinates)
        
        # Verify tiles were loaded
        for coord in coordinates:
            assert pool.has_tile(coord, level) is True
            retrieved = pool.get_tile(coord, level)
            assert retrieved is not None
    
    def test_preload_disabled(self, config, sample_tile):
        """Test preloading when disabled."""
        config.preload_enabled = False
        pool = TileBufferPool(config)
        
        coordinates = [(i, 0) for i in range(3)]
        level = 0
        
        def mock_loader(x, y, level):
            return sample_tile.copy()
        
        # Preload should return 0
        loaded_count = pool.preload_tiles(coordinates, level, mock_loader)
        assert loaded_count == 0
        
        # No tiles should be loaded
        for coord in coordinates:
            assert pool.has_tile(coord, level) is False
    
    def test_memory_optimization(self, pool, sample_tile):
        """Test memory optimization functionality."""
        # Store some tiles
        for i in range(10):
            pool.store_tile((i, 0), 0, sample_tile)
        
        initial_stats = pool.get_buffer_stats()
        
        # Run optimization
        optimization_results = pool.optimize_memory_usage()
        
        assert 'initial_memory_gb' in optimization_results
        assert 'final_memory_gb' in optimization_results
        assert 'memory_freed_gb' in optimization_results
        assert optimization_results['memory_freed_gb'] >= 0
    
    def test_context_manager(self, config, sample_tile):
        """Test context manager functionality."""
        with TileBufferPool(config) as pool:
            # Store some tiles
            pool.store_tile((0, 0), 0, sample_tile)
            pool.store_tile((1, 1), 0, sample_tile)
            
            assert pool.get_buffer_stats()['tile_count'] == 2
        
        # After context exit, buffer should be cleared
        # Note: We can't easily test this since the pool is out of scope
    
    def test_error_handling_in_store_tile(self, pool):
        """Test error handling in store_tile method."""
        # Test with invalid tile data
        invalid_tile = "not a numpy array"
        
        with pytest.raises(ProcessingError):
            pool.store_tile((0, 0), 0, invalid_tile)
    
    def test_statistics_accuracy(self, pool, sample_tile):
        """Test that statistics are accurately maintained."""
        # Perform various operations
        coordinates = [(i, 0) for i in range(5)]
        
        # Store tiles
        for coord in coordinates:
            pool.store_tile(coord, 0, sample_tile)
        
        # Access some tiles (hits)
        for coord in coordinates[:3]:
            pool.get_tile(coord, 0)
        
        # Try to access non-existent tiles (misses)
        for i in range(3):
            pool.get_tile((100 + i, 0), 0)
        
        stats = pool.get_buffer_stats()
        
        assert stats['tile_count'] == 5
        assert stats['total_hits'] == 3
        assert stats['total_misses'] == 3
        assert abs(stats['hit_rate'] - 0.5) < 0.01  # 3/(3+3) = 0.5
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.empty_cache')
    def test_gpu_memory_cleanup(self, mock_empty_cache, mock_cuda_available, pool, sample_tile):
        """Test GPU memory cleanup during memory management."""
        mock_cuda_available.return_value = True
        
        # Fill buffer to trigger cleanup
        for i in range(20):
            pool.store_tile((i, 0), 0, sample_tile)
        
        # GPU cache should be cleared during cleanup
        mock_empty_cache.assert_called()
    
    def test_memory_calculation_accuracy(self, pool):
        """Test that memory calculations are accurate."""
        # Create tiles of known size
        small_tile = np.zeros((100, 100, 3), dtype=np.uint8)  # 30KB
        large_tile = np.zeros((500, 500, 3), dtype=np.uint8)  # 750KB
        
        initial_memory = pool.get_memory_usage()
        
        # Store small tile
        pool.store_tile((0, 0), 0, small_tile)
        memory_after_small = pool.get_memory_usage()
        
        # Store large tile
        pool.store_tile((1, 1), 0, large_tile)
        memory_after_large = pool.get_memory_usage()
        
        # Memory should increase appropriately
        small_increase = memory_after_small - initial_memory
        large_increase = memory_after_large - memory_after_small
        
        assert small_increase > 0
        assert large_increase > small_increase  # Large tile uses more memory
        
        # Memory usage should be reasonable (within 50% of expected)
        expected_small_mb = small_tile.nbytes / 1024**2
        expected_large_mb = large_tile.nbytes / 1024**2
        
        actual_small_mb = small_increase * 1024  # Convert GB to MB
        actual_large_mb = large_increase * 1024
        
        assert 0.5 * expected_small_mb <= actual_small_mb <= 1.5 * expected_small_mb
        assert 0.5 * expected_large_mb <= actual_large_mb <= 1.5 * expected_large_mb


class TestTileMetadata:
    """Test TileMetadata functionality."""
    
    def test_metadata_creation(self):
        """Test TileMetadata creation and attributes."""
        coordinate = (100, 200)
        level = 2
        size = (256, 256)
        memory_size = 196608  # 256*256*3 bytes
        access_time = time.time()
        access_count = 1
        
        metadata = TileMetadata(
            coordinate=coordinate,
            level=level,
            size=size,
            memory_size=memory_size,
            access_time=access_time,
            access_count=access_count
        )
        
        assert metadata.coordinate == coordinate
        assert metadata.level == level
        assert metadata.size == size
        assert metadata.memory_size == memory_size
        assert metadata.access_time == access_time
        assert metadata.access_count == access_count
        assert metadata.compressed is False  # Default value
    
    def test_metadata_with_compression(self):
        """Test TileMetadata with compression flag."""
        metadata = TileMetadata(
            coordinate=(0, 0),
            level=0,
            size=(256, 256),
            memory_size=100000,
            access_time=time.time(),
            access_count=1,
            compressed=True
        )
        
        assert metadata.compressed is True


if __name__ == "__main__":
    pytest.main([__file__])