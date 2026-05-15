"""Tests for batch optimizer."""

import pytest

from src.streaming.memory.batch_optimizer import BatchOptimizer, OptimalSizes


class TestBatchOptimizer:
    """Tests for BatchOptimizer."""

    def test_init(self):
        """Test init."""
        opt = BatchOptimizer(available_memory_gb=4.0, safety_margin=0.8)

        assert opt.available_memory_gb == 4.0
        assert opt.safety_margin == 0.8
        assert opt.min_batch_size == 1
        assert opt.max_batch_size == 64

    def test_optimize_batch_size_small_tiles(self):
        """Test batch size optimization for small tiles."""
        opt = BatchOptimizer(available_memory_gb=2.0)

        batch_size = opt.optimize_batch_size(tile_size=224, feature_dim=512)

        assert batch_size >= opt.min_batch_size
        assert batch_size <= opt.max_batch_size
        assert batch_size & (batch_size - 1) == 0  # Power of 2

    def test_optimize_batch_size_large_tiles(self):
        """Test batch size optimization for large tiles."""
        opt = BatchOptimizer(available_memory_gb=2.0, max_batch_size=128)

        batch_small = opt.optimize_batch_size(tile_size=224)
        batch_large = opt.optimize_batch_size(tile_size=512)

        # Larger tiles → smaller or equal batch (may hit max)
        assert batch_large <= batch_small

    def test_optimize_batch_size_memory_constraint(self):
        """Test batch size respects memory constraint."""
        opt = BatchOptimizer(available_memory_gb=0.5)  # Very limited

        batch_size = opt.optimize_batch_size(tile_size=512, feature_dim=2048)

        # Should still return valid batch size
        assert batch_size >= opt.min_batch_size
        assert batch_size <= opt.max_batch_size

    def test_optimize_tile_size_high_mag(self):
        """Test tile size for high magnification."""
        opt = BatchOptimizer()

        tile_size = opt.optimize_tile_size(
            slide_dimensions=(50000, 50000), magnification=40.0, target_mpp=0.5
        )

        # High mag → larger tiles preferred
        assert tile_size in opt.tile_size_options
        assert tile_size >= 256

    def test_optimize_tile_size_low_mag(self):
        """Test tile size for low magnification."""
        opt = BatchOptimizer()

        tile_size = opt.optimize_tile_size(
            slide_dimensions=(50000, 50000), magnification=10.0, target_mpp=0.5
        )

        # Low mag → smaller tiles OK
        assert tile_size in opt.tile_size_options

    def test_optimize_for_workload(self):
        """Test full workload optimization."""
        opt = BatchOptimizer(available_memory_gb=4.0)

        result = opt.optimize_for_workload(
            slide_dimensions=(50000, 50000),
            magnification=20.0,
            target_mpp=0.5,
            feature_dim=512,
        )

        assert isinstance(result, OptimalSizes)
        assert result.batch_size >= opt.min_batch_size
        assert result.batch_size <= opt.max_batch_size
        assert result.tile_size in opt.tile_size_options
        assert result.estimated_memory_gb > 0
        assert result.throughput_estimate > 0

    def test_optimize_for_workload_with_patches(self):
        """Test workload optimization with explicit patch count."""
        opt = BatchOptimizer(available_memory_gb=4.0)

        result = opt.optimize_for_workload(
            slide_dimensions=(50000, 50000),
            magnification=20.0,
            estimated_patches=5000,
        )

        assert result.batch_size > 0
        assert result.tile_size > 0

    def test_adjust_for_oom_reduce_batch(self):
        """Test OOM adjustment reduces batch size."""
        opt = BatchOptimizer()

        new_batch, new_tile = opt.adjust_for_oom(current_batch_size=32, current_tile_size=224)

        # Should halve batch size
        assert new_batch == 16
        assert new_tile == 224  # Tile unchanged

    def test_adjust_for_oom_min_batch(self):
        """Test OOM adjustment at minimum batch."""
        opt = BatchOptimizer(min_batch_size=1)

        new_batch, new_tile = opt.adjust_for_oom(current_batch_size=1, current_tile_size=512)

        # Batch at min, should reduce tile
        assert new_batch == 1
        assert new_tile < 512

    def test_adjust_for_oom_min_both(self):
        """Test OOM adjustment at minimum batch and tile."""
        opt = BatchOptimizer(min_batch_size=1)

        new_batch, new_tile = opt.adjust_for_oom(current_batch_size=1, current_tile_size=96)

        # Both at minimum
        assert new_batch == 1
        assert new_tile == 96  # Can't reduce further

    def test_get_memory_estimate(self):
        """Test memory estimation."""
        opt = BatchOptimizer()

        mem_gb = opt.get_memory_estimate(
            batch_size=16, tile_size=224, feature_dim=512, num_patches=1000
        )

        assert mem_gb > 0
        assert mem_gb < 10  # Reasonable range

    def test_get_memory_estimate_scales(self):
        """Test memory estimate scales with batch size."""
        opt = BatchOptimizer()

        mem_small = opt.get_memory_estimate(
            batch_size=8, tile_size=224, feature_dim=512, num_patches=1000
        )

        mem_large = opt.get_memory_estimate(
            batch_size=32, tile_size=224, feature_dim=512, num_patches=1000
        )

        # Larger batch → more memory
        assert mem_large > mem_small

    def test_batch_size_power_of_two(self):
        """Test batch sizes are powers of 2."""
        opt = BatchOptimizer(available_memory_gb=8.0)

        for tile_size in [96, 224, 512]:
            batch_size = opt.optimize_batch_size(tile_size=tile_size)

            # Check power of 2
            assert batch_size > 0
            assert batch_size & (batch_size - 1) == 0

    def test_safety_margin_effect(self):
        """Test safety margin reduces batch size."""
        opt_conservative = BatchOptimizer(available_memory_gb=4.0, safety_margin=0.5)
        opt_aggressive = BatchOptimizer(available_memory_gb=4.0, safety_margin=0.9)

        batch_conservative = opt_conservative.optimize_batch_size(tile_size=224)
        batch_aggressive = opt_aggressive.optimize_batch_size(tile_size=224)

        # Higher margin → larger batch
        assert batch_aggressive >= batch_conservative

    def test_repr(self):
        """Test string representation."""
        opt = BatchOptimizer(available_memory_gb=4.0, safety_margin=0.8)

        repr_str = repr(opt)

        assert "BatchOptimizer" in repr_str
        assert "4.0" in repr_str or "4.00" in repr_str
        assert "0.8" in repr_str or "0.80" in repr_str
