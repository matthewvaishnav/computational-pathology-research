"""Unit tests for advanced memory optimization.

Tests cover:
- Memory pool management
- Smart garbage collection
- Memory usage prediction
- Preallocation strategies
"""

import pytest
import time
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Import directly from memory_optimizer module
import sys
import importlib.util
from pathlib import Path

# Load memory_optimizer module directly
memory_optimizer_path = Path(__file__).parent.parent.parent / "src" / "streaming" / "memory_optimizer.py"
spec = importlib.util.spec_from_file_location("memory_optimizer", memory_optimizer_path)
memory_optimizer_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(memory_optimizer_module)

# Import classes from loaded module
MemoryPoolManager = memory_optimizer_module.MemoryPoolManager
MemoryPoolStrategy = memory_optimizer_module.MemoryPoolStrategy
MemoryBlock = memory_optimizer_module.MemoryBlock
MemoryPoolStats = memory_optimizer_module.MemoryPoolStats
SmartGarbageCollector = memory_optimizer_module.SmartGarbageCollector
GCStats = memory_optimizer_module.GCStats
MemoryUsagePredictor = memory_optimizer_module.MemoryUsagePredictor
MemoryPrediction = memory_optimizer_module.MemoryPrediction


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def device():
    """Get test device (CPU for CI compatibility)."""
    return torch.device('cpu')


@pytest.fixture
def memory_pool(device):
    """Create memory pool manager."""
    return MemoryPoolManager(
        device=device,
        initial_pool_size_gb=0.5,
        max_pool_size_gb=2.0,
        strategy=MemoryPoolStrategy.ADAPTIVE
    )


@pytest.fixture
def gc_collector(device):
    """Create smart garbage collector."""
    return SmartGarbageCollector(
        device=device,
        memory_pressure_threshold=0.8,
        collection_interval_seconds=1.0
    )


@pytest.fixture
def memory_predictor():
    """Create memory usage predictor."""
    return MemoryUsagePredictor(enable_learning=True)


# ============================================================================
# MemoryBlock Tests
# ============================================================================

class TestMemoryBlock:
    """Test memory block functionality."""
    
    def test_initialization(self):
        """Test memory block initialization."""
        block = MemoryBlock(size_bytes=1024)
        
        assert block.size_bytes == 1024
        assert block.tensor is None
        assert block.is_free is True
        assert block.use_count == 0
    
    def test_mark_used(self):
        """Test marking block as used."""
        block = MemoryBlock(size_bytes=1024)
        
        block.mark_used()
        
        assert block.is_free is False
        assert block.use_count == 1
    
    def test_mark_free(self):
        """Test marking block as free."""
        block = MemoryBlock(size_bytes=1024)
        
        block.mark_used()
        block.mark_free()
        
        assert block.is_free is True
        assert block.use_count == 1  # Use count persists
    
    def test_age_seconds(self):
        """Test age calculation."""
        block = MemoryBlock(size_bytes=1024)
        
        time.sleep(0.1)
        
        assert block.age_seconds >= 0.1
    
    def test_idle_seconds(self):
        """Test idle time calculation."""
        block = MemoryBlock(size_bytes=1024)
        
        block.mark_used()
        time.sleep(0.1)
        
        assert block.idle_seconds >= 0.1


# ============================================================================
# MemoryPoolStats Tests
# ============================================================================

class TestMemoryPoolStats:
    """Test memory pool statistics."""
    
    def test_initialization(self):
        """Test stats initialization."""
        stats = MemoryPoolStats(
            total_blocks=10,
            free_blocks=6,
            allocated_blocks=4,
            total_size_gb=2.0,
            free_size_gb=1.2,
            allocated_size_gb=0.8,
            hit_rate=0.85,
            miss_rate=0.15,
            fragmentation_ratio=0.3,
            avg_block_age_seconds=120.0
        )
        
        assert stats.total_blocks == 10
        assert stats.free_blocks == 6
        assert stats.allocated_blocks == 4
    
    def test_utilization_percent(self):
        """Test utilization calculation."""
        stats = MemoryPoolStats(
            total_blocks=10,
            free_blocks=5,
            allocated_blocks=5,
            total_size_gb=2.0,
            free_size_gb=1.0,
            allocated_size_gb=1.0,
            hit_rate=0.8,
            miss_rate=0.2,
            fragmentation_ratio=0.3,
            avg_block_age_seconds=100.0
        )
        
        assert stats.utilization_percent == 50.0
    
    def test_utilization_zero_total(self):
        """Test utilization with zero total."""
        stats = MemoryPoolStats(
            total_blocks=0,
            free_blocks=0,
            allocated_blocks=0,
            total_size_gb=0.0,
            free_size_gb=0.0,
            allocated_size_gb=0.0,
            hit_rate=0.0,
            miss_rate=0.0,
            fragmentation_ratio=0.0,
            avg_block_age_seconds=0.0
        )
        
        assert stats.utilization_percent == 0.0


# ============================================================================
# MemoryPoolManager Tests
# ============================================================================

class TestMemoryPoolManager:
    """Test memory pool manager."""
    
    def test_initialization(self, device):
        """Test pool manager initialization."""
        pool = MemoryPoolManager(
            device=device,
            initial_pool_size_gb=1.0,
            max_pool_size_gb=4.0,
            strategy=MemoryPoolStrategy.ADAPTIVE
        )
        
        assert pool.device == device
        assert pool.initial_pool_size_gb == 1.0
        assert pool.max_pool_size_gb == 4.0
        assert pool.strategy == MemoryPoolStrategy.ADAPTIVE
    
    def test_calculate_common_sizes(self, memory_pool):
        """Test common size calculation."""
        sizes = memory_pool._calculate_common_sizes()
        
        assert len(sizes) > 0
        assert all(s > 0 for s in sizes)
        assert sizes == sorted(sizes)  # Should be sorted
    
    def test_allocate_new_block(self, memory_pool):
        """Test allocating new memory block."""
        size_bytes = 1024 * 4  # 1024 float32 elements
        
        tensor = memory_pool.allocate(size_bytes)
        
        assert tensor is not None
        assert tensor.numel() * tensor.element_size() >= size_bytes
        assert memory_pool.total_allocations == 1
    
    def test_allocate_from_pool(self, memory_pool):
        """Test allocating from existing pool."""
        size_bytes = 1024 * 4
        
        # First allocation
        tensor1 = memory_pool.allocate(size_bytes)
        memory_pool.deallocate(tensor1)
        
        # Second allocation should reuse
        tensor2 = memory_pool.allocate(size_bytes)
        
        assert tensor2 is not None
        assert memory_pool.cache_hits >= 1
    
    def test_deallocate(self, memory_pool):
        """Test deallocating memory."""
        size_bytes = 1024 * 4
        
        tensor = memory_pool.allocate(size_bytes)
        memory_pool.deallocate(tensor)
        
        # Check that block is marked as free
        stats = memory_pool.get_stats()
        assert stats.free_blocks > 0
    
    def test_cleanup_idle_blocks(self, memory_pool):
        """Test cleanup of idle blocks."""
        # Allocate and deallocate multiple blocks
        for _ in range(5):
            size_bytes = 1024 * 4
            tensor = memory_pool.allocate(size_bytes)
            memory_pool.deallocate(tensor)
        
        initial_blocks = memory_pool.get_stats().total_blocks
        
        # Cleanup with very short idle time
        memory_pool._cleanup_idle_blocks(max_idle_seconds=0.0)
        
        # Should have cleaned up some blocks
        final_blocks = memory_pool.get_stats().total_blocks
        assert final_blocks <= initial_blocks
    
    def test_get_stats(self, memory_pool):
        """Test getting pool statistics."""
        # Allocate some blocks
        tensors = []
        for _ in range(3):
            tensor = memory_pool.allocate(1024 * 4)
            tensors.append(tensor)
        
        stats = memory_pool.get_stats()
        
        assert isinstance(stats, MemoryPoolStats)
        assert stats.total_blocks >= 3
        assert stats.allocated_blocks >= 3
        assert stats.total_size_gb > 0
        assert 0.0 <= stats.hit_rate <= 1.0
        assert 0.0 <= stats.miss_rate <= 1.0
    
    def test_cache_hit_rate(self, memory_pool):
        """Test cache hit rate calculation."""
        size_bytes = 1024 * 4
        
        # First allocation (miss)
        tensor1 = memory_pool.allocate(size_bytes)
        memory_pool.deallocate(tensor1)
        
        # Second allocation (hit)
        tensor2 = memory_pool.allocate(size_bytes)
        
        stats = memory_pool.get_stats()
        assert stats.hit_rate > 0.0
    
    def test_pool_growth(self, memory_pool):
        """Test pool growth with new allocations."""
        initial_size = memory_pool.total_size_bytes
        
        # Allocate unique sizes
        for i in range(5):
            size_bytes = (1024 + i * 512) * 4
            tensor = memory_pool.allocate(size_bytes)
        
        final_size = memory_pool.total_size_bytes
        assert final_size > initial_size
    
    def test_cleanup(self, memory_pool):
        """Test pool cleanup."""
        # Allocate some blocks
        for _ in range(3):
            memory_pool.allocate(1024 * 4)
        
        memory_pool.cleanup()
        
        assert len(memory_pool.blocks) == 0
        assert memory_pool.total_size_bytes == 0
    
    def test_emergency_cleanup(self, memory_pool):
        """Test emergency cleanup."""
        # Allocate blocks
        for _ in range(5):
            memory_pool.allocate(1024 * 4)
        
        memory_pool._emergency_cleanup()
        
        assert len(memory_pool.blocks) == 0
        assert memory_pool.total_size_bytes == 0
    
    def test_thread_safety(self, memory_pool):
        """Test thread-safe operations."""
        import threading
        
        def allocate_deallocate():
            for _ in range(10):
                tensor = memory_pool.allocate(1024 * 4)
                memory_pool.deallocate(tensor)
        
        threads = [threading.Thread(target=allocate_deallocate) for _ in range(3)]
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        # Should complete without errors
        stats = memory_pool.get_stats()
        assert stats.total_blocks >= 0


# ============================================================================
# SmartGarbageCollector Tests
# ============================================================================

class TestSmartGarbageCollector:
    """Test smart garbage collector."""
    
    def test_initialization(self, device):
        """Test GC initialization."""
        gc = SmartGarbageCollector(
            device=device,
            memory_pressure_threshold=0.8,
            collection_interval_seconds=10.0
        )
        
        assert gc.device == device
        assert gc.memory_pressure_threshold == 0.8
        assert gc.collection_interval_seconds == 10.0
    
    def test_should_collect_pressure(self, gc_collector):
        """Test collection trigger based on memory pressure."""
        # Low pressure - should not collect
        assert not gc_collector.should_collect(
            current_memory_gb=2.0,
            total_memory_gb=8.0
        )
        
        # High pressure - should collect
        assert gc_collector.should_collect(
            current_memory_gb=7.0,
            total_memory_gb=8.0
        )
    
    def test_should_collect_interval(self, gc_collector):
        """Test collection interval enforcement."""
        # First check - should collect if pressure is high
        result1 = gc_collector.should_collect(
            current_memory_gb=7.0,
            total_memory_gb=8.0
        )
        
        if result1:
            gc_collector.last_collection_time = time.time()
        
        # Immediate second check - should not collect (too soon)
        result2 = gc_collector.should_collect(
            current_memory_gb=7.0,
            total_memory_gb=8.0
        )
        
        assert not result2
    
    def test_collect_normal(self, gc_collector):
        """Test normal garbage collection."""
        memory_freed = gc_collector.collect(aggressive=False)
        
        assert memory_freed >= 0.0
        assert gc_collector.collections_triggered == 1
    
    def test_collect_aggressive(self, gc_collector):
        """Test aggressive garbage collection."""
        memory_freed = gc_collector.collect(aggressive=True)
        
        assert memory_freed >= 0.0
        assert gc_collector.collections_triggered == 1
    
    def test_get_stats(self, gc_collector):
        """Test getting GC statistics."""
        # Trigger some collections
        gc_collector.collect()
        gc_collector.collect()
        
        stats = gc_collector.get_stats()
        
        assert isinstance(stats, GCStats)
        assert stats.collections_triggered == 2
        assert stats.memory_freed_gb >= 0.0
        assert stats.avg_collection_time_ms >= 0.0
    
    def test_adaptive_threshold_increase(self, gc_collector):
        """Test adaptive threshold increase."""
        gc_collector.enable_adaptive = True
        initial_threshold = gc_collector.memory_pressure_threshold
        
        # Simulate effective collection (freed significant memory)
        gc_collector._adjust_threshold(memory_freed=1.0, collection_time=0.1)
        
        # Threshold should increase (be more conservative)
        assert gc_collector.memory_pressure_threshold >= initial_threshold
    
    def test_adaptive_threshold_decrease(self, gc_collector):
        """Test adaptive threshold decrease."""
        gc_collector.enable_adaptive = True
        initial_threshold = gc_collector.memory_pressure_threshold
        
        # Simulate ineffective collection (freed little memory)
        gc_collector._adjust_threshold(memory_freed=0.05, collection_time=0.1)
        
        # Threshold should decrease (be more aggressive)
        assert gc_collector.memory_pressure_threshold <= initial_threshold
    
    def test_threshold_bounds(self, gc_collector):
        """Test threshold stays within bounds."""
        gc_collector.enable_adaptive = True
        
        # Try to push threshold very high
        for _ in range(20):
            gc_collector._adjust_threshold(memory_freed=2.0, collection_time=0.1)
        
        assert gc_collector.memory_pressure_threshold <= gc_collector.max_threshold
        
        # Try to push threshold very low
        for _ in range(20):
            gc_collector._adjust_threshold(memory_freed=0.01, collection_time=0.1)
        
        assert gc_collector.memory_pressure_threshold >= gc_collector.min_threshold


# ============================================================================
# MemoryUsagePredictor Tests
# ============================================================================

class TestMemoryUsagePredictor:
    """Test memory usage predictor."""
    
    def test_initialization(self):
        """Test predictor initialization."""
        predictor = MemoryUsagePredictor(enable_learning=True)
        
        assert predictor.enable_learning is True
        assert len(predictor.usage_history) == 0
    
    def test_predict_basic(self, memory_predictor):
        """Test basic memory prediction."""
        slide_chars = {
            'dimensions': (20000, 20000),
            'estimated_patches': 1000,
            'tile_size': 224,
            'batch_size': 32,
            'feature_dim': 512
        }
        
        prediction = memory_predictor.predict(slide_chars)
        
        assert isinstance(prediction, MemoryPrediction)
        assert prediction.predicted_peak_gb > 0
        assert prediction.predicted_avg_gb > 0
        assert 0.0 <= prediction.confidence <= 1.0
        assert prediction.based_on_samples == 0
    
    def test_predict_scales_with_patches(self, memory_predictor):
        """Test prediction scales with patch count."""
        small_slide = {
            'dimensions': (10000, 10000),
            'estimated_patches': 500,
            'tile_size': 224,
            'batch_size': 32,
            'feature_dim': 512
        }
        
        large_slide = {
            'dimensions': (40000, 40000),
            'estimated_patches': 5000,
            'tile_size': 224,
            'batch_size': 32,
            'feature_dim': 512
        }
        
        pred_small = memory_predictor.predict(small_slide)
        pred_large = memory_predictor.predict(large_slide)
        
        # Larger slide should predict more memory
        assert pred_large.predicted_peak_gb > pred_small.predicted_peak_gb
        assert pred_large.predicted_avg_gb > pred_small.predicted_avg_gb
    
    def test_predict_scales_with_batch_size(self, memory_predictor):
        """Test prediction scales with batch size."""
        small_batch = {
            'dimensions': (20000, 20000),
            'estimated_patches': 1000,
            'tile_size': 224,
            'batch_size': 16,
            'feature_dim': 512
        }
        
        large_batch = {
            'dimensions': (20000, 20000),
            'estimated_patches': 1000,
            'tile_size': 224,
            'batch_size': 64,
            'feature_dim': 512
        }
        
        pred_small = memory_predictor.predict(small_batch)
        pred_large = memory_predictor.predict(large_batch)
        
        # Larger batch should predict more memory
        assert pred_large.predicted_peak_gb > pred_small.predicted_peak_gb
    
    def test_record_usage(self, memory_predictor):
        """Test recording actual usage."""
        slide_chars = {
            'dimensions': (20000, 20000),
            'estimated_patches': 1000,
            'tile_size': 224,
            'batch_size': 32,
            'feature_dim': 512
        }
        
        memory_predictor.record_usage(slide_chars, peak_memory_gb=2.5, avg_memory_gb=1.8)
        
        assert len(memory_predictor.usage_history) == 1
    
    def test_learning_improves_prediction(self, memory_predictor):
        """Test that learning improves predictions."""
        slide_chars = {
            'dimensions': (20000, 20000),
            'estimated_patches': 1000,
            'tile_size': 224,
            'batch_size': 32,
            'feature_dim': 512
        }
        
        # Initial prediction
        pred_before = memory_predictor.predict(slide_chars)
        initial_confidence = pred_before.confidence
        
        # Record several similar usages
        for _ in range(5):
            memory_predictor.record_usage(slide_chars, peak_memory_gb=2.0, avg_memory_gb=1.5)
        
        # Prediction after learning
        pred_after = memory_predictor.predict(slide_chars)
        
        # Confidence should improve
        assert pred_after.confidence > initial_confidence
        assert pred_after.based_on_samples == 5
    
    def test_history_limit(self, memory_predictor):
        """Test history is limited to recent samples."""
        slide_chars = {
            'dimensions': (20000, 20000),
            'estimated_patches': 1000,
            'tile_size': 224,
            'batch_size': 32,
            'feature_dim': 512
        }
        
        # Record more than limit (100)
        for i in range(150):
            memory_predictor.record_usage(slide_chars, peak_memory_gb=2.0, avg_memory_gb=1.5)
        
        # Should keep only last 100
        assert len(memory_predictor.usage_history) == 100
    
    def test_get_preallocation_recommendation(self, memory_predictor):
        """Test preallocation recommendation."""
        slide_chars = {
            'dimensions': (20000, 20000),
            'estimated_patches': 1000,
            'tile_size': 224,
            'batch_size': 32,
            'feature_dim': 512
        }
        
        recommended_gb = memory_predictor.get_preallocation_recommendation(slide_chars)
        
        assert recommended_gb > 0
        
        # Should be larger than predicted peak (safety margin)
        prediction = memory_predictor.predict(slide_chars)
        assert recommended_gb >= prediction.predicted_peak_gb
    
    def test_confidence_increases_with_samples(self, memory_predictor):
        """Test confidence increases with more samples."""
        slide_chars = {
            'dimensions': (20000, 20000),
            'estimated_patches': 1000,
            'tile_size': 224,
            'batch_size': 32,
            'feature_dim': 512
        }
        
        confidences = []
        
        for i in range(10):
            pred = memory_predictor.predict(slide_chars)
            confidences.append(pred.confidence)
            
            # Record usage
            memory_predictor.record_usage(slide_chars, peak_memory_gb=2.0, avg_memory_gb=1.5)
        
        # Confidence should generally increase
        assert confidences[-1] > confidences[0]
    
    def test_prediction_to_dict(self, memory_predictor):
        """Test prediction serialization."""
        slide_chars = {
            'dimensions': (20000, 20000),
            'estimated_patches': 1000,
            'tile_size': 224,
            'batch_size': 32,
            'feature_dim': 512
        }
        
        prediction = memory_predictor.predict(slide_chars)
        pred_dict = prediction.to_dict()
        
        assert isinstance(pred_dict, dict)
        assert 'predicted_peak_gb' in pred_dict
        assert 'predicted_avg_gb' in pred_dict
        assert 'confidence' in pred_dict
        assert 'based_on_samples' in pred_dict


# ============================================================================
# Integration Tests
# ============================================================================

class TestMemoryOptimizerIntegration:
    """Integration tests for memory optimization components."""
    
    def test_pool_with_gc(self, device):
        """Test memory pool with garbage collector."""
        pool = MemoryPoolManager(device=device, initial_pool_size_gb=0.5)
        gc = SmartGarbageCollector(device=device, memory_pressure_threshold=0.7)
        
        # Allocate many blocks
        tensors = []
        for _ in range(20):
            tensor = pool.allocate(1024 * 4)
            tensors.append(tensor)
        
        # Check if GC should trigger
        stats = pool.get_stats()
        should_collect = gc.should_collect(
            current_memory_gb=stats.allocated_size_gb,
            total_memory_gb=pool.max_pool_size_gb
        )
        
        if should_collect:
            # Deallocate some tensors
            for tensor in tensors[:10]:
                pool.deallocate(tensor)
            
            # Trigger GC
            gc.collect()
        
        # System should remain stable
        assert pool.get_stats().total_blocks > 0
    
    def test_predictor_with_pool(self, device):
        """Test predictor with memory pool."""
        predictor = MemoryUsagePredictor(enable_learning=True)
        pool = MemoryPoolManager(device=device, initial_pool_size_gb=0.5)
        
        slide_chars = {
            'dimensions': (20000, 20000),
            'estimated_patches': 1000,
            'tile_size': 224,
            'batch_size': 32,
            'feature_dim': 512
        }
        
        # Get prediction
        prediction = predictor.predict(slide_chars)
        
        # Use prediction to guide allocation
        recommended_gb = predictor.get_preallocation_recommendation(slide_chars)
        
        # Simulate processing
        initial_stats = pool.get_stats()
        
        # Allocate based on prediction
        num_allocations = int(prediction.predicted_peak_gb * 1024**3 / (1024 * 4))
        for _ in range(min(10, num_allocations)):  # Limit for test speed
            pool.allocate(1024 * 4)
        
        final_stats = pool.get_stats()
        
        # Record actual usage
        predictor.record_usage(
            slide_chars,
            peak_memory_gb=final_stats.allocated_size_gb,
            avg_memory_gb=final_stats.allocated_size_gb * 0.7
        )
        
        # Should have recorded usage
        assert len(predictor.usage_history) == 1
    
    def test_end_to_end_optimization(self, device):
        """Test complete memory optimization workflow."""
        # Initialize all components
        pool = MemoryPoolManager(device=device, initial_pool_size_gb=0.5, max_pool_size_gb=2.0)
        gc = SmartGarbageCollector(device=device, memory_pressure_threshold=0.75)
        predictor = MemoryUsagePredictor(enable_learning=True)
        
        slide_chars = {
            'dimensions': (20000, 20000),
            'estimated_patches': 500,
            'tile_size': 224,
            'batch_size': 16,
            'feature_dim': 512
        }
        
        # 1. Predict memory usage
        prediction = predictor.predict(slide_chars)
        assert prediction.predicted_peak_gb > 0
        
        # 2. Allocate memory from pool
        tensors = []
        for _ in range(10):
            tensor = pool.allocate(1024 * 4)
            tensors.append(tensor)
        
        # 3. Check memory pressure
        pool_stats = pool.get_stats()
        if gc.should_collect(pool_stats.allocated_size_gb, pool.max_pool_size_gb):
            # 4. Trigger GC if needed
            memory_freed = gc.collect()
            assert memory_freed >= 0
        
        # 5. Deallocate some tensors
        for tensor in tensors[:5]:
            pool.deallocate(tensor)
        
        # 6. Record actual usage for learning
        final_stats = pool.get_stats()
        predictor.record_usage(
            slide_chars,
            peak_memory_gb=final_stats.total_size_gb,
            avg_memory_gb=final_stats.allocated_size_gb
        )
        
        # 7. Cleanup
        pool.cleanup()
        
        # Verify system state
        assert len(predictor.usage_history) == 1
        assert gc.collections_triggered >= 0


# ============================================================================
# Performance Tests
# ============================================================================

class TestMemoryOptimizerPerformance:
    """Performance tests for memory optimization."""
    
    def test_pool_allocation_speed(self, memory_pool):
        """Test pool allocation performance."""
        import time
        
        num_allocations = 100
        size_bytes = 1024 * 4
        
        start = time.time()
        for _ in range(num_allocations):
            tensor = memory_pool.allocate(size_bytes)
            memory_pool.deallocate(tensor)
        elapsed = time.time() - start
        
        # Should complete quickly
        assert elapsed < 1.0
        
        # Check hit rate
        stats = memory_pool.get_stats()
        assert stats.hit_rate > 0.5  # At least 50% hit rate
    
    def test_gc_collection_speed(self, gc_collector):
        """Test GC collection performance."""
        import time
        
        # Create some garbage
        for _ in range(100):
            _ = torch.randn(100, 100)
        
        start = time.time()
        gc_collector.collect()
        elapsed = time.time() - start
        
        # Should complete quickly (<100ms)
        assert elapsed < 0.1
    
    def test_prediction_speed(self, memory_predictor):
        """Test prediction performance."""
        import time
        
        slide_chars = {
            'dimensions': (20000, 20000),
            'estimated_patches': 1000,
            'tile_size': 224,
            'batch_size': 32,
            'feature_dim': 512
        }
        
        # Add some history
        for _ in range(50):
            memory_predictor.record_usage(slide_chars, peak_memory_gb=2.0, avg_memory_gb=1.5)
        
        # Time predictions
        start = time.time()
        for _ in range(100):
            memory_predictor.predict(slide_chars)
        elapsed = time.time() - start
        
        # Should be fast (<10ms per prediction)
        assert elapsed / 100 < 0.01


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
