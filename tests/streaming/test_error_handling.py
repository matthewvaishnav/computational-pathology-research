"""Error handling tests for streaming components."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from src.streaming.gpu_pipeline import (
    GPUPipeline,
    GPUMemoryManager,
    BatchSizeOptimizer
)


class TestGPUOOMRecovery:
    """Test GPU out-of-memory recovery scenarios."""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = Mock(spec=torch.nn.Module)
        model.eval = Mock()
        model.to = Mock(return_value=model)
        model.half = Mock(return_value=model)
        model.return_value = torch.randn(1, 10)
        return model
    
    def test_oom_error_handling_exists(self, mock_model):
        """Test OOM error handling mechanism exists."""
        pipeline = GPUPipeline(
            model=mock_model,
            batch_size=32,
            memory_limit_gb=1.0
        )
        
        # Verify pipeline has error handling components
        assert hasattr(pipeline, 'memory_manager')
        assert hasattr(pipeline, 'batch_optimizer')
    
    def test_batch_size_reduction_on_failure(self, mock_model):
        """Test batch size can be reduced."""
        pipeline = GPUPipeline(
            model=mock_model,
            batch_size=64
        )
        
        # Reduce via OOM handler
        original_size = pipeline.batch_optimizer.current_batch_size
        pipeline.batch_optimizer.handle_oom()
        
        # Should be smaller
        assert pipeline.batch_optimizer.current_batch_size < original_size
    
    def test_min_batch_size_limit(self, mock_model):
        """Test batch size respects minimum."""
        pipeline = GPUPipeline(
            model=mock_model,
            batch_size=4
        )
        
        # Reduce multiple times via OOM
        for _ in range(10):
            pipeline.batch_optimizer.handle_oom()
        
        # Should not go below min
        assert pipeline.batch_optimizer.current_batch_size >= pipeline.batch_optimizer.min_batch_size


class TestMemoryMonitoring:
    """Test memory monitoring."""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_memory_usage_tracking(self, device):
        """Test memory usage tracked."""
        manager = GPUMemoryManager(device, memory_limit_gb=8.0)
        
        usage = manager.get_memory_usage()
        assert usage >= 0.0
        assert isinstance(usage, float)
    
    def test_memory_limit_enforcement(self, device):
        """Test memory limit enforced."""
        manager = GPUMemoryManager(device, memory_limit_gb=1.0)
        
        # Check availability
        assert not manager.is_memory_available(10.0)  # 10GB > 1GB
        assert manager.is_memory_available(0.1)  # 0.1GB < 1GB
    
    def test_memory_cleanup(self, device):
        """Test cleanup works."""
        manager = GPUMemoryManager(device)
        
        # Should not raise
        manager.cleanup()


class TestBatchSizeOptimization:
    """Test batch size optimization."""
    
    def test_batch_size_reduction(self):
        """Test batch size reduces on high memory pressure."""
        optimizer = BatchSizeOptimizer(
            initial_batch_size=64,
            min_batch_size=4,
            max_batch_size=128
        )
        
        original = optimizer.current_batch_size
        optimizer.optimize(memory_pressure=0.95)  # High pressure
        
        assert optimizer.current_batch_size < original
    
    def test_batch_size_increase(self):
        """Test batch size increases on low memory pressure."""
        optimizer = BatchSizeOptimizer(
            initial_batch_size=32,
            min_batch_size=4,
            max_batch_size=128
        )
        
        # Record some fast batches
        for _ in range(5):
            optimizer.record_batch(batch_time=0.1, memory_used_gb=1.0)
        
        original = optimizer.current_batch_size
        optimizer.optimize(memory_pressure=0.3)  # Low pressure
        
        assert optimizer.current_batch_size >= original
    
    def test_batch_size_limits(self):
        """Test limits respected."""
        optimizer = BatchSizeOptimizer(
            initial_batch_size=64,
            min_batch_size=8,
            max_batch_size=128
        )
        
        # Reduce to min
        for _ in range(20):
            optimizer.optimize(memory_pressure=0.95)
        assert optimizer.current_batch_size >= 8
        
        # Increase to max
        for _ in range(5):
            optimizer.record_batch(batch_time=0.1, memory_used_gb=1.0)
        for _ in range(20):
            optimizer.optimize(memory_pressure=0.2)
        assert optimizer.current_batch_size <= 128


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
