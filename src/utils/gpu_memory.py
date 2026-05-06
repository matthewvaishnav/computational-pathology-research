"""
Optimized GPU memory management utilities for HistoCore.

This module provides intelligent GPU memory management with automatic
cleanup, memory pressure detection, and efficient resource allocation
for deep learning workloads.
"""

import gc
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

import torch

from ..utils.constants import (
    MAX_MEMORY_PERCENT,
    MEMORY_PRESSURE_THRESHOLD,
    DEFAULT_BATCH_SIZE,
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class GPUMemoryStats:
    """GPU memory statistics."""
    
    device_id: int
    allocated_mb: float
    reserved_mb: float
    max_allocated_mb: float
    max_reserved_mb: float
    utilization_percent: float
    
    @property
    def pressure_level(self) -> str:
        """Get memory pressure level."""
        if self.utilization_percent > 90:
            return "critical"
        elif self.utilization_percent > 80:
            return "high"
        elif self.utilization_percent > 60:
            return "moderate"
        else:
            return "low"


class GPUMemoryManager:
    """
    Intelligent GPU memory manager with automatic cleanup and optimization.
    
    Features:
    - Automatic memory pressure detection
    - Smart batch size adjustment
    - Memory fragmentation reduction
    - Automatic garbage collection
    - Memory leak detection
    """
    
    def __init__(self, target_utilization: float = 0.8, cleanup_threshold: float = 0.9):
        self.target_utilization = target_utilization
        self.cleanup_threshold = cleanup_threshold
        self._memory_history: List[Tuple[float, float]] = []  # (timestamp, allocated_mb)
        self._last_cleanup = 0.0
        self._cleanup_interval = 30.0  # seconds
        
    def get_memory_stats(self, device_id: int = 0) -> GPUMemoryStats:
        """Get comprehensive GPU memory statistics."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        
        device = torch.device(f'cuda:{device_id}')
        
        # Get memory info
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)  # MB
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)  # MB
        max_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB
        max_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)  # MB
        
        # Calculate utilization
        total_memory = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
        utilization = (allocated / total_memory) * 100
        
        return GPUMemoryStats(
            device_id=device_id,
            allocated_mb=allocated,
            reserved_mb=reserved,
            max_allocated_mb=max_allocated,
            max_reserved_mb=max_reserved,
            utilization_percent=utilization
        )
    
    def check_memory_pressure(self, device_id: int = 0) -> bool:
        """Check if GPU is under memory pressure."""
        try:
            stats = self.get_memory_stats(device_id)
            return stats.utilization_percent > (self.cleanup_threshold * 100)
        except RuntimeError:
            return False
    
    def cleanup_memory(self, device_id: int = 0, force: bool = False) -> float:
        """
        Clean up GPU memory with intelligent strategies.
        
        Returns:
            Amount of memory freed in MB
        """
        if not torch.cuda.is_available():
            return 0.0
        
        current_time = time.time()
        
        # Rate limit cleanup operations
        if not force and current_time - self._last_cleanup < self._cleanup_interval:
            return 0.0
        
        # Get memory before cleanup
        stats_before = self.get_memory_stats(device_id)
        
        # Clear PyTorch cache
        torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        # Clear cache again after GC
        torch.cuda.empty_cache()
        
        # Get memory after cleanup
        stats_after = self.get_memory_stats(device_id)
        
        freed_mb = stats_before.allocated_mb - stats_after.allocated_mb
        
        if freed_mb > 0:
            logger.info(f"GPU memory cleanup freed {freed_mb:.1f} MB on device {device_id}")
        
        self._last_cleanup = current_time
        return freed_mb
    
    def optimize_batch_size(self, base_batch_size: int, model: torch.nn.Module,
                           input_shape: Tuple[int, ...], device_id: int = 0) -> int:
        """
        Optimize batch size based on available GPU memory.
        
        Args:
            base_batch_size: Starting batch size
            model: PyTorch model
            input_shape: Shape of input tensors (without batch dimension)
            device_id: GPU device ID
            
        Returns:
            Optimized batch size
        """
        if not torch.cuda.is_available():
            return base_batch_size
        
        device = torch.device(f'cuda:{device_id}')
        model = model.to(device)
        
        # Start with base batch size and test
        current_batch_size = base_batch_size
        max_working_batch_size = 1
        
        # Binary search for optimal batch size
        min_batch = 1
        max_batch = base_batch_size * 4  # Test up to 4x base size
        
        while min_batch <= max_batch:
            test_batch_size = (min_batch + max_batch) // 2
            
            try:
                # Test memory allocation with dummy data
                dummy_input = torch.randn(test_batch_size, *input_shape, device=device)
                
                with torch.no_grad():
                    _ = model(dummy_input)
                
                # Check memory usage
                stats = self.get_memory_stats(device_id)
                
                if stats.utilization_percent < (self.target_utilization * 100):
                    max_working_batch_size = test_batch_size
                    min_batch = test_batch_size + 1
                else:
                    max_batch = test_batch_size - 1
                
                # Clean up test tensors
                del dummy_input
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    max_batch = test_batch_size - 1
                    torch.cuda.empty_cache()
                else:
                    raise
        
        optimized_size = max(1, max_working_batch_size)
        
        if optimized_size != base_batch_size:
            logger.info(f"Optimized batch size from {base_batch_size} to {optimized_size}")
        
        return optimized_size
    
    def track_memory_usage(self, device_id: int = 0):
        """Track memory usage over time for leak detection."""
        if not torch.cuda.is_available():
            return
        
        stats = self.get_memory_stats(device_id)
        current_time = time.time()
        
        self._memory_history.append((current_time, stats.allocated_mb))
        
        # Keep only last hour of data
        cutoff_time = current_time - 3600
        self._memory_history = [
            (t, mem) for t, mem in self._memory_history if t > cutoff_time
        ]
    
    def detect_memory_leak(self, device_id: int = 0) -> bool:
        """
        Detect potential memory leaks based on usage trends.
        
        Returns:
            True if potential memory leak detected
        """
        if len(self._memory_history) < 10:
            return False
        
        # Check if memory usage is consistently increasing
        recent_usage = [mem for _, mem in self._memory_history[-10:]]
        
        # Simple trend detection: check if last 5 values are higher than first 5
        first_half_avg = sum(recent_usage[:5]) / 5
        second_half_avg = sum(recent_usage[5:]) / 5
        
        # Consider it a leak if memory increased by more than 100MB consistently
        return second_half_avg - first_half_avg > 100.0


@contextmanager
def gpu_memory_context(device_id: int = 0, cleanup_on_exit: bool = True):
    """
    Context manager for GPU memory management.
    
    Args:
        device_id: GPU device ID
        cleanup_on_exit: Whether to cleanup memory on exit
    
    Example:
        with gpu_memory_context(device_id=0):
            # GPU operations
            model_output = model(input_tensor)
    """
    manager = GPUMemoryManager()
    
    # Track initial memory
    initial_stats = None
    if torch.cuda.is_available():
        initial_stats = manager.get_memory_stats(device_id)
    
    try:
        yield manager
    finally:
        if cleanup_on_exit and torch.cuda.is_available():
            # Cleanup and report
            freed_mb = manager.cleanup_memory(device_id, force=True)
            
            if initial_stats:
                final_stats = manager.get_memory_stats(device_id)
                net_change = final_stats.allocated_mb - initial_stats.allocated_mb
                
                if abs(net_change) > 10:  # Report significant changes
                    logger.info(f"GPU memory change: {net_change:+.1f} MB "
                               f"(freed {freed_mb:.1f} MB during cleanup)")


def gpu_memory_efficient(cleanup_interval: int = 10):
    """
    Decorator for GPU memory efficient function execution.
    
    Args:
        cleanup_interval: Clean up memory every N calls
    
    Example:
        @gpu_memory_efficient(cleanup_interval=5)
        def process_batch(data):
            # GPU processing
            return results
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        call_count = 0
        manager = GPUMemoryManager()
        
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            nonlocal call_count
            call_count += 1
            
            # Periodic cleanup
            if call_count % cleanup_interval == 0:
                if torch.cuda.is_available():
                    for device_id in range(torch.cuda.device_count()):
                        if manager.check_memory_pressure(device_id):
                            manager.cleanup_memory(device_id)
            
            try:
                return func(*args, **kwargs)
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and torch.cuda.is_available():
                    # Emergency cleanup and retry
                    logger.warning("GPU OOM detected, attempting cleanup and retry")
                    
                    for device_id in range(torch.cuda.device_count()):
                        manager.cleanup_memory(device_id, force=True)
                    
                    # Retry once
                    return func(*args, **kwargs)
                else:
                    raise
        
        return wrapper
    return decorator


def auto_batch_size(model: torch.nn.Module, input_shape: Tuple[int, ...],
                   base_batch_size: int = DEFAULT_BATCH_SIZE, device_id: int = 0) -> int:
    """
    Automatically determine optimal batch size for a model.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape (without batch dimension)
        base_batch_size: Starting batch size for optimization
        device_id: GPU device ID
        
    Returns:
        Optimal batch size
    """
    manager = GPUMemoryManager()
    return manager.optimize_batch_size(base_batch_size, model, input_shape, device_id)


def get_gpu_recommendations() -> Dict[str, Any]:
    """
    Get GPU optimization recommendations based on current state.
    
    Returns:
        Dictionary with optimization recommendations
    """
    recommendations = {
        'cleanup_needed': False,
        'memory_pressure': False,
        'suggested_actions': [],
        'device_stats': {}
    }
    
    if not torch.cuda.is_available():
        recommendations['suggested_actions'].append("CUDA not available")
        return recommendations
    
    manager = GPUMemoryManager()
    
    for device_id in range(torch.cuda.device_count()):
        try:
            stats = manager.get_memory_stats(device_id)
            recommendations['device_stats'][device_id] = {
                'allocated_mb': stats.allocated_mb,
                'utilization_percent': stats.utilization_percent,
                'pressure_level': stats.pressure_level
            }
            
            if stats.pressure_level in ['high', 'critical']:
                recommendations['memory_pressure'] = True
                recommendations['cleanup_needed'] = True
                recommendations['suggested_actions'].append(
                    f"Device {device_id}: {stats.pressure_level} memory pressure "
                    f"({stats.utilization_percent:.1f}% used)"
                )
            
            if manager.detect_memory_leak(device_id):
                recommendations['suggested_actions'].append(
                    f"Device {device_id}: Potential memory leak detected"
                )
                
        except Exception as e:
            recommendations['suggested_actions'].append(
                f"Device {device_id}: Error getting stats - {e}"
            )
    
    return recommendations


# Global GPU memory manager instance
gpu_memory_manager = GPUMemoryManager()