"""GPU Pipeline for async batch processing with memory optimization."""

import asyncio
import logging
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from collections import deque
import gc

import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ThroughputMetrics:
    """Throughput and performance metrics."""
    patches_per_second: float
    batches_per_second: float
    avg_batch_time: float
    gpu_utilization: float
    gpu_memory_used_gb: float
    gpu_memory_total_gb: float
    current_batch_size: int
    total_patches_processed: int
    
    @property
    def gpu_memory_percent(self) -> float:
        """Calculate GPU memory usage percentage."""
        if self.gpu_memory_total_gb == 0:
            return 0.0
        return (self.gpu_memory_used_gb / self.gpu_memory_total_gb) * 100.0


class GPUMemoryManager:
    """Manages GPU memory allocation and cleanup."""
    
    def __init__(self, device: torch.device, memory_limit_gb: Optional[float] = None):
        self.device = device
        self.memory_limit_gb = memory_limit_gb
        
        # Get total GPU memory
        if device.type == 'cuda':
            self.total_memory_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        else:
            self.total_memory_gb = 0.0
        
        # Set memory limit
        if memory_limit_gb is None:
            self.memory_limit_gb = self.total_memory_gb * 0.8  # Use 80% by default
        
        logger.info(f"GPU Memory Manager initialized: {self.total_memory_gb:.2f}GB total, {self.memory_limit_gb:.2f}GB limit")
    
    def get_memory_usage(self) -> float:
        """Get current GPU memory usage in GB."""
        if self.device.type == 'cuda':
            return torch.cuda.memory_allocated(self.device) / (1024**3)
        return 0.0
    
    def get_memory_reserved(self) -> float:
        """Get reserved GPU memory in GB."""
        if self.device.type == 'cuda':
            return torch.cuda.memory_reserved(self.device) / (1024**3)
        return 0.0
    
    def is_memory_available(self, required_gb: float) -> bool:
        """Check if required memory is available."""
        current_usage = self.get_memory_usage()
        return (current_usage + required_gb) <= self.memory_limit_gb
    
    def cleanup(self):
        """Clean up GPU memory."""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
    
    def get_available_memory(self) -> float:
        """Get available GPU memory in GB."""
        return max(0.0, self.memory_limit_gb - self.get_memory_usage())


class BatchSizeOptimizer:
    """Dynamically optimizes batch size based on memory usage."""
    
    def __init__(self, initial_batch_size: int, min_batch_size: int = 1, max_batch_size: int = 256):
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        
        # Track batch processing history
        self.batch_times = deque(maxlen=10)
        self.memory_usage = deque(maxlen=10)
        
        # OOM tracking
        self.oom_count = 0
        self.last_oom_batch_size = None
    
    def record_batch(self, batch_time: float, memory_used_gb: float):
        """Record batch processing metrics."""
        self.batch_times.append(batch_time)
        self.memory_usage.append(memory_used_gb)
    
    def handle_oom(self):
        """Handle out-of-memory error."""
        self.oom_count += 1
        self.last_oom_batch_size = self.current_batch_size
        
        # Reduce batch size significantly
        self.current_batch_size = max(self.min_batch_size, self.current_batch_size // 4)
        logger.warning(f"OOM detected. Reduced batch size to {self.current_batch_size}")
    
    def optimize(self, memory_pressure: float) -> int:
        """Optimize batch size based on memory pressure.
        
        Args:
            memory_pressure: Memory usage as fraction of limit (0.0 to 1.0)
        
        Returns:
            Optimized batch size
        """
        if memory_pressure > 0.9:
            # Critical memory pressure - reduce aggressively
            self.current_batch_size = max(self.min_batch_size, self.current_batch_size // 2)
            logger.info(f"Critical memory pressure. Reduced batch size to {self.current_batch_size}")
        
        elif memory_pressure > 0.8:
            # High memory pressure - reduce conservatively
            self.current_batch_size = max(self.min_batch_size, int(self.current_batch_size * 0.75))
            logger.info(f"High memory pressure. Reduced batch size to {self.current_batch_size}")
        
        elif memory_pressure < 0.4 and len(self.batch_times) >= 5:
            # Low memory pressure and stable - try to increase
            avg_time = np.mean(self.batch_times)
            
            # Only increase if processing is fast and stable
            if avg_time < 0.5 and self.oom_count == 0:
                new_size = min(self.max_batch_size, int(self.current_batch_size * 1.2))
                
                # Don't increase past last OOM size
                if self.last_oom_batch_size is not None:
                    new_size = min(new_size, self.last_oom_batch_size - 1)
                
                if new_size > self.current_batch_size:
                    self.current_batch_size = new_size
                    logger.info(f"Low memory pressure. Increased batch size to {self.current_batch_size}")
        
        return self.current_batch_size
    
    def get_batch_size(self) -> int:
        """Get current batch size."""
        return self.current_batch_size


class GPUPipeline:
    """Parallel GPU processing pipeline with async processing and memory optimization."""
    
    def __init__(self, 
                 model: nn.Module, 
                 batch_size: int = 64, 
                 gpu_ids: Optional[List[int]] = None,
                 memory_limit_gb: Optional[float] = None,
                 enable_fp16: bool = False):
        """Initialize GPU pipeline.
        
        Args:
            model: PyTorch model for feature extraction
            batch_size: Initial batch size
            gpu_ids: List of GPU IDs to use (None = auto-detect)
            memory_limit_gb: GPU memory limit in GB
            enable_fp16: Enable FP16 precision for memory reduction
        """
        self.model = model
        self.initial_batch_size = batch_size
        self.enable_fp16 = enable_fp16
        
        # Setup devices
        self.devices = self._setup_devices(gpu_ids)
        self.primary_device = self.devices[0]
        
        # Move model to device
        self.model = self.model.to(self.primary_device)
        self.model.eval()
        
        # Enable FP16 if requested
        if self.enable_fp16 and self.primary_device.type == 'cuda':
            self.model = self.model.half()
            logger.info("Enabled FP16 precision")
        
        # Multi-GPU support
        if len(self.devices) > 1:
            self.model = nn.DataParallel(self.model, device_ids=[d.index for d in self.devices if d.type == 'cuda'])
            logger.info(f"Enabled DataParallel across {len(self.devices)} GPUs")
        
        # Initialize memory manager
        self.memory_manager = GPUMemoryManager(self.primary_device, memory_limit_gb)
        
        # Initialize batch size optimizer
        self.batch_optimizer = BatchSizeOptimizer(
            initial_batch_size=batch_size,
            min_batch_size=1,
            max_batch_size=256
        )
        
        # Performance tracking
        self.total_patches_processed = 0
        self.total_batches_processed = 0
        self.total_processing_time = 0.0
        self.batch_times = deque(maxlen=100)
        
        # Async processing
        self.processing_queue = asyncio.Queue()
        self.result_queue = asyncio.Queue()
        
        logger.info(f"GPUPipeline initialized on {self.primary_device} with batch_size={batch_size}")
    
    def _setup_devices(self, gpu_ids: Optional[List[int]]) -> List[torch.device]:
        """Setup GPU devices."""
        if gpu_ids is None:
            # Auto-detect GPUs
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_ids = list(range(gpu_count))
            else:
                logger.warning("No CUDA GPUs available, using CPU")
                return [torch.device('cpu')]
        
        devices = []
        for gpu_id in gpu_ids:
            if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
                devices.append(torch.device(f'cuda:{gpu_id}'))
            else:
                logger.warning(f"GPU {gpu_id} not available")
        
        if not devices:
            devices = [torch.device('cpu')]
        
        return devices
    
    async def process_batch_async(self, patches: torch.Tensor) -> torch.Tensor:
        """Asynchronously process patch batch through CNN.
        
        Args:
            patches: Tensor of patches [batch_size, channels, height, width]
        
        Returns:
            Feature tensor [batch_size, feature_dim]
        """
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        features = await loop.run_in_executor(None, self._process_batch_sync, patches)
        return features
    
    def _process_batch_sync(self, patches: torch.Tensor) -> torch.Tensor:
        """Synchronous batch processing with memory optimization."""
        start_time = time.time()
        
        try:
            # Move to device
            patches = patches.to(self.primary_device)
            
            # Convert to FP16 if enabled
            if self.enable_fp16 and self.primary_device.type == 'cuda':
                patches = patches.half()
            
            # Check if we need to split batch due to memory
            current_batch_size = patches.shape[0]
            optimal_batch_size = self.batch_optimizer.get_batch_size()
            
            if current_batch_size > optimal_batch_size:
                # Process in sub-batches
                features = self._process_in_subbatches(patches, optimal_batch_size)
            else:
                # Process entire batch
                with torch.no_grad():
                    features = self.model(patches)
            
            # Record metrics
            batch_time = time.time() - start_time
            memory_used = self.memory_manager.get_memory_usage()
            
            self.batch_optimizer.record_batch(batch_time, memory_used)
            self.batch_times.append(batch_time)
            self.total_patches_processed += current_batch_size
            self.total_batches_processed += 1
            self.total_processing_time += batch_time
            
            # Optimize batch size
            memory_pressure = memory_used / self.memory_manager.memory_limit_gb
            self.batch_optimizer.optimize(memory_pressure)
            
            # Periodic cleanup
            if self.total_batches_processed % 10 == 0:
                self.memory_manager.cleanup()
            
            return features.cpu()
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"GPU OOM error: {e}")
                
                # Handle OOM
                self.memory_manager.cleanup()
                self.batch_optimizer.handle_oom()
                
                # Retry with smaller batch
                optimal_batch_size = self.batch_optimizer.get_batch_size()
                return self._process_in_subbatches(patches, optimal_batch_size)
            else:
                raise
    
    def _process_in_subbatches(self, patches: torch.Tensor, subbatch_size: int) -> torch.Tensor:
        """Process large batch in smaller sub-batches."""
        num_patches = patches.shape[0]
        features_list = []
        
        for i in range(0, num_patches, subbatch_size):
            end_idx = min(i + subbatch_size, num_patches)
            subbatch = patches[i:end_idx]
            
            with torch.no_grad():
                subbatch_features = self.model(subbatch)
            
            features_list.append(subbatch_features.cpu())
            
            # Cleanup between sub-batches
            if self.primary_device.type == 'cuda':
                torch.cuda.empty_cache()
        
        return torch.cat(features_list, dim=0)
    
    def optimize_batch_size(self, memory_usage: float) -> int:
        """Dynamically adjust batch size based on memory usage.
        
        Args:
            memory_usage: Current memory usage in GB
        
        Returns:
            Optimized batch size
        """
        memory_pressure = memory_usage / self.memory_manager.memory_limit_gb
        return self.batch_optimizer.optimize(memory_pressure)
    
    def get_throughput_stats(self) -> ThroughputMetrics:
        """Get current processing throughput metrics."""
        # Calculate throughput
        if self.total_processing_time > 0:
            patches_per_sec = self.total_patches_processed / self.total_processing_time
            batches_per_sec = self.total_batches_processed / self.total_processing_time
        else:
            patches_per_sec = 0.0
            batches_per_sec = 0.0
        
        # Calculate average batch time
        avg_batch_time = np.mean(self.batch_times) if self.batch_times else 0.0
        
        # Get GPU stats
        memory_used = self.memory_manager.get_memory_usage()
        memory_total = self.memory_manager.total_memory_gb
        
        # Estimate GPU utilization (simplified)
        if avg_batch_time > 0:
            gpu_utilization = min(100.0, (1.0 / avg_batch_time) * 10.0)  # Rough estimate
        else:
            gpu_utilization = 0.0
        
        return ThroughputMetrics(
            patches_per_second=patches_per_sec,
            batches_per_second=batches_per_sec,
            avg_batch_time=avg_batch_time,
            gpu_utilization=gpu_utilization,
            gpu_memory_used_gb=memory_used,
            gpu_memory_total_gb=memory_total,
            current_batch_size=self.batch_optimizer.get_batch_size(),
            total_patches_processed=self.total_patches_processed
        )
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.total_patches_processed = 0
        self.total_batches_processed = 0
        self.total_processing_time = 0.0
        self.batch_times.clear()
    
    def cleanup(self):
        """Clean up GPU resources."""
        self.memory_manager.cleanup()
        
        # Move model to CPU to free GPU memory
        if self.primary_device.type == 'cuda':
            self.model = self.model.cpu()
            torch.cuda.empty_cache()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


def get_optimal_batch_size(model: nn.Module, 
                          input_shape: tuple, 
                          device: torch.device,
                          memory_limit_gb: float = 8.0) -> int:
    """Estimate optimal batch size for given model and memory limit.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape (C, H, W)
        device: Target device
        memory_limit_gb: Memory limit in GB
    
    Returns:
        Estimated optimal batch size
    """
    model = model.to(device)
    model.eval()
    
    # Start with batch size of 1
    batch_size = 1
    max_batch_size = 256
    
    try:
        while batch_size < max_batch_size:
            # Create dummy input
            dummy_input = torch.randn(batch_size, *input_shape).to(device)
            
            # Try forward pass
            with torch.no_grad():
                _ = model(dummy_input)
            
            # Check memory usage
            if device.type == 'cuda':
                memory_used = torch.cuda.memory_allocated(device) / (1024**3)
                if memory_used > memory_limit_gb * 0.7:  # 70% threshold
                    break
            
            # Double batch size
            batch_size *= 2
            
            # Cleanup
            del dummy_input
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Return 75% of max successful batch size for safety
        optimal = max(1, int(batch_size * 0.75))
        logger.info(f"Estimated optimal batch size: {optimal}")
        return optimal
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            # Return last successful batch size
            return max(1, batch_size // 2)
        raise
    finally:
        if device.type == 'cuda':
            torch.cuda.empty_cache()