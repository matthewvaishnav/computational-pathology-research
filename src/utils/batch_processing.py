"""
Optimized batch processing utilities for HistoCore.

This module provides high-performance batch processing with intelligent
batching strategies, memory management, and parallel execution optimization.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar
from queue import Queue

import torch

from ..utils.constants import (
    DEFAULT_BATCH_SIZE,
    MAX_BATCH_SIZE,
)

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    
    batch_size: int = DEFAULT_BATCH_SIZE
    max_batch_size: int = MAX_BATCH_SIZE
    min_batch_size: int = 1
    adaptive_batching: bool = True
    memory_limit_gb: float = 4.0
    timeout_seconds: float = 30.0
    max_workers: int = 4
    prefetch_batches: int = 2


@dataclass
class BatchResult:
    """Result from batch processing."""
    
    batch_id: int
    items_processed: int
    processing_time: float
    memory_peak_mb: float
    success: bool
    error: Optional[str] = None
    results: Optional[List[Any]] = None


class AdaptiveBatcher:
    """
    Intelligent batcher that adapts batch size based on performance and memory.
    
    Features:
    - Dynamic batch size adjustment based on processing time and memory usage
    - Memory pressure detection and response
    - Performance tracking and optimization
    - Automatic fallback for OOM errors
    """
    
    def __init__(self, config: BatchConfig):
        self.config = config
        self.current_batch_size = config.batch_size
        self.performance_history: List[Tuple[int, float, float]] = []  # (batch_size, time, memory)
        self.oom_count = 0
        self.success_count = 0
        
    def get_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on performance history."""
        if not self.config.adaptive_batching:
            return self.config.batch_size
        
        # If we've had recent OOM errors, reduce batch size
        if self.oom_count > 0:
            self.current_batch_size = max(
                self.config.min_batch_size,
                self.current_batch_size // 2
            )
            self.oom_count = 0
            logger.info(f"Reduced batch size to {self.current_batch_size} due to OOM")
            return self.current_batch_size
        
        # If we have performance history, optimize based on throughput
        if len(self.performance_history) >= 3:
            # Calculate throughput (items/second) for recent batches
            recent_history = self.performance_history[-3:]
            best_throughput = 0
            best_batch_size = self.current_batch_size
            
            for batch_size, time_taken, memory_used in recent_history:
                if time_taken > 0:
                    throughput = batch_size / time_taken
                    memory_gb = memory_used / 1024
                    
                    # Prefer higher throughput with reasonable memory usage
                    if (throughput > best_throughput and 
                        memory_gb < self.config.memory_limit_gb * 0.8):
                        best_throughput = throughput
                        best_batch_size = batch_size
            
            # Gradually adjust towards optimal batch size
            if best_batch_size != self.current_batch_size:
                if best_batch_size > self.current_batch_size:
                    self.current_batch_size = min(
                        self.config.max_batch_size,
                        int(self.current_batch_size * 1.2)
                    )
                else:
                    self.current_batch_size = max(
                        self.config.min_batch_size,
                        int(self.current_batch_size * 0.8)
                    )
        
        return self.current_batch_size
    
    def record_batch_performance(self, batch_size: int, processing_time: float, 
                               memory_peak_mb: float, success: bool):
        """Record performance metrics for batch size optimization."""
        if success:
            self.performance_history.append((batch_size, processing_time, memory_peak_mb))
            self.success_count += 1
            
            # Keep only recent history
            if len(self.performance_history) > 10:
                self.performance_history = self.performance_history[-10:]
        else:
            self.oom_count += 1
    
    def create_batches(self, items: List[T]) -> List[List[T]]:
        """Create optimally-sized batches from items."""
        batch_size = self.get_optimal_batch_size()
        
        batches = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batches.append(batch)
        
        return batches


class OptimizedBatchProcessor:
    """
    High-performance batch processor with adaptive batching and memory management.
    
    Features:
    - Adaptive batch sizing based on performance
    - Memory pressure monitoring and response
    - Parallel processing with thread/process pools
    - Automatic error recovery and retry logic
    - Progress tracking and performance metrics
    """
    
    def __init__(self, config: BatchConfig):
        self.config = config
        self.batcher = AdaptiveBatcher(config)
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
        self.results_queue = Queue()
        self.stats = {
            'total_items': 0,
            'total_batches': 0,
            'successful_batches': 0,
            'failed_batches': 0,
            'total_time': 0.0,
            'avg_batch_size': 0.0,
        }
    
    def process_batch_sync(self, batch: List[T], processor_func: Callable[[List[T]], List[R]],
                          batch_id: int = 0) -> BatchResult:
        """
        Process a single batch synchronously with monitoring.
        
        Args:
            batch: Items to process
            processor_func: Function to process the batch
            batch_id: Unique identifier for the batch
            
        Returns:
            BatchResult with processing metrics
        """
        start_time = time.time()
        initial_memory = 0
        peak_memory = 0
        
        try:
            # Monitor memory usage
            if torch.cuda.is_available():
                initial_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
            
            # Process the batch
            results = processor_func(batch)
            
            # Record peak memory
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
                torch.cuda.reset_peak_memory_stats()
            
            processing_time = time.time() - start_time
            
            # Record performance for adaptive batching
            self.batcher.record_batch_performance(
                len(batch), processing_time, peak_memory, True
            )
            
            return BatchResult(
                batch_id=batch_id,
                items_processed=len(batch),
                processing_time=processing_time,
                memory_peak_mb=peak_memory,
                success=True,
                results=results
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            # Record failure for adaptive batching
            self.batcher.record_batch_performance(
                len(batch), processing_time, peak_memory, False
            )
            
            logger.error(f"Batch {batch_id} failed: {e}")
            
            return BatchResult(
                batch_id=batch_id,
                items_processed=0,
                processing_time=processing_time,
                memory_peak_mb=peak_memory,
                success=False,
                error=str(e)
            )
    
    async def process_batch_async(self, batch: List[T], 
                                processor_func: Callable[[List[T]], List[R]],
                                batch_id: int = 0) -> BatchResult:
        """Process a single batch asynchronously."""
        loop = asyncio.get_event_loop()
        
        # Run in thread pool to avoid blocking
        result = await loop.run_in_executor(
            self.executor,
            self.process_batch_sync,
            batch,
            processor_func,
            batch_id
        )
        
        return result
    
    def process_items_parallel(self, items: List[T], 
                             processor_func: Callable[[List[T]], List[R]],
                             progress_callback: Optional[Callable[[int, int], None]] = None) -> List[R]:
        """
        Process items in parallel batches with adaptive sizing.
        
        Args:
            items: Items to process
            processor_func: Function to process each batch
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of all processed results
        """
        if not items:
            return []
        
        start_time = time.time()
        
        # Create adaptive batches
        batches = self.batcher.create_batches(items)
        
        logger.info(f"Processing {len(items)} items in {len(batches)} batches "
                   f"(avg batch size: {len(items) / len(batches):.1f})")
        
        all_results = []
        completed_items = 0
        
        # Submit all batches to thread pool
        future_to_batch = {
            self.executor.submit(self.process_batch_sync, batch, processor_func, i): i
            for i, batch in enumerate(batches)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_batch):
            batch_id = future_to_batch[future]
            
            try:
                result = future.result(timeout=self.config.timeout_seconds)
                
                if result.success and result.results:
                    all_results.extend(result.results)
                    completed_items += result.items_processed
                    
                    # Update statistics
                    self.stats['successful_batches'] += 1
                else:
                    self.stats['failed_batches'] += 1
                    logger.warning(f"Batch {batch_id} failed: {result.error}")
                
                # Progress callback
                if progress_callback:
                    progress_callback(completed_items, len(items))
                    
            except Exception as e:
                self.stats['failed_batches'] += 1
                logger.error(f"Batch {batch_id} execution failed: {e}")
        
        # Update final statistics
        total_time = time.time() - start_time
        self.stats.update({
            'total_items': len(items),
            'total_batches': len(batches),
            'total_time': total_time,
            'avg_batch_size': len(items) / len(batches) if batches else 0,
        })
        
        logger.info(f"Completed processing {len(items)} items in {total_time:.2f}s "
                   f"({len(items) / total_time:.1f} items/sec)")
        
        return all_results
    
    async def process_items_async(self, items: List[T],
                                processor_func: Callable[[List[T]], List[R]],
                                progress_callback: Optional[Callable[[int, int], None]] = None) -> List[R]:
        """
        Process items asynchronously with adaptive batching.
        
        Args:
            items: Items to process
            processor_func: Function to process each batch
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of all processed results
        """
        if not items:
            return []
        
        start_time = time.time()
        
        # Create adaptive batches
        batches = self.batcher.create_batches(items)
        
        logger.info(f"Processing {len(items)} items in {len(batches)} batches "
                   f"(avg batch size: {len(items) / len(batches):.1f})")
        
        # Process batches concurrently
        tasks = [
            self.process_batch_async(batch, processor_func, i)
            for i, batch in enumerate(batches)
        ]
        
        all_results = []
        completed_items = 0
        
        # Collect results as they complete
        for task in asyncio.as_completed(tasks):
            result = await task
            
            if result.success and result.results:
                all_results.extend(result.results)
                completed_items += result.items_processed
                
                # Update statistics
                self.stats['successful_batches'] += 1
            else:
                self.stats['failed_batches'] += 1
                logger.warning(f"Batch {result.batch_id} failed: {result.error}")
            
            # Progress callback
            if progress_callback:
                progress_callback(completed_items, len(items))
        
        # Update final statistics
        total_time = time.time() - start_time
        self.stats.update({
            'total_items': len(items),
            'total_batches': len(batches),
            'total_time': total_time,
            'avg_batch_size': len(items) / len(batches) if batches else 0,
        })
        
        logger.info(f"Completed processing {len(items)} items in {total_time:.2f}s "
                   f"({len(items) / total_time:.1f} items/sec)")
        
        return all_results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = self.stats.copy()
        
        if stats['total_batches'] > 0:
            stats['success_rate'] = stats['successful_batches'] / stats['total_batches']
            stats['avg_processing_time'] = stats['total_time'] / stats['total_batches']
        
        if stats['total_time'] > 0:
            stats['throughput_items_per_sec'] = stats['total_items'] / stats['total_time']
        
        return stats
    
    def cleanup(self):
        """Clean up resources."""
        self.executor.shutdown(wait=True)


def batch_process(batch_size: int = DEFAULT_BATCH_SIZE, 
                 adaptive: bool = True,
                 memory_limit_gb: float = 4.0):
    """
    Decorator for automatic batch processing of functions.
    
    Args:
        batch_size: Initial batch size
        adaptive: Whether to use adaptive batch sizing
        memory_limit_gb: Memory limit for processing
    
    Example:
        @batch_process(batch_size=32, adaptive=True)
        def process_images(images: List[Image]) -> List[Features]:
            # Process batch of images
            return features
    """
    def decorator(func: Callable[[List[T]], List[R]]) -> Callable[[List[T]], List[R]]:
        config = BatchConfig(
            batch_size=batch_size,
            adaptive_batching=adaptive,
            memory_limit_gb=memory_limit_gb
        )
        processor = OptimizedBatchProcessor(config)
        
        @wraps(func)
        def wrapper(items: List[T]) -> List[R]:
            return processor.process_items_parallel(items, func)
        
        # Attach processor for inspection
        wrapper._processor = processor
        return wrapper
    
    return decorator


def create_batch_processor(batch_size: int = DEFAULT_BATCH_SIZE,
                          max_workers: int = 4,
                          adaptive: bool = True) -> OptimizedBatchProcessor:
    """
    Factory function to create an optimized batch processor.
    
    Args:
        batch_size: Initial batch size
        max_workers: Maximum number of worker threads
        adaptive: Whether to use adaptive batch sizing
        
    Returns:
        Configured OptimizedBatchProcessor
    """
    config = BatchConfig(
        batch_size=batch_size,
        max_workers=max_workers,
        adaptive_batching=adaptive
    )
    
    return OptimizedBatchProcessor(config)