"""
Multi-GPU pipeline parallelism for HistoCore Real-Time WSI Streaming.

Implements data parallelism and pipeline parallelism across multiple GPUs
for maximum throughput and <30s processing on gigapixel slides.
"""

import asyncio
import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel

from .gpu_pipeline import GPUPipeline, ThroughputMetrics
from .metrics import record_throughput_measurement, timed_operation
from .model_optimizer import ModelOptimizer, OptimizationConfig

logger = logging.getLogger(__name__)


@dataclass
class ParallelConfig:
    """Configuration for parallel processing."""
    
    # GPU allocation
    gpu_ids: List[int]
    primary_gpu: int = 0
    
    # Pipeline parallelism
    enable_pipeline_parallel: bool = True
    pipeline_stages: int = 2  # feature_extraction, attention_aggregation
    stage_overlap: bool = True
    
    # Data parallelism
    enable_data_parallel: bool = True
    batch_distribution: str = "round_robin"  # round_robin, load_balanced
    
    # Queue management
    queue_size: int = 100
    timeout_seconds: float = 30.0
    
    # Load balancing
    enable_load_balancing: bool = True
    load_balance_interval: float = 5.0
    
    # Performance
    prefetch_batches: int = 2
    async_processing: bool = True


class GPUWorker:
    """Individual GPU worker for parallel processing."""
    
    def __init__(
        self,
        gpu_id: int,
        model: nn.Module,
        config: ParallelConfig,
        optimization_config: OptimizationConfig
    ):
        """Initialize GPU worker."""
        self.gpu_id = gpu_id
        self.device = torch.device(f'cuda:{gpu_id}')
        self.config = config
        
        # Move model to GPU
        self.model = model.to(self.device)
        self.model.eval()
        
        # Initialize GPU pipeline
        self.pipeline = GPUPipeline(
            model=self.model,
            batch_size=64,
            gpu_ids=[gpu_id],
            enable_fp16=optimization_config.enable_mixed_precision,
            enable_advanced_memory_optimization=True
        )
        
        # Processing queues
        self.input_queue = queue.Queue(maxsize=config.queue_size)
        self.output_queue = queue.Queue(maxsize=config.queue_size)
        
        # Worker state
        self.is_running = False
        self.worker_thread = None
        self.processed_batches = 0
        self.total_processing_time = 0.0
        self.current_load = 0.0
        
        logger.info("GPU worker initialized: gpu_id=%d device=%s", gpu_id, self.device)
    
    def start(self):
        """Start worker thread."""
        if self.is_running:
            return
        
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        
        logger.info("GPU worker started: gpu_id=%d", self.gpu_id)
    
    def stop(self):
        """Stop worker thread."""
        self.is_running = False
        
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
        
        logger.info("GPU worker stopped: gpu_id=%d", self.gpu_id)
    
    def _worker_loop(self):
        """Main worker processing loop."""
        while self.is_running:
            try:
                # Get batch from input queue
                batch_data = self.input_queue.get(timeout=1.0)
                
                if batch_data is None:  # Shutdown signal
                    break
                
                batch_id, patches, metadata = batch_data
                
                # Process batch
                start_time = time.time()
                
                with torch.no_grad():
                    features = self.pipeline._process_batch_sync(patches)
                
                processing_time = time.time() - start_time
                
                # Update statistics
                self.processed_batches += 1
                self.total_processing_time += processing_time
                self.current_load = self.input_queue.qsize() / self.config.queue_size
                
                # Put result in output queue
                result = {
                    'batch_id': batch_id,
                    'features': features,
                    'metadata': metadata,
                    'processing_time': processing_time,
                    'gpu_id': self.gpu_id
                }
                
                self.output_queue.put(result, timeout=self.config.timeout_seconds)
                
            except queue.Empty:
                continue
            except queue.Full:
                logger.warning("Output queue full for GPU %d", self.gpu_id)
            except Exception as e:
                logger.error("Worker error on GPU %d: %s", self.gpu_id, e)
    
    def submit_batch(self, batch_id: int, patches: torch.Tensor, metadata: Dict[str, Any]) -> bool:
        """Submit batch for processing."""
        try:
            batch_data = (batch_id, patches, metadata)
            self.input_queue.put(batch_data, timeout=0.1)
            return True
        except queue.Full:
            return False
    
    def get_result(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """Get processing result."""
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        avg_time = (
            self.total_processing_time / self.processed_batches
            if self.processed_batches > 0 else 0.0
        )
        
        return {
            'gpu_id': self.gpu_id,
            'processed_batches': self.processed_batches,
            'avg_processing_time': avg_time,
            'current_load': self.current_load,
            'input_queue_size': self.input_queue.qsize(),
            'output_queue_size': self.output_queue.qsize(),
            'throughput_batches_per_sec': (
                self.processed_batches / self.total_processing_time
                if self.total_processing_time > 0 else 0.0
            )
        }
    
    def cleanup(self):
        """Clean up worker resources."""
        self.stop()
        self.pipeline.cleanup()


class LoadBalancer:
    """Load balancer for distributing work across GPU workers."""
    
    def __init__(self, workers: List[GPUWorker], config: ParallelConfig):
        """Initialize load balancer."""
        self.workers = workers
        self.config = config
        self.current_worker_idx = 0
        self.worker_loads = {worker.gpu_id: 0.0 for worker in workers}
        
        logger.info("Load balancer initialized: %d workers", len(workers))
    
    def select_worker(self) -> GPUWorker:
        """Select best worker for next batch."""
        if self.config.batch_distribution == "round_robin":
            return self._round_robin_selection()
        elif self.config.batch_distribution == "load_balanced":
            return self._load_balanced_selection()
        else:
            return self.workers[0]
    
    def _round_robin_selection(self) -> GPUWorker:
        """Round-robin worker selection."""
        worker = self.workers[self.current_worker_idx]
        self.current_worker_idx = (self.current_worker_idx + 1) % len(self.workers)
        return worker
    
    def _load_balanced_selection(self) -> GPUWorker:
        """Load-balanced worker selection."""
        # Update worker loads
        for worker in self.workers:
            self.worker_loads[worker.gpu_id] = worker.current_load
        
        # Select worker with lowest load
        best_worker = min(self.workers, key=lambda w: self.worker_loads[w.gpu_id])
        return best_worker
    
    def get_load_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics."""
        return {
            'worker_loads': self.worker_loads.copy(),
            'total_workers': len(self.workers),
            'distribution_strategy': self.config.batch_distribution
        }


class PipelineStage:
    """Individual stage in pipeline parallelism."""
    
    def __init__(
        self,
        stage_id: int,
        stage_name: str,
        model_part: nn.Module,
        gpu_id: int,
        config: ParallelConfig
    ):
        """Initialize pipeline stage."""
        self.stage_id = stage_id
        self.stage_name = stage_name
        self.gpu_id = gpu_id
        self.device = torch.device(f'cuda:{gpu_id}')
        self.config = config
        
        # Move model part to GPU
        self.model_part = model_part.to(self.device)
        self.model_part.eval()
        
        # Stage queues
        self.input_queue = queue.Queue(maxsize=config.queue_size)
        self.output_queue = queue.Queue(maxsize=config.queue_size)
        
        # Stage state
        self.is_running = False
        self.stage_thread = None
        self.processed_items = 0
        
        logger.info("Pipeline stage initialized: stage_id=%d name=%s gpu_id=%d",
                   stage_id, stage_name, gpu_id)
    
    def start(self):
        """Start stage processing."""
        if self.is_running:
            return
        
        self.is_running = True
        self.stage_thread = threading.Thread(target=self._stage_loop, daemon=True)
        self.stage_thread.start()
        
        logger.info("Pipeline stage started: %s", self.stage_name)
    
    def stop(self):
        """Stop stage processing."""
        self.is_running = False
        
        if self.stage_thread:
            self.stage_thread.join(timeout=5.0)
        
        logger.info("Pipeline stage stopped: %s", self.stage_name)
    
    def _stage_loop(self):
        """Main stage processing loop."""
        while self.is_running:
            try:
                # Get input from previous stage
                stage_input = self.input_queue.get(timeout=1.0)
                
                if stage_input is None:  # Shutdown signal
                    break
                
                # Process through model part
                with torch.no_grad():
                    stage_output = self.model_part(stage_input)
                
                # Send to next stage
                self.output_queue.put(stage_output, timeout=self.config.timeout_seconds)
                self.processed_items += 1
                
            except queue.Empty:
                continue
            except queue.Full:
                logger.warning("Output queue full for stage %s", self.stage_name)
            except Exception as e:
                logger.error("Stage error in %s: %s", self.stage_name, e)
    
    def submit_input(self, stage_input: torch.Tensor) -> bool:
        """Submit input to stage."""
        try:
            self.input_queue.put(stage_input, timeout=0.1)
            return True
        except queue.Full:
            return False
    
    def get_output(self, timeout: float = 1.0) -> Optional[torch.Tensor]:
        """Get stage output."""
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None


class ParallelPipeline:
    """Multi-GPU parallel processing pipeline."""
    
    def __init__(
        self,
        model: nn.Module,
        config: ParallelConfig,
        optimization_config: OptimizationConfig
    ):
        """Initialize parallel pipeline."""
        self.model = model
        self.config = config
        self.optimization_config = optimization_config
        
        # Validate GPU availability
        available_gpus = list(range(torch.cuda.device_count()))
        self.gpu_ids = [gpu_id for gpu_id in config.gpu_ids if gpu_id in available_gpus]
        
        if not self.gpu_ids:
            raise RuntimeError("No valid GPUs available for parallel processing")
        
        # Initialize workers
        self.workers = []
        self.load_balancer = None
        self.pipeline_stages = []
        
        # Processing state
        self.is_running = False
        self.batch_counter = 0
        self.pending_results = {}
        self.result_queue = queue.Queue()
        
        # Performance tracking
        self.total_batches_processed = 0
        self.total_processing_time = 0.0
        
        self._setup_parallel_processing()
        
        logger.info("Parallel pipeline initialized: gpus=%s data_parallel=%s pipeline_parallel=%s",
                   self.gpu_ids, config.enable_data_parallel, config.enable_pipeline_parallel)
    
    def _setup_parallel_processing(self):
        """Setup parallel processing based on configuration."""
        if self.config.enable_data_parallel and len(self.gpu_ids) > 1:
            self._setup_data_parallel()
        elif self.config.enable_pipeline_parallel and len(self.gpu_ids) > 1:
            self._setup_pipeline_parallel()
        else:
            # Single GPU fallback
            self._setup_single_gpu()
    
    def _setup_data_parallel(self):
        """Setup data parallelism across GPUs."""
        logger.info("Setting up data parallelism across %d GPUs", len(self.gpu_ids))
        
        # Create workers for each GPU
        for gpu_id in self.gpu_ids:
            worker = GPUWorker(gpu_id, self.model, self.config, self.optimization_config)
            self.workers.append(worker)
        
        # Initialize load balancer
        self.load_balancer = LoadBalancer(self.workers, self.config)
    
    def _setup_pipeline_parallel(self):
        """Setup pipeline parallelism across GPUs."""
        logger.info("Setting up pipeline parallelism across %d GPUs", len(self.gpu_ids))
        
        # Split model into stages
        # This is model-specific and would need customization
        # For now, implement a simple 2-stage split
        
        if hasattr(self.model, 'feature_proj'):
            # Stage 1: Feature projection
            stage1_model = nn.Sequential(self.model.feature_proj)
            stage1 = PipelineStage(0, "feature_projection", stage1_model, self.gpu_ids[0], self.config)
            self.pipeline_stages.append(stage1)
            
            # Stage 2: Attention and classification
            remaining_modules = []
            for name, module in self.model.named_children():
                if name != 'feature_proj':
                    remaining_modules.append(module)
            
            stage2_model = nn.Sequential(*remaining_modules)
            stage2_gpu = self.gpu_ids[1] if len(self.gpu_ids) > 1 else self.gpu_ids[0]
            stage2 = PipelineStage(1, "attention_classification", stage2_model, stage2_gpu, self.config)
            self.pipeline_stages.append(stage2)
        else:
            logger.warning("Model structure not suitable for pipeline parallelism, falling back to data parallel")
            self._setup_data_parallel()
    
    def _setup_single_gpu(self):
        """Setup single GPU processing."""
        logger.info("Setting up single GPU processing: gpu_id=%d", self.gpu_ids[0])
        
        worker = GPUWorker(self.gpu_ids[0], self.model, self.config, self.optimization_config)
        self.workers.append(worker)
    
    def start(self):
        """Start parallel processing."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start workers
        for worker in self.workers:
            worker.start()
        
        # Start pipeline stages
        for stage in self.pipeline_stages:
            stage.start()
        
        # Start result collection thread
        self.result_thread = threading.Thread(target=self._collect_results, daemon=True)
        self.result_thread.start()
        
        logger.info("Parallel pipeline started")
    
    def stop(self):
        """Stop parallel processing."""
        self.is_running = False
        
        # Stop workers
        for worker in self.workers:
            worker.stop()
        
        # Stop pipeline stages
        for stage in self.pipeline_stages:
            stage.stop()
        
        logger.info("Parallel pipeline stopped")
    
    def _collect_results(self):
        """Collect results from workers."""
        while self.is_running:
            try:
                # Collect from data parallel workers
                for worker in self.workers:
                    result = worker.get_result(timeout=0.1)
                    if result:
                        self.result_queue.put(result)
                
                # Collect from pipeline stages
                if self.pipeline_stages:
                    final_stage = self.pipeline_stages[-1]
                    result = final_stage.get_output(timeout=0.1)
                    if result is not None:
                        self.result_queue.put({
                            'batch_id': self.batch_counter,
                            'features': result,
                            'metadata': {},
                            'processing_time': 0.0,
                            'gpu_id': final_stage.gpu_id
                        })
                
                time.sleep(0.01)  # Small delay to prevent busy waiting
                
            except Exception as e:
                logger.error("Result collection error: %s", e)
    
    @timed_operation("parallel_batch_processing")
    def process_batch(
        self,
        patches: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """Process batch through parallel pipeline."""
        if not self.is_running:
            raise RuntimeError("Pipeline not started")
        
        batch_id = self.batch_counter
        self.batch_counter += 1
        
        if metadata is None:
            metadata = {}
        
        start_time = time.time()
        
        # Submit to appropriate processing path
        if self.workers and not self.pipeline_stages:
            # Data parallel processing
            success = self._submit_to_data_parallel(batch_id, patches, metadata)
        elif self.pipeline_stages:
            # Pipeline parallel processing
            success = self._submit_to_pipeline_parallel(batch_id, patches, metadata)
        else:
            raise RuntimeError("No processing path available")
        
        if not success:
            raise RuntimeError("Failed to submit batch for processing")
        
        # Wait for result
        result = self._wait_for_result(batch_id)
        
        processing_time = time.time() - start_time
        self.total_batches_processed += 1
        self.total_processing_time += processing_time
        
        # Record metrics
        throughput = patches.shape[0] / processing_time
        record_throughput_measurement(throughput, str(result.get('gpu_id', 0)))
        
        return result['features']
    
    def _submit_to_data_parallel(
        self,
        batch_id: int,
        patches: torch.Tensor,
        metadata: Dict[str, Any]
    ) -> bool:
        """Submit batch to data parallel workers."""
        if self.load_balancer:
            worker = self.load_balancer.select_worker()
        else:
            worker = self.workers[0]
        
        return worker.submit_batch(batch_id, patches, metadata)
    
    def _submit_to_pipeline_parallel(
        self,
        batch_id: int,
        patches: torch.Tensor,
        metadata: Dict[str, Any]
    ) -> bool:
        """Submit batch to pipeline parallel stages."""
        if not self.pipeline_stages:
            return False
        
        # Submit to first stage
        first_stage = self.pipeline_stages[0]
        return first_stage.submit_input(patches)
    
    def _wait_for_result(self, batch_id: int, timeout: float = 30.0) -> Dict[str, Any]:
        """Wait for processing result."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                result = self.result_queue.get(timeout=1.0)
                
                # For pipeline parallel, batch_id might not match exactly
                if self.pipeline_stages or result['batch_id'] == batch_id:
                    return result
                else:
                    # Put back if not our result
                    self.result_queue.put(result)
                
            except queue.Empty:
                continue
        
        raise TimeoutError(f"Timeout waiting for batch {batch_id} result")
    
    def get_throughput_stats(self) -> Dict[str, Any]:
        """Get parallel processing throughput statistics."""
        # Aggregate worker stats
        worker_stats = [worker.get_stats() for worker in self.workers]
        
        # Calculate overall throughput
        total_throughput = sum(stats['throughput_batches_per_sec'] for stats in worker_stats)
        avg_processing_time = (
            self.total_processing_time / self.total_batches_processed
            if self.total_batches_processed > 0 else 0.0
        )
        
        stats = {
            'total_throughput_batches_per_sec': total_throughput,
            'avg_processing_time_ms': avg_processing_time * 1000,
            'total_batches_processed': self.total_batches_processed,
            'active_gpus': len(self.gpu_ids),
            'worker_stats': worker_stats
        }
        
        # Add load balancer stats
        if self.load_balancer:
            stats['load_balancer'] = self.load_balancer.get_load_stats()
        
        return stats
    
    def cleanup(self):
        """Clean up parallel processing resources."""
        self.stop()
        
        for worker in self.workers:
            worker.cleanup()
        
        logger.info("Parallel pipeline cleaned up")
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


# Convenience functions
def create_parallel_pipeline(
    model: nn.Module,
    gpu_ids: List[int],
    enable_data_parallel: bool = True,
    enable_pipeline_parallel: bool = False,
    optimization_config: Optional[OptimizationConfig] = None
) -> ParallelPipeline:
    """Create parallel processing pipeline."""
    
    if optimization_config is None:
        from .model_optimizer import get_optimization_config
        optimization_config = get_optimization_config()
    
    parallel_config = ParallelConfig(
        gpu_ids=gpu_ids,
        enable_data_parallel=enable_data_parallel,
        enable_pipeline_parallel=enable_pipeline_parallel
    )
    
    return ParallelPipeline(model, parallel_config, optimization_config)


def benchmark_parallel_performance(
    model: nn.Module,
    gpu_ids: List[int],
    dummy_input: torch.Tensor,
    num_batches: int = 100
) -> Dict[str, Any]:
    """Benchmark parallel processing performance."""
    
    results = {}
    
    # Test single GPU
    single_config = ParallelConfig(gpu_ids=[gpu_ids[0]], enable_data_parallel=False)
    single_pipeline = ParallelPipeline(model, single_config, OptimizationConfig())
    
    with single_pipeline:
        start_time = time.time()
        for _ in range(num_batches):
            _ = single_pipeline.process_batch(dummy_input)
        single_time = time.time() - start_time
    
    results['single_gpu'] = {
        'total_time_s': single_time,
        'avg_batch_time_ms': (single_time / num_batches) * 1000,
        'throughput_batches_per_sec': num_batches / single_time
    }
    
    # Test multi-GPU if available
    if len(gpu_ids) > 1:
        multi_config = ParallelConfig(gpu_ids=gpu_ids, enable_data_parallel=True)
        multi_pipeline = ParallelPipeline(model, multi_config, OptimizationConfig())
        
        with multi_pipeline:
            start_time = time.time()
            for _ in range(num_batches):
                _ = multi_pipeline.process_batch(dummy_input)
            multi_time = time.time() - start_time
        
        results['multi_gpu'] = {
            'total_time_s': multi_time,
            'avg_batch_time_ms': (multi_time / num_batches) * 1000,
            'throughput_batches_per_sec': num_batches / multi_time,
            'speedup': single_time / multi_time,
            'efficiency': (single_time / multi_time) / len(gpu_ids)
        }
    
    return results