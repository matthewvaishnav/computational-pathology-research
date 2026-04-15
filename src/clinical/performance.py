"""
Performance optimization module for real-time clinical inference.

This module provides GPU acceleration, batch processing, and performance monitoring
for the clinical workflow system to achieve <5 second inference times.
"""

import logging
import time
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ..utils.monitoring import MetricsTracker, ResourceMonitor

logger = logging.getLogger(__name__)


class InferenceProfiler:
    """
    Profiler for measuring inference performance and identifying bottlenecks.

    Tracks timing for different stages of the inference pipeline:
    - Feature extraction
    - WSI encoding
    - Classification
    - Post-processing
    """

    def __init__(self):
        self.timings = {}
        self.current_stage = None
        self.stage_start_time = None

    @contextmanager
    def profile_stage(self, stage_name: str):
        """Context manager for profiling a specific stage."""
        self.current_stage = stage_name
        self.stage_start_time = time.perf_counter()

        try:
            yield
        finally:
            elapsed = time.perf_counter() - self.stage_start_time
            if stage_name not in self.timings:
                self.timings[stage_name] = []
            self.timings[stage_name].append(elapsed)

    def get_average_timings(self) -> Dict[str, float]:
        """Get average timing for each stage."""
        return {stage: sum(times) / len(times) for stage, times in self.timings.items()}

    def get_total_time(self) -> float:
        """Get total average inference time."""
        averages = self.get_average_timings()
        return sum(averages.values())

    def reset(self):
        """Reset all timing data."""
        self.timings.clear()

    def log_performance_summary(self):
        """Log performance summary."""
        averages = self.get_average_timings()
        total_time = self.get_total_time()

        logger.info("=== Inference Performance Summary ===")
        logger.info(f"Total inference time: {total_time:.3f}s")

        for stage, avg_time in averages.items():
            percentage = (avg_time / total_time) * 100 if total_time > 0 else 0
            logger.info(f"  {stage}: {avg_time:.3f}s ({percentage:.1f}%)")


class GPUAccelerator:
    """
    GPU acceleration utilities for optimized inference.

    Provides:
    - Automatic device selection
    - Memory management
    - Batch optimization
    - Mixed precision support
    """

    def __init__(self, device: Optional[str] = None, use_mixed_precision: bool = True):
        self.device = self._select_device(device)
        self.use_mixed_precision = use_mixed_precision and self.device.type == "cuda"

        if self.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        logger.info(f"GPU Accelerator initialized on device: {self.device}")
        if self.use_mixed_precision:
            logger.info("Mixed precision enabled")

    def _select_device(self, device: Optional[str] = None) -> torch.device:
        """Select optimal device for inference."""
        if device is not None:
            return torch.device(device)

        if torch.cuda.is_available():
            # Select GPU with most free memory
            gpu_count = torch.cuda.device_count()
            if gpu_count > 1:
                max_free_memory = 0
                best_gpu = 0

                for i in range(gpu_count):
                    torch.cuda.set_device(i)
                    free_memory = torch.cuda.get_device_properties(
                        i
                    ).total_memory - torch.cuda.memory_allocated(i)
                    if free_memory > max_free_memory:
                        max_free_memory = free_memory
                        best_gpu = i

                return torch.device(f"cuda:{best_gpu}")
            else:
                return torch.device("cuda:0")
        else:
            logger.warning("CUDA not available, falling back to CPU")
            return torch.device("cpu")

    def move_to_device(
        self, data: Union[torch.Tensor, Dict, List]
    ) -> Union[torch.Tensor, Dict, List]:
        """Move data to the selected device."""
        if isinstance(data, torch.Tensor):
            return data.to(self.device, non_blocking=True)
        elif isinstance(data, dict):
            return {k: self.move_to_device(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.move_to_device(item) for item in data]
        else:
            return data

    @contextmanager
    def autocast_context(self):
        """Context manager for mixed precision inference."""
        if self.use_mixed_precision:
            with torch.cuda.amp.autocast():
                yield
        else:
            yield

    def optimize_memory(self):
        """Optimize GPU memory usage."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


class BatchProcessor:
    """
    Batch processing system for handling multiple concurrent inference requests.

    Features:
    - Dynamic batch sizing
    - Queue management
    - Load balancing
    - Performance monitoring
    """

    def __init__(
        self, max_batch_size: int = 32, max_queue_size: int = 100, timeout_seconds: float = 1.0
    ):
        self.max_batch_size = max_batch_size
        self.max_queue_size = max_queue_size
        self.timeout_seconds = timeout_seconds

        self.request_queue = []
        self.metrics_tracker = MetricsTracker()

        logger.info(f"BatchProcessor initialized with max_batch_size={max_batch_size}")

    def add_request(self, request_data: Dict) -> str:
        """Add inference request to queue."""
        if len(self.request_queue) >= self.max_queue_size:
            raise RuntimeError(f"Request queue full (max_size={self.max_queue_size})")

        request_id = f"req_{int(time.time() * 1000000)}"
        request = {
            "id": request_id,
            "data": request_data,
            "timestamp": time.time(),
            "processed": False,
            "result": None,
        }

        self.request_queue.append(request)
        return request_id

    def get_batch(self) -> Tuple[List[Dict], List[str]]:
        """Get next batch of requests to process."""
        if not self.request_queue:
            return [], []

        # Get up to max_batch_size unprocessed requests
        batch_requests = []
        batch_ids = []

        for request in self.request_queue:
            if not request["processed"] and len(batch_requests) < self.max_batch_size:
                batch_requests.append(request["data"])
                batch_ids.append(request["id"])

        return batch_requests, batch_ids

    def mark_processed(self, request_ids: List[str], results: List[Dict]):
        """Mark requests as processed with results."""
        id_to_result = dict(zip(request_ids, results))

        for request in self.request_queue:
            if request["id"] in id_to_result:
                request["processed"] = True
                request["result"] = id_to_result[request["id"]]

        # Log processing metrics
        processing_time = time.time() - min(
            req["timestamp"] for req in self.request_queue if req["id"] in request_ids
        )

        self.metrics_tracker.log_metric("batch_processing_time", processing_time)
        self.metrics_tracker.log_metric("batch_size", len(request_ids))

    def get_result(self, request_id: str) -> Optional[Dict]:
        """Get result for a specific request."""
        for request in self.request_queue:
            if request["id"] == request_id and request["processed"]:
                return request["result"]
        return None

    def cleanup_old_requests(self, max_age_seconds: float = 300):
        """Remove old processed requests from queue."""
        current_time = time.time()
        self.request_queue = [
            req for req in self.request_queue if current_time - req["timestamp"] < max_age_seconds
        ]


class OptimizedInferencePipeline:
    """
    Optimized inference pipeline for real-time clinical workflow.

    Combines GPU acceleration, batch processing, and performance monitoring
    to achieve <5 second inference times for slides with up to 10,000 patches.
    """

    def __init__(
        self,
        feature_extractor: nn.Module,
        wsi_encoder: nn.Module,
        classifier: nn.Module,
        device: Optional[str] = None,
        max_batch_size: int = 32,
        target_patches_per_second: int = 100,
        use_mixed_precision: bool = True,
    ):
        self.feature_extractor = feature_extractor
        self.wsi_encoder = wsi_encoder
        self.classifier = classifier

        # Initialize GPU acceleration
        self.gpu_accelerator = GPUAccelerator(device, use_mixed_precision)

        # Move models to device
        self.feature_extractor = self.feature_extractor.to(self.gpu_accelerator.device)
        self.wsi_encoder = self.wsi_encoder.to(self.gpu_accelerator.device)
        self.classifier = self.classifier.to(self.gpu_accelerator.device)

        # Set models to eval mode
        self.feature_extractor.eval()
        self.wsi_encoder.eval()
        self.classifier.eval()

        # Initialize batch processor
        self.batch_processor = BatchProcessor(max_batch_size=max_batch_size)

        # Initialize profiler and monitoring
        self.profiler = InferenceProfiler()
        self.resource_monitor = ResourceMonitor()
        self.metrics_tracker = MetricsTracker()

        self.target_patches_per_second = target_patches_per_second

        logger.info("OptimizedInferencePipeline initialized")

    def process_wsi_patches(
        self, patches: torch.Tensor, batch_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Process WSI patches with optimized batching for target throughput.

        Args:
            patches: WSI patches [num_patches, channels, height, width]
            batch_size: Override batch size for patch processing

        Returns:
            Patch features [num_patches, feature_dim]
        """
        if batch_size is None:
            # Calculate optimal batch size based on target throughput
            batch_size = self._calculate_optimal_batch_size(patches.shape[0])

        patches = self.gpu_accelerator.move_to_device(patches)

        with self.profiler.profile_stage("feature_extraction"):
            patch_features = []

            with torch.no_grad():
                for i in range(0, patches.shape[0], batch_size):
                    batch_patches = patches[i : i + batch_size]

                    with self.gpu_accelerator.autocast_context():
                        batch_features = self.feature_extractor(batch_patches)

                    patch_features.append(batch_features.cpu())

                    # Clear GPU cache periodically
                    if i % (batch_size * 10) == 0:
                        self.gpu_accelerator.optimize_memory()

            features = torch.cat(patch_features, dim=0)

        return features

    def _calculate_optimal_batch_size(self, num_patches: int) -> int:
        """Calculate optimal batch size based on target throughput and GPU memory."""
        # Start with a reasonable default
        base_batch_size = 64

        # Adjust based on GPU memory
        if self.gpu_accelerator.device.type == "cuda":
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

            if gpu_memory_gb >= 24:  # RTX 4090, A6000, etc.
                base_batch_size = 128
            elif gpu_memory_gb >= 12:  # RTX 4070, RTX 3080, etc.
                base_batch_size = 64
            elif gpu_memory_gb >= 8:  # RTX 3070, etc.
                base_batch_size = 32
            else:
                base_batch_size = 16

        # Ensure we can process patches fast enough
        target_time_per_patch = 1.0 / self.target_patches_per_second
        estimated_time_per_batch = target_time_per_patch * base_batch_size

        # If estimated time is too high, reduce batch size
        if estimated_time_per_batch > 0.5:  # 500ms max per batch
            base_batch_size = max(16, int(0.5 / target_time_per_patch))

        return min(base_batch_size, num_patches)

    def inference_single(
        self, wsi_patches: torch.Tensor, patient_context: Optional[Dict] = None
    ) -> Dict:
        """
        Perform inference on a single WSI case.

        Args:
            wsi_patches: WSI patches [num_patches, channels, height, width]
            patient_context: Optional patient context data

        Returns:
            Inference results including predictions and timing metrics
        """
        start_time = time.time()

        # Process patches to features
        with self.profiler.profile_stage("patch_processing"):
            patch_features = self.process_wsi_patches(wsi_patches)

        # Move to device for WSI encoding
        patch_features = self.gpu_accelerator.move_to_device(patch_features)

        # Encode WSI
        with self.profiler.profile_stage("wsi_encoding"):
            with torch.no_grad():
                with self.gpu_accelerator.autocast_context():
                    # Add batch dimension
                    patch_features_batch = patch_features.unsqueeze(
                        0
                    )  # [1, num_patches, feature_dim]
                    wsi_embedding = self.wsi_encoder(patch_features_batch)  # [1, embed_dim]

        # Classification
        with self.profiler.profile_stage("classification"):
            with torch.no_grad():
                with self.gpu_accelerator.autocast_context():
                    predictions = self.classifier(wsi_embedding)

        # Post-processing
        with self.profiler.profile_stage("post_processing"):
            # Move results to CPU
            for key, value in predictions.items():
                if isinstance(value, torch.Tensor):
                    predictions[key] = value.cpu()

        total_time = time.time() - start_time

        # Log performance metrics
        self.metrics_tracker.log_metric("inference_time", total_time)
        self.metrics_tracker.log_metric("num_patches", wsi_patches.shape[0])
        self.metrics_tracker.log_metric("patches_per_second", wsi_patches.shape[0] / total_time)

        # Log warning if inference exceeds 5 seconds
        if total_time > 5.0:
            logger.warning(
                f"Inference time exceeded 5 seconds: {total_time:.2f}s "
                f"for {wsi_patches.shape[0]} patches"
            )
            self.metrics_tracker.log_metric("slow_inference_count", 1)

        # Add timing information to results
        predictions["inference_time"] = total_time
        predictions["num_patches"] = wsi_patches.shape[0]
        predictions["patches_per_second"] = wsi_patches.shape[0] / total_time

        return predictions

    def inference_batch(self, batch_data: List[Dict]) -> List[Dict]:
        """
        Perform batch inference on multiple cases.

        Args:
            batch_data: List of inference requests, each containing 'wsi_patches' and optional 'patient_context'

        Returns:
            List of inference results
        """
        if not batch_data:
            return []

        start_time = time.time()
        results = []

        # Process each case in the batch
        for case_data in batch_data:
            wsi_patches = case_data["wsi_patches"]
            patient_context = case_data.get("patient_context")

            case_result = self.inference_single(wsi_patches, patient_context)
            results.append(case_result)

        batch_time = time.time() - start_time

        # Log batch metrics
        self.metrics_tracker.log_metric("batch_inference_time", batch_time)
        self.metrics_tracker.log_metric("batch_size", len(batch_data))
        self.metrics_tracker.log_metric("avg_time_per_case", batch_time / len(batch_data))

        return results

    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics."""
        gpu_stats = (
            self.resource_monitor.get_gpu_usage()
            if self.gpu_accelerator.device.type == "cuda"
            else {}
        )

        return {
            "profiler_timings": self.profiler.get_average_timings(),
            "total_inference_time": self.profiler.get_total_time(),
            "metrics": {
                "avg_inference_time": self.metrics_tracker.get_average("inference_time"),
                "avg_patches_per_second": self.metrics_tracker.get_average("patches_per_second"),
                "slow_inference_count": self.metrics_tracker.get_latest("slow_inference_count")
                or 0,
            },
            "gpu_stats": gpu_stats,
            "device": str(self.gpu_accelerator.device),
            "mixed_precision": self.gpu_accelerator.use_mixed_precision,
        }

    def reset_metrics(self):
        """Reset all performance metrics."""
        self.profiler.reset()
        self.metrics_tracker.reset()
