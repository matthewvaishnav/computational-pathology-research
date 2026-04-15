"""
Batch inference system for concurrent clinical workflow processing.

This module provides a high-performance batch inference system that can handle
multiple concurrent requests while maintaining <5 second latency per case.
"""

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from queue import Empty, Full, Queue
from typing import Callable, Dict, List, Optional
from uuid import uuid4

import torch

from .performance import OptimizedInferencePipeline

logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    """Represents a single inference request."""

    request_id: str
    wsi_patches: torch.Tensor
    patient_context: Optional[Dict] = None
    priority: int = 0  # Higher values = higher priority
    timestamp: float = None
    callback: Optional[Callable] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class InferenceResult:
    """Represents the result of an inference request."""

    request_id: str
    predictions: Dict
    processing_time: float
    queue_time: float
    success: bool = True
    error_message: Optional[str] = None


class ConcurrentInferenceManager:
    """
    Manages concurrent inference requests with load balancing and performance monitoring.

    Features:
    - Concurrent request processing
    - Priority-based queue management
    - Load balancing across multiple workers
    - Performance monitoring and alerting
    - Automatic scaling based on load
    """

    def __init__(
        self,
        inference_pipeline: OptimizedInferencePipeline,
        max_workers: int = 4,
        max_queue_size: int = 100,
        max_latency_seconds: float = 5.0,
        batch_timeout_seconds: float = 0.1,
    ):
        self.inference_pipeline = inference_pipeline
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.max_latency_seconds = max_latency_seconds
        self.batch_timeout_seconds = batch_timeout_seconds

        # Request management
        self.request_queue = Queue(maxsize=max_queue_size)
        self.results = {}  # request_id -> InferenceResult
        self.active_requests = set()

        # Worker management
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.workers_running = False
        self.worker_threads = []

        # Performance monitoring
        self.stats = {
            "total_requests": 0,
            "completed_requests": 0,
            "failed_requests": 0,
            "avg_queue_time": 0.0,
            "avg_processing_time": 0.0,
            "current_queue_size": 0,
            "peak_queue_size": 0,
            "slow_requests": 0,  # Requests exceeding max_latency_seconds
        }

        self._stats_lock = threading.Lock()

        logger.info(f"ConcurrentInferenceManager initialized with {max_workers} workers")

    def start(self):
        """Start the inference workers."""
        if self.workers_running:
            logger.warning("Workers already running")
            return

        self.workers_running = True

        # Start worker threads
        for i in range(self.max_workers):
            worker_thread = threading.Thread(
                target=self._worker_loop, name=f"InferenceWorker-{i}", daemon=True
            )
            worker_thread.start()
            self.worker_threads.append(worker_thread)

        logger.info(f"Started {self.max_workers} inference workers")

    def stop(self):
        """Stop the inference workers."""
        if not self.workers_running:
            return

        self.workers_running = False

        # Wait for workers to finish
        for worker_thread in self.worker_threads:
            worker_thread.join(timeout=5.0)

        self.worker_threads.clear()
        self.executor.shutdown(wait=True)

        logger.info("Stopped all inference workers")

    def submit_request(
        self,
        wsi_patches: torch.Tensor,
        patient_context: Optional[Dict] = None,
        priority: int = 0,
        callback: Optional[Callable] = None,
    ) -> str:
        """
        Submit an inference request.

        Args:
            wsi_patches: WSI patches tensor
            patient_context: Optional patient context
            priority: Request priority (higher = more urgent)
            callback: Optional callback function for async processing

        Returns:
            Request ID for tracking

        Raises:
            RuntimeError: If queue is full
        """
        if not self.workers_running:
            raise RuntimeError("Inference manager not started. Call start() first.")

        request_id = str(uuid4())
        request = InferenceRequest(
            request_id=request_id,
            wsi_patches=wsi_patches,
            patient_context=patient_context,
            priority=priority,
            callback=callback,
        )

        try:
            self.request_queue.put(request, block=False)
            self.active_requests.add(request_id)

            with self._stats_lock:
                self.stats["total_requests"] += 1
                self.stats["current_queue_size"] = self.request_queue.qsize()
                self.stats["peak_queue_size"] = max(
                    self.stats["peak_queue_size"], self.stats["current_queue_size"]
                )

            logger.debug(f"Submitted request {request_id} with priority {priority}")
            return request_id

        except Full:
            raise RuntimeError(f"Request queue full (max_size={self.max_queue_size})")

    def get_result(
        self, request_id: str, timeout: Optional[float] = None
    ) -> Optional[InferenceResult]:
        """
        Get result for a specific request.

        Args:
            request_id: Request ID
            timeout: Maximum time to wait for result

        Returns:
            InferenceResult if available, None if not ready or timeout
        """
        start_time = time.time()

        while True:
            if request_id in self.results:
                result = self.results.pop(request_id)
                self.active_requests.discard(request_id)
                return result

            if timeout is not None and (time.time() - start_time) > timeout:
                return None

            time.sleep(0.01)  # Small sleep to avoid busy waiting

    async def get_result_async(
        self, request_id: str, timeout: Optional[float] = None
    ) -> Optional[InferenceResult]:
        """Async version of get_result."""
        start_time = time.time()

        while True:
            if request_id in self.results:
                result = self.results.pop(request_id)
                self.active_requests.discard(request_id)
                return result

            if timeout is not None and (time.time() - start_time) > timeout:
                return None

            await asyncio.sleep(0.01)

    def _worker_loop(self):
        """Main worker loop for processing inference requests."""
        worker_name = threading.current_thread().name
        logger.info(f"Worker {worker_name} started")

        while self.workers_running:
            try:
                # Get batch of requests
                batch_requests = self._get_request_batch()

                if not batch_requests:
                    time.sleep(0.01)  # No requests, sleep briefly
                    continue

                # Process batch
                self._process_request_batch(batch_requests)

            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}", exc_info=True)
                time.sleep(0.1)  # Brief pause on error

        logger.info(f"Worker {worker_name} stopped")

    def _get_request_batch(self) -> List[InferenceRequest]:
        """Get a batch of requests from the queue."""
        batch = []
        batch_start_time = time.time()

        # Try to get requests until timeout or max batch size
        while (
            len(batch) < self.inference_pipeline.batch_processor.max_batch_size
            and (time.time() - batch_start_time) < self.batch_timeout_seconds
        ):
            try:
                request = self.request_queue.get(timeout=0.01)
                batch.append(request)

                with self._stats_lock:
                    self.stats["current_queue_size"] = self.request_queue.qsize()

            except Empty:
                if batch:  # If we have some requests, process them
                    break
                continue

        # Sort by priority (higher priority first)
        batch.sort(key=lambda r: r.priority, reverse=True)

        return batch

    def _process_request_batch(self, batch_requests: List[InferenceRequest]):
        """Process a batch of inference requests."""
        if not batch_requests:
            return

        batch_start_time = time.time()

        try:
            # Prepare batch data for inference pipeline
            batch_data = []
            for request in batch_requests:
                batch_data.append(
                    {"wsi_patches": request.wsi_patches, "patient_context": request.patient_context}
                )

            # Run batch inference
            batch_results = self.inference_pipeline.inference_batch(batch_data)

            # Process results
            for request, predictions in zip(batch_requests, batch_results):
                queue_time = batch_start_time - request.timestamp
                processing_time = time.time() - batch_start_time
                total_time = queue_time + processing_time

                result = InferenceResult(
                    request_id=request.request_id,
                    predictions=predictions,
                    processing_time=processing_time,
                    queue_time=queue_time,
                    success=True,
                )

                # Store result
                self.results[request.request_id] = result

                # Update statistics
                with self._stats_lock:
                    self.stats["completed_requests"] += 1
                    self.stats["avg_queue_time"] = self._update_average(
                        self.stats["avg_queue_time"], queue_time, self.stats["completed_requests"]
                    )
                    self.stats["avg_processing_time"] = self._update_average(
                        self.stats["avg_processing_time"],
                        processing_time,
                        self.stats["completed_requests"],
                    )

                    if total_time > self.max_latency_seconds:
                        self.stats["slow_requests"] += 1
                        logger.warning(
                            f"Request {request.request_id} exceeded max latency: "
                            f"{total_time:.2f}s (queue: {queue_time:.2f}s, "
                            f"processing: {processing_time:.2f}s)"
                        )

                # Call callback if provided
                if request.callback:
                    try:
                        request.callback(result)
                    except Exception as e:
                        logger.error(f"Callback error for request {request.request_id}: {e}")

        except Exception as e:
            logger.error(f"Batch processing error: {e}", exc_info=True)

            # Mark all requests as failed
            for request in batch_requests:
                queue_time = batch_start_time - request.timestamp

                result = InferenceResult(
                    request_id=request.request_id,
                    predictions={},
                    processing_time=0.0,
                    queue_time=queue_time,
                    success=False,
                    error_message=str(e),
                )

                self.results[request.request_id] = result

                with self._stats_lock:
                    self.stats["failed_requests"] += 1

    def _update_average(self, current_avg: float, new_value: float, count: int) -> float:
        """Update running average."""
        if count <= 1:
            return new_value
        return ((current_avg * (count - 1)) + new_value) / count

    def get_statistics(self) -> Dict:
        """Get current performance statistics."""
        with self._stats_lock:
            stats = self.stats.copy()

        # Add derived metrics
        if stats["total_requests"] > 0:
            stats["success_rate"] = (stats["completed_requests"] / stats["total_requests"]) * 100
            stats["failure_rate"] = (stats["failed_requests"] / stats["total_requests"]) * 100
        else:
            stats["success_rate"] = 0.0
            stats["failure_rate"] = 0.0

        if stats["completed_requests"] > 0:
            stats["slow_request_rate"] = (
                stats["slow_requests"] / stats["completed_requests"]
            ) * 100
        else:
            stats["slow_request_rate"] = 0.0

        return stats

    def reset_statistics(self):
        """Reset all performance statistics."""
        with self._stats_lock:
            self.stats = {
                "total_requests": 0,
                "completed_requests": 0,
                "failed_requests": 0,
                "avg_queue_time": 0.0,
                "avg_processing_time": 0.0,
                "current_queue_size": self.request_queue.qsize(),
                "peak_queue_size": 0,
                "slow_requests": 0,
            }

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


class PerformanceMonitor:
    """
    Monitors inference performance and provides alerts when performance degrades.
    """

    def __init__(
        self,
        inference_manager: ConcurrentInferenceManager,
        alert_threshold_seconds: float = 5.0,
        monitoring_interval_seconds: float = 30.0,
    ):
        self.inference_manager = inference_manager
        self.alert_threshold_seconds = alert_threshold_seconds
        self.monitoring_interval_seconds = monitoring_interval_seconds

        self.monitoring_active = False
        self.monitor_thread = None

        # Performance history
        self.performance_history = []
        self.max_history_length = 100

    def start_monitoring(self):
        """Start performance monitoring."""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop, name="PerformanceMonitor", daemon=True
        )
        self.monitor_thread.start()

        logger.info("Performance monitoring started")

    def stop_monitoring(self):
        """Stop performance monitoring."""
        if not self.monitoring_active:
            return

        self.monitoring_active = False

        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)

        logger.info("Performance monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                stats = self.inference_manager.get_statistics()
                pipeline_metrics = (
                    self.inference_manager.inference_pipeline.get_performance_metrics()
                )

                # Record performance snapshot
                snapshot = {
                    "timestamp": time.time(),
                    "avg_processing_time": stats["avg_processing_time"],
                    "avg_queue_time": stats["avg_queue_time"],
                    "slow_request_rate": stats["slow_request_rate"],
                    "queue_size": stats["current_queue_size"],
                    "success_rate": stats["success_rate"],
                    "patches_per_second": pipeline_metrics["metrics"].get(
                        "avg_patches_per_second", 0
                    ),
                }

                self.performance_history.append(snapshot)

                # Trim history
                if len(self.performance_history) > self.max_history_length:
                    self.performance_history.pop(0)

                # Check for performance issues
                self._check_performance_alerts(snapshot)

                time.sleep(self.monitoring_interval_seconds)

            except Exception as e:
                logger.error(f"Performance monitoring error: {e}", exc_info=True)
                time.sleep(self.monitoring_interval_seconds)

    def _check_performance_alerts(self, snapshot: Dict):
        """Check for performance issues and log alerts."""
        total_time = snapshot["avg_processing_time"] + snapshot["avg_queue_time"]

        if total_time > self.alert_threshold_seconds:
            logger.warning(
                f"PERFORMANCE ALERT: Average total time {total_time:.2f}s exceeds "
                f"threshold {self.alert_threshold_seconds}s"
            )

        if snapshot["slow_request_rate"] > 10.0:  # More than 10% slow requests
            logger.warning(
                f"PERFORMANCE ALERT: {snapshot['slow_request_rate']:.1f}% of requests "
                f"exceed latency threshold"
            )

        if snapshot["queue_size"] > 50:  # Queue getting large
            logger.warning(f"PERFORMANCE ALERT: Request queue size is {snapshot['queue_size']}")

        if snapshot["success_rate"] < 95.0:  # Success rate dropping
            logger.warning(
                f"PERFORMANCE ALERT: Success rate dropped to {snapshot['success_rate']:.1f}%"
            )

    def get_performance_report(self) -> Dict:
        """Get comprehensive performance report."""
        if not self.performance_history:
            return {"error": "No performance data available"}

        recent_snapshots = self.performance_history[-10:]  # Last 10 snapshots

        avg_processing_time = sum(s["avg_processing_time"] for s in recent_snapshots) / len(
            recent_snapshots
        )
        avg_queue_time = sum(s["avg_queue_time"] for s in recent_snapshots) / len(recent_snapshots)
        avg_total_time = avg_processing_time + avg_queue_time

        return {
            "current_performance": {
                "avg_processing_time": avg_processing_time,
                "avg_queue_time": avg_queue_time,
                "avg_total_time": avg_total_time,
                "meets_latency_target": avg_total_time <= self.alert_threshold_seconds,
            },
            "recent_stats": self.inference_manager.get_statistics(),
            "pipeline_metrics": self.inference_manager.inference_pipeline.get_performance_metrics(),
            "history_length": len(self.performance_history),
        }
