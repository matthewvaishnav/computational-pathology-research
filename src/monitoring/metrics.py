"""Prometheus metrics for production monitoring."""

import logging
import time
from functools import wraps
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Try to import prometheus_client
try:
    from prometheus_client import Counter, Gauge, Histogram, Summary

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not available, metrics disabled")


class MetricsCollector:
    """Collect Prometheus metrics for HistoCore.

    Tracks:
    - Inference requests (counter)
    - Inference latency (histogram)
    - GPU memory usage (gauge)
    - Active requests (gauge)
    - Error rate (counter)
    """

    def __init__(self, enabled: bool = True):
        """Init metrics collector.

        Args:
            enabled: Enable metrics collection
        """
        self.enabled = enabled and PROMETHEUS_AVAILABLE

        if not self.enabled:
            logger.info("Metrics collection disabled")
            return

        # Counters
        self.inference_requests = Counter(
            "histocore_inference_requests_total",
            "Total inference requests",
            ["model", "status"],
        )

        self.errors = Counter(
            "histocore_errors_total",
            "Total errors",
            ["error_type"],
        )

        # Gauges
        self.active_requests = Gauge(
            "histocore_active_requests",
            "Active inference requests",
        )

        self.gpu_memory_used = Gauge(
            "histocore_gpu_memory_bytes",
            "GPU memory used in bytes",
            ["device"],
        )

        self.gpu_utilization = Gauge(
            "histocore_gpu_utilization_percent",
            "GPU utilization percentage",
            ["device"],
        )

        # Histograms
        self.inference_latency = Histogram(
            "histocore_inference_duration_seconds",
            "Inference latency in seconds",
            ["model"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
        )

        self.batch_size = Histogram(
            "histocore_batch_size",
            "Batch size distribution",
            buckets=[1, 4, 8, 16, 32, 64, 128, 256],
        )

        # Summaries
        self.tile_processing_time = Summary(
            "histocore_tile_processing_seconds",
            "Tile processing time",
        )

        logger.info("Metrics collection enabled")

    def record_inference(
        self,
        model_name: str,
        duration: float,
        batch_size: int,
        success: bool = True,
    ):
        """Record inference metrics.

        Args:
            model_name: Model name
            duration: Inference duration in seconds
            batch_size: Batch size
            success: Whether inference succeeded
        """
        if not self.enabled:
            return

        status = "success" if success else "error"
        self.inference_requests.labels(model=model_name, status=status).inc()
        self.inference_latency.labels(model=model_name).observe(duration)
        self.batch_size.observe(batch_size)

    def record_error(self, error_type: str):
        """Record error.

        Args:
            error_type: Error type (e.g., "gpu_oom", "model_load_failed")
        """
        if not self.enabled:
            return

        self.errors.labels(error_type=error_type).inc()

    def update_gpu_metrics(self):
        """Update GPU metrics."""
        if not self.enabled:
            return

        try:
            import torch

            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    device = f"cuda:{i}"

                    # Memory usage
                    memory_used = torch.cuda.memory_allocated(i)
                    self.gpu_memory_used.labels(device=device).set(memory_used)

                    # Utilization (requires nvidia-ml-py3)
                    try:
                        import pynvml

                        pynvml.nvmlInit()
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        self.gpu_utilization.labels(device=device).set(util.gpu)
                    except ImportError:
                        pass

        except Exception as e:
            logger.warning(f"Failed to update GPU metrics: {e}")

    def track_active_request(self, increment: bool = True):
        """Track active requests.

        Args:
            increment: Increment (True) or decrement (False)
        """
        if not self.enabled:
            return

        if increment:
            self.active_requests.inc()
        else:
            self.active_requests.dec()


# Global metrics collector
_metrics = MetricsCollector()


def get_metrics() -> MetricsCollector:
    """Get global metrics collector.

    Returns:
        MetricsCollector instance
    """
    return _metrics


def track_inference(model_name: str = "default"):
    """Decorator to track inference metrics.

    Args:
        model_name: Model name for metrics

    Example:
        @track_inference("resnet50")
        def run_inference(images):
            return model(images)
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            metrics = get_metrics()
            metrics.track_active_request(increment=True)

            start_time = time.time()
            success = True

            try:
                result = func(*args, **kwargs)
                return result

            except Exception as e:
                success = False
                metrics.record_error(type(e).__name__)
                raise

            finally:
                duration = time.time() - start_time

                # Try to get batch size from args/kwargs
                batch_size = 1
                if args and hasattr(args[0], "shape"):
                    batch_size = args[0].shape[0]
                elif "batch_size" in kwargs:
                    batch_size = kwargs["batch_size"]

                metrics.record_inference(model_name, duration, batch_size, success)
                metrics.track_active_request(increment=False)

        return wrapper

    return decorator


def create_metrics_endpoint():
    """Create FastAPI metrics endpoint.

    Returns:
        FastAPI router with metrics endpoint
    """
    if not PROMETHEUS_AVAILABLE:
        logger.warning("prometheus_client not available, metrics endpoint not created")
        return None

    try:
        from fastapi import APIRouter, Response
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

        router = APIRouter()

        @router.get("/metrics")
        def metrics():
            """Prometheus metrics endpoint."""
            # Update GPU metrics before serving
            get_metrics().update_gpu_metrics()

            return Response(
                content=generate_latest(),
                media_type=CONTENT_TYPE_LATEST,
            )

        return router

    except ImportError:
        logger.warning("FastAPI not available, metrics endpoint not created")
        return None
