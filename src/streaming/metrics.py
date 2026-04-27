"""Prometheus metrics for HistoCore streaming performance tracking."""

import time
from typing import Dict, Optional, Any
from prometheus_client import (
    Counter, Histogram, Gauge, Info, CollectorRegistry, 
    generate_latest, CONTENT_TYPE_LATEST
)
import psutil
import torch
import threading
from functools import wraps


class StreamingMetrics:
    """Centralized metrics collection for streaming pipeline."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        self._setup_metrics()
        self._gpu_monitor_thread = None
        self._monitoring = False
        
    def _setup_metrics(self):
        """Initialize all Prometheus metrics."""
        
        # Processing metrics
        self.slides_processed = Counter(
            'histocore_slides_processed_total',
            'Total number of slides processed',
            ['status', 'slide_type'],
            registry=self.registry
        )
        
        self.patches_processed = Counter(
            'histocore_patches_processed_total', 
            'Total number of patches processed',
            ['gpu_id', 'batch_size'],
            registry=self.registry
        )
        
        self.processing_duration = Histogram(
            'histocore_processing_duration_seconds',
            'Time spent processing slides',
            ['stage', 'slide_type'],
            buckets=[1, 5, 10, 30, 60, 120, 300, 600],
            registry=self.registry
        )
        
        # Memory metrics
        self.memory_usage = Gauge(
            'histocore_memory_usage_bytes',
            'Memory usage by component',
            ['component', 'type'],
            registry=self.registry
        )
        
        self.gpu_memory_usage = Gauge(
            'histocore_gpu_memory_usage_bytes',
            'GPU memory usage',
            ['gpu_id', 'type'],
            registry=self.registry
        )
        
        # Performance metrics
        self.throughput = Gauge(
            'histocore_throughput_patches_per_second',
            'Processing throughput in patches per second',
            ['gpu_id'],
            registry=self.registry
        )
        
        self.attention_computation_time = Histogram(
            'histocore_attention_computation_seconds',
            'Time for attention weight computation',
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
            registry=self.registry
        )
        
        # Quality metrics
        self.confidence_score = Histogram(
            'histocore_confidence_score',
            'Model confidence scores',
            ['slide_id'],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
            registry=self.registry
        )
        
        self.early_stopping_rate = Gauge(
            'histocore_early_stopping_rate',
            'Rate of early stopping triggers',
            registry=self.registry
        )
        
        # Error metrics
        self.errors = Counter(
            'histocore_errors_total',
            'Total number of errors',
            ['error_type', 'component'],
            registry=self.registry
        )
        
        self.oom_events = Counter(
            'histocore_oom_events_total',
            'Out of memory events',
            ['gpu_id', 'recovery_action'],
            registry=self.registry
        )
        
        # Network metrics
        self.pacs_requests = Counter(
            'histocore_pacs_requests_total',
            'PACS requests',
            ['operation', 'status'],
            registry=self.registry
        )
        
        self.pacs_response_time = Histogram(
            'histocore_pacs_response_time_seconds',
            'PACS response times',
            ['operation'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
            registry=self.registry
        )
        
        # Cache metrics
        self.cache_hits = Counter(
            'histocore_cache_hits_total',
            'Cache hits',
            ['cache_type'],
            registry=self.registry
        )
        
        self.cache_misses = Counter(
            'histocore_cache_misses_total',
            'Cache misses', 
            ['cache_type'],
            registry=self.registry
        )
        
        self.cache_size = Gauge(
            'histocore_cache_size_bytes',
            'Cache size in bytes',
            ['cache_type'],
            registry=self.registry
        )
        
        self.cache_operations = Histogram(
            'histocore_cache_operations_duration_seconds',
            'Cache operation duration',
            ['operation'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
            registry=self.registry
        )
        
        # Storage metrics
        self.storage_operations = Histogram(
            'histocore_storage_operations_duration_seconds',
            'Storage operation duration',
            ['operation'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
            registry=self.registry
        )
        
        self.storage_size = Gauge(
            'histocore_storage_size_bytes',
            'Storage size in bytes',
            ['storage_type'],
            registry=self.registry
        )
        
        # System info
        self.system_info = Info(
            'histocore_system_info',
            'System information',
            registry=self.registry
        )
        
        # Set system info
        self.system_info.info({
            'version': '1.0.0',
            'python_version': f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}",
            'cuda_available': str(torch.cuda.is_available()),
            'gpu_count': str(torch.cuda.device_count()) if torch.cuda.is_available() else '0'
        })
        
    def start_monitoring(self):
        """Start background monitoring thread."""
        if self._monitoring:
            return
            
        self._monitoring = True
        self._gpu_monitor_thread = threading.Thread(
            target=self._monitor_resources,
            daemon=True
        )
        self._gpu_monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop background monitoring."""
        self._monitoring = False
        if self._gpu_monitor_thread:
            self._gpu_monitor_thread.join(timeout=1.0)
            
    def _monitor_resources(self):
        """Background thread to monitor system resources."""
        while self._monitoring:
            try:
                # CPU memory
                memory = psutil.virtual_memory()
                self.memory_usage.labels(
                    component='system', 
                    type='used'
                ).set(memory.used)
                self.memory_usage.labels(
                    component='system',
                    type='available'
                ).set(memory.available)
                
                # GPU memory
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        try:
                            mem_info = torch.cuda.mem_get_info(i)
                            free_mem, total_mem = mem_info
                            used_mem = total_mem - free_mem
                            
                            self.gpu_memory_usage.labels(
                                gpu_id=str(i),
                                type='used'
                            ).set(used_mem)
                            self.gpu_memory_usage.labels(
                                gpu_id=str(i), 
                                type='total'
                            ).set(total_mem)
                        except Exception:
                            pass  # GPU might be in use
                            
                time.sleep(5.0)  # Update every 5 seconds
                
            except Exception:
                pass  # Continue monitoring despite errors
                
    def record_slide_processed(self, status: str, slide_type: str = 'wsi'):
        """Record slide processing completion."""
        self.slides_processed.labels(
            status=status,
            slide_type=slide_type
        ).inc()
        
    def record_patches_processed(self, count: int, gpu_id: str, batch_size: int):
        """Record patch processing."""
        self.patches_processed.labels(
            gpu_id=gpu_id,
            batch_size=str(batch_size)
        ).inc(count)
        
    def record_processing_time(self, duration: float, stage: str, slide_type: str = 'wsi'):
        """Record processing duration."""
        self.processing_duration.labels(
            stage=stage,
            slide_type=slide_type
        ).observe(duration)
        
    def record_throughput(self, patches_per_second: float, gpu_id: str):
        """Record processing throughput."""
        self.throughput.labels(gpu_id=gpu_id).set(patches_per_second)
        
    def record_attention_time(self, duration: float):
        """Record attention computation time."""
        self.attention_computation_time.observe(duration)
        
    def record_confidence(self, score: float, slide_id: str):
        """Record confidence score."""
        self.confidence_score.labels(slide_id=slide_id).observe(score)
        
    def record_early_stopping(self, rate: float):
        """Record early stopping rate."""
        self.early_stopping_rate.set(rate)
        
    def record_error(self, error_type: str, component: str):
        """Record error occurrence."""
        self.errors.labels(
            error_type=error_type,
            component=component
        ).inc()
        
    def record_oom_event(self, gpu_id: str, recovery_action: str):
        """Record out-of-memory event."""
        self.oom_events.labels(
            gpu_id=gpu_id,
            recovery_action=recovery_action
        ).inc()
        
    def record_pacs_request(self, operation: str, status: str, response_time: float):
        """Record PACS request metrics."""
        self.pacs_requests.labels(
            operation=operation,
            status=status
        ).inc()
        
        self.pacs_response_time.labels(
            operation=operation
        ).observe(response_time)
        
    def record_cache_hit(self, cache_type: str):
        """Record cache hit."""
        self.cache_hits.labels(cache_type=cache_type).inc()
        
    def record_cache_miss(self, cache_type: str):
        """Record cache miss."""
        self.cache_misses.labels(cache_type=cache_type).inc()
        
    def get_metrics(self) -> bytes:
        """Get Prometheus metrics in text format."""
        return generate_latest(self.registry)
        
    def get_content_type(self) -> str:
        """Get content type for metrics endpoint."""
        return CONTENT_TYPE_LATEST


# Global metrics instance
_metrics_instance: Optional[StreamingMetrics] = None


def get_metrics() -> StreamingMetrics:
    """Get global metrics instance."""
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = StreamingMetrics()
        _metrics_instance.start_monitoring()
    return _metrics_instance


def timed_operation(stage: str, slide_type: str = 'wsi'):
    """Decorator to time operations and record metrics."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                get_metrics().record_processing_time(duration, stage, slide_type)
                return result
            except Exception as e:
                duration = time.time() - start_time
                get_metrics().record_processing_time(duration, f"{stage}_error", slide_type)
                get_metrics().record_error(type(e).__name__, stage)
                raise
        return wrapper
    return decorator


def track_gpu_memory(gpu_id: str):
    """Decorator to track GPU memory usage."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(int(gpu_id))
                
            try:
                result = func(*args, **kwargs)
                
                if torch.cuda.is_available():
                    peak_memory = torch.cuda.max_memory_allocated(int(gpu_id))
                    get_metrics().memory_usage.labels(
                        component=f'gpu_{gpu_id}',
                        type='peak'
                    ).set(peak_memory)
                    
                return result
            except torch.cuda.OutOfMemoryError:
                get_metrics().record_oom_event(gpu_id, 'exception')
                raise
                
        return wrapper
    return decorator


class MetricsContext:
    """Context manager for tracking operation metrics."""
    
    def __init__(self, operation: str, **labels):
        self.operation = operation
        self.labels = labels
        self.start_time = None
        self.metrics = get_metrics()
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if exc_type is None:
            status = 'success'
        else:
            status = 'error'
            self.metrics.record_error(
                error_type=exc_type.__name__,
                component=self.operation
            )
            
        self.metrics.record_processing_time(
            duration, 
            self.operation,
            self.labels.get('slide_type', 'wsi')
        )


# Convenience functions
def record_slide_success(slide_type: str = 'wsi'):
    """Record successful slide processing."""
    get_metrics().record_slide_processed('success', slide_type)


def record_slide_error(slide_type: str = 'wsi'):
    """Record failed slide processing."""
    get_metrics().record_slide_processed('error', slide_type)


def record_throughput_measurement(patches_per_second: float, gpu_id: str = '0'):
    """Record throughput measurement."""
    get_metrics().record_throughput(patches_per_second, gpu_id)


def record_confidence_measurement(score: float, slide_id: str):
    """Record confidence measurement."""
    get_metrics().record_confidence(score, slide_id)


# Cache and storage metric accessors
cache_hits_total = get_metrics().cache_hits
cache_misses_total = get_metrics().cache_misses
cache_size_bytes = get_metrics().cache_size
cache_operations_duration = get_metrics().cache_operations
storage_operations_duration = get_metrics().storage_operations
storage_size_bytes = get_metrics().storage_size