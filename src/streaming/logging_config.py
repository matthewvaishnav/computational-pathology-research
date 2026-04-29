"""Structured logging with correlation IDs for HistoCore streaming."""

import json
import logging
import os
import sys
import threading
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Dict, Optional

import structlog
from structlog.stdlib import LoggerFactory

# Context variable for correlation ID
correlation_id: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)


class CorrelationIDProcessor:
    """Processor to add correlation ID to log records."""

    def __call__(self, logger, method_name, event_dict):
        corr_id = correlation_id.get()
        if corr_id:
            event_dict["correlation_id"] = corr_id
        return event_dict


class TimestampProcessor:
    """Processor to add ISO timestamp."""

    def __call__(self, logger, method_name, event_dict):
        event_dict["timestamp"] = datetime.now(timezone.utc).isoformat()
        return event_dict


class ComponentProcessor:
    """Processor to add component information."""

    def __init__(self, component: str = "streaming"):
        self.component = component

    def __call__(self, logger, method_name, event_dict):
        event_dict["component"] = self.component
        event_dict["level"] = method_name.upper()
        return event_dict


class PerformanceProcessor:
    """Processor to add performance metrics."""

    def __call__(self, logger, method_name, event_dict):
        # Add thread info
        event_dict["thread_id"] = threading.get_ident()
        event_dict["thread_name"] = threading.current_thread().name

        # Add process info
        event_dict["process_id"] = os.getpid()

        return event_dict


def configure_logging(
    level: str = "INFO",
    format_type: str = "json",
    component: str = "streaming",
    log_file: Optional[str] = None,
):
    """Configure structured logging."""

    # Configure structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        TimestampProcessor(),
        ComponentProcessor(component),
        CorrelationIDProcessor(),
        PerformanceProcessor(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if format_type == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s", stream=sys.stdout, level=getattr(logging, level.upper())
    )

    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        logging.getLogger().addHandler(file_handler)

    # Suppress noisy loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.BoundLogger:
    """Get structured logger."""
    return structlog.get_logger(name)


def generate_correlation_id() -> str:
    """Generate new correlation ID."""
    return str(uuid.uuid4())


def set_correlation_id(corr_id: str):
    """Set correlation ID for current context."""
    correlation_id.set(corr_id)


def get_correlation_id() -> Optional[str]:
    """Get current correlation ID."""
    return correlation_id.get()


def with_correlation_id(corr_id: Optional[str] = None):
    """Decorator to set correlation ID for function execution."""

    def decorator(func):
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Generate ID if not provided
            if corr_id is None:
                new_id = generate_correlation_id()
            else:
                new_id = corr_id

            # Set correlation ID
            token = correlation_id.set(new_id)

            try:
                return func(*args, **kwargs)
            finally:
                correlation_id.reset(token)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate ID if not provided
            if corr_id is None:
                new_id = generate_correlation_id()
            else:
                new_id = corr_id

            # Set correlation ID
            token = correlation_id.set(new_id)

            try:
                return await func(*args, **kwargs)
            finally:
                correlation_id.reset(token)

        # Return appropriate wrapper
        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


class LoggingContext:
    """Context manager for logging with additional context."""

    def __init__(self, logger: structlog.BoundLogger, **context):
        self.logger = logger
        self.context = context
        self.bound_logger = None

    def __enter__(self):
        self.bound_logger = self.logger.bind(**self.context)
        return self.bound_logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.bound_logger.error(
                "Exception in logging context", exc_type=exc_type.__name__, exc_message=str(exc_val)
            )


class StreamingLogger:
    """High-level logger for streaming operations."""

    def __init__(self, component: str = "streaming"):
        self.logger = get_logger(component)
        self.component = component

    def log_slide_processing_start(self, slide_id: str, slide_path: str):
        """Log slide processing start."""
        self.logger.info(
            "Starting slide processing",
            slide_id=slide_id,
            slide_path=slide_path,
            operation="slide_processing_start",
        )

    def log_slide_processing_complete(self, slide_id: str, duration: float, patch_count: int):
        """Log slide processing completion."""
        self.logger.info(
            "Slide processing complete",
            slide_id=slide_id,
            duration_seconds=duration,
            patch_count=patch_count,
            operation="slide_processing_complete",
        )

    def log_gpu_operation(self, gpu_id: int, operation: str, batch_size: int, duration: float):
        """Log GPU operation."""
        self.logger.info(
            "GPU operation complete",
            gpu_id=gpu_id,
            operation=operation,
            batch_size=batch_size,
            duration_seconds=duration,
            throughput=batch_size / duration if duration > 0 else 0,
        )

    def log_memory_usage(
        self, component: str, memory_mb: float, gpu_memory_mb: Optional[float] = None
    ):
        """Log memory usage."""
        log_data = {"component": component, "memory_mb": memory_mb, "operation": "memory_usage"}

        if gpu_memory_mb is not None:
            log_data["gpu_memory_mb"] = gpu_memory_mb

        self.logger.info("Memory usage report", **log_data)

    def log_error(self, error: Exception, context: Dict[str, Any]):
        """Log error with context."""
        self.logger.error(
            "Error occurred",
            error_type=type(error).__name__,
            error_message=str(error),
            operation="error",
            **context,
        )

    def log_performance_metric(self, metric_name: str, value: float, unit: str, **context):
        """Log performance metric."""
        self.logger.info(
            "Performance metric",
            metric_name=metric_name,
            metric_value=value,
            metric_unit=unit,
            operation="performance_metric",
            **context,
        )

    def log_pacs_operation(self, operation: str, study_id: str, status: str, duration: float):
        """Log PACS operation."""
        self.logger.info(
            "PACS operation",
            pacs_operation=operation,
            study_id=study_id,
            status=status,
            duration_seconds=duration,
            operation="pacs_operation",
        )

    def log_cache_operation(self, cache_type: str, operation: str, key: str, hit: bool):
        """Log cache operation."""
        self.logger.info(
            "Cache operation",
            cache_type=cache_type,
            cache_operation=operation,
            cache_key=key,
            cache_hit=hit,
            operation="cache_operation",
        )

    def with_context(self, **context):
        """Create logging context."""
        return LoggingContext(self.logger, **context)


# Global logger instances
_loggers: Dict[str, StreamingLogger] = {}


def get_streaming_logger(component: str = "streaming") -> StreamingLogger:
    """Get streaming logger for component."""
    if component not in _loggers:
        _loggers[component] = StreamingLogger(component)
    return _loggers[component]


# Convenience functions
def log_slide_start(slide_id: str, slide_path: str):
    """Log slide processing start."""
    logger = get_streaming_logger()
    logger.log_slide_processing_start(slide_id, slide_path)


def log_slide_complete(slide_id: str, duration: float, patch_count: int):
    """Log slide processing completion."""
    logger = get_streaming_logger()
    logger.log_slide_processing_complete(slide_id, duration, patch_count)


def log_gpu_batch(gpu_id: int, operation: str, batch_size: int, duration: float):
    """Log GPU batch processing."""
    logger = get_streaming_logger("gpu_pipeline")
    logger.log_gpu_operation(gpu_id, operation, batch_size, duration)


def log_memory_report(component: str, memory_mb: float, gpu_memory_mb: Optional[float] = None):
    """Log memory usage report."""
    logger = get_streaming_logger("memory_monitor")
    logger.log_memory_usage(component, memory_mb, gpu_memory_mb)


def log_error_with_context(error: Exception, **context):
    """Log error with context."""
    logger = get_streaming_logger()
    logger.log_error(error, context)


# Initialize logging on import
configure_logging(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format_type=os.getenv("LOG_FORMAT", "json"),
    component=os.getenv("COMPONENT_NAME", "streaming"),
    log_file=os.getenv("LOG_FILE"),
)
