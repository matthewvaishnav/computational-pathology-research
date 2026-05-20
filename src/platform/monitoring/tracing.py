#!/usr/bin/env python3
"""
OpenTelemetry Distributed Tracing

System-wide distributed tracing for all HistoCore services.
"""

import logging
import os
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# Check if OpenTelemetry is available
try:
    from opentelemetry import trace
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
    from opentelemetry.instrumentation.torch import TorchInstrumentor
    from opentelemetry.propagate import extract, inject
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.trace import Status, StatusCode

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    logger.warning(
        "OpenTelemetry not available. Install with: pip install opentelemetry-api opentelemetry-sdk opentelemetry-instrumentation-fastapi"
    )


class DistributedTracer:
    """System-wide distributed tracing configuration."""

    def __init__(self, service_name: str = "histocore"):
        """Initialize tracer.

        Args:
            service_name: Service name for tracing
        """
        self.service_name = service_name
        self.tracer_provider: Optional[Any] = None
        self.tracer: Optional[Any] = None
        self._initialized = False

        if not OTEL_AVAILABLE:
            logger.warning(f"Tracing disabled for {service_name} - OpenTelemetry not available")

    def initialize(
        self,
        jaeger_endpoint: Optional[str] = None,
        otlp_endpoint: Optional[str] = None,
        service_version: str = "1.0.0",
        environment: str = None,
        enable_console: bool = False,
    ):
        """Initialize distributed tracing.

        Args:
            jaeger_endpoint: Jaeger endpoint (host:port)
            otlp_endpoint: OTLP endpoint URL
            service_version: Service version
            environment: Deployment environment
            enable_console: Enable console exporter for debugging
        """
        if not OTEL_AVAILABLE:
            return

        if self._initialized:
            return

        # Get environment from env var if not provided
        if environment is None:
            environment = os.getenv("ENVIRONMENT", "development")

        # Create resource
        resource = Resource.create(
            {
                "service.name": self.service_name,
                "service.version": service_version,
                "deployment.environment": environment,
                "host.name": os.getenv("HOSTNAME", "localhost"),
            }
        )

        # Create tracer provider
        self.tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(self.tracer_provider)

        # Add exporters
        exporters_added = False

        if jaeger_endpoint:
            if self._add_jaeger_exporter(jaeger_endpoint):
                exporters_added = True

        if otlp_endpoint:
            if self._add_otlp_exporter(otlp_endpoint):
                exporters_added = True

        # Add console exporter if requested or no other exporters
        if enable_console or not exporters_added:
            self._add_console_exporter()

        # Get tracer
        self.tracer = trace.get_tracer(self.service_name, service_version)

        # Instrument libraries
        self._instrument_libraries()

        self._initialized = True
        logger.info(f"Distributed tracing initialized for {self.service_name}")

    def _add_jaeger_exporter(self, endpoint: str) -> bool:
        """Add Jaeger exporter.

        Args:
            endpoint: Jaeger endpoint (host:port)

        Returns:
            True if successful
        """
        try:
            # Parse endpoint
            if ":" in endpoint:
                host, port = endpoint.split(":")
                port = int(port)
            else:
                host = endpoint
                port = 6831  # Default Jaeger UDP port

            jaeger_exporter = JaegerExporter(
                agent_host_name=host,
                agent_port=port,
            )

            span_processor = BatchSpanProcessor(jaeger_exporter)
            self.tracer_provider.add_span_processor(span_processor)
            logger.info(f"Added Jaeger exporter: {endpoint}")
            return True

        except Exception as e:
            logger.error(f"Failed to add Jaeger exporter: {e}")
            return False

    def _add_otlp_exporter(self, endpoint: str) -> bool:
        """Add OTLP exporter.

        Args:
            endpoint: OTLP endpoint URL

        Returns:
            True if successful
        """
        try:
            otlp_exporter = OTLPSpanExporter(endpoint=endpoint)
            span_processor = BatchSpanProcessor(otlp_exporter)
            self.tracer_provider.add_span_processor(span_processor)
            logger.info(f"Added OTLP exporter: {endpoint}")
            return True

        except Exception as e:
            logger.error(f"Failed to add OTLP exporter: {e}")
            return False

    def _add_console_exporter(self):
        """Add console exporter for development."""
        try:
            console_exporter = ConsoleSpanExporter()
            span_processor = BatchSpanProcessor(console_exporter)
            self.tracer_provider.add_span_processor(span_processor)
            logger.info("Added console exporter")

        except Exception as e:
            logger.error(f"Failed to add console exporter: {e}")

    def _instrument_libraries(self):
        """Instrument common libraries."""
        try:
            # HTTP requests
            RequestsInstrumentor().instrument()
            logger.info("Instrumented requests library")

        except Exception as e:
            logger.warning(f"Failed to instrument requests: {e}")

        try:
            # PyTorch (if available)
            TorchInstrumentor().instrument()
            logger.info("Instrumented PyTorch")

        except Exception as e:
            logger.debug(f"PyTorch instrumentation not available: {e}")

    def instrument_fastapi(self, app):
        """Instrument FastAPI application.

        Args:
            app: FastAPI application instance
        """
        if not OTEL_AVAILABLE or not self._initialized:
            return

        try:
            FastAPIInstrumentor.instrument_app(app)
            logger.info(f"Instrumented FastAPI app: {app.title}")

        except Exception as e:
            logger.error(f"Failed to instrument FastAPI: {e}")

    def instrument_sqlalchemy(self, engine):
        """Instrument SQLAlchemy engine.

        Args:
            engine: SQLAlchemy engine instance
        """
        if not OTEL_AVAILABLE or not self._initialized:
            return

        try:
            SQLAlchemyInstrumentor().instrument(engine=engine)
            logger.info("Instrumented SQLAlchemy engine")

        except Exception as e:
            logger.error(f"Failed to instrument SQLAlchemy: {e}")

    def get_tracer(self):
        """Get tracer instance.

        Returns:
            Tracer instance or None if not available
        """
        if not OTEL_AVAILABLE:
            return None

        if not self._initialized:
            self.initialize()

        return self.tracer

    def shutdown(self):
        """Shutdown tracing."""
        if OTEL_AVAILABLE and self.tracer_provider:
            self.tracer_provider.shutdown()
            self._initialized = False
            logger.info("Tracing shutdown")


# Global tracer instance
_tracer_instance: Optional[DistributedTracer] = None


def get_tracer(service_name: str = "histocore") -> DistributedTracer:
    """Get global tracer instance.

    Args:
        service_name: Service name

    Returns:
        DistributedTracer instance
    """
    global _tracer_instance

    if _tracer_instance is None:
        _tracer_instance = DistributedTracer(service_name)

        # Auto-initialize from environment
        jaeger_endpoint = os.getenv("JAEGER_ENDPOINT")
        otlp_endpoint = os.getenv("OTLP_ENDPOINT")
        enable_console = os.getenv("OTEL_CONSOLE", "false").lower() == "true"

        _tracer_instance.initialize(
            jaeger_endpoint=jaeger_endpoint,
            otlp_endpoint=otlp_endpoint,
            enable_console=enable_console,
        )

    return _tracer_instance


def traced(
    operation_name: Optional[str] = None,
    component: str = "histocore",
    record_exception: bool = True,
):
    """Decorator to trace operations.

    Args:
        operation_name: Operation name (defaults to function name)
        component: Component name
        record_exception: Record exceptions in span

    Returns:
        Decorated function
    """
    if not OTEL_AVAILABLE:
        # Return no-op decorator if OpenTelemetry not available
        def noop_decorator(func):
            return func

        return noop_decorator

    def decorator(func: Callable) -> Callable:
        nonlocal operation_name
        if operation_name is None:
            operation_name = f"{func.__module__}.{func.__name__}"

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            tracer = get_tracer().get_tracer()
            if tracer is None:
                return func(*args, **kwargs)

            with tracer.start_as_current_span(
                operation_name,
                attributes={
                    "component": component,
                    "function": func.__name__,
                    "module": func.__module__,
                },
            ) as span:
                try:
                    result = func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    if record_exception:
                        span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracer = get_tracer().get_tracer()
            if tracer is None:
                return await func(*args, **kwargs)

            with tracer.start_as_current_span(
                operation_name,
                attributes={
                    "component": component,
                    "function": func.__name__,
                    "module": func.__module__,
                },
            ) as span:
                try:
                    result = await func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    if record_exception:
                        span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

        # Return appropriate wrapper
        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


@contextmanager
def trace_span(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
    component: str = "histocore",
):
    """Context manager for manual span creation.

    Args:
        name: Span name
        attributes: Span attributes
        component: Component name

    Yields:
        Span instance or None
    """
    if not OTEL_AVAILABLE:
        yield None
        return

    tracer = get_tracer().get_tracer()
    if tracer is None:
        yield None
        return

    span_attributes = {"component": component}
    if attributes:
        span_attributes.update(attributes)

    with tracer.start_as_current_span(name, attributes=span_attributes) as span:
        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise


def add_span_attributes(**attributes):
    """Add attributes to current span.

    Args:
        **attributes: Attributes to add
    """
    if not OTEL_AVAILABLE:
        return

    current_span = trace.get_current_span()
    if current_span.is_recording():
        for key, value in attributes.items():
            current_span.set_attribute(key, value)


def add_span_event(name: str, attributes: Optional[Dict[str, Any]] = None):
    """Add event to current span.

    Args:
        name: Event name
        attributes: Event attributes
    """
    if not OTEL_AVAILABLE:
        return

    current_span = trace.get_current_span()
    if current_span.is_recording():
        current_span.add_event(name, attributes or {})


def get_trace_context() -> Dict[str, str]:
    """Get current trace context for propagation.

    Returns:
        Trace context dict
    """
    if not OTEL_AVAILABLE:
        return {}

    context = {}
    inject(context)
    return context


def set_trace_context(context: Dict[str, str]):
    """Set trace context from propagated headers.

    Args:
        context: Trace context dict

    Returns:
        Context object
    """
    if not OTEL_AVAILABLE:
        return None

    return extract(context)


# Convenience functions for common operations
def trace_inference(model_name: str, batch_size: int):
    """Trace inference operation.

    Args:
        model_name: Model name
        batch_size: Batch size

    Returns:
        Context manager
    """
    return trace_span(
        "inference",
        attributes={
            "model.name": model_name,
            "batch.size": batch_size,
        },
        component="inference",
    )


def trace_data_loading(dataset_name: str, num_samples: int):
    """Trace data loading operation.

    Args:
        dataset_name: Dataset name
        num_samples: Number of samples

    Returns:
        Context manager
    """
    return trace_span(
        "data_loading",
        attributes={
            "dataset.name": dataset_name,
            "dataset.samples": num_samples,
        },
        component="data",
    )


def trace_model_training(model_name: str, epoch: int):
    """Trace model training operation.

    Args:
        model_name: Model name
        epoch: Current epoch

    Returns:
        Context manager
    """
    return trace_span(
        "training",
        attributes={
            "model.name": model_name,
            "training.epoch": epoch,
        },
        component="training",
    )


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)

    # Initialize tracer
    tracer = get_tracer("test-service")
    tracer.initialize(enable_console=True)

    # Test traced function
    @traced("test_operation")
    def test_function(x: int) -> int:
        add_span_attributes(input_value=x)
        result = x * 2
        add_span_event("calculation_complete", {"result": result})
        return result

    # Test span context
    with trace_span("test_span", {"test": "value"}):
        result = test_function(5)
        print(f"Result: {result}")

    # Shutdown
    tracer.shutdown()
