"""OpenTelemetry distributed tracing for HistoCore streaming."""

import logging
import os
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, Optional

from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.propagate import extract, inject
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode

logger = logging.getLogger(__name__)


class StreamingTracer:
    """Centralized tracing configuration for streaming pipeline."""

    def __init__(self, service_name: str = "histocore-streaming"):
        self.service_name = service_name
        self.tracer_provider: Optional[TracerProvider] = None
        self.tracer: Optional[trace.Tracer] = None
        self._initialized = False

    def initialize(
        self,
        jaeger_endpoint: Optional[str] = None,
        otlp_endpoint: Optional[str] = None,
        service_version: str = "1.0.0",
    ):
        """Initialize tracing with exporters."""
        if self._initialized:
            return

        # Create resource
        resource = Resource.create(
            {
                "service.name": self.service_name,
                "service.version": service_version,
                "deployment.environment": os.getenv("ENVIRONMENT", "development"),
            }
        )

        # Create tracer provider
        self.tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(self.tracer_provider)

        # Add exporters
        if jaeger_endpoint:
            self._add_jaeger_exporter(jaeger_endpoint)

        if otlp_endpoint:
            self._add_otlp_exporter(otlp_endpoint)

        # Default to console if no exporters
        if not jaeger_endpoint and not otlp_endpoint:
            self._add_console_exporter()

        # Get tracer
        self.tracer = trace.get_tracer(self.service_name)

        # Instrument libraries
        self._instrument_libraries()

        self._initialized = True
        logger.info(f"Tracing initialized for {self.service_name}")

    def _add_jaeger_exporter(self, endpoint: str):
        """Add Jaeger exporter."""
        try:
            jaeger_exporter = JaegerExporter(
                agent_host_name=endpoint.split(":")[0],
                agent_port=int(endpoint.split(":")[1]) if ":" in endpoint else 14268,
            )

            span_processor = BatchSpanProcessor(jaeger_exporter)
            self.tracer_provider.add_span_processor(span_processor)
            logger.info(f"Added Jaeger exporter: {endpoint}")

        except Exception as e:
            logger.error(f"Failed to add Jaeger exporter: {e}")

    def _add_otlp_exporter(self, endpoint: str):
        """Add OTLP exporter."""
        try:
            otlp_exporter = OTLPSpanExporter(endpoint=endpoint)
            span_processor = BatchSpanProcessor(otlp_exporter)
            self.tracer_provider.add_span_processor(span_processor)
            logger.info(f"Added OTLP exporter: {endpoint}")

        except Exception as e:
            logger.error(f"Failed to add OTLP exporter: {e}")

    def _add_console_exporter(self):
        """Add console exporter for development."""
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter

        console_exporter = ConsoleSpanExporter()
        span_processor = BatchSpanProcessor(console_exporter)
        self.tracer_provider.add_span_processor(span_processor)
        logger.info("Added console exporter")

    def _instrument_libraries(self):
        """Instrument common libraries."""
        try:
            RequestsInstrumentor().instrument()
            AsyncioInstrumentor().instrument()
            logger.info("Instrumented common libraries")
        except Exception as e:
            logger.error(f"Failed to instrument libraries: {e}")

    def get_tracer(self) -> trace.Tracer:
        """Get tracer instance."""
        if not self._initialized:
            self.initialize()
        return self.tracer

    def shutdown(self):
        """Shutdown tracing."""
        if self.tracer_provider:
            self.tracer_provider.shutdown()
            self._initialized = False
            logger.info("Tracing shutdown")


# Global tracer instance
_tracer_instance: Optional[StreamingTracer] = None


def get_tracer() -> StreamingTracer:
    """Get global tracer instance."""
    global _tracer_instance
    if _tracer_instance is None:
        _tracer_instance = StreamingTracer()

        # Auto-initialize from environment
        jaeger_endpoint = os.getenv("JAEGER_ENDPOINT")
        otlp_endpoint = os.getenv("OTLP_ENDPOINT")

        _tracer_instance.initialize(jaeger_endpoint=jaeger_endpoint, otlp_endpoint=otlp_endpoint)

    return _tracer_instance


def traced_operation(
    operation_name: str, component: str = "streaming", record_exception: bool = True
):
    """Decorator to trace operations."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            tracer = get_tracer().get_tracer()

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

        # Return appropriate wrapper based on function type
        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


@contextmanager
def trace_span(
    name: str, attributes: Optional[Dict[str, Any]] = None, component: str = "streaming"
):
    """Context manager for manual span creation."""
    tracer = get_tracer().get_tracer()

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
    """Add attributes to current span."""
    current_span = trace.get_current_span()
    if current_span.is_recording():
        for key, value in attributes.items():
            current_span.set_attribute(key, value)


def add_span_event(name: str, attributes: Optional[Dict[str, Any]] = None):
    """Add event to current span."""
    current_span = trace.get_current_span()
    if current_span.is_recording():
        current_span.add_event(name, attributes or {})


def get_trace_context() -> Dict[str, str]:
    """Get current trace context for propagation."""
    context = {}
    inject(context)
    return context


def set_trace_context(context: Dict[str, str]):
    """Set trace context from propagated headers."""
    return extract(context)


class TracedWSIProcessor:
    """Example traced WSI processor."""

    @traced_operation("wsi.load", "wsi_reader")
    def load_slide(self, slide_path: str):
        """Load WSI slide with tracing."""
        add_span_attributes(slide_path=slide_path, slide_format=slide_path.split(".")[-1])

        # Simulate loading
        import time

        time.sleep(0.1)

        add_span_event("slide_loaded", {"dimensions": "40000x30000", "levels": 5})

        return {"width": 40000, "height": 30000}

    @traced_operation("wsi.extract_patches", "patch_extractor")
    def extract_patches(self, slide_data: dict, patch_size: int = 224):
        """Extract patches with tracing."""
        add_span_attributes(patch_size=patch_size, total_patches=100)

        with trace_span("patch_extraction_loop"):
            for i in range(10):  # Simulate patch extraction
                with trace_span(f"extract_patch_{i}", {"patch_id": i}):
                    # Simulate patch extraction
                    import time

                    time.sleep(0.01)

        add_span_event("patches_extracted", {"count": 10})
        return [f"patch_{i}" for i in range(10)]

    @traced_operation("wsi.process_gpu", "gpu_pipeline")
    async def process_on_gpu(self, patches: list, gpu_id: int = 0):
        """Process patches on GPU with tracing."""
        add_span_attributes(gpu_id=gpu_id, batch_size=len(patches))

        # Simulate GPU processing
        import asyncio

        await asyncio.sleep(0.05)

        add_span_event("gpu_processing_complete", {"throughput": len(patches) / 0.05})

        return [f"feature_{i}" for i in range(len(patches))]


# Convenience functions
def trace_slide_processing(slide_id: str):
    """Start tracing for slide processing."""
    return trace_span("slide_processing", attributes={"slide_id": slide_id})


def trace_gpu_operation(gpu_id: int, operation: str):
    """Start tracing for GPU operation."""
    return trace_span(f"gpu_{operation}", attributes={"gpu_id": gpu_id, "operation": operation})


def trace_pacs_operation(operation: str, study_id: str):
    """Start tracing for PACS operation."""
    return trace_span(
        f"pacs_{operation}", attributes={"operation": operation, "study_id": study_id}
    )
