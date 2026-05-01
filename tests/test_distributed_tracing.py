#!/usr/bin/env python3
"""
Tests for OpenTelemetry Distributed Tracing

Tests the centralized tracing module and its integration with services.
"""

import os
import time
from unittest.mock import MagicMock, patch

import pytest

# Test if OpenTelemetry is available
try:
    from opentelemetry import trace

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

from src.monitoring.tracing import (
    DistributedTracer,
    add_span_attributes,
    add_span_event,
    get_trace_context,
    get_tracer,
    set_trace_context,
    trace_data_loading,
    trace_inference,
    trace_model_training,
    trace_span,
    traced,
)

pytestmark = pytest.mark.skipif(not OTEL_AVAILABLE, reason="OpenTelemetry not available")


class TestDistributedTracer:
    """Test DistributedTracer class."""

    def test_tracer_initialization(self):
        """Test tracer initialization."""
        tracer = DistributedTracer("test-service")
        assert tracer.service_name == "test-service"
        assert not tracer._initialized

    def test_tracer_initialize_console(self):
        """Test tracer initialization with console exporter."""
        tracer = DistributedTracer("test-service")
        tracer.initialize(enable_console=True)

        assert tracer._initialized
        assert tracer.tracer is not None
        assert tracer.tracer_provider is not None

        tracer.shutdown()

    def test_tracer_initialize_jaeger(self):
        """Test tracer initialization with Jaeger exporter."""
        tracer = DistributedTracer("test-service")

        # Mock Jaeger exporter to avoid actual connection
        with patch("src.monitoring.tracing.JaegerExporter"):
            tracer.initialize(jaeger_endpoint="localhost:6831")

            assert tracer._initialized
            assert tracer.tracer is not None

        tracer.shutdown()

    def test_tracer_initialize_otlp(self):
        """Test tracer initialization with OTLP exporter."""
        tracer = DistributedTracer("test-service")

        # Mock OTLP exporter to avoid actual connection
        with patch("src.monitoring.tracing.OTLPSpanExporter"):
            tracer.initialize(otlp_endpoint="http://localhost:4317")

            assert tracer._initialized
            assert tracer.tracer is not None

        tracer.shutdown()

    def test_tracer_get_tracer(self):
        """Test getting tracer instance."""
        tracer = DistributedTracer("test-service")
        tracer.initialize(enable_console=True)

        tracer_instance = tracer.get_tracer()
        assert tracer_instance is not None

        tracer.shutdown()

    def test_tracer_shutdown(self):
        """Test tracer shutdown."""
        tracer = DistributedTracer("test-service")
        tracer.initialize(enable_console=True)

        assert tracer._initialized

        tracer.shutdown()
        assert not tracer._initialized


class TestGlobalTracer:
    """Test global tracer functions."""

    def test_get_global_tracer(self):
        """Test getting global tracer instance."""
        tracer = get_tracer("test-service")
        assert isinstance(tracer, DistributedTracer)
        assert tracer.service_name == "test-service"

    def test_global_tracer_auto_initialize(self):
        """Test global tracer auto-initialization from environment."""
        # Clear environment
        os.environ.pop("JAEGER_ENDPOINT", None)
        os.environ.pop("OTLP_ENDPOINT", None)

        tracer = get_tracer("test-service")
        assert tracer._initialized  # Should auto-initialize


class TestTracedDecorator:
    """Test @traced decorator."""

    def test_traced_sync_function(self):
        """Test tracing synchronous function."""

        @traced("test.sync_function")
        def sync_function(x: int) -> int:
            return x * 2

        result = sync_function(5)
        assert result == 10

    def test_traced_async_function(self):
        """Test tracing asynchronous function."""

        @traced("test.async_function")
        async def async_function(x: int) -> int:
            return x * 2

        import asyncio

        result = asyncio.run(async_function(5))
        assert result == 10

    def test_traced_function_with_exception(self):
        """Test tracing function that raises exception."""

        @traced("test.failing_function", record_exception=True)
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            failing_function()

    def test_traced_function_custom_component(self):
        """Test tracing with custom component name."""

        @traced("test.custom", component="custom_component")
        def custom_function():
            return "success"

        result = custom_function()
        assert result == "success"


class TestTraceSpan:
    """Test trace_span context manager."""

    def test_trace_span_basic(self):
        """Test basic span creation."""
        with trace_span("test.span"):
            time.sleep(0.01)

    def test_trace_span_with_attributes(self):
        """Test span with attributes."""
        with trace_span("test.span", attributes={"key": "value", "count": 42}):
            time.sleep(0.01)

    def test_trace_span_with_exception(self):
        """Test span with exception."""
        with pytest.raises(ValueError):
            with trace_span("test.failing_span"):
                raise ValueError("Test error")

    def test_trace_span_custom_component(self):
        """Test span with custom component."""
        with trace_span("test.span", component="custom"):
            time.sleep(0.01)


class TestSpanAttributes:
    """Test span attribute functions."""

    def test_add_span_attributes(self):
        """Test adding attributes to current span."""
        with trace_span("test.span"):
            add_span_attributes(key1="value1", key2=42, key3=True)

    def test_add_span_event(self):
        """Test adding event to current span."""
        with trace_span("test.span"):
            add_span_event("test_event", {"detail": "test"})


class TestTraceContext:
    """Test trace context propagation."""

    def test_get_trace_context(self):
        """Test getting trace context."""
        with trace_span("test.span"):
            context = get_trace_context()
            assert isinstance(context, dict)

    def test_set_trace_context(self):
        """Test setting trace context."""
        # Create a context
        with trace_span("test.span"):
            context = get_trace_context()

        # Set context in new span
        result = set_trace_context(context)
        # Result should be a context object (or None if empty)
        assert result is not None or len(context) == 0


class TestConvenienceFunctions:
    """Test convenience tracing functions."""

    def test_trace_inference(self):
        """Test inference tracing."""
        with trace_inference("attention_mil", batch_size=32):
            time.sleep(0.01)

    def test_trace_data_loading(self):
        """Test data loading tracing."""
        with trace_data_loading("pcam", num_samples=1000):
            time.sleep(0.01)

    def test_trace_model_training(self):
        """Test model training tracing."""
        with trace_model_training("attention_mil", epoch=5):
            time.sleep(0.01)


class TestFastAPIInstrumentation:
    """Test FastAPI instrumentation."""

    def test_instrument_fastapi(self):
        """Test instrumenting FastAPI application."""
        from fastapi import FastAPI

        app = FastAPI(title="Test API")

        tracer = DistributedTracer("test-api")
        tracer.initialize(enable_console=True)

        # Should not raise exception
        tracer.instrument_fastapi(app)

        tracer.shutdown()


class TestSQLAlchemyInstrumentation:
    """Test SQLAlchemy instrumentation."""

    def test_instrument_sqlalchemy(self):
        """Test instrumenting SQLAlchemy engine."""
        from sqlalchemy import create_engine

        engine = create_engine("sqlite:///:memory:")

        tracer = DistributedTracer("test-db")
        tracer.initialize(enable_console=True)

        # Should not raise exception
        tracer.instrument_sqlalchemy(engine)

        tracer.shutdown()


class TestEndToEndTracing:
    """Test end-to-end tracing scenarios."""

    def test_nested_spans(self):
        """Test nested span creation."""
        with trace_span("parent"):
            add_span_attributes(level="parent")

            with trace_span("child1"):
                add_span_attributes(level="child1")
                time.sleep(0.01)

            with trace_span("child2"):
                add_span_attributes(level="child2")
                time.sleep(0.01)

    def test_traced_function_with_manual_spans(self):
        """Test combining @traced decorator with manual spans."""

        @traced("test.function")
        def function_with_spans():
            with trace_span("internal_operation"):
                add_span_attributes(operation="internal")
                time.sleep(0.01)

            return "success"

        result = function_with_spans()
        assert result == "success"

    def test_distributed_trace_simulation(self):
        """Test simulating distributed trace across services."""

        @traced("service_a.process")
        def service_a():
            add_span_attributes(service="a")

            # Get trace context
            context = get_trace_context()

            # Call service B with context
            return service_b(context)

        @traced("service_b.process")
        def service_b(trace_context):
            # Set trace context
            set_trace_context(trace_context)

            add_span_attributes(service="b")
            return "success"

        result = service_a()
        assert result == "success"


class TestErrorHandling:
    """Test error handling in tracing."""

    def test_tracing_without_initialization(self):
        """Test that tracing works even without explicit initialization."""

        @traced("test.function")
        def test_function():
            return "success"

        # Should not raise exception
        result = test_function()
        assert result == "success"

    def test_span_with_invalid_attributes(self):
        """Test span with invalid attribute types."""
        with trace_span("test.span"):
            # Should handle various types gracefully
            add_span_attributes(
                string="value",
                integer=42,
                float_val=3.14,
                boolean=True,
                none_val=None,
            )


class TestPerformance:
    """Test tracing performance impact."""

    def test_tracing_overhead(self):
        """Test that tracing overhead is minimal."""

        @traced("test.performance")
        def fast_function():
            return 42

        # Measure with tracing
        start = time.time()
        for _ in range(1000):
            fast_function()
        duration_with_tracing = time.time() - start

        # Overhead should be minimal (< 10ms per call)
        assert duration_with_tracing < 10.0

    def test_span_creation_performance(self):
        """Test span creation performance."""
        start = time.time()

        for _ in range(1000):
            with trace_span("test.span"):
                pass

        duration = time.time() - start

        # Should be fast (< 5 seconds for 1000 spans)
        assert duration < 5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
