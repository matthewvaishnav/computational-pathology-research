#!/usr/bin/env python3
"""
OpenTelemetry Distributed Tracing Integration Examples

This file demonstrates how to use distributed tracing across HistoCore services.
"""

import asyncio
import logging
import time

from src.monitoring.tracing import (
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Example 1: FastAPI Service with Tracing
def example_fastapi_integration():
    """Example: Integrate tracing with FastAPI application."""
    from fastapi import FastAPI

    app = FastAPI(title="HistoCore API")

    # Initialize tracer on startup
    @app.on_event("startup")
    async def startup_event():
        tracer = get_tracer("histocore-api")
        tracer.initialize(
            jaeger_endpoint="localhost:6831",
            service_version="2.0.0",
            environment="development",
        )
        # Automatically instrument FastAPI
        tracer.instrument_fastapi(app)
        logger.info("Tracing initialized")

    # Endpoints are automatically traced
    @app.get("/process/{slide_id}")
    async def process_slide(slide_id: str):
        """Process slide - automatically traced by FastAPI instrumentation."""
        # Add custom attributes to the span
        add_span_attributes(slide_id=slide_id, slide_type="WSI")

        # Simulate processing
        result = await process_slide_internal(slide_id)

        # Add event to mark completion
        add_span_event("processing_complete", {"result_size": len(result)})

        return result

    @traced("internal.process_slide")
    async def process_slide_internal(slide_id: str):
        """Internal processing with explicit tracing."""
        await asyncio.sleep(0.1)  # Simulate work
        return {"slide_id": slide_id, "status": "processed"}

    return app


# Example 2: Inference Pipeline with Tracing
class TracedInferenceEngine:
    """Example: Inference engine with distributed tracing."""

    def __init__(self, model_name: str = "attention_mil"):
        self.model_name = model_name

    @traced("inference.load_model")
    def load_model(self, model_path: str):
        """Load model with tracing."""
        add_span_attributes(model_path=model_path, model_name=self.model_name)

        # Simulate model loading
        time.sleep(0.2)

        add_span_event("model_loaded", {"model_size_mb": 120})
        logger.info(f"Model loaded: {self.model_name}")

    @traced("inference.preprocess")
    def preprocess(self, image_data: dict):
        """Preprocess image with tracing."""
        add_span_attributes(
            image_width=image_data.get("width", 0),
            image_height=image_data.get("height", 0),
        )

        # Simulate preprocessing
        time.sleep(0.05)

        add_span_event("preprocessing_complete", {"patches_extracted": 100})
        return {"patches": 100}

    @traced("inference.predict")
    def predict(self, preprocessed_data: dict):
        """Run inference with tracing."""
        batch_size = preprocessed_data.get("patches", 0)
        add_span_attributes(batch_size=batch_size, model_name=self.model_name)

        # Use convenience function for inference tracing
        with trace_inference(self.model_name, batch_size):
            # Simulate inference
            time.sleep(0.3)

            predictions = {"class": "malignant", "confidence": 0.92}

        add_span_event("inference_complete", {"prediction_class": predictions["class"]})
        return predictions

    def run_full_pipeline(self, image_data: dict):
        """Run full inference pipeline with nested tracing."""
        with trace_span("inference.full_pipeline", {"pipeline": "end_to_end"}):
            self.load_model("/models/attention_mil.pth")
            preprocessed = self.preprocess(image_data)
            predictions = self.predict(preprocessed)
            return predictions


# Example 3: Training Pipeline with Tracing
class TracedTrainingPipeline:
    """Example: Training pipeline with distributed tracing."""

    def __init__(self, model_name: str = "attention_mil"):
        self.model_name = model_name

    @traced("training.load_data")
    def load_training_data(self, dataset_name: str, num_samples: int):
        """Load training data with tracing."""
        with trace_data_loading(dataset_name, num_samples):
            # Simulate data loading
            time.sleep(0.1)

            add_span_event("data_loaded", {"dataset_size_gb": 2.5})
            return {"samples": num_samples}

    @traced("training.train_epoch")
    def train_epoch(self, epoch: int, data: dict):
        """Train one epoch with tracing."""
        with trace_model_training(self.model_name, epoch):
            num_batches = 100

            for batch_idx in range(num_batches):
                # Don't trace individual batches (too granular)
                # Just simulate training
                time.sleep(0.001)

                # Log progress every 10 batches
                if batch_idx % 10 == 0:
                    add_span_attributes(
                        current_batch=batch_idx,
                        total_batches=num_batches,
                        current_loss=0.5 - (batch_idx * 0.001),
                    )

            # Mark epoch complete
            add_span_event(
                "epoch_complete", {"final_loss": 0.4, "accuracy": 0.85, "epoch": epoch}
            )

            return {"loss": 0.4, "accuracy": 0.85}

    def run_training(self, num_epochs: int = 5):
        """Run full training with tracing."""
        with trace_span("training.full_training", {"num_epochs": num_epochs}):
            # Load data
            data = self.load_training_data("pcam", num_samples=10000)

            # Train epochs
            for epoch in range(num_epochs):
                metrics = self.train_epoch(epoch, data)
                logger.info(f"Epoch {epoch}: loss={metrics['loss']}, acc={metrics['accuracy']}")

            add_span_event("training_complete", {"total_epochs": num_epochs})


# Example 4: Distributed Service Communication
class ServiceA:
    """Example: Service A that calls Service B with trace context propagation."""

    @traced("service_a.process_request")
    def process_request(self, request_id: str):
        """Process request and call Service B."""
        add_span_attributes(request_id=request_id, service="service_a")

        # Get current trace context
        trace_context = get_trace_context()

        # Call Service B with trace context
        result = self.call_service_b(request_id, trace_context)

        add_span_event("service_b_called", {"result": result})
        return result

    def call_service_b(self, request_id: str, trace_context: dict):
        """Simulate calling Service B with trace propagation."""
        # In real implementation, this would be an HTTP request
        # with trace_context as headers
        import requests

        # Example (not executed):
        # response = requests.post(
        #     "http://service-b/process",
        #     json={"request_id": request_id},
        #     headers=trace_context,  # Propagate trace context
        # )

        # Simulate response
        return {"status": "success", "request_id": request_id}


class ServiceB:
    """Example: Service B that receives trace context from Service A."""

    @traced("service_b.process_request")
    def process_request(self, request_id: str, trace_context: dict):
        """Process request with extracted trace context."""
        # Extract trace context from headers
        context = set_trace_context(trace_context)

        # This span will be part of the same trace as Service A
        add_span_attributes(request_id=request_id, service="service_b")

        # Do work
        time.sleep(0.1)

        add_span_event("processing_complete", {"request_id": request_id})
        return {"status": "processed"}


# Example 5: Error Handling with Tracing
class TracedErrorHandling:
    """Example: Error handling with automatic exception recording."""

    @traced("operation.risky", record_exception=True)
    def risky_operation(self, should_fail: bool = False):
        """Operation that might fail - exceptions automatically recorded."""
        add_span_attributes(should_fail=should_fail)

        if should_fail:
            # Exception will be automatically recorded in span
            raise ValueError("Operation failed as requested")

        return {"status": "success"}

    def safe_operation_with_manual_error_handling(self):
        """Manual error handling with tracing."""
        with trace_span("operation.safe") as span:
            try:
                # Do work
                result = self.risky_operation(should_fail=False)
                add_span_event("operation_succeeded")
                return result

            except Exception as e:
                # Manually record exception if needed
                span.record_exception(e)
                add_span_event("operation_failed", {"error": str(e)})
                raise


# Example 6: Federated Learning with Tracing
class TracedFederatedLearning:
    """Example: Federated learning with distributed tracing."""

    @traced("fl.coordinator.start_round")
    def coordinator_start_round(self, round_id: int, client_ids: list):
        """Coordinator starts training round."""
        add_span_attributes(round_id=round_id, num_clients=len(client_ids))

        # Get trace context to propagate to clients
        trace_context = get_trace_context()

        # Send to clients (with trace context)
        for client_id in client_ids:
            self.send_to_client(client_id, round_id, trace_context)

        add_span_event("round_started", {"clients_notified": len(client_ids)})

    def send_to_client(self, client_id: str, round_id: int, trace_context: dict):
        """Send training request to client."""
        # In real implementation, this would be gRPC/HTTP with trace context
        logger.info(f"Sending round {round_id} to client {client_id}")

    @traced("fl.client.train_round")
    def client_train_round(self, round_id: int, trace_context: dict):
        """Client trains on local data."""
        # Extract trace context (continues the trace from coordinator)
        context = set_trace_context(trace_context)

        add_span_attributes(round_id=round_id, client_id="hospital_a")

        # Train locally
        with trace_span("fl.client.local_training"):
            time.sleep(0.5)  # Simulate training
            add_span_event("training_complete", {"samples_trained": 1000})

        # Send update back to coordinator
        add_span_event("update_sent", {"model_size_mb": 50})


def main():
    """Run all examples."""
    # Initialize tracer
    tracer = get_tracer("tracing-examples")
    tracer.initialize(enable_console=True)

    print("\n=== Example 1: Inference Pipeline ===")
    engine = TracedInferenceEngine()
    result = engine.run_full_pipeline({"width": 1024, "height": 1024})
    print(f"Inference result: {result}")

    print("\n=== Example 2: Training Pipeline ===")
    trainer = TracedTrainingPipeline()
    trainer.run_training(num_epochs=2)

    print("\n=== Example 3: Distributed Services ===")
    service_a = ServiceA()
    service_b = ServiceB()
    result = service_a.process_request("req-123")
    print(f"Service result: {result}")

    print("\n=== Example 4: Error Handling ===")
    error_handler = TracedErrorHandling()
    try:
        error_handler.risky_operation(should_fail=True)
    except ValueError as e:
        print(f"Caught expected error: {e}")

    print("\n=== Example 5: Federated Learning ===")
    fl = TracedFederatedLearning()
    fl.coordinator_start_round(round_id=1, client_ids=["hospital_a", "hospital_b"])

    # Shutdown
    tracer.shutdown()
    print("\n=== All examples complete ===")


if __name__ == "__main__":
    main()
