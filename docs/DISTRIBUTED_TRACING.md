# Distributed Tracing with OpenTelemetry

Distributed tracing provides end-to-end visibility into request flows across HistoCore services.

## Overview

**Benefits**:
- Track requests across multiple services
- Identify performance bottlenecks
- Debug complex distributed systems
- Monitor service dependencies
- Analyze latency distributions

**Supported Backends**:
- Jaeger (recommended for development)
- OTLP (OpenTelemetry Protocol)
- Console (debugging)

## Quick Start

### 1. Install Dependencies

```bash
pip install opentelemetry-api opentelemetry-sdk \
    opentelemetry-instrumentation-fastapi \
    opentelemetry-instrumentation-requests \
    opentelemetry-exporter-jaeger \
    opentelemetry-exporter-otlp
```

### 2. Start Jaeger (Development)

```bash
# Using Docker
docker run -d --name jaeger \
  -p 6831:6831/udp \
  -p 16686:16686 \
  jaegertracing/all-in-one:latest

# Access UI at http://localhost:16686
```

### 3. Enable Tracing

```python
from src.monitoring.tracing import get_tracer

# Initialize tracer
tracer = get_tracer("my-service")
tracer.initialize(
    jaeger_endpoint="localhost:6831",
    service_version="1.0.0",
    environment="development",
)
```

### 4. Instrument FastAPI

```python
from fastapi import FastAPI
from src.monitoring.tracing import get_tracer

app = FastAPI(title="My Service")

# Instrument app
tracer = get_tracer("my-service")
tracer.initialize(jaeger_endpoint="localhost:6831")
tracer.instrument_fastapi(app)
```

## Usage

### Automatic Tracing (Decorator)

```python
from src.monitoring.tracing import traced

@traced("process_slide")
def process_slide(slide_id: str):
    """Process slide with automatic tracing."""
    # Function automatically traced
    return {"slide_id": slide_id, "status": "processed"}

# Async functions also supported
@traced("async_process")
async def async_process(data: dict):
    """Async function with tracing."""
    await some_async_operation()
    return result
```

### Manual Tracing (Context Manager)

```python
from src.monitoring.tracing import trace_span, add_span_attributes, add_span_event

def complex_operation():
    """Complex operation with manual tracing."""
    
    with trace_span("complex_operation", {"operation_type": "batch"}):
        # Add attributes
        add_span_attributes(
            batch_size=100,
            model_name="attention_mil",
        )
        
        # Process data
        for i in range(100):
            with trace_span(f"process_item_{i}", {"item_id": i}):
                process_item(i)
        
        # Add event
        add_span_event("processing_complete", {"items_processed": 100})
```

### Convenience Functions

```python
from src.monitoring.tracing import (
    trace_inference,
    trace_data_loading,
    trace_model_training,
)

# Trace inference
with trace_inference("attention_mil", batch_size=32):
    predictions = model(batch)

# Trace data loading
with trace_data_loading("pcam", num_samples=1000):
    dataset = load_dataset()

# Trace training
with trace_model_training("attention_mil", epoch=5):
    train_one_epoch()
```

## Configuration

### Environment Variables

```bash
# Jaeger endpoint
export JAEGER_ENDPOINT="localhost:6831"

# OTLP endpoint
export OTLP_ENDPOINT="http://localhost:4317"

# Enable console exporter (debugging)
export OTEL_CONSOLE="true"

# Environment name
export ENVIRONMENT="production"
```

### Programmatic Configuration

```python
from src.monitoring.tracing import get_tracer

tracer = get_tracer("my-service")
tracer.initialize(
    jaeger_endpoint="localhost:6831",
    otlp_endpoint="http://localhost:4317",
    service_version="1.0.0",
    environment="production",
    enable_console=False,  # Disable console output
)
```

## Integration Examples

### FastAPI Application

```python
from fastapi import FastAPI
from src.monitoring.tracing import get_tracer, traced

app = FastAPI(title="HistoCore API")

# Initialize and instrument
tracer = get_tracer("histocore-api")
tracer.initialize(jaeger_endpoint="localhost:6831")
tracer.instrument_fastapi(app)

@app.get("/process/{slide_id}")
@traced("api.process_slide")
async def process_slide(slide_id: str):
    """Process slide endpoint with tracing."""
    result = await process_slide_async(slide_id)
    return result
```

### Inference Pipeline

```python
from src.monitoring.tracing import traced, add_span_attributes

class InferenceEngine:
    @traced("inference.load_model")
    def load_model(self, model_path: str):
        """Load model with tracing."""
        add_span_attributes(model_path=model_path)
        model = torch.load(model_path)
        return model
    
    @traced("inference.preprocess")
    def preprocess(self, image):
        """Preprocess with tracing."""
        add_span_attributes(
            image_size=image.size,
            image_mode=image.mode,
        )
        return transform(image)
    
    @traced("inference.predict")
    def predict(self, batch):
        """Predict with tracing."""
        add_span_attributes(batch_size=len(batch))
        
        with torch.no_grad():
            predictions = self.model(batch)
        
        add_span_event("prediction_complete", {
            "num_predictions": len(predictions)
        })
        
        return predictions
```

### Training Pipeline

```python
from src.monitoring.tracing import trace_model_training, add_span_attributes

def train_model(model, train_loader, num_epochs):
    """Train model with tracing."""
    
    for epoch in range(num_epochs):
        with trace_model_training("attention_mil", epoch=epoch):
            # Training loop
            for batch_idx, (data, target) in enumerate(train_loader):
                loss = train_step(model, data, target)
                
                # Add metrics
                if batch_idx % 10 == 0:
                    add_span_attributes(
                        epoch=epoch,
                        batch=batch_idx,
                        loss=float(loss),
                    )
```

### Distributed Services

```python
from src.monitoring.tracing import get_trace_context, set_trace_context
import requests

# Service A: Propagate context
def call_service_b():
    """Call service B with trace context."""
    
    # Get current trace context
    context = get_trace_context()
    
    # Send request with context headers
    response = requests.post(
        "http://service-b/process",
        json={"data": "value"},
        headers=context,  # Propagate trace context
    )
    
    return response.json()

# Service B: Extract context
from fastapi import Request

@app.post("/process")
async def process(request: Request):
    """Process with extracted trace context."""
    
    # Extract trace context from headers
    context = set_trace_context(dict(request.headers))
    
    # Continue trace
    with trace_span("service_b.process"):
        result = process_data()
    
    return result
```

## Viewing Traces

### Jaeger UI

1. Open http://localhost:16686
2. Select service from dropdown
3. Click "Find Traces"
4. Click on trace to view details

**Trace View**:
- Timeline of all spans
- Service dependencies
- Latency breakdown
- Error details

### Analyzing Performance

```
Example Trace:
├─ api.process_slide (250ms)
│  ├─ data.load_slide (50ms)
│  ├─ inference.preprocess (30ms)
│  ├─ inference.predict (150ms)  ← Bottleneck
│  └─ data.save_results (20ms)
```

## Best Practices

### 1. Meaningful Span Names

```python
# Good: Descriptive, hierarchical
@traced("inference.model.forward")
@traced("data.wsi.load_tile")
@traced("pacs.query.study")

# Bad: Generic, unclear
@traced("process")
@traced("do_work")
@traced("function1")
```

### 2. Add Relevant Attributes

```python
# Good: Useful context
add_span_attributes(
    slide_id="12345",
    model_name="attention_mil",
    batch_size=32,
    gpu_id=0,
)

# Bad: Too much detail
add_span_attributes(
    every_single_pixel_value=[...],  # Too large
    internal_counter=i,  # Not useful
)
```

### 3. Use Events for Milestones

```python
# Mark important events
add_span_event("model_loaded", {"model_size_mb": 120})
add_span_event("preprocessing_complete", {"tiles_extracted": 1000})
add_span_event("inference_started", {"batch_count": 10})
```

### 4. Handle Errors Properly

```python
@traced("operation", record_exception=True)
def operation():
    """Errors automatically recorded in span."""
    try:
        risky_operation()
    except Exception as e:
        # Exception recorded in span
        raise
```

### 5. Avoid Over-Tracing

```python
# Good: Trace significant operations
@traced("process_batch")
def process_batch(items):
    for item in items:
        process_item(item)  # Don't trace each item

# Bad: Too granular
@traced("process_batch")
def process_batch(items):
    for item in items:
        with trace_span(f"item_{item}"):  # Too many spans
            process_item(item)
```

## Production Deployment

### 1. Use OTLP Collector

```yaml
# docker-compose.yml
services:
  otel-collector:
    image: otel/opentelemetry-collector:latest
    ports:
      - "4317:4317"  # OTLP gRPC
      - "4318:4318"  # OTLP HTTP
    volumes:
      - ./otel-config.yaml:/etc/otel/config.yaml
    command: ["--config=/etc/otel/config.yaml"]
  
  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"  # UI
```

### 2. Configure Sampling

```python
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

# Sample 10% of traces in production
tracer_provider = TracerProvider(
    sampler=TraceIdRatioBased(0.1),
    resource=resource,
)
```

### 3. Set Resource Limits

```python
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure batch processor
span_processor = BatchSpanProcessor(
    exporter,
    max_queue_size=2048,
    max_export_batch_size=512,
    schedule_delay_millis=5000,
)
```

## Troubleshooting

### Issue: No traces appearing

**Causes**:
- Tracer not initialized
- Exporter not configured
- Network connectivity issues

**Solutions**:
```python
# Enable console exporter for debugging
tracer.initialize(enable_console=True)

# Check logs
logging.basicConfig(level=logging.DEBUG)

# Verify endpoint
curl http://localhost:16686
```

### Issue: High overhead

**Causes**:
- Too many spans
- Large span attributes
- Synchronous export

**Solutions**:
- Reduce span granularity
- Use sampling in production
- Use batch span processor (default)

### Issue: Missing context propagation

**Cause**: Context not propagated between services

**Solution**:
```python
# Always propagate context
context = get_trace_context()
requests.post(url, headers=context)

# Always extract context
context = set_trace_context(request.headers)
```

## Performance Impact

Typical overhead:
- **Latency**: <1ms per span
- **Memory**: ~1KB per span
- **CPU**: <1% with sampling

**Recommendations**:
- Use sampling in production (10-20%)
- Batch span export (default)
- Avoid tracing hot loops

## References

- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [Jaeger Documentation](https://www.jaegertracing.io/docs/)
- [Distributed Tracing Best Practices](https://opentelemetry.io/docs/concepts/signals/traces/)

## Next Steps

1. Install dependencies
2. Start Jaeger locally
3. Instrument your services
4. View traces in Jaeger UI
5. Optimize based on insights

For production deployment, see [DEPLOYMENT.md](DEPLOYMENT.md).
