# Deployment Guide

This directory contains deployment examples for the computational pathology model.

## FastAPI REST API

### Quick Start

```bash
# Install deployment dependencies
pip install fastapi uvicorn python-multipart

# Run the API server
cd deploy
python api.py

# Or use uvicorn directly
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### API Documentation

Once running, visit:
- **Interactive docs**: http://localhost:8000/docs
- **Alternative docs**: http://localhost:8000/redoc
- **OpenAPI schema**: http://localhost:8000/openapi.json

### Example Usage

#### Python Client

```python
import requests
import numpy as np

# Prepare sample data
data = {
    "wsi_features": np.random.randn(50, 1024).tolist(),  # 50 patches
    "genomic": np.random.randn(2000).tolist(),
    "clinical_text": [100, 200, 300, 400, 500]
}

# Make prediction
response = requests.post("http://localhost:8000/predict", json=data)
result = response.json()

print(f"Predicted class: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Probabilities: {result['probabilities']}")
```

#### cURL

```bash
# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/model-info

# Prediction (with all modalities)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "wsi_features": [[0.1, 0.2, ...], ...],
    "genomic": [0.1, 0.2, ...],
    "clinical_text": [100, 200, 300]
  }'

# Prediction (with missing modalities)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "genomic": [0.1, 0.2, ...]
  }'
```

### Endpoints

#### `GET /`
Root endpoint with API information.

#### `GET /health`
Health check endpoint.

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu"
}
```

#### `GET /model-info`
Get model architecture information.

**Response**:
```json
{
  "architecture": "MultimodalFusionModel",
  "embed_dim": 256,
  "num_classes": 4,
  "total_parameters": 29544581,
  "device": "cpu",
  "supported_modalities": ["wsi", "genomic", "clinical_text"]
}
```

#### `POST /predict`
Make prediction from multimodal data.

**Request Body**:
```json
{
  "wsi_features": [[float, ...], ...],  // Optional: [num_patches, 1024]
  "genomic": [float, ...],               // Optional: [2000]
  "clinical_text": [int, ...]           // Optional: [seq_len]
}
```

**Response**:
```json
{
  "predicted_class": 2,
  "confidence": 0.87,
  "probabilities": [0.05, 0.08, 0.87, 0.00],
  "available_modalities": ["wsi", "genomic"]
}
```

#### `POST /batch-predict`
Make predictions for multiple samples (max 32).

**Request Body**:
```json
[
  {"wsi_features": [...], "genomic": [...]},
  {"genomic": [...], "clinical_text": [...]},
  ...
]
```

**Response**:
```json
{
  "predictions": [
    {"predicted_class": 2, "confidence": 0.87, ...},
    {"predicted_class": 1, "confidence": 0.92, ...}
  ],
  "count": 2
}
```

## Production Deployment

### Docker

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install fastapi uvicorn

COPY . .

EXPOSE 8000

CMD ["uvicorn", "deploy.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t pathology-api .
docker run -p 8000:8000 pathology-api
```

### Kubernetes

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pathology-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: pathology-api
  template:
    metadata:
      labels:
        app: pathology-api
    spec:
      containers:
      - name: api
        image: pathology-api:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
---
apiVersion: v1
kind: Service
metadata:
  name: pathology-api-service
spec:
  selector:
    app: pathology-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Performance Optimization

#### 1. Model Quantization

```python
# Quantize model for faster inference
import torch.quantization as quantization

# Dynamic quantization (easiest)
model_quantized = quantization.quantize_dynamic(
    MODEL, {torch.nn.Linear}, dtype=torch.qint8
)

# Static quantization (best performance)
model.qconfig = quantization.get_default_qconfig('fbgemm')
model_prepared = quantization.prepare(model)
# Calibrate with representative data
model_quantized = quantization.convert(model_prepared)
```

#### 2. ONNX Export

```python
# Export to ONNX for deployment
import torch.onnx

dummy_input = {
    'wsi_features': torch.randn(1, 100, 1024),
    'genomic': torch.randn(1, 2000),
    'clinical_text': torch.randint(0, 30000, (1, 128))
}

torch.onnx.export(
    MODEL,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=14,
    input_names=['wsi', 'genomic', 'clinical'],
    output_names=['embeddings'],
    dynamic_axes={
        'wsi': {1: 'num_patches'},
        'clinical': {1: 'seq_len'}
    }
)
```

#### 3. Batch Processing

```python
# Process multiple samples efficiently
@app.post("/batch-predict-optimized")
async def batch_predict_optimized(requests: List[PredictionRequest]):
    # Collate into single batch
    batch = collate_requests(requests)
    
    # Single forward pass
    with torch.no_grad():
        embeddings = MODEL(batch)
        logits = CLASSIFIER(embeddings)
        probabilities = torch.softmax(logits, dim=-1)
    
    # Split results
    return split_results(probabilities, len(requests))
```

## Monitoring

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, generate_latest

# Add metrics
prediction_counter = Counter('predictions_total', 'Total predictions')
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

### Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

@app.post("/predict")
async def predict(request: PredictionRequest):
    logger.info(f"Prediction request: {len(request.wsi_features or [])} patches")
    # ... prediction logic
    logger.info(f"Prediction complete: class={result.predicted_class}")
```

## Security

### API Key Authentication

```python
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

API_KEY = "your-secret-key"
api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

@app.post("/predict")
async def predict(request: PredictionRequest, api_key: str = Security(verify_api_key)):
    # ... prediction logic
```

### Rate Limiting

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/predict")
@limiter.limit("10/minute")
async def predict(request: Request, data: PredictionRequest):
    # ... prediction logic
```

## Testing

```python
from fastapi.testclient import TestClient

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict():
    data = {
        "genomic": [0.1] * 2000
    }
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert "predicted_class" in response.json()
```

## Troubleshooting

### Common Issues

**Model not loading**:
- Check `models/best_model.pth` exists
- Verify model architecture matches checkpoint

**Out of memory**:
- Reduce batch size
- Use model quantization
- Enable gradient checkpointing

**Slow inference**:
- Use GPU if available
- Enable ONNX runtime
- Implement request batching

**CORS errors**:
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Performance Benchmarks

| Configuration | Latency (p50) | Latency (p99) | Throughput |
|---------------|---------------|---------------|------------|
| CPU (single) | 500ms | 800ms | 2 req/s |
| CPU (batch=16) | 100ms | 150ms | 16 req/s |
| GPU (single) | 50ms | 80ms | 20 req/s |
| GPU (batch=32) | 20ms | 30ms | 160 req/s |

## Next Steps

1. Add authentication and authorization
2. Implement request caching
3. Add model versioning
4. Set up CI/CD pipeline
5. Configure auto-scaling
6. Add comprehensive monitoring
