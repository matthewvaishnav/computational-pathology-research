# HistoCore API Documentation

Complete API reference for HistoCore Real-Time WSI Streaming.

## Quick Links

- **[OpenAPI Specification](openapi.yaml)** - Machine-readable API spec
- **[Interactive API Docs](#interactive-documentation)** - Try the API in your browser
- **[Authentication Guide](#authentication)** - OAuth 2.0 setup
- **[Rate Limits](#rate-limits)** - Usage quotas and throttling
- **[Error Handling](#error-handling)** - Error codes and recovery
- **[WebSocket Streaming](#websocket-streaming)** - Real-time updates
- **[Code Examples](#code-examples)** - Python, JavaScript, cURL

## Base URLs

| Environment | URL | Description |
|-------------|-----|-------------|
| Production | `https://api.histocore.ai/v1` | Production API |
| Staging | `https://staging-api.histocore.ai/v1` | Testing environment |
| Local | `http://localhost:8000/v1` | Development server |

## Authentication

All API endpoints require OAuth 2.0 JWT token authentication.

### Getting a Token

```bash
# Request token
curl -X POST https://api.histocore.ai/auth/token \
  -H "Content-Type: application/json" \
  -d '{
    "client_id": "your_client_id",
    "client_secret": "your_client_secret",
    "grant_type": "client_credentials"
  }'

# Response
{
  "access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 3600
}
```

### Using the Token

Include the token in the `Authorization` header:

```bash
curl -H "Authorization: Bearer <your_token>" \
  https://api.histocore.ai/v1/health
```

### Token Expiration

- Tokens expire after 1 hour
- Refresh tokens before expiration
- Handle 401 responses by refreshing token

## Rate Limits

| Tier | Requests/Minute | Concurrent Slides | Burst Limit |
|------|-----------------|-------------------|-------------|
| Standard | 100 | 10 | 200/5min |
| Premium | 500 | 50 | 1000/5min |
| Enterprise | Unlimited | Unlimited | N/A |

### Rate Limit Headers

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640000000
```

### Handling Rate Limits

```python
import time
import requests

def make_request_with_retry(url, headers):
    while True:
        response = requests.get(url, headers=headers)
        
        if response.status_code == 429:
            # Rate limited - wait and retry
            reset_time = int(response.headers['X-RateLimit-Reset'])
            wait_time = reset_time - time.time()
            time.sleep(max(wait_time, 0) + 1)
            continue
            
        return response
```

## Interactive Documentation

### Swagger UI

View and test the API interactively:

```bash
# Start local server
docker run -p 8080:8080 \
  -e SWAGGER_JSON=/docs/openapi.yaml \
  -v $(pwd)/docs/api:/docs \
  swaggerapi/swagger-ui

# Open browser
open http://localhost:8080
```

### Redoc

Alternative documentation viewer:

```bash
# Start Redoc
docker run -p 8080:80 \
  -e SPEC_URL=https://raw.githubusercontent.com/histocore/histocore/main/docs/api/openapi.yaml \
  redocly/redoc

# Open browser
open http://localhost:8080
```

## Core Endpoints

### Health Checks

#### Basic Health Check

```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-04-26T10:30:00Z",
  "version": "1.0.0"
}
```

#### Detailed Health Status

```bash
GET /health/detailed
Authorization: Bearer <token>
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-04-26T10:30:00Z",
  "components": {
    "system": {
      "status": "healthy",
      "message": "CPU: 45%, Memory: 60%, Disk: 70%"
    },
    "gpu": {
      "status": "healthy",
      "message": "GPU 0: 8GB/12GB (67%), Utilization: 85%"
    },
    "metrics": {
      "status": "healthy",
      "message": "Prometheus collecting metrics"
    },
    "cache": {
      "status": "healthy",
      "message": "Redis connected, 1000 keys"
    }
  }
}
```

### WSI Processing

#### Start Processing

```bash
POST /process/wsi
Authorization: Bearer <token>
Content-Type: application/json

{
  "wsi_path": "/data/slides/patient_001.svs",
  "config": {
    "tile_size": 1024,
    "batch_size": 32,
    "memory_budget_gb": 2.0,
    "target_time": 30.0,
    "confidence_threshold": 0.95
  },
  "metadata": {
    "patient_id": "P001",
    "study_id": "S001",
    "priority": "high"
  }
}
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "created_at": "2026-04-26T10:30:00Z",
  "stream_url": "wss://api.histocore.ai/v1/process/wsi/550e8400-e29b-41d4-a716-446655440000/stream"
}
```

#### Get Processing Status

```bash
GET /process/wsi/{job_id}
Authorization: Bearer <token>
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "progress": {
    "patches_processed": 50000,
    "total_patches": 100000,
    "percent_complete": 50.0,
    "current_confidence": 0.87,
    "estimated_time_remaining": 15.0
  }
}
```

**Completed Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "progress": {
    "patches_processed": 100000,
    "total_patches": 100000,
    "percent_complete": 100.0,
    "current_confidence": 0.96,
    "estimated_time_remaining": 0.0
  },
  "result": {
    "prediction": {
      "class": 1,
      "probability": 0.96,
      "confidence": 0.96
    },
    "processing_time": 28.5,
    "attention_heatmap_url": "https://api.histocore.ai/v1/results/550e8400.../heatmap.png"
  }
}
```

#### Cancel Processing

```bash
POST /process/wsi/{job_id}/cancel
Authorization: Bearer <token>
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "cancelled",
  "message": "Processing cancelled successfully"
}
```

### PACS Integration

#### Query Studies

```bash
GET /pacs/studies?patient_id=P001&modality=SM
Authorization: Bearer <token>
```

**Response:**
```json
[
  {
    "study_uid": "1.2.840.113619.2.55.3.604688119.868.1234567890.1",
    "patient_id": "P001",
    "patient_name": "Doe^John",
    "study_date": "2026-04-26",
    "modality": "SM",
    "description": "Breast biopsy H&E",
    "series_count": 1
  }
]
```

#### Retrieve Study

```bash
POST /pacs/studies/{study_uid}/retrieve
Authorization: Bearer <token>
```

**Response:**
```json
{
  "job_id": "660e8400-e29b-41d4-a716-446655440000",
  "study_uid": "1.2.840.113619.2.55.3.604688119.868.1234567890.1",
  "status": "retrieving"
}
```

### Clinical Reports

#### Generate Report

```bash
POST /clinical/reports
Authorization: Bearer <token>
Content-Type: application/json

{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "template": "detailed",
  "include_heatmaps": true,
  "language": "en"
}
```

**Response:** PDF file download

## WebSocket Streaming

Real-time processing updates via WebSocket.

### Connection

```javascript
const ws = new WebSocket(
  'wss://api.histocore.ai/v1/process/wsi/550e8400.../stream',
  {
    headers: {
      'Authorization': 'Bearer <token>'
    }
  }
);

ws.onopen = () => {
  console.log('Connected to processing stream');
};

ws.onmessage = (event) => {
  const update = JSON.parse(event.data);
  console.log('Progress:', update.progress.percent_complete);
  console.log('Confidence:', update.progress.current_confidence);
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('Stream closed');
};
```

### Message Format

```json
{
  "type": "progress",
  "timestamp": "2026-04-26T10:30:15Z",
  "progress": {
    "patches_processed": 50000,
    "total_patches": 100000,
    "percent_complete": 50.0,
    "current_confidence": 0.87,
    "estimated_time_remaining": 15.0
  }
}
```

### Completion Message

```json
{
  "type": "completed",
  "timestamp": "2026-04-26T10:30:30Z",
  "result": {
    "prediction": {
      "class": 1,
      "probability": 0.96,
      "confidence": 0.96
    },
    "processing_time": 28.5,
    "attention_heatmap_url": "https://api.histocore.ai/v1/results/550e8400.../heatmap.png"
  }
}
```

## Error Handling

### Error Response Format

```json
{
  "error": "ValidationError",
  "message": "Invalid tile_size: must be between 256 and 2048",
  "details": {
    "field": "config.tile_size",
    "value": 4096,
    "constraint": "256 <= value <= 2048"
  },
  "timestamp": "2026-04-26T10:30:00Z"
}
```

### HTTP Status Codes

| Code | Meaning | Action |
|------|---------|--------|
| 200 | Success | Continue |
| 202 | Accepted | Poll for status |
| 400 | Bad Request | Fix request parameters |
| 401 | Unauthorized | Refresh token |
| 404 | Not Found | Check resource ID |
| 429 | Rate Limited | Wait and retry |
| 500 | Server Error | Retry with backoff |
| 503 | Unavailable | Service maintenance |

### Error Types

| Error | Description | Recovery |
|-------|-------------|----------|
| `ValidationError` | Invalid request parameters | Fix parameters |
| `AuthenticationError` | Invalid or expired token | Refresh token |
| `RateLimitError` | Too many requests | Wait for reset |
| `ResourceNotFound` | Job/resource not found | Check ID |
| `ProcessingError` | Processing failed | Check logs, retry |
| `GPUOutOfMemory` | GPU memory exhausted | Reduce batch size |
| `TimeoutError` | Processing timeout | Increase timeout |

### Retry Strategy

```python
import time
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

def create_session_with_retries():
    session = requests.Session()
    
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET", "POST"]
    )
    
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    
    return session

# Usage
session = create_session_with_retries()
response = session.post(
    'https://api.histocore.ai/v1/process/wsi',
    headers={'Authorization': f'Bearer {token}'},
    json=request_data
)
```

## Code Examples

### Python

```python
import requests
import time

class HistoCoreClient:
    def __init__(self, base_url, token):
        self.base_url = base_url
        self.headers = {'Authorization': f'Bearer {token}'}
    
    def process_wsi(self, wsi_path, config=None):
        """Start WSI processing"""
        response = requests.post(
            f'{self.base_url}/process/wsi',
            headers=self.headers,
            json={
                'wsi_path': wsi_path,
                'config': config or {}
            }
        )
        response.raise_for_status()
        return response.json()
    
    def get_status(self, job_id):
        """Get processing status"""
        response = requests.get(
            f'{self.base_url}/process/wsi/{job_id}',
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def wait_for_completion(self, job_id, poll_interval=5):
        """Wait for processing to complete"""
        while True:
            status = self.get_status(job_id)
            
            if status['status'] == 'completed':
                return status['result']
            elif status['status'] == 'failed':
                raise Exception(f"Processing failed: {status.get('error')}")
            
            print(f"Progress: {status['progress']['percent_complete']:.1f}%")
            time.sleep(poll_interval)

# Usage
client = HistoCoreClient('https://api.histocore.ai/v1', 'your_token')

# Start processing
job = client.process_wsi('/data/slides/patient_001.svs')
print(f"Job started: {job['job_id']}")

# Wait for completion
result = client.wait_for_completion(job['job_id'])
print(f"Prediction: {result['prediction']}")
print(f"Processing time: {result['processing_time']}s")
```

### JavaScript

```javascript
class HistoCoreClient {
  constructor(baseUrl, token) {
    this.baseUrl = baseUrl;
    this.headers = {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json'
    };
  }

  async processWSI(wsiPath, config = {}) {
    const response = await fetch(`${this.baseUrl}/process/wsi`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify({
        wsi_path: wsiPath,
        config: config
      })
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${await response.text()}`);
    }
    
    return await response.json();
  }

  async getStatus(jobId) {
    const response = await fetch(
      `${this.baseUrl}/process/wsi/${jobId}`,
      { headers: this.headers }
    );
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${await response.text()}`);
    }
    
    return await response.json();
  }

  async waitForCompletion(jobId, pollInterval = 5000) {
    while (true) {
      const status = await this.getStatus(jobId);
      
      if (status.status === 'completed') {
        return status.result;
      } else if (status.status === 'failed') {
        throw new Error(`Processing failed: ${status.error}`);
      }
      
      console.log(`Progress: ${status.progress.percent_complete.toFixed(1)}%`);
      await new Promise(resolve => setTimeout(resolve, pollInterval));
    }
  }
}

// Usage
const client = new HistoCoreClient('https://api.histocore.ai/v1', 'your_token');

// Start processing
const job = await client.processWSI('/data/slides/patient_001.svs');
console.log(`Job started: ${job.job_id}`);

// Wait for completion
const result = await client.waitForCompletion(job.job_id);
console.log(`Prediction:`, result.prediction);
console.log(`Processing time: ${result.processing_time}s`);
```

### cURL

```bash
#!/bin/bash

# Configuration
BASE_URL="https://api.histocore.ai/v1"
TOKEN="your_token_here"

# Start processing
JOB_RESPONSE=$(curl -s -X POST "$BASE_URL/process/wsi" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "wsi_path": "/data/slides/patient_001.svs",
    "config": {
      "batch_size": 32,
      "confidence_threshold": 0.95
    }
  }')

JOB_ID=$(echo $JOB_RESPONSE | jq -r '.job_id')
echo "Job started: $JOB_ID"

# Poll for completion
while true; do
  STATUS_RESPONSE=$(curl -s -X GET "$BASE_URL/process/wsi/$JOB_ID" \
    -H "Authorization: Bearer $TOKEN")
  
  STATUS=$(echo $STATUS_RESPONSE | jq -r '.status')
  
  if [ "$STATUS" = "completed" ]; then
    echo "Processing completed!"
    echo $STATUS_RESPONSE | jq '.result'
    break
  elif [ "$STATUS" = "failed" ]; then
    echo "Processing failed!"
    echo $STATUS_RESPONSE | jq '.error'
    exit 1
  fi
  
  PROGRESS=$(echo $STATUS_RESPONSE | jq -r '.progress.percent_complete')
  echo "Progress: $PROGRESS%"
  
  sleep 5
done
```

## Performance Optimization

### Batch Processing

Process multiple slides efficiently:

```python
import asyncio
import aiohttp

async def process_slides_batch(client, slide_paths):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for path in slide_paths:
            task = client.process_wsi_async(session, path)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return results

# Process 10 slides concurrently
slide_paths = [f'/data/slides/patient_{i:03d}.svs' for i in range(10)]
results = asyncio.run(process_slides_batch(client, slide_paths))
```

### Configuration Tuning

Optimize for your hardware:

```json
{
  "config": {
    "tile_size": 1024,
    "batch_size": 64,
    "memory_budget_gb": 8.0,
    "target_time": 20.0,
    "confidence_threshold": 0.95
  }
}
```

**Guidelines:**
- **tile_size**: 512-2048 (larger = faster but more memory)
- **batch_size**: 16-128 (adjust based on GPU memory)
- **memory_budget_gb**: 1-16 (match available RAM)
- **target_time**: 10-60 seconds (quality vs speed tradeoff)
- **confidence_threshold**: 0.8-0.99 (higher = more processing)

## Support

### Documentation
- **API Reference**: [openapi.yaml](openapi.yaml)
- **User Guide**: [../USER_GUIDE.md](../USER_GUIDE.md)
- **Deployment**: [../deployment/](../deployment/)

### Contact
- **Email**: support@histocore.ai
- **GitHub**: https://github.com/histocore/histocore/issues
- **Slack**: https://histocore.slack.com

### SLA
- **Uptime**: 99.9% availability
- **Response Time**: <200ms (p95)
- **Processing Time**: <30s for gigapixel slides
- **Support**: 24/7 for Enterprise tier
