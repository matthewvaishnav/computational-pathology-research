# Real-Time WSI Streaming Web Dashboard

Web-based dashboard for monitoring and controlling real-time whole-slide image (WSI) processing with interactive visualizations.

## Features

### Backend (FastAPI)
- **REST API Endpoints**: Status, heatmap, confidence data, and parameter management
- **WebSocket Support**: Real-time bidirectional communication for live updates
- **CORS Configuration**: Ready for clinical system integration
- **Async Processing**: Non-blocking operations for high performance
- **OpenAPI Documentation**: Auto-generated API docs at `/docs`

### Frontend (HTML/JavaScript)
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Real-Time Updates**: WebSocket-based live data streaming
- **Interactive Visualizations**: Plotly.js-powered charts with zoom/pan
- **Parameter Controls**: Adjust processing parameters on-the-fly
- **Progress Tracking**: Visual progress bars and time estimates

### Visualizations
- **Attention Heatmap**: Interactive 2D heatmap showing attention weights across WSI tiles
- **Confidence Progression**: Real-time plot of confidence building over time
- **Processing Statistics**: Throughput, elapsed time, and remaining time estimates
- **Coverage Tracking**: Visual representation of processed vs. unprocessed regions

## Quick Start

### 1. Start the Dashboard Server

```bash
# From project root
python -m uvicorn src.streaming.web_dashboard:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Access the Dashboard

Open your browser and navigate to:
```
http://localhost:8000
```

### 3. API Documentation

Interactive API documentation available at:
```
http://localhost:8000/docs
```

## API Endpoints

### REST Endpoints

#### GET `/api/status`
Get current processing status.

**Response:**
```json
{
  "slide_id": "test_slide",
  "status": "processing",
  "patches_processed": 500,
  "total_patches": 1000,
  "progress_percent": 50.0,
  "current_confidence": 0.85,
  "elapsed_time": 10.5,
  "estimated_remaining": 10.5,
  "throughput": 47.6
}
```

#### GET `/api/heatmap`
Get attention heatmap data.

**Response:**
```json
{
  "slide_id": "test_slide",
  "heatmap": [[0.1, 0.2], [0.3, 0.4]],
  "dimensions": [100, 100],
  "coverage_percent": 75.0,
  "timestamp": 1234567890.0
}
```

#### GET `/api/confidence`
Get confidence progression data.

**Response:**
```json
{
  "slide_id": "test_slide",
  "timestamps": [0.0, 1.0, 2.0, 3.0],
  "confidences": [0.5, 0.7, 0.85, 0.92],
  "target_threshold": 0.95
}
```

#### GET `/api/parameters`
Get current processing parameters.

**Response:**
```json
{
  "confidence_threshold": 0.95,
  "batch_size": 64,
  "tile_size": 1024,
  "update_interval": 1.0,
  "enable_early_stopping": true
}
```

#### POST `/api/parameters`
Update processing parameters.

**Request Body:**
```json
{
  "confidence_threshold": 0.90,
  "batch_size": 128,
  "tile_size": 2048,
  "update_interval": 2.0,
  "enable_early_stopping": false
}
```

#### POST `/api/stop`
Stop current processing.

**Response:**
```json
{
  "status": "success",
  "message": "Processing stopped"
}
```

#### GET `/health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": 1234567890.0,
  "active_connections": 3
}
```

### WebSocket Endpoint

#### WS `/ws`
WebSocket endpoint for real-time updates.

**Client → Server Messages:**
```json
{"type": "ping"}
{"type": "request_status"}
```

**Server → Client Messages:**
```json
{"type": "connected", "message": "WebSocket connected", "current_status": "idle"}
{"type": "status_update", "data": {...}}
{"type": "processing_complete", "slide_id": "...", "final_confidence": 0.96}
{"type": "error", "error": "...", "timestamp": 1234567890.0}
{"type": "parameters_updated", "parameters": {...}}
```

## Integration with Processing Pipeline

### Updating Dashboard from Processing Code

```python
from src.streaming.web_dashboard import (
    update_dashboard_status,
    update_dashboard_error,
    update_dashboard_complete
)
import asyncio

# During processing loop
async def process_wsi():
    for batch_idx, (patches, coords) in enumerate(tile_batches):
        # Process patches
        features = model(patches)
        attention_weights = attention_model(features)
        confidence = get_confidence(attention_weights)
        
        # Update dashboard
        await update_dashboard_status(
            patches_processed=batch_idx * batch_size,
            total_patches=total_patches,
            confidence=confidence,
            attention_weights=attention_weights.cpu().numpy(),
            coordinates=coords
        )
    
    # Mark complete
    await update_dashboard_complete()

# Run with asyncio
asyncio.run(process_wsi())
```

### Error Handling

```python
try:
    # Processing code
    result = process_wsi(wsi_path)
except Exception as e:
    await update_dashboard_error(str(e))
```

## Configuration

### CORS Settings

For production, configure CORS to allow only specific origins:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://clinical-system.hospital.org"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

### Server Configuration

```bash
# Development
uvicorn src.streaming.web_dashboard:app --reload

# Production
uvicorn src.streaming.web_dashboard:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4 \
    --log-level info
```

### Environment Variables

```bash
# Optional configuration
export DASHBOARD_HOST="0.0.0.0"
export DASHBOARD_PORT="8000"
export DASHBOARD_LOG_LEVEL="info"
```

## Testing

Run the test suite:

```bash
# Run all dashboard tests
pytest tests/streaming/test_web_dashboard.py -v

# Run with coverage
pytest tests/streaming/test_web_dashboard.py --cov=src.streaming.web_dashboard --cov-report=html

# Run specific test class
pytest tests/streaming/test_web_dashboard.py::TestRESTEndpoints -v
```

## Architecture

### Components

1. **FastAPI Application**: Main web server handling HTTP and WebSocket connections
2. **DashboardState**: Centralized state management for processing data
3. **ConnectionManager**: WebSocket connection lifecycle management
4. **Pydantic Models**: Type-safe request/response validation
5. **Update Functions**: Async functions for pushing updates from processing pipeline

### Data Flow

```
Processing Pipeline → update_dashboard_status() → DashboardState → ConnectionManager → WebSocket Clients
                                                                  ↓
                                                            REST API Endpoints
```

### State Management

The dashboard maintains state in `DashboardState`:
- Current processing status
- Attention heatmap data
- Confidence history
- Processing parameters
- Error tracking

## Performance Considerations

### Update Frequency

Control update frequency with `update_interval` parameter:
- **High frequency (0.1-0.5s)**: Smooth real-time updates, higher CPU usage
- **Medium frequency (1.0-2.0s)**: Balanced performance (recommended)
- **Low frequency (5.0-10.0s)**: Minimal overhead, less responsive

### Memory Management

- Heatmap data is stored as NumPy arrays for efficiency
- Confidence history is pruned automatically (configurable)
- WebSocket connections are cleaned up on disconnect

### Scalability

- Multiple clients can connect simultaneously
- Broadcast updates to all connected clients efficiently
- Async operations prevent blocking

## Troubleshooting

### WebSocket Connection Issues

**Problem**: WebSocket disconnects frequently

**Solution**: Check firewall settings and proxy configuration. Some proxies don't support WebSocket upgrades.

### CORS Errors

**Problem**: Browser blocks requests due to CORS

**Solution**: Update CORS middleware configuration to allow your origin:
```python
allow_origins=["http://your-frontend-domain.com"]
```

### Slow Updates

**Problem**: Dashboard updates lag behind processing

**Solution**: 
1. Increase `update_interval` to reduce update frequency
2. Check network latency
3. Verify WebSocket connection is active

### Missing Visualizations

**Problem**: Heatmap or confidence plot not displaying

**Solution**:
1. Check browser console for JavaScript errors
2. Verify Plotly.js is loaded (check network tab)
3. Ensure data is available via REST API endpoints

## Requirements

### Python Dependencies
- `fastapi>=0.104.0`
- `uvicorn>=0.24.0`
- `pydantic>=2.0.0`
- `numpy>=1.24.0`
- `python-multipart>=0.0.6`

### Frontend Dependencies
- Plotly.js 2.26.0+ (loaded via CDN)
- Modern browser with WebSocket support

## Security Considerations

### Production Deployment

1. **Use HTTPS**: Always use TLS/SSL in production
2. **Configure CORS**: Restrict origins to trusted domains
3. **Authentication**: Add authentication middleware for protected endpoints
4. **Rate Limiting**: Implement rate limiting to prevent abuse
5. **Input Validation**: All inputs are validated via Pydantic models

### Example Authentication

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer

security = HTTPBearer()

async def verify_token(credentials = Depends(security)):
    # Implement token verification
    if not verify_jwt(credentials.credentials):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    return credentials

@app.get("/api/status", dependencies=[Depends(verify_token)])
async def get_status():
    # Protected endpoint
    pass
```

## Future Enhancements

- [ ] User authentication and authorization
- [ ] Multi-slide processing queue
- [ ] Historical processing logs
- [ ] Export visualizations as images
- [ ] Configurable alert thresholds
- [ ] Mobile app integration
- [ ] Real-time collaboration features

## License

Part of the HistoCore computational pathology research platform.

## Support

For issues or questions, please refer to the main project documentation or open an issue on the project repository.
