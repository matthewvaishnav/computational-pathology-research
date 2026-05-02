"""Web-based dashboard for Real-Time WSI Streaming.

FastAPI backend providing REST endpoints and WebSocket connections for
real-time visualization of WSI processing progress.
"""

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import BackgroundTasks, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ========== Pydantic Models for API ==========


class ProcessingStatus(BaseModel):
    """Current processing status."""

    slide_id: str
    status: str = Field(..., description="Status: idle, processing, completed, error")
    patches_processed: int = Field(ge=0)
    total_patches: int = Field(ge=0)
    progress_percent: float = Field(ge=0.0, le=100.0)
    current_confidence: float = Field(ge=0.0, le=1.0)
    elapsed_time: float = Field(ge=0.0, description="Elapsed time in seconds")
    estimated_remaining: float = Field(ge=0.0, description="Estimated remaining time in seconds")
    throughput: float = Field(ge=0.0, description="Patches per second")


class HeatmapData(BaseModel):
    """Attention heatmap data."""

    slide_id: str
    heatmap: List[List[float]] = Field(..., description="2D heatmap array")
    dimensions: tuple[int, int] = Field(..., description="(width, height) of heatmap")
    coverage_percent: float = Field(ge=0.0, le=100.0)
    timestamp: float


class ConfidenceData(BaseModel):
    """Confidence progression data."""

    slide_id: str
    timestamps: List[float] = Field(..., description="Time points in seconds")
    confidences: List[float] = Field(..., description="Confidence values")
    target_threshold: float = Field(default=0.95)


class ProcessingParameters(BaseModel):
    """Configurable processing parameters."""

    confidence_threshold: float = Field(default=0.95, ge=0.5, le=1.0)
    batch_size: int = Field(default=64, ge=1, le=512)
    tile_size: int = Field(default=1024, ge=256, le=4096)
    update_interval: float = Field(default=1.0, ge=0.1, le=10.0)
    enable_early_stopping: bool = Field(default=True)


class ProcessingRequest(BaseModel):
    """Request to start WSI processing."""

    wsi_path: str = Field(..., description="Path to WSI file")
    parameters: Optional[ProcessingParameters] = None


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    detail: Optional[str] = None
    timestamp: float


# ========== Dashboard State Manager ==========


@dataclass
class DashboardState:
    """Manages state for web dashboard."""

    slide_id: str = ""
    status: str = "idle"
    patches_processed: int = 0
    total_patches: int = 0
    current_confidence: float = 0.0
    start_time: float = 0.0

    # Visualization data
    attention_heatmap: Optional[np.ndarray] = None
    heatmap_dimensions: tuple[int, int] = (0, 0)
    coverage_mask: Optional[np.ndarray] = None
    confidence_history: List[tuple[float, float]] = None  # (timestamp, confidence)

    # Processing parameters
    parameters: ProcessingParameters = None

    # Error tracking
    last_error: Optional[str] = None

    def __post_init__(self):
        if self.confidence_history is None:
            self.confidence_history = []
        if self.parameters is None:
            self.parameters = ProcessingParameters()

    def reset(self):
        """Reset state for new processing."""
        self.slide_id = ""
        self.status = "idle"
        self.patches_processed = 0
        self.total_patches = 0
        self.current_confidence = 0.0
        self.start_time = 0.0
        self.attention_heatmap = None
        self.coverage_mask = None
        self.confidence_history = []
        self.last_error = None

    def get_progress_percent(self) -> float:
        """Calculate progress percentage."""
        if self.total_patches == 0:
            return 0.0
        return (self.patches_processed / self.total_patches) * 100.0

    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time == 0.0:
            return 0.0
        return time.time() - self.start_time

    def get_estimated_remaining(self) -> float:
        """Estimate remaining time in seconds."""
        if self.patches_processed == 0 or self.total_patches == 0:
            return 0.0

        elapsed = self.get_elapsed_time()
        progress = self.patches_processed / self.total_patches

        if progress == 0:
            return 0.0

        total_estimated = elapsed / progress
        return max(0.0, total_estimated - elapsed)

    def get_throughput(self) -> float:
        """Calculate current throughput (patches/sec)."""
        elapsed = self.get_elapsed_time()
        if elapsed == 0:
            return 0.0
        return self.patches_processed / elapsed


# ========== WebSocket Connection Manager ==========


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        disconnected = []

        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.append(connection)

        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

    async def send_personal(self, message: dict, websocket: WebSocket):
        """Send message to specific client."""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending to client: {e}")
            self.disconnect(websocket)


# ========== FastAPI Application ==========

# Global state and connection manager
dashboard_state = DashboardState()
connection_manager = ConnectionManager()

# Create FastAPI app
app = FastAPI(
    title="Real-Time WSI Streaming Dashboard",
    description="Web-based dashboard for monitoring real-time WSI processing",
    version="1.0.0",
)

# Configure CORS for clinical system integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== REST API Endpoints ==========


@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Serve main dashboard HTML page."""
    html_path = Path(__file__).parent / "static" / "dashboard.html"

    if html_path.exists():
        return FileResponse(html_path)
    else:
        # Return inline HTML if static file doesn't exist
        return HTMLResponse(content=get_inline_dashboard_html())


@app.get("/api/status", response_model=ProcessingStatus)
async def get_status():
    """Get current processing status.

    Returns:
        ProcessingStatus: Current status including progress, confidence, and timing
    """
    return ProcessingStatus(
        slide_id=dashboard_state.slide_id or "none",
        status=dashboard_state.status,
        patches_processed=dashboard_state.patches_processed,
        total_patches=dashboard_state.total_patches,
        progress_percent=dashboard_state.get_progress_percent(),
        current_confidence=dashboard_state.current_confidence,
        elapsed_time=dashboard_state.get_elapsed_time(),
        estimated_remaining=dashboard_state.get_estimated_remaining(),
        throughput=dashboard_state.get_throughput(),
    )


@app.get("/api/heatmap", response_model=HeatmapData)
async def get_heatmap():
    """Get current attention heatmap data.

    Returns:
        HeatmapData: Attention heatmap with dimensions and coverage

    Raises:
        HTTPException: If no heatmap data available
    """
    if dashboard_state.attention_heatmap is None:
        raise HTTPException(status_code=404, detail="No heatmap data available")

    # Normalize heatmap
    heatmap = dashboard_state.attention_heatmap.copy()
    if dashboard_state.coverage_mask is not None:
        # Only normalize covered areas
        covered = dashboard_state.coverage_mask
        if covered.any():
            max_val = heatmap[covered].max()
            if max_val > 0:
                heatmap[covered] = heatmap[covered] / max_val

    # Convert to list for JSON serialization
    heatmap_list = heatmap.tolist()

    # Calculate coverage
    coverage_percent = 0.0
    if dashboard_state.coverage_mask is not None:
        coverage_percent = (
            dashboard_state.coverage_mask.sum() / dashboard_state.coverage_mask.size
        ) * 100.0

    return HeatmapData(
        slide_id=dashboard_state.slide_id or "none",
        heatmap=heatmap_list,
        dimensions=dashboard_state.heatmap_dimensions,
        coverage_percent=coverage_percent,
        timestamp=time.time(),
    )


@app.get("/api/confidence", response_model=ConfidenceData)
async def get_confidence():
    """Get confidence progression data.

    Returns:
        ConfidenceData: Time series of confidence values

    Raises:
        HTTPException: If no confidence data available
    """
    if not dashboard_state.confidence_history:
        raise HTTPException(status_code=404, detail="No confidence data available")

    # Extract timestamps and confidences
    timestamps = [t for t, _ in dashboard_state.confidence_history]
    confidences = [c for _, c in dashboard_state.confidence_history]

    # Normalize timestamps to start at 0
    if timestamps:
        start_time = timestamps[0]
        timestamps = [t - start_time for t in timestamps]

    return ConfidenceData(
        slide_id=dashboard_state.slide_id or "none",
        timestamps=timestamps,
        confidences=confidences,
        target_threshold=dashboard_state.parameters.confidence_threshold,
    )


@app.post("/api/parameters")
async def update_parameters(params: ProcessingParameters):
    """Update processing parameters.

    Args:
        params: New processing parameters

    Returns:
        dict: Confirmation with updated parameters
    """
    dashboard_state.parameters = params

    # Broadcast parameter update to connected clients
    await connection_manager.broadcast(
        {"type": "parameters_updated", "parameters": params.model_dump()}
    )

    logger.info(f"Parameters updated: {params}")

    return {"status": "success", "message": "Parameters updated", "parameters": params.model_dump()}


@app.get("/api/parameters", response_model=ProcessingParameters)
async def get_parameters():
    """Get current processing parameters.

    Returns:
        ProcessingParameters: Current parameters
    """
    return dashboard_state.parameters


@app.post("/api/process")
async def start_processing(request: ProcessingRequest, background_tasks: BackgroundTasks):
    """Start WSI processing (async mode).

    Args:
        request: Processing request with WSI path and parameters
        background_tasks: FastAPI background tasks

    Returns:
        dict: Processing started confirmation

    Raises:
        HTTPException: If already processing or invalid request
    """
    if dashboard_state.status == "processing":
        raise HTTPException(status_code=400, detail="Processing already in progress")

    # Validate WSI path
    wsi_path = Path(request.wsi_path)
    if not wsi_path.exists():
        raise HTTPException(status_code=404, detail=f"WSI file not found: {request.wsi_path}")

    # Update parameters if provided
    if request.parameters:
        dashboard_state.parameters = request.parameters

    # Reset state for new processing
    dashboard_state.reset()
    dashboard_state.slide_id = wsi_path.stem
    dashboard_state.status = "processing"
    dashboard_state.start_time = time.time()

    # Start processing in background
    # Note: Actual processing integration would happen here
    # For now, this is a placeholder that would call the streaming processor

    logger.info(f"Started processing: {request.wsi_path}")

    return {
        "status": "success",
        "message": "Processing started",
        "slide_id": dashboard_state.slide_id,
    }


@app.post("/api/stop")
async def stop_processing():
    """Stop current processing.

    Returns:
        dict: Processing stopped confirmation
    """
    if dashboard_state.status != "processing":
        raise HTTPException(status_code=400, detail="No processing in progress")

    dashboard_state.status = "idle"

    # Broadcast stop event
    await connection_manager.broadcast(
        {"type": "processing_stopped", "slide_id": dashboard_state.slide_id}
    )

    logger.info("Processing stopped")

    return {"status": "success", "message": "Processing stopped"}


# ========== WebSocket Endpoint ==========


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates.

    Clients connect to this endpoint to receive real-time processing updates.

    Message types sent to clients:
    - status_update: Processing status changes
    - heatmap_update: New heatmap data
    - confidence_update: New confidence data
    - parameters_updated: Parameter changes
    - processing_stopped: Processing stopped
    - error: Error occurred
    """
    await connection_manager.connect(websocket)

    try:
        # Send initial state
        await websocket.send_json(
            {
                "type": "connected",
                "message": "WebSocket connected",
                "current_status": dashboard_state.status,
            }
        )

        # Keep connection alive and handle incoming messages
        timeout = time.time() + 3600

        while time.time() < timeout:
            # Receive messages from client (e.g., parameter updates, commands)
            data = await websocket.receive_json()

            # Handle client messages
            message_type = data.get("type")

            if message_type == "ping":
                await websocket.send_json({"type": "pong"})

            elif message_type == "request_status":
                status = await get_status()
                await websocket.send_json({"type": "status_update", "data": status.dict()})

            else:
                logger.warning(f"Unknown message type: {message_type}")

    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
        logger.info("Client disconnected")

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        connection_manager.disconnect(websocket)


# ========== Update Functions (called by processing pipeline) ==========


async def update_dashboard_status(
    patches_processed: int,
    total_patches: int,
    confidence: float,
    attention_weights: Optional[np.ndarray] = None,
    coordinates: Optional[np.ndarray] = None,
):
    """Update dashboard with new processing data.

    This function should be called by the streaming processor to push updates.

    Args:
        patches_processed: Number of patches processed so far
        total_patches: Total number of patches
        confidence: Current prediction confidence
        attention_weights: Optional attention weights for heatmap
        coordinates: Optional tile coordinates for heatmap
    """
    # Update state
    dashboard_state.patches_processed = patches_processed
    dashboard_state.total_patches = total_patches
    dashboard_state.current_confidence = confidence

    # Update confidence history
    timestamp = time.time()
    dashboard_state.confidence_history.append((timestamp, confidence))

    # Update heatmap if data provided
    if attention_weights is not None and coordinates is not None:
        if dashboard_state.attention_heatmap is None:
            # Initialize heatmap (dimensions would come from processor)
            # This is a placeholder - actual dimensions from WSI metadata
            dashboard_state.heatmap_dimensions = (100, 100)
            dashboard_state.attention_heatmap = np.zeros(
                dashboard_state.heatmap_dimensions, dtype=np.float32
            )
            dashboard_state.coverage_mask = np.zeros(dashboard_state.heatmap_dimensions, dtype=bool)

        # Update heatmap with new attention weights
        for weight, (x, y) in zip(attention_weights, coordinates):
            if (
                0 <= x < dashboard_state.heatmap_dimensions[0]
                and 0 <= y < dashboard_state.heatmap_dimensions[1]
            ):
                dashboard_state.attention_heatmap[y, x] += weight
                dashboard_state.coverage_mask[y, x] = True

    # Broadcast update to connected clients
    await connection_manager.broadcast(
        {
            "type": "status_update",
            "data": {
                "patches_processed": patches_processed,
                "total_patches": total_patches,
                "progress_percent": dashboard_state.get_progress_percent(),
                "confidence": confidence,
                "elapsed_time": dashboard_state.get_elapsed_time(),
                "estimated_remaining": dashboard_state.get_estimated_remaining(),
                "throughput": dashboard_state.get_throughput(),
            },
        }
    )


async def update_dashboard_error(error_message: str):
    """Update dashboard with error.

    Args:
        error_message: Error message to display
    """
    dashboard_state.status = "error"
    dashboard_state.last_error = error_message

    await connection_manager.broadcast(
        {"type": "error", "error": error_message, "timestamp": time.time()}
    )


async def update_dashboard_complete():
    """Mark processing as complete."""
    dashboard_state.status = "completed"

    await connection_manager.broadcast(
        {
            "type": "processing_complete",
            "slide_id": dashboard_state.slide_id,
            "final_confidence": dashboard_state.current_confidence,
            "total_time": dashboard_state.get_elapsed_time(),
        }
    )


# ========== Inline HTML Dashboard ==========


def get_inline_dashboard_html() -> str:
    """Get inline HTML for dashboard (fallback if static file not found)."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Real-Time WSI Streaming Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: #f5f5f5;
                padding: 20px;
            }
            .container {
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                padding: 20px;
            }
            h1 {
                color: #333;
                margin-bottom: 20px;
                border-bottom: 3px solid #4CAF50;
                padding-bottom: 10px;
            }
            .status-bar {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-bottom: 20px;
            }
            .status-card {
                background: #f9f9f9;
                padding: 15px;
                border-radius: 6px;
                border-left: 4px solid #4CAF50;
            }
            .status-card h3 {
                font-size: 14px;
                color: #666;
                margin-bottom: 5px;
            }
            .status-card .value {
                font-size: 24px;
                font-weight: bold;
                color: #333;
            }
            .visualization-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin-top: 20px;
            }
            .viz-panel {
                background: #fafafa;
                padding: 15px;
                border-radius: 6px;
                border: 1px solid #ddd;
            }
            .viz-panel h2 {
                font-size: 18px;
                margin-bottom: 10px;
                color: #333;
            }
            #heatmap, #confidence-plot {
                width: 100%;
                height: 400px;
            }
            .connection-status {
                position: fixed;
                top: 20px;
                right: 20px;
                padding: 10px 15px;
                border-radius: 4px;
                font-size: 14px;
                font-weight: bold;
            }
            .connected {
                background: #4CAF50;
                color: white;
            }
            .disconnected {
                background: #f44336;
                color: white;
            }
        </style>
    </head>
    <body>
        <div class="connection-status disconnected" id="connection-status">Disconnected</div>
        
        <div class="container">
            <h1>🔬 Real-Time WSI Streaming Dashboard</h1>
            
            <div class="status-bar">
                <div class="status-card">
                    <h3>Status</h3>
                    <div class="value" id="status">Idle</div>
                </div>
                <div class="status-card">
                    <h3>Progress</h3>
                    <div class="value" id="progress">0%</div>
                </div>
                <div class="status-card">
                    <h3>Confidence</h3>
                    <div class="value" id="confidence">0.000</div>
                </div>
                <div class="status-card">
                    <h3>Throughput</h3>
                    <div class="value" id="throughput">0 p/s</div>
                </div>
                <div class="status-card">
                    <h3>Elapsed Time</h3>
                    <div class="value" id="elapsed">0.0s</div>
                </div>
                <div class="status-card">
                    <h3>Est. Remaining</h3>
                    <div class="value" id="remaining">0.0s</div>
                </div>
            </div>
            
            <div class="visualization-grid">
                <div class="viz-panel">
                    <h2>Attention Heatmap</h2>
                    <div id="heatmap"></div>
                </div>
                <div class="viz-panel">
                    <h2>Confidence Progression</h2>
                    <div id="confidence-plot"></div>
                </div>
            </div>
        </div>
        
        <script>
            // WebSocket connection
            let ws = null;
            let reconnectInterval = null;
            
            function connectWebSocket() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/ws`;
                
                ws = new WebSocket(wsUrl);
                
                ws.onopen = () => {
                    console.log('WebSocket connected');
                    document.getElementById('connection-status').textContent = 'Connected';
                    document.getElementById('connection-status').className = 'connection-status connected';
                    
                    if (reconnectInterval) {
                        clearInterval(reconnectInterval);
                        reconnectInterval = null;
                    }
                };
                
                ws.onmessage = (event) => {
                    const message = JSON.parse(event.data);
                    handleWebSocketMessage(message);
                };
                
                ws.onclose = () => {
                    console.log('WebSocket disconnected');
                    document.getElementById('connection-status').textContent = 'Disconnected';
                    document.getElementById('connection-status').className = 'connection-status disconnected';
                    
                    // Attempt reconnection
                    if (!reconnectInterval) {
                        reconnectInterval = setInterval(() => {
                            console.log('Attempting to reconnect...');
                            connectWebSocket();
                        }, 3000);
                    }
                };
                
                ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                };
            }
            
            function handleWebSocketMessage(message) {
                console.log('Received:', message);
                
                if (message.type === 'status_update') {
                    updateStatus(message.data);
                } else if (message.type === 'processing_complete') {
                    document.getElementById('status').textContent = 'Completed';
                } else if (message.type === 'error') {
                    document.getElementById('status').textContent = 'Error';
                    console.error('Processing error:', message.error);
                }
            }
            
            function updateStatus(data) {
                document.getElementById('status').textContent = 'Processing';
                document.getElementById('progress').textContent = data.progress_percent.toFixed(1) + '%';
                document.getElementById('confidence').textContent = data.confidence.toFixed(3);
                document.getElementById('throughput').textContent = data.throughput.toFixed(1) + ' p/s';
                document.getElementById('elapsed').textContent = data.elapsed_time.toFixed(1) + 's';
                document.getElementById('remaining').textContent = data.estimated_remaining.toFixed(1) + 's';
            }
            
            // Polling for REST API updates (fallback)
            async function pollStatus() {
                try {
                    const response = await fetch('/api/status');
                    const data = await response.json();
                    updateStatus(data);
                    
                    // Update visualizations
                    await updateHeatmap();
                    await updateConfidencePlot();
                } catch (error) {
                    console.error('Error polling status:', error);
                }
            }
            
            async function updateHeatmap() {
                try {
                    const response = await fetch('/api/heatmap');
                    if (response.ok) {
                        const data = await response.json();
                        
                        const plotData = [{
                            z: data.heatmap,
                            type: 'heatmap',
                            colorscale: 'Jet',
                            showscale: true
                        }];
                        
                        const layout = {
                            title: `Coverage: ${data.coverage_percent.toFixed(1)}%`,
                            xaxis: { title: 'Tile X' },
                            yaxis: { title: 'Tile Y' },
                            margin: { t: 40, b: 40, l: 50, r: 50 }
                        };
                        
                        Plotly.react('heatmap', plotData, layout);
                    }
                } catch (error) {
                    console.error('Error updating heatmap:', error);
                }
            }
            
            async function updateConfidencePlot() {
                try {
                    const response = await fetch('/api/confidence');
                    if (response.ok) {
                        const data = await response.json();
                        
                        const plotData = [
                            {
                                x: data.timestamps,
                                y: data.confidences,
                                type: 'scatter',
                                mode: 'lines+markers',
                                name: 'Confidence',
                                line: { color: 'blue', width: 2 }
                            },
                            {
                                x: [data.timestamps[0], data.timestamps[data.timestamps.length - 1]],
                                y: [data.target_threshold, data.target_threshold],
                                type: 'scatter',
                                mode: 'lines',
                                name: 'Target',
                                line: { color: 'red', dash: 'dash' }
                            }
                        ];
                        
                        const layout = {
                            xaxis: { title: 'Time (seconds)' },
                            yaxis: { title: 'Confidence', range: [0, 1] },
                            margin: { t: 20, b: 40, l: 50, r: 50 }
                        };
                        
                        Plotly.react('confidence-plot', plotData, layout);
                    }
                } catch (error) {
                    console.error('Error updating confidence plot:', error);
                }
            }
            
            // Initialize
            connectWebSocket();
            
            // Poll every 2 seconds as fallback
            setInterval(pollStatus, 2000);
            
            // Initial load
            pollStatus();
        </script>
    </body>
    </html>
    """


# ========== Health Check ==========


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "active_connections": len(connection_manager.active_connections),
    }


if __name__ == "__main__":
    import uvicorn

    # Run server
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
