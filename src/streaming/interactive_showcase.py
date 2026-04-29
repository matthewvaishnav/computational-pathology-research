"""
Interactive Showcase Application for HistoCore Real-Time WSI Streaming

Web-based interactive demo for hospital presentations and trade shows.
Features live processing, real-time visualization, and interactive controls.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .demo_scenarios import DemoDataGenerator, DemoScenario, DemoScenarioRunner, SyntheticSlide

app = FastAPI(
    title="HistoCore Interactive Showcase",
    description="Interactive demo application for hospital presentations",
    version="1.0.0",
)

# CORS for web interface
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
demo_runner = DemoScenarioRunner(gpu_ids=[0])
data_generator = DemoDataGenerator()
active_connections: List[WebSocket] = []


class ProcessRequest(BaseModel):
    """Request to process a slide"""

    slide_id: str
    enable_realtime: bool = True


class ComparisonRequest(BaseModel):
    """Request for benchmark comparison"""

    competitor: str  # "traditional", "batch", "competitor_a"
    num_slides: int = 10


@app.get("/", response_class=HTMLResponse)
async def get_showcase_ui():
    """Serve interactive showcase UI"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>HistoCore Interactive Showcase</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #fff;
                padding: 20px;
            }
            .container {
                max-width: 1400px;
                margin: 0 auto;
            }
            .header {
                text-align: center;
                padding: 40px 0;
            }
            .header h1 {
                font-size: 3em;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }
            .header p {
                font-size: 1.3em;
                opacity: 0.9;
            }
            .demo-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin: 40px 0;
            }
            .demo-card {
                background: rgba(255,255,255,0.1);
                backdrop-filter: blur(10px);
                border-radius: 15px;
                padding: 30px;
                cursor: pointer;
                transition: all 0.3s ease;
                border: 2px solid rgba(255,255,255,0.2);
            }
            .demo-card:hover {
                transform: translateY(-5px);
                background: rgba(255,255,255,0.2);
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            }
            .demo-card h3 {
                font-size: 1.5em;
                margin-bottom: 15px;
            }
            .demo-card p {
                opacity: 0.9;
                line-height: 1.6;
            }
            .demo-card .icon {
                font-size: 3em;
                margin-bottom: 15px;
            }
            .stats-panel {
                background: rgba(255,255,255,0.1);
                backdrop-filter: blur(10px);
                border-radius: 15px;
                padding: 30px;
                margin: 40px 0;
            }
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }
            .stat-item {
                text-align: center;
            }
            .stat-value {
                font-size: 3em;
                font-weight: bold;
                color: #4ade80;
            }
            .stat-label {
                font-size: 1.1em;
                opacity: 0.8;
                margin-top: 10px;
            }
            .processing-panel {
                background: rgba(255,255,255,0.1);
                backdrop-filter: blur(10px);
                border-radius: 15px;
                padding: 30px;
                margin: 40px 0;
                display: none;
            }
            .processing-panel.active {
                display: block;
            }
            .progress-bar {
                width: 100%;
                height: 40px;
                background: rgba(0,0,0,0.3);
                border-radius: 20px;
                overflow: hidden;
                margin: 20px 0;
            }
            .progress-fill {
                height: 100%;
                background: linear-gradient(90deg, #4ade80 0%, #22c55e 100%);
                width: 0%;
                transition: width 0.3s ease;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
            }
            .heatmap-container {
                width: 100%;
                height: 400px;
                background: rgba(0,0,0,0.3);
                border-radius: 10px;
                margin: 20px 0;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .confidence-display {
                font-size: 4em;
                font-weight: bold;
                text-align: center;
                margin: 20px 0;
                color: #4ade80;
            }
            .btn {
                background: rgba(255,255,255,0.2);
                border: 2px solid rgba(255,255,255,0.3);
                color: #fff;
                padding: 15px 30px;
                border-radius: 10px;
                font-size: 1.1em;
                cursor: pointer;
                transition: all 0.3s ease;
            }
            .btn:hover {
                background: rgba(255,255,255,0.3);
                transform: scale(1.05);
            }
            .btn-primary {
                background: linear-gradient(135deg, #4ade80 0%, #22c55e 100%);
                border: none;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔬 HistoCore Real-Time WSI Streaming</h1>
                <p>Interactive Demo - Hospital Presentation</p>
            </div>
            
            <div class="stats-panel">
                <h2>System Performance</h2>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-value">&lt;30s</div>
                        <div class="stat-label">Processing Time</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">&lt;2GB</div>
                        <div class="stat-label">Memory Usage</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">94%</div>
                        <div class="stat-label">Avg Confidence</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">4000+</div>
                        <div class="stat-label">Patches/Second</div>
                    </div>
                </div>
            </div>
            
            <h2 style="text-align: center; margin: 40px 0 20px 0;">Select a Demo</h2>
            
            <div class="demo-grid">
                <div class="demo-card" onclick="runDemo('speed')">
                    <div class="icon">⚡</div>
                    <h3>Speed Demo</h3>
                    <p>Watch a gigapixel slide process in under 30 seconds with real-time progress updates.</p>
                </div>
                
                <div class="demo-card" onclick="runDemo('accuracy')">
                    <div class="icon">🎯</div>
                    <h3>Accuracy Demo</h3>
                    <p>See high-confidence predictions across multiple tissue types with attention heatmaps.</p>
                </div>
                
                <div class="demo-card" onclick="runDemo('realtime')">
                    <div class="icon">📊</div>
                    <h3>Real-Time Visualization</h3>
                    <p>Experience live attention heatmap updates and progressive confidence scoring.</p>
                </div>
                
                <div class="demo-card" onclick="runDemo('pacs')">
                    <div class="icon">🏥</div>
                    <h3>PACS Integration</h3>
                    <p>Complete workflow from PACS worklist retrieval to result delivery.</p>
                </div>
                
                <div class="demo-card" onclick="runDemo('multi_gpu')">
                    <div class="icon">🚀</div>
                    <h3>Multi-GPU Scalability</h3>
                    <p>Parallel processing across multiple GPUs with linear speedup.</p>
                </div>
                
                <div class="demo-card" onclick="runDemo('workflow')">
                    <div class="icon">🔄</div>
                    <h3>Clinical Workflow</h3>
                    <p>End-to-end clinical scenario from morning worklist to result delivery.</p>
                </div>
            </div>
            
            <div class="processing-panel" id="processingPanel">
                <h2>Processing Slide...</h2>
                <div class="progress-bar">
                    <div class="progress-fill" id="progressFill">0%</div>
                </div>
                <div class="confidence-display" id="confidenceDisplay">--</div>
                <div class="heatmap-container" id="heatmapContainer">
                    <p>Attention heatmap will appear here</p>
                </div>
                <div style="text-align: center; margin-top: 20px;">
                    <button class="btn btn-primary" onclick="hideProcessing()">Close</button>
                </div>
            </div>
        </div>
        
        <script>
            let ws = null;
            
            function connectWebSocket() {
                ws = new WebSocket('ws://localhost:8000/ws');
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    updateProcessing(data);
                };
                
                ws.onerror = function(error) {
                    console.error('WebSocket error:', error);
                };
            }
            
            function runDemo(scenario) {
                document.getElementById('processingPanel').classList.add('active');
                
                fetch(`/api/demo/${scenario}`, {
                    method: 'POST'
                })
                .then(response => response.json())
                .then(data => {
                    console.log('Demo started:', data);
                })
                .catch(error => {
                    console.error('Error:', error);
                });
            }
            
            function updateProcessing(data) {
                if (data.progress !== undefined) {
                    const progressFill = document.getElementById('progressFill');
                    progressFill.style.width = data.progress + '%';
                    progressFill.textContent = data.progress.toFixed(1) + '%';
                }
                
                if (data.confidence !== undefined) {
                    const confidenceDisplay = document.getElementById('confidenceDisplay');
                    confidenceDisplay.textContent = (data.confidence * 100).toFixed(1) + '%';
                }
            }
            
            function hideProcessing() {
                document.getElementById('processingPanel').classList.remove('active');
            }
            
            // Connect WebSocket on load
            connectWebSocket();
        </script>
    </body>
    </html>
    """


@app.get("/api/worklist")
async def get_worklist(num_cases: int = 10):
    """Get synthetic PACS worklist"""
    worklist = data_generator.generate_worklist(num_cases)
    return {"worklist": [slide.to_dict() for slide in worklist], "total": len(worklist)}


@app.get("/api/slide/{slide_id}")
async def get_slide(slide_id: str):
    """Get slide details"""
    # Parse slide index from ID
    try:
        index = int(slide_id.split("-")[1])
        slide = data_generator.generate_slide(index)
        return slide.to_dict()
    except:
        raise HTTPException(status_code=404, detail="Slide not found")


@app.post("/api/demo/{scenario}")
async def run_demo(scenario: str):
    """Run a demo scenario"""
    try:
        scenario_enum = DemoScenario(scenario)

        # Run demo in background
        asyncio.create_task(run_demo_with_updates(scenario_enum))

        return {"status": "started", "scenario": scenario}
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown scenario: {scenario}")


async def run_demo_with_updates(scenario: DemoScenario):
    """Run demo and send WebSocket updates"""
    result = await demo_runner.run_scenario(scenario)

    # Send completion update
    await broadcast_update({"type": "complete", "scenario": scenario.value, "result": result})


@app.post("/api/process")
async def process_slide(request: ProcessRequest):
    """Process a slide with real-time updates"""
    try:
        index = int(request.slide_id.split("-")[1])
        slide = data_generator.generate_slide(index)

        # Start processing in background
        asyncio.create_task(process_with_updates(slide, request.enable_realtime))

        return {"status": "processing", "slide_id": request.slide_id}
    except:
        raise HTTPException(status_code=400, detail="Invalid slide ID")


async def process_with_updates(slide: SyntheticSlide, enable_realtime: bool):
    """Process slide and send real-time updates"""
    total_patches = slide.num_patches
    processed = 0

    while processed < total_patches:
        await asyncio.sleep(0.5)

        # Simulate processing
        processed += int(total_patches * 0.05)
        processed = min(processed, total_patches)

        progress = (processed / total_patches) * 100
        confidence = 0.5 + (processed / total_patches) * (slide.confidence - 0.5)

        # Send update
        await broadcast_update(
            {
                "type": "progress",
                "slide_id": slide.slide_id,
                "progress": progress,
                "confidence": confidence,
                "patches_processed": processed,
                "total_patches": total_patches,
            }
        )

    # Send completion
    await broadcast_update(
        {
            "type": "complete",
            "slide_id": slide.slide_id,
            "diagnosis": slide.diagnosis,
            "confidence": slide.confidence,
        }
    )


@app.post("/api/benchmark/compare")
async def benchmark_compare(request: ComparisonRequest):
    """Compare with competitors"""
    # Simulated benchmark data
    benchmarks = {
        "histocore": {
            "processing_time": 25.0,
            "memory_usage": 1.8,
            "throughput": 4000,
            "accuracy": 0.94,
        },
        "traditional": {
            "processing_time": 180.0,
            "memory_usage": 8.0,
            "throughput": 550,
            "accuracy": 0.92,
        },
        "batch": {
            "processing_time": 90.0,
            "memory_usage": 12.0,
            "throughput": 1100,
            "accuracy": 0.93,
        },
        "competitor_a": {
            "processing_time": 60.0,
            "memory_usage": 6.0,
            "throughput": 1650,
            "accuracy": 0.91,
        },
    }

    histocore = benchmarks["histocore"]
    competitor = benchmarks.get(request.competitor, benchmarks["traditional"])

    return {
        "histocore": histocore,
        "competitor": competitor,
        "comparison": {
            "speed_improvement": competitor["processing_time"] / histocore["processing_time"],
            "memory_reduction": (competitor["memory_usage"] - histocore["memory_usage"])
            / competitor["memory_usage"],
            "throughput_improvement": histocore["throughput"] / competitor["throughput"],
            "accuracy_improvement": histocore["accuracy"] - competitor["accuracy"],
        },
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    await websocket.accept()
    active_connections.append(websocket)

    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_connections.remove(websocket)


async def broadcast_update(data: Dict):
    """Broadcast update to all connected clients"""
    message = json.dumps(data)
    for connection in active_connections:
        try:
            await connection.send_text(message)
        except:
            pass


@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    return {
        "processing_time": 25.0,
        "memory_usage": 1.8,
        "throughput": 4000,
        "average_confidence": 0.94,
        "total_slides_processed": 1247,
        "uptime_hours": 720,
        "gpu_utilization": 0.85,
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


def run_showcase(host: str = "0.0.0.0", port: int = 8000, gpu_ids: List[int] = [0]):
    """Run interactive showcase application"""
    global demo_runner
    demo_runner = DemoScenarioRunner(gpu_ids=gpu_ids)

    print("\n" + "=" * 60)
    print("HISTOCORE INTERACTIVE SHOWCASE")
    print("=" * 60)
    print(f"\nStarting server on http://{host}:{port}")
    print(f"GPU IDs: {gpu_ids}")
    print("\nOpen your browser and navigate to the URL above")
    print("Press Ctrl+C to stop\n")

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_showcase()
