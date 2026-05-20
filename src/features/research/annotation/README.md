# Expert Annotation Interface

Web-based annotation tool for pathologists to review and annotate high-uncertainty cases flagged by the active learning system.

## Features

- **Zero-footprint browser interface** - No installation required
- **OpenSeadragon WSI viewer** - Industry-standard gigapixel image viewing
- **Annotation tools** - Polygon, circle, rectangle, freehand drawing
- **AI prediction overlays** - Display model predictions alongside annotations
- **Real-time collaboration** - Multiple pathologists can work simultaneously via WebSocket
- **Annotation queue** - Prioritized list of high-uncertainty cases
- **REST API** - Full CRUD operations for annotations

## Architecture

```
annotation_interface/
├── backend/
│   ├── annotation_api.py       # FastAPI REST endpoints
│   ├── annotation_models.py    # Pydantic data models
│   └── websocket_handler.py    # Real-time collaboration
└── frontend/
    ├── src/
    │   ├── components/          # React components
    │   │   ├── WSIViewer.tsx    # OpenSeadragon viewer
    │   │   ├── AnnotationQueue.tsx
    │   │   ├── AnnotationPanel.tsx
    │   │   └── AIOverlay.tsx
    │   ├── store/
    │   │   └── annotationStore.ts  # Zustand state management
    │   ├── App.tsx
    │   └── main.tsx
    ├── package.json
    └── vite.config.ts
```

## Quick Start

### Backend

```bash
# Install dependencies
pip install fastapi uvicorn websockets

# Run backend server
cd src/annotation_interface/backend
python annotation_api.py

# Server runs on http://localhost:8001
# API docs at http://localhost:8001/docs
```

### Frontend

```bash
# Install dependencies
cd src/annotation_interface/frontend
npm install

# Run development server
npm run dev

# Frontend runs on http://localhost:3000
```

## API Endpoints

### Annotation Queue
- `GET /api/queue` - Get annotation queue
- `POST /api/queue/{task_id}/assign` - Assign task to expert
- `POST /api/queue/{task_id}/complete` - Mark task complete

### Slides
- `GET /api/slides/{slide_id}` - Get slide info
- `GET /api/slides/{slide_id}/tile/{z}/{x}/{y}` - Get tile for OpenSeadragon
- `GET /api/slides/{slide_id}/ai-prediction` - Get AI prediction overlay

### Annotations
- `POST /api/annotations` - Create annotation
- `GET /api/annotations` - List annotations (filterable by slide_id, expert_id)
- `GET /api/annotations/{id}` - Get specific annotation
- `PUT /api/annotations/{id}` - Update annotation
- `DELETE /api/annotations/{id}` - Delete annotation

### WebSocket
- `WS /ws/{slide_id}` - Real-time collaboration for slide

## Integration with Active Learning

The annotation interface integrates with the active learning system:

```python
from src.annotation_interface import annotation_app
from src.annotation_interface.backend.annotation_api import add_task_to_queue
from src.annotation_interface.backend.annotation_models import AnnotationQueueItem

# Add high-uncertainty case to annotation queue
task = AnnotationQueueItem(
    task_id="task_001",
    slide_id="slide_123",
    priority=0.9,
    uncertainty_score=0.87,
    ai_prediction={"diagnosis": "tumor", "confidence": 0.65},
    status="pending",
    created_at=datetime.now()
)

add_task_to_queue(task)
```

## Data Models

### Annotation
```python
{
    "id": "uuid",
    "slide_id": "slide_123",
    "task_id": "task_001",
    "label": "tumor",  # tumor, normal, necrosis, etc.
    "geometry": {
        "type": "polygon",  # polygon, circle, rectangle, freehand
        "points": [{"x": 100, "y": 200}, ...]
    },
    "confidence": 0.95,
    "comments": "Clear tumor margins",
    "expert_id": "expert_001",
    "created_at": "2024-01-01T12:00:00",
    "updated_at": "2024-01-01T12:00:00"
}
```

### Queue Item
```python
{
    "task_id": "task_001",
    "slide_id": "slide_123",
    "priority": 0.9,
    "uncertainty_score": 0.87,
    "ai_prediction": {"diagnosis": "tumor", "confidence": 0.65},
    "status": "pending",  # pending, in_progress, completed
    "created_at": "2024-01-01T12:00:00",
    "assigned_expert": "expert_001"
}
```

## Real-Time Collaboration

Multiple pathologists can annotate the same slide simultaneously:

1. Connect to WebSocket: `ws://localhost:8001/ws/{slide_id}`
2. Receive real-time updates when others create/update/delete annotations
3. See cursor positions of other users
4. View recent annotation actions

## TODO: Production Integration

- [ ] Integrate with actual WSI streaming system for tile serving
- [ ] Connect to foundation model for real AI predictions
- [ ] Implement authentication and authorization
- [ ] Add database persistence (PostgreSQL/MongoDB)
- [ ] Implement annotation quality control
- [ ] Add inter-rater agreement calculations
- [ ] Export annotations for model retraining
- [ ] Add annotation history and versioning
- [ ] Implement undo/redo functionality
- [ ] Add keyboard shortcuts for tools
- [ ] Optimize WebSocket for large-scale deployment

## Testing

```bash
# Backend tests
pytest tests/test_annotation_interface.py

# Frontend tests
cd frontend
npm test
```

## License

Part of HistoCore Medical AI Platform
