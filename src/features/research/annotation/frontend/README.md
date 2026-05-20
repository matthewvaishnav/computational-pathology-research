# Annotation Interface Frontend

React + TypeScript frontend for the expert annotation interface.

## Features

- **OpenSeadragon WSI Viewer** - Gigapixel whole-slide image viewing
- **Annotation Tools** - Polygon, circle, rectangle, freehand drawing
- **Real-time Collaboration** - WebSocket-based multi-user support
- **Annotation Queue** - Prioritized list of high-uncertainty cases
- **AI Prediction Overlay** - Display model predictions alongside annotations
- **State Management** - Zustand for efficient state handling

## Tech Stack

- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Fast build tool
- **OpenSeadragon** - WSI viewer
- **Fabric.js** - Canvas drawing (planned)
- **Zustand** - State management
- **Axios** - HTTP client

## Quick Start

### Prerequisites

- Node.js 18+ and npm
- Backend server running on http://localhost:8001

### Installation

```bash
# Install dependencies
npm install
```

### Development

```bash
# Start development server
npm run dev

# Frontend will be available at http://localhost:3000
```

### Build for Production

```bash
# Build optimized production bundle
npm run build

# Preview production build
npm run preview
```

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── WSIViewer.tsx          # OpenSeadragon viewer component
│   │   ├── AnnotationQueue.tsx    # Queue of cases to annotate
│   │   ├── AnnotationPanel.tsx    # List of annotations
│   │   └── AIOverlay.tsx          # AI prediction display
│   ├── store/
│   │   └── annotationStore.ts     # Zustand state management
│   ├── App.tsx                    # Main application component
│   ├── main.tsx                   # Application entry point
│   └── index.css                  # Global styles
├── index.html                     # HTML template
├── package.json                   # Dependencies
├── tsconfig.json                  # TypeScript configuration
└── vite.config.ts                 # Vite configuration
```

## Components

### WSIViewer

Main viewer component using OpenSeadragon for gigapixel image viewing.

**Features:**
- Pan and zoom navigation
- Annotation drawing tools
- Tool selection (polygon, circle, rectangle, freehand)
- Save annotations with labels

**Props:**
- `slideId: string` - Slide identifier
- `expertId: string` - Expert/pathologist identifier

### AnnotationQueue

Displays prioritized list of cases requiring annotation.

**Features:**
- Priority-based sorting
- Uncertainty score display
- AI prediction preview
- Click to load slide

**Props:**
- `expertId: string` - Expert identifier
- `onSelectSlide: (slideId: string) => void` - Callback when slide selected

### AnnotationPanel

Shows list of annotations for current slide.

**Features:**
- List all annotations
- Select annotation to view details
- Delete annotations
- Display annotation metadata

**Props:**
- `slideId: string` - Current slide identifier
- `expertId: string` - Expert identifier

### AIOverlay

Displays AI prediction information.

**Features:**
- Prediction type and confidence
- Model metadata
- Confidence visualization
- Warning about AI limitations

**Props:**
- `slideId: string` - Slide identifier

## State Management

The application uses Zustand for state management:

```typescript
interface AnnotationStore {
  annotations: Annotation[]
  queue: QueueItem[]
  currentAnnotation: Annotation | null
  loading: boolean
  error: string | null
  
  fetchAnnotations: (slideId: string) => Promise<void>
  createAnnotation: (data: any) => Promise<void>
  updateAnnotation: (id: string, data: any) => Promise<void>
  deleteAnnotation: (id: string) => Promise<void>
  fetchQueue: (expertId: string) => Promise<void>
  setCurrentAnnotation: (annotation: Annotation | null) => void
}
```

## API Integration

The frontend communicates with the backend via REST API and WebSocket:

### REST Endpoints

- `GET /api/queue` - Get annotation queue
- `GET /api/slides/{id}` - Get slide info
- `GET /api/annotations` - List annotations
- `POST /api/annotations` - Create annotation
- `PUT /api/annotations/{id}` - Update annotation
- `DELETE /api/annotations/{id}` - Delete annotation

### WebSocket

- `WS /ws/{slide_id}` - Real-time collaboration

## Customization

### Styling

Edit `src/index.css` to customize the appearance.

### Adding Annotation Tools

1. Add tool type to `AnnotationTool` type in `WSIViewer.tsx`
2. Add button to toolbar
3. Implement drawing logic
4. Update geometry model in backend

### Integrating with WSI Streaming

Replace the placeholder tile source in `WSIViewer.tsx`:

```typescript
tileSources: {
  type: 'image',
  url: `/api/slides/${slideId}/dzi`  // Your DZI endpoint
}
```

## TODO

- [ ] Implement Fabric.js for advanced drawing
- [ ] Add undo/redo functionality
- [ ] Implement keyboard shortcuts
- [ ] Add annotation history
- [ ] Implement collaborative cursors
- [ ] Add annotation export
- [ ] Implement annotation validation
- [ ] Add measurement tools
- [ ] Implement annotation search
- [ ] Add user preferences

## Troubleshooting

### Backend Connection Issues

Ensure backend is running on http://localhost:8001:

```bash
cd src/annotation_interface
python start_annotation_server.py
```

### CORS Errors

The backend is configured to allow all origins in development. For production, update CORS settings in `annotation_api.py`.

### OpenSeadragon Not Loading

Check browser console for errors. Ensure tile source URL is correct.

## License

Part of HistoCore Medical AI Platform
