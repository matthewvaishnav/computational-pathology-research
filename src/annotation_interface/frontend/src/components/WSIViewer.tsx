import React, { useEffect, useRef, useState } from 'react'
import OpenSeadragon from 'openseadragon'
import { useAnnotationStore } from '../store/annotationStore'

interface Props {
  slideId: string
  expertId: string
}

type AnnotationTool = 'select' | 'polygon' | 'circle' | 'rectangle' | 'freehand'

const WSIViewer: React.FC<Props> = ({ slideId, expertId }) => {
  const viewerRef = useRef<HTMLDivElement>(null)
  const osdViewerRef = useRef<any>(null)
  const [activeTool, setActiveTool] = useState<AnnotationTool>('select')
  const [isDrawing, setIsDrawing] = useState(false)
  const [currentPoints, setCurrentPoints] = useState<any[]>([])
  const { createAnnotation, fetchAnnotations } = useAnnotationStore()

  useEffect(() => {
    if (!viewerRef.current) return

    // Initialize OpenSeadragon viewer
    const viewer = OpenSeadragon({
      element: viewerRef.current,
      prefixUrl: 'https://cdn.jsdelivr.net/npm/openseadragon@4.1/build/openseadragon/images/',
      tileSources: {
        // TODO: Replace with actual DZI tile source from backend
        type: 'image',
        url: 'https://openseadragon.github.io/example-images/highsmith/highsmith.dzi'
      },
      showNavigationControl: true,
      showNavigator: true,
      navigatorPosition: 'BOTTOM_RIGHT',
      animationTime: 0.5,
      blendTime: 0.1,
      constrainDuringPan: true,
      maxZoomPixelRatio: 2,
      minZoomLevel: 0.5,
      visibilityRatio: 1,
      zoomPerScroll: 2
    })

    osdViewerRef.current = viewer

    // Fetch existing annotations
    fetchAnnotations(slideId)

    return () => {
      if (viewer) {
        viewer.destroy()
      }
    }
  }, [slideId, fetchAnnotations])

  const handleToolClick = (tool: AnnotationTool) => {
    setActiveTool(tool)
    setIsDrawing(false)
    setCurrentPoints([])
  }

  const handleSaveAnnotation = async (label: string) => {
    if (currentPoints.length === 0) return

    try {
      await createAnnotation({
        slide_id: slideId,
        expert_id: expertId,
        label: label,
        geometry: {
          type: activeTool,
          points: currentPoints
        },
        confidence: 1.0,
        comments: ''
      })

      // Clear current drawing
      setCurrentPoints([])
      setIsDrawing(false)
    } catch (error) {
      console.error('Failed to save annotation:', error)
    }
  }

  return (
    <>
      <div
        ref={viewerRef}
        style={{
          width: '100%',
          height: '100%',
          background: '#000'
        }}
      />

      <div className="annotation-tools">
        <button
          className={`tool-button ${activeTool === 'select' ? 'active' : ''}`}
          onClick={() => handleToolClick('select')}
          title="Select"
        >
          Select
        </button>
        <button
          className={`tool-button ${activeTool === 'polygon' ? 'active' : ''}`}
          onClick={() => handleToolClick('polygon')}
          title="Polygon"
        >
          Polygon
        </button>
        <button
          className={`tool-button ${activeTool === 'circle' ? 'active' : ''}`}
          onClick={() => handleToolClick('circle')}
          title="Circle"
        >
          Circle
        </button>
        <button
          className={`tool-button ${activeTool === 'rectangle' ? 'active' : ''}`}
          onClick={() => handleToolClick('rectangle')}
          title="Rectangle"
        >
          Rectangle
        </button>
        <button
          className={`tool-button ${activeTool === 'freehand' ? 'active' : ''}`}
          onClick={() => handleToolClick('freehand')}
          title="Freehand"
        >
          Freehand
        </button>
      </div>

      {isDrawing && (
        <div style={{
          position: 'absolute',
          bottom: '1rem',
          left: '50%',
          transform: 'translateX(-50%)',
          background: 'white',
          padding: '1rem',
          borderRadius: '8px',
          boxShadow: '0 2px 8px rgba(0,0,0,0.2)',
          display: 'flex',
          gap: '0.5rem'
        }}>
          <button
            className="primary"
            onClick={() => handleSaveAnnotation('tumor')}
          >
            Save as Tumor
          </button>
          <button
            className="secondary"
            onClick={() => handleSaveAnnotation('normal')}
          >
            Save as Normal
          </button>
          <button
            className="danger"
            onClick={() => {
              setCurrentPoints([])
              setIsDrawing(false)
            }}
          >
            Cancel
          </button>
        </div>
      )}
    </>
  )
}

export default WSIViewer
