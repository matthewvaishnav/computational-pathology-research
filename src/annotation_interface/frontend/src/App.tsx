import React, { useState, useEffect } from 'react'
import AnnotationQueue from './components/AnnotationQueue'
import WSIViewer from './components/WSIViewer'
import AnnotationPanel from './components/AnnotationPanel'
import AIOverlay from './components/AIOverlay'
import { useAnnotationStore } from './store/annotationStore'

function App() {
  const [currentSlideId, setCurrentSlideId] = useState<string | null>(null)
  const [expertId] = useState('expert_001') // TODO: Get from auth
  const { fetchQueue } = useAnnotationStore()

  useEffect(() => {
    // Fetch annotation queue on mount
    fetchQueue(expertId)
  }, [expertId, fetchQueue])

  const handleSelectSlide = (slideId: string) => {
    setCurrentSlideId(slideId)
  }

  return (
    <div className="app-container">
      <header className="header">
        <h1>Expert Annotation Interface</h1>
        <div>
          <span>Expert: {expertId}</span>
        </div>
      </header>

      <div className="main-content">
        <aside className="sidebar">
          <h2>Annotation Queue</h2>
          <AnnotationQueue
            expertId={expertId}
            onSelectSlide={handleSelectSlide}
          />
        </aside>

        <main className="viewer-container">
          {currentSlideId ? (
            <>
              <WSIViewer slideId={currentSlideId} expertId={expertId} />
              <AIOverlay slideId={currentSlideId} />
            </>
          ) : (
            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              height: '100%',
              color: 'white',
              fontSize: '1.5rem'
            }}>
              Select a slide from the queue to begin annotation
            </div>
          )}
        </main>

        {currentSlideId && (
          <aside className="sidebar">
            <AnnotationPanel slideId={currentSlideId} expertId={expertId} />
          </aside>
        )}
      </div>
    </div>
  )
}

export default App
