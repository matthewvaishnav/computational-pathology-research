import React, { useEffect } from 'react'
import { useAnnotationStore } from '../store/annotationStore'

interface Props {
  slideId: string
  expertId: string
}

const AnnotationPanel: React.FC<Props> = ({ slideId, expertId }) => {
  const {
    annotations,
    currentAnnotation,
    setCurrentAnnotation,
    deleteAnnotation,
    fetchAnnotations
  } = useAnnotationStore()

  useEffect(() => {
    fetchAnnotations(slideId)
  }, [slideId, fetchAnnotations])

  const handleDelete = async (id: string) => {
    if (confirm('Delete this annotation?')) {
      try {
        await deleteAnnotation(id)
      } catch (error) {
        console.error('Failed to delete annotation:', error)
      }
    }
  }

  const slideAnnotations = annotations.filter(a => a.slide_id === slideId)

  return (
    <div>
      <h2>Annotations ({slideAnnotations.length})</h2>

      <div className="annotation-list">
        {slideAnnotations.length === 0 ? (
          <div style={{ padding: '1rem', textAlign: 'center', color: '#7f8c8d' }}>
            No annotations yet
          </div>
        ) : (
          slideAnnotations.map(annotation => (
            <div
              key={annotation.id}
              className={`annotation-item ${
                currentAnnotation?.id === annotation.id ? 'active' : ''
              }`}
              onClick={() => setCurrentAnnotation(annotation)}
            >
              <div style={{ fontWeight: 600, marginBottom: '0.5rem' }}>
                {annotation.label}
              </div>
              <div style={{ fontSize: '0.875rem', color: '#7f8c8d' }}>
                Confidence: {(annotation.confidence * 100).toFixed(0)}%
              </div>
              {annotation.comments && (
                <div style={{ fontSize: '0.875rem', marginTop: '0.5rem' }}>
                  {annotation.comments}
                </div>
              )}
              <div style={{
                marginTop: '0.5rem',
                display: 'flex',
                gap: '0.5rem'
              }}>
                <button
                  className="danger"
                  style={{ fontSize: '0.75rem', padding: '0.25rem 0.5rem' }}
                  onClick={(e) => {
                    e.stopPropagation()
                    handleDelete(annotation.id)
                  }}
                >
                  Delete
                </button>
              </div>
            </div>
          ))
        )}
      </div>

      {currentAnnotation && (
        <div style={{
          marginTop: '1rem',
          padding: '1rem',
          background: 'white',
          borderRadius: '4px',
          border: '1px solid #bdc3c7'
        }}>
          <h3>Selected Annotation</h3>
          <div style={{ marginTop: '0.5rem' }}>
            <strong>Label:</strong> {currentAnnotation.label}
          </div>
          <div>
            <strong>Confidence:</strong> {(currentAnnotation.confidence * 100).toFixed(0)}%
          </div>
          <div>
            <strong>Expert:</strong> {currentAnnotation.expert_id}
          </div>
          <div>
            <strong>Created:</strong> {new Date(currentAnnotation.created_at).toLocaleString()}
          </div>
        </div>
      )}
    </div>
  )
}

export default AnnotationPanel
