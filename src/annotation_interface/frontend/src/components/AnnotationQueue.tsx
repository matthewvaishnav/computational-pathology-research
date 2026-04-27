import React from 'react'
import { useAnnotationStore } from '../store/annotationStore'

interface Props {
  expertId: string
  onSelectSlide: (slideId: string) => void
}

const AnnotationQueue: React.FC<Props> = ({ expertId, onSelectSlide }) => {
  const { queue, loading } = useAnnotationStore()

  const getConfidenceClass = (uncertainty: number) => {
    if (uncertainty > 0.7) return 'confidence-low'
    if (uncertainty > 0.4) return 'confidence-medium'
    return 'confidence-high'
  }

  const getPriorityClass = (priority: number) => {
    return priority > 0.7 ? 'high-priority' : ''
  }

  if (loading) {
    return <div>Loading queue...</div>
  }

  if (queue.length === 0) {
    return <div>No pending annotations</div>
  }

  return (
    <div className="annotation-list">
      {queue.map(item => (
        <div
          key={item.task_id}
          className={`queue-item ${getPriorityClass(item.priority)}`}
          onClick={() => onSelectSlide(item.slide_id)}
        >
          <div style={{ fontWeight: 600, marginBottom: '0.5rem' }}>
            Slide: {item.slide_id}
          </div>
          <div style={{ fontSize: '0.875rem', color: '#7f8c8d' }}>
            Priority: {(item.priority * 100).toFixed(0)}%
          </div>
          <div style={{ marginTop: '0.5rem' }}>
            <span className={`confidence-badge ${getConfidenceClass(item.uncertainty_score)}`}>
              Uncertainty: {(item.uncertainty_score * 100).toFixed(0)}%
            </span>
          </div>
          {item.ai_prediction && (
            <div style={{ fontSize: '0.875rem', marginTop: '0.5rem' }}>
              AI: {item.ai_prediction.diagnosis || 'N/A'}
            </div>
          )}
        </div>
      ))}
    </div>
  )
}

export default AnnotationQueue
