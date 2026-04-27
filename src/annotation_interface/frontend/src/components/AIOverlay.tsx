import React, { useEffect, useState } from 'react'
import axios from 'axios'

interface Props {
  slideId: string
}

interface AIPrediction {
  slide_id: string
  prediction_type: string
  confidence: number
  metadata: any
}

const AIOverlay: React.FC<Props> = ({ slideId }) => {
  const [prediction, setPrediction] = useState<AIPrediction | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchPrediction = async () => {
      try {
        const response = await axios.get(`/api/slides/${slideId}/ai-prediction`)
        setPrediction(response.data)
      } catch (error) {
        console.error('Failed to fetch AI prediction:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchPrediction()
  }, [slideId])

  if (loading) {
    return (
      <div className="ai-overlay">
        <div>Loading AI prediction...</div>
      </div>
    )
  }

  if (!prediction) {
    return null
  }

  const getConfidenceClass = (confidence: number) => {
    if (confidence >= 0.85) return 'confidence-high'
    if (confidence >= 0.7) return 'confidence-medium'
    return 'confidence-low'
  }

  return (
    <div className="ai-overlay">
      <h3 style={{ marginBottom: '0.75rem' }}>AI Prediction</h3>
      
      <div style={{ marginBottom: '0.5rem' }}>
        <strong>Type:</strong> {prediction.prediction_type}
      </div>
      
      <div style={{ marginBottom: '0.5rem' }}>
        <strong>Confidence:</strong>
        <span
          className={`confidence-badge ${getConfidenceClass(prediction.confidence)}`}
          style={{ marginLeft: '0.5rem' }}
        >
          {(prediction.confidence * 100).toFixed(1)}%
        </span>
      </div>

      {prediction.metadata && (
        <div style={{
          marginTop: '0.75rem',
          padding: '0.5rem',
          background: '#ecf0f1',
          borderRadius: '4px',
          fontSize: '0.875rem'
        }}>
          <strong>Model:</strong> {prediction.metadata.model || 'N/A'}
        </div>
      )}

      <div style={{
        marginTop: '0.75rem',
        padding: '0.5rem',
        background: '#fff3cd',
        borderRadius: '4px',
        fontSize: '0.875rem',
        color: '#856404'
      }}>
        <strong>Note:</strong> AI predictions are for reference only. 
        Expert review is required for final diagnosis.
      </div>
    </div>
  )
}

export default AIOverlay
