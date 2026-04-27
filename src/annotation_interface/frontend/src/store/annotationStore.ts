import { create } from 'zustand'
import axios from 'axios'

interface Annotation {
  id: string
  slide_id: string
  label: string
  geometry: any
  confidence: number
  comments: string
  expert_id: string
  created_at: string
  updated_at: string
}

interface QueueItem {
  task_id: string
  slide_id: string
  priority: number
  uncertainty_score: number
  ai_prediction: any
  status: string
  created_at: string
}

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

const API_BASE = '/api'

export const useAnnotationStore = create<AnnotationStore>((set, get) => ({
  annotations: [],
  queue: [],
  currentAnnotation: null,
  loading: false,
  error: null,

  fetchAnnotations: async (slideId: string) => {
    set({ loading: true, error: null })
    try {
      const response = await axios.get(`${API_BASE}/annotations`, {
        params: { slide_id: slideId }
      })
      set({ annotations: response.data, loading: false })
    } catch (error: any) {
      set({ error: error.message, loading: false })
    }
  },

  createAnnotation: async (data: any) => {
    set({ loading: true, error: null })
    try {
      const response = await axios.post(`${API_BASE}/annotations`, data)
      const newAnnotation = response.data.annotation
      set(state => ({
        annotations: [...state.annotations, newAnnotation],
        loading: false
      }))
    } catch (error: any) {
      set({ error: error.message, loading: false })
      throw error
    }
  },

  updateAnnotation: async (id: string, data: any) => {
    set({ loading: true, error: null })
    try {
      const response = await axios.put(`${API_BASE}/annotations/${id}`, data)
      const updatedAnnotation = response.data.annotation
      set(state => ({
        annotations: state.annotations.map(a =>
          a.id === id ? updatedAnnotation : a
        ),
        loading: false
      }))
    } catch (error: any) {
      set({ error: error.message, loading: false })
      throw error
    }
  },

  deleteAnnotation: async (id: string) => {
    set({ loading: true, error: null })
    try {
      await axios.delete(`${API_BASE}/annotations/${id}`)
      set(state => ({
        annotations: state.annotations.filter(a => a.id !== id),
        loading: false
      }))
    } catch (error: any) {
      set({ error: error.message, loading: false })
      throw error
    }
  },

  fetchQueue: async (expertId: string) => {
    set({ loading: true, error: null })
    try {
      const response = await axios.get(`${API_BASE}/queue`, {
        params: { expert_id: expertId, limit: 20 }
      })
      set({ queue: response.data, loading: false })
    } catch (error: any) {
      set({ error: error.message, loading: false })
    }
  },

  setCurrentAnnotation: (annotation: Annotation | null) => {
    set({ currentAnnotation: annotation })
  }
}))
