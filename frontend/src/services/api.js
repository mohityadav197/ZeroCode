import axios from 'axios'

const BASE_URL = 'http://localhost:8000'

const api = axios.create({
  baseURL: BASE_URL,
})

export const uploadFile = async (file) => {
  const formData = new FormData()
  formData.append('file', file)
  const res = await api.post('/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return res.data
}

export const analyzeData = async (filename, confirmedTarget = null, visualOptions = null, sessionId = null) => {
  const res = await api.post('/analyze', {
    filename,
    confirmed_target: confirmedTarget,
    visual_options: visualOptions,
    session_id: sessionId,
  })
  return res.data
}

export const trainModels = async (filename, targetCol, problemType, extraModels = null, sessionId = null) => {
  const res = await api.post('/train', {
    filename,
    target_col: targetCol,
    problem_type: problemType,
    extra_models: extraModels,
    session_id: sessionId,
  })
  return res.data
}

export const generateReport = async (filename, targetCol, problemType, sessionId = null) => {
  const res = await api.post('/report', {
    filename,
    target_col: targetCol,
    problem_type: problemType,
    session_id: sessionId,
  })
  return res.data
}

export const sendChat = async (message, sessionId = null) => {
  const res = await api.post('/chat', { message, session_id: sessionId })
  return res.data
}

export const updateContext = async (key, value) => {
  const res = await api.post('/context/update', { key, value })
  return res.data
}

export const getSessions = async () => {
  const res = await api.get('/sessions')
  return res.data
}

export const getSession = async (sessionId) => {
  const res = await api.get(`/sessions/${sessionId}`)
  return res.data
}

export const getSessionFiles = async (sessionId) => {
  const res = await api.get(`/sessions/${sessionId}/files`)
  return res.data
}

export const getDownloadUrl = (item) => `${BASE_URL}/download/${item}`

export default api
