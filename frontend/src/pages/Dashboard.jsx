import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from 'react'
import toast from 'react-hot-toast'
import { motion } from 'framer-motion'
import { Clock, Plus, RotateCcw } from 'lucide-react'
import FileUpload from '../components/FileUpload'
import DataPreview from '../components/DataPreview'
import PipelineTracker from '../components/PipelineTracker'
import AnalysisResults from '../components/AnalysisResults'
import ModelResults from '../components/ModelResults'
import ModelSelector from '../components/ModelSelector'
import DownloadPanel from '../components/DownloadPanel'
import ChartDisplay from '../components/ChartDisplay'
import ChatPanel from '../components/ChatPanel'
import VisualChecklist from '../components/VisualChecklist'
import ConfirmDialog from '../components/ConfirmDialog'
import {
  analyzeData,
  generateReport,
  getSession,
  getSessions,
  trainModels,
  updateContext,
} from '../services/api'

const INITIAL_STEPS = [
  { name: 'Upload CSV', status: 'waiting', description: 'Upload a dataset to get started' },
  { name: 'Analysis', status: 'waiting', description: 'Profile data and generate insights' },
  { name: 'Model Training', status: 'waiting', description: 'Train and tune ML models' },
  { name: 'Report Generation', status: 'waiting', description: 'Generate the final report' },
]

const STATUS_BADGE = {
  uploaded: 'bg-slate-500/20 text-slate-300',
  analyzed: 'bg-indigo-500/20 text-indigo-300',
  trained: 'bg-purple-500/20 text-purple-300',
  completed: 'bg-emerald-500/20 text-emerald-300',
}

const STATUS_STEP_INDEX = { uploaded: 0, analyzed: 1, trained: 2, completed: 3 }

function SkeletonBlock({ lines = 3 }) {
  return (
    <div className="glass-card p-5 space-y-3">
      <div className="shimmer h-4 w-1/3 rounded" />
      {Array.from({ length: lines }).map((_, i) => (
        <div key={i} className="shimmer h-3 w-full rounded" />
      ))}
    </div>
  )
}

function RecentSessions({ sessions, onRestore, restoringId }) {
  if (!sessions.length) return null

  return (
    <div className="glass-card p-5">
      <div className="flex items-center gap-2 mb-4">
        <Clock className="w-4 h-4 text-indigo-400" />
        <h3 className="text-sm font-semibold text-slate-300">Recent Sessions</h3>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {sessions.slice(0, 5).map((s, i) => (
          <motion.div
            key={s.session_id}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: i * 0.06 }}
            className="rounded-xl border border-white/10 bg-white/[0.03] p-3.5 flex flex-col gap-2"
          >
            <div className="flex items-center justify-between gap-2">
              <p className="text-sm font-semibold text-slate-100 truncate">{s.filename}</p>
              <span className={`text-[10px] font-bold uppercase px-2 py-0.5 rounded-full shrink-0 ${STATUS_BADGE[s.status] || STATUS_BADGE.uploaded}`}>
                {s.status}
              </span>
            </div>
            {s.best_model && (
              <p className="text-xs text-slate-400">
                Best: <span className="text-emerald-300 font-medium">{s.best_model}</span>
                {s.accuracy != null && ` (${(s.accuracy * 100).toFixed(1)}%)`}
              </p>
            )}
            <div className="flex items-center justify-between mt-1">
              <span className="text-[11px] text-slate-500">
                {s.created_at ? new Date(s.created_at).toLocaleDateString() : ''}
              </span>
              <button
                onClick={() => onRestore(s)}
                disabled={restoringId === s.session_id}
                className="flex items-center gap-1 text-xs font-semibold text-indigo-300 hover:text-indigo-200 transition-colors disabled:opacity-50"
              >
                <RotateCcw className="w-3 h-3" />
                {restoringId === s.session_id ? 'Restoring…' : 'Restore'}
              </button>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  )
}

const Dashboard = forwardRef(function Dashboard({ onDatasetChange }, ref) {
  const [sessionId, setSessionId] = useState(null)
  const [sessions, setSessions] = useState([])
  const [restoringId, setRestoringId] = useState(null)
  const [chatHistory, setChatHistory] = useState([])
  const [uploadedFile, setUploadedFile] = useState(null)
  const [uploadInfo, setUploadInfo] = useState(null)
  const [analysisResult, setAnalysisResult] = useState(null)
  const [mlResult, setMlResult] = useState(null)
  const [reportResult, setReportResult] = useState(null)
  const [pipelineSteps, setPipelineSteps] = useState(INITIAL_STEPS)
  const [selectedExtraModels, setSelectedExtraModels] = useState([])
  const [loadingMessage, setLoadingMessage] = useState('')
  const [errorMessage, setErrorMessage] = useState('')
  const [checklistOpen, setChecklistOpen] = useState(false)
  const [newSessionModalOpen, setNewSessionModalOpen] = useState(false)

  const scrollContainerRef = useRef(null)
  const previewRef = useRef(null)
  const analysisRef = useRef(null)
  const modelRef = useRef(null)
  const reportRef = useRef(null)

  const refreshSessions = () => {
    getSessions()
      .then((data) => setSessions(data || []))
      .catch(() => setSessions([]))
  }

  useEffect(() => {
    refreshSessions()
  }, [])

  const updateStep = (index, status) => {
    setPipelineSteps((prev) => prev.map((s, i) => (i === index ? { ...s, status } : s)))
  }

  const handleUploadSuccess = (filename, data) => {
    setUploadedFile(filename)
    setUploadInfo(data)
    setSessionId(data.session_id ?? null)
    updateStep(0, 'done')
    onDatasetChange?.(filename)
    setChecklistOpen(true)
    toast.success('File uploaded successfully!')
  }

  const handleRunAnalysis = async (visualOptions) => {
    if (!uploadedFile) return
    setChecklistOpen(false)
    setErrorMessage('')
    updateStep(1, 'running')
    setLoadingMessage('Analyzing your data...')
    try {
      const result = await analyzeData(uploadedFile, null, visualOptions ?? null, sessionId)
      setAnalysisResult(result)
      updateStep(1, 'done')
      await updateContext('filename', uploadedFile)
      await updateContext('analysis_result', result)
      const insightCount = result.insights?.length ?? 0
      toast.success(`Analysis complete! ${insightCount} insights found.`)
    } catch (err) {
      updateStep(1, 'error')
      const msg = err.response?.data?.detail || err.message || 'Analysis failed'
      setErrorMessage(msg)
      toast.error(msg)
    } finally {
      setLoadingMessage('')
    }
  }

  const handleTrainModels = async () => {
    if (!uploadedFile || !analysisResult) return
    const targetCol = analysisResult.target_suggestion?.suggested_target
    const problemType = analysisResult.problem_type?.problem_type
    setErrorMessage('')
    updateStep(2, 'running')
    setLoadingMessage('Training models...')
    try {
      const result = await trainModels(
        uploadedFile,
        targetCol,
        problemType,
        selectedExtraModels.length ? selectedExtraModels : null,
        sessionId
      )
      console.log('Train result:', result)
      setMlResult(result)
      updateStep(2, 'done')
      await updateContext('ml_result', result)
      await updateContext('target_col', targetCol)
      await updateContext('problem_type', problemType)

      const bestEntry = result.leaderboard?.find((r) => r.is_best) || result.leaderboard?.[0]
      const metric = bestEntry?.accuracy ?? bestEntry?.r2
      const pct = metric != null ? (metric * 100).toFixed(1) : '—'
      toast.success(`Best model: ${result.best_model_name} (${pct}%)`)
    } catch (err) {
      updateStep(2, 'error')
      const msg = err.response?.data?.detail || err.message || 'Training failed'
      setErrorMessage(msg)
      toast.error(msg)
    } finally {
      setLoadingMessage('')
    }
  }

  const handleGenerateReport = async () => {
    if (!uploadedFile || !analysisResult) return
    const targetCol = analysisResult.target_suggestion?.suggested_target
    const problemType = analysisResult.problem_type?.problem_type
    setErrorMessage('')
    updateStep(3, 'running')
    setLoadingMessage('Generating report...')
    try {
      const result = await generateReport(uploadedFile, targetCol, problemType, sessionId)
      setReportResult(result)
      updateStep(3, 'done')
      toast.success('Report ready to download!')
      refreshSessions()
    } catch (err) {
      updateStep(3, 'error')
      const msg = err.response?.data?.detail || err.message || 'Report generation failed'
      setErrorMessage(msg)
      toast.error(msg)
    } finally {
      setLoadingMessage('')
    }
  }

  const handleRetry = (index) => {
    if (index === 1) handleRunAnalysis(null)
    if (index === 2) handleTrainModels()
    if (index === 3) handleGenerateReport()
  }

  const handleRestoreSession = async (session) => {
    setRestoringId(session.session_id)
    try {
      const data = await getSession(session.session_id)
      setSessionId(session.session_id)
      setUploadedFile(session.filename)
      setUploadInfo(null)
      setAnalysisResult(data.analysis_result)
      setMlResult(data.ml_result)
      setReportResult(data.files?.some((f) => f.file_type === 'report') ? { verification: { stamp: 'Restored session' } } : null)
      setChatHistory(data.chat_history || [])
      setChecklistOpen(false)
      setErrorMessage('')

      const doneUpTo = STATUS_STEP_INDEX[session.status] ?? 0
      setPipelineSteps(INITIAL_STEPS.map((s, i) => ({ ...s, status: i <= doneUpTo ? 'done' : 'waiting' })))
      onDatasetChange?.(session.filename)
      toast.success(`Restored session for ${session.filename}`)
    } catch (err) {
      toast.error(err.response?.data?.detail || err.message || 'Failed to restore session')
    } finally {
      setRestoringId(null)
    }
  }

  const handleStepClick = (index) => {
    if (pipelineSteps[index]?.status !== 'done') return
    const refs = [previewRef, analysisRef, modelRef, reportRef]
    const target = refs[index]?.current
    if (target) {
      target.scrollIntoView({ behavior: 'smooth', block: 'start' })
    } else {
      scrollContainerRef.current?.scrollTo({ top: 0, behavior: 'smooth' })
    }
  }

  const resetSession = () => {
    setSessionId(null)
    setChatHistory([])
    setUploadedFile(null)
    setUploadInfo(null)
    setAnalysisResult(null)
    setMlResult(null)
    setReportResult(null)
    setSelectedExtraModels([])
    setPipelineSteps(INITIAL_STEPS)
    setLoadingMessage('')
    setErrorMessage('')
    setChecklistOpen(false)
    setNewSessionModalOpen(false)
    onDatasetChange?.(null)
    refreshSessions()
    scrollContainerRef.current?.scrollTo({ top: 0, behavior: 'smooth' })
  }

  useImperativeHandle(ref, () => ({
    requestNewSession: () => setNewSessionModalOpen(true),
  }))

  return (
    <div className="flex flex-col lg:flex-row flex-1 min-h-0 w-full gap-6 p-6 bg-[#0f0f1a]">
      <VisualChecklist open={checklistOpen} onRunAnalysis={handleRunAnalysis} />

      <div ref={scrollContainerRef} className="w-full lg:w-[65%] h-full overflow-y-auto space-y-6">
        {Boolean(uploadedFile) && (
          <div className="sticky top-0 z-20 pb-2 bg-[#0f0f1a]/95 backdrop-blur-sm">
            <button
              onClick={() => setNewSessionModalOpen(true)}
              className="glass-card flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold text-slate-200 hover:text-white transition-shadow duration-200 hover:shadow-[0_0_20px_rgba(139,92,246,0.5)]"
            >
              <Plus className="w-3.5 h-3.5" />
              New Session
            </button>
          </div>
        )}

        <PipelineTracker steps={pipelineSteps} onRetry={handleRetry} onStepClick={handleStepClick} />

        {!uploadInfo && !analysisResult && (
          <RecentSessions sessions={sessions} onRestore={handleRestoreSession} restoringId={restoringId} />
        )}

        {!uploadInfo && !analysisResult && (
          <div className="glass-card p-6">
            <FileUpload onUploadSuccess={handleUploadSuccess} />
          </div>
        )}

        {uploadInfo && (
          <div ref={previewRef} className="glass-card p-6">
            <h3 className="text-sm font-semibold text-slate-300 mb-3">Data Preview</h3>
            <DataPreview data={uploadInfo} />
          </div>
        )}

        {loadingMessage && (
          <div className="space-y-3">
            <div className="flex items-center gap-2 text-sm text-indigo-300">
              <span className="w-2 h-2 rounded-full bg-indigo-400 animate-pulse" />
              {loadingMessage}
            </div>
            <SkeletonBlock lines={4} />
          </div>
        )}

        {errorMessage && (
          <div className="text-sm text-red-300 bg-red-500/10 border border-red-500/30 rounded-lg p-3">
            {errorMessage}
          </div>
        )}

        {analysisResult && (
          <div ref={analysisRef} className="space-y-4">
            <AnalysisResults result={analysisResult} />

            <div className="glass-card p-5">
              <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wide mb-3">Charts</h3>
              <ChartDisplay charts={analysisResult.charts?.charts || {}} />
            </div>

            {!mlResult && (
              <div className="space-y-4">
                <ModelSelector
                  problemType={analysisResult.problem_type?.problem_type}
                  onSelectionChange={setSelectedExtraModels}
                />
                <button
                  onClick={handleTrainModels}
                  disabled={loadingMessage !== ''}
                  className="gradient-btn"
                >
                  Train Models
                </button>
              </div>
            )}
          </div>
        )}

        {mlResult && (
          <div ref={modelRef} className="space-y-4">
            <ModelResults result={mlResult} />
            {!reportResult && (
              <button
                onClick={handleGenerateReport}
                disabled={loadingMessage !== ''}
                className="gradient-btn"
              >
                Generate Report
              </button>
            )}
          </div>
        )}

        {reportResult && (
          <motion.div ref={reportRef} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="space-y-4">
            <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-xl p-4 text-sm text-emerald-300">
              Report generated successfully — {reportResult.verification?.stamp}
            </div>
            <DownloadPanel reportReady={Boolean(reportResult)} modelReady={Boolean(mlResult)} />
          </motion.div>
        )}
      </div>

      <div className="hidden lg:block lg:w-[35%] h-full">
        <ChatPanel
          analysisResult={analysisResult}
          mlResult={mlResult}
          sessionId={sessionId}
          initialMessages={chatHistory}
        />
      </div>

      <ConfirmDialog
        open={newSessionModalOpen}
        title="Start New Session?"
        message="Your current results will remain saved in history. Start fresh with a new dataset?"
        confirmText="Yes, Start Fresh"
        cancelText="Cancel"
        danger
        onConfirm={resetSession}
        onCancel={() => setNewSessionModalOpen(false)}
      />
    </div>
  )
})

export default Dashboard
