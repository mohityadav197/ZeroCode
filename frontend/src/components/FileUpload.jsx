import { useRef, useState } from 'react'
import { motion } from 'framer-motion'
import { CheckCircle2, FileSpreadsheet, Loader2, UploadCloud, XCircle } from 'lucide-react'
import { uploadFile } from '../services/api'

function formatBytes(bytes) {
  if (!bytes) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(1024))
  return `${(bytes / 1024 ** i).toFixed(1)} ${units[i]}`
}

function DashBorder({ active }) {
  return (
    <svg className="absolute inset-0 w-full h-full pointer-events-none" style={{ borderRadius: 24 }}>
      <rect
        x="2"
        y="2"
        width="calc(100% - 4px)"
        height="calc(100% - 4px)"
        rx="22"
        fill="none"
        stroke={active ? '#a78bfa' : '#8b5cf6'}
        strokeOpacity={active ? 1 : 0.6}
        strokeWidth={active ? 2.5 : 2}
        strokeDasharray="10 8"
        className="dash-border-anim"
        style={{ animationDuration: active ? '0.4s' : '0.9s' }}
      />
    </svg>
  )
}

function FileUpload({ onUploadSuccess }) {
  const [file, setFile] = useState(null)
  const [dragActive, setDragActive] = useState(false)
  const [status, setStatus] = useState('idle')
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const inputRef = useRef(null)

  const handleFile = (f) => {
    if (!f) return
    setFile(f)
    setStatus('idle')
    setResult(null)
    setError('')
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragActive(false)
    handleFile(e.dataTransfer.files?.[0])
  }

  const handleUpload = async () => {
    if (!file) return
    setStatus('uploading')
    setError('')
    try {
      const data = await uploadFile(file)
      setResult(data)
      setStatus('success')
      onUploadSuccess?.(file.name, data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Upload failed')
      setStatus('error')
    }
  }

  if (status === 'success' && result) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: 'easeOut' }}
        className="glass-card p-5"
      >
        <div className="flex items-center gap-3">
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ type: 'spring', stiffness: 260, damping: 15, delay: 0.1 }}
            className="w-11 h-11 rounded-xl bg-emerald-500/15 flex items-center justify-center shrink-0"
          >
            <FileSpreadsheet className="w-5 h-5 text-emerald-400" />
          </motion.div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-bold text-slate-100 truncate">{result.filename}</p>
            <p className="text-xs text-slate-500">Uploaded successfully</p>
          </div>
          <motion.div
            initial={{ scale: 0, rotate: -45 }}
            animate={{ scale: 1, rotate: 0 }}
            transition={{ type: 'spring', stiffness: 260, damping: 15, delay: 0.2 }}
          >
            <CheckCircle2 className="w-6 h-6 text-emerald-400 shrink-0" />
          </motion.div>
        </div>
        <div className="mt-3 flex flex-wrap gap-2">
          <span className="text-xs font-medium bg-white/10 text-slate-300 px-2.5 py-1 rounded-full">
            {formatBytes(file?.size)}
          </span>
          <span className="text-xs font-medium bg-indigo-500/15 text-indigo-300 px-2.5 py-1 rounded-full">
            {result.rows} rows
          </span>
          <span className="text-xs font-medium bg-purple-500/15 text-purple-300 px-2.5 py-1 rounded-full">
            {result.columns} columns
          </span>
        </div>
        <button
          onClick={() => {
            setFile(null)
            setResult(null)
            setStatus('idle')
          }}
          className="mt-3 text-xs text-indigo-300 hover:text-indigo-200 font-medium"
        >
          Change file
        </button>
      </motion.div>
    )
  }

  return (
    <div className="w-full">
      <div
        onDragOver={(e) => {
          e.preventDefault()
          setDragActive(true)
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        className="glass-card relative flex flex-col items-center justify-center gap-3 p-12 text-center cursor-pointer transition-colors duration-300"
        style={{ background: dragActive ? 'rgba(139,92,246,0.12)' : undefined }}
      >
        <DashBorder active={dragActive} />
        <motion.div animate={{ scale: dragActive ? 1.2 : 1 }} transition={{ type: 'spring', stiffness: 300, damping: 15 }}>
          <UploadCloud className="w-12 h-12" style={{ color: dragActive ? '#a78bfa' : '#8b5cf6' }} />
        </motion.div>
        <p className="text-slate-100 font-semibold text-lg">Drop your CSV here</p>
        <p className="text-sm text-slate-500">or click to browse files</p>
        <span className="mt-2 text-[11px] font-medium bg-white/5 border border-white/10 text-slate-400 px-2.5 py-1 rounded-full">
          CSV files only
        </span>
        <input
          ref={inputRef}
          type="file"
          accept=".csv"
          className="hidden"
          onChange={(e) => handleFile(e.target.files?.[0])}
        />
      </div>

      {file && (
        <div className="mt-4 flex items-center justify-between glass-card p-3">
          <div className="flex items-center gap-2 min-w-0">
            <FileSpreadsheet className="w-5 h-5 text-cyan-400 shrink-0" />
            <div className="min-w-0">
              <p className="text-sm font-medium text-slate-100 truncate">{file.name}</p>
              <p className="text-xs text-slate-500">{formatBytes(file.size)}</p>
            </div>
          </div>
          <button
            onClick={handleUpload}
            disabled={status === 'uploading'}
            className="gradient-btn flex items-center gap-2 text-sm py-2 px-4"
          >
            {status === 'uploading' && <Loader2 className="w-4 h-4 animate-spin" />}
            {status === 'uploading' ? 'Uploading...' : 'Upload'}
          </button>
        </div>
      )}

      {status === 'error' && (
        <div className="mt-4 flex items-center gap-2 bg-red-500/10 border border-red-500/30 rounded-lg p-3">
          <XCircle className="w-5 h-5 text-red-400 shrink-0" />
          <p className="text-sm text-red-300">{error}</p>
        </div>
      )}
    </div>
  )
}

export default FileUpload
