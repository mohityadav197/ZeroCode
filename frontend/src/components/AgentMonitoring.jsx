import { useState } from 'react'
import api from '../services/api'

export default function AgentMonitoring({ sessionId }) {
  const [trace, setTrace] = useState(null)
  const [status, setStatus] = useState('')
  const [heatmap, setHeatmap] = useState(false)
  const [refresh, setRefresh] = useState(0)
  if (!sessionId) return null

  const inspect = async () => {
    try {
      const { data } = await api.get(`/observability/trace/${sessionId}`)
      setTrace(data.url)
      setStatus(data.url ? '' : 'Run analysis to collect agent traces.')
    } catch {
      setStatus('Monitoring is currently unavailable.')
    }
  }

  return (
    <div className="glass-card p-5 space-y-3">
      <h3 className="text-sm font-semibold text-slate-300">Agent performance</h3>
      <div className="flex gap-4 text-sm text-indigo-300">
        <button onClick={inspect}>Find session trace</button>
        <button onClick={() => { setRefresh(Date.now()); setHeatmap((visible) => !visible) }}>Toggle duration heatmap</button>
        {trace && <a href={trace} target="_blank" rel="noreferrer">Open Jaeger trace ↗</a>}
      </div>
      {status && <p role="status" className="text-xs text-slate-400">{status}</p>}
      {heatmap && <img className="w-full rounded-lg" alt="Agent durations by pipeline stage and run"
        src={`${api.defaults.baseURL}/monitoring/heatmap/${sessionId}?refresh=${refresh}`}
        onError={() => { setHeatmap(false); setStatus('Run analysis to collect agent timings.') }} />}
    </div>
  )
}
