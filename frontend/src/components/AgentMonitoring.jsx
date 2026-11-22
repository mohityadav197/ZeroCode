import { useEffect, useState } from 'react'
import api from '../services/api'

const STATUS_LABELS = { start: 'Started', complete: 'Completed', error: 'Error' }
const STATUS_STYLES = { start: 'text-amber-300', complete: 'text-emerald-300', error: 'text-red-400' }

export default function AgentMonitoring({ sessionId }) {
  const [trace, setTrace] = useState(null)
  const [status, setStatus] = useState('')
  const [heatmap, setHeatmap] = useState(false)
  const [refresh, setRefresh] = useState(0)
  const [logs, setLogs] = useState([])

  useEffect(() => {
    if (!sessionId) return undefined
    let active = true
    const load = async () => {
      try {
        const { data } = await api.get(`/monitoring/logs/${sessionId}`)
        if (active) setLogs(Array.isArray(data) ? data : [])
      } catch {
        // Keep the previous rows; the next poll will retry.
      }
    }
    load()
    const timer = setInterval(load, 5000)
    return () => { active = false; clearInterval(timer) }
  }, [sessionId])

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

      <div className="space-y-2">
        <h4 className="text-xs font-semibold text-slate-400">Live agent logs (refreshes every 5s)</h4>
        {logs.length === 0 ? (
          <p className="text-xs text-slate-500">No logs yet</p>
        ) : (
          <div className="max-h-64 overflow-y-auto">
            <table className="w-full text-xs text-left text-slate-300">
              <thead className="text-slate-400">
                <tr>
                  <th className="py-1 pr-3">Agent</th>
                  <th className="py-1 pr-3">Status</th>
                  <th className="py-1 pr-3">Duration</th>
                  <th className="py-1">Timestamp</th>
                </tr>
              </thead>
              <tbody>
                {logs.map((log) => (
                  <tr key={log.id} className="border-t border-slate-700/50" title={log.error || undefined}>
                    <td className="py-1 pr-3">{log.agent_name}</td>
                    <td className={`py-1 pr-3 ${STATUS_STYLES[log.event_type] || ''}`}>
                      {STATUS_LABELS[log.event_type] || log.event_type}
                    </td>
                    <td className="py-1 pr-3">{log.duration_ms != null ? `${Math.round(log.duration_ms)} ms` : '—'}</td>
                    <td className="py-1">{log.timestamp ? new Date(log.timestamp).toLocaleTimeString() : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
