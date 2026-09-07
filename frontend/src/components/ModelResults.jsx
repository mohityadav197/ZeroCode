import { motion } from 'framer-motion'
import { Info, Trophy } from 'lucide-react'
import Leaderboard from './Leaderboard'

const MODEL_COLORS = ['#6366f1', '#06b6d4', '#8b5cf6', '#10b981', '#f59e0b', '#ec4899']

function colorFor(name) {
  let hash = 0
  for (let i = 0; i < name.length; i++) hash = name.charCodeAt(i) + ((hash << 5) - hash)
  return MODEL_COLORS[Math.abs(hash) % MODEL_COLORS.length]
}

function Card({ title, children }) {
  return (
    <div className="glass-card p-5">
      <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wide mb-3">{title}</h3>
      {children}
    </div>
  )
}

function ModelResults({ result }) {
  if (!result) return null

  const reasoning = result?.model_reasoning || {}
  const leaderboard = result?.leaderboard || []
  const bestEntry = leaderboard.find((r) => r?.is_best) || leaderboard[0]
  const bestParams = result?.best_params || {}
  const preprocessingLog = result?.preprocessing_log || []

  return (
    <div className="space-y-5">
      <Card title="Model Selection Reasoning">
        {Object.keys(reasoning).length === 0 ? (
          <p className="text-sm text-slate-500">No model reasoning available.</p>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {Object.entries(reasoning).map(([name, reason], i) => {
              const color = colorFor(name)
              return (
                <motion.div
                  key={name}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  whileHover={{ y: -3 }}
                  transition={{ duration: 0.3, delay: i * 0.06 }}
                  className="flex items-start gap-2 rounded-xl border px-3 py-2.5 cursor-default"
                  style={{ backgroundColor: `${color}14`, borderColor: `${color}33` }}
                >
                  <Info className="w-4 h-4 shrink-0 mt-0.5" style={{ color }} />
                  <div>
                    <p className="text-sm font-semibold" style={{ color }}>
                      {name}
                    </p>
                    <p className="text-xs text-slate-400">{reason}</p>
                  </div>
                </motion.div>
              )
            })}
          </div>
        )}
      </Card>

      <Card title="Leaderboard">
        <Leaderboard leaderboard={leaderboard} />
      </Card>

      {bestEntry && (
        <div className="relative rounded-2xl p-[1.5px] bg-gradient-to-r from-purple-500 to-cyan-500">
          <div className="rounded-[15px] p-5 border border-white/5" style={{ backgroundColor: '#14142a' }}>
            <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wide mb-3">Best Model Details</h3>
            <div className="flex items-center gap-2 mb-4">
              <Trophy className="w-7 h-7 text-yellow-400" />
              <p className="text-3xl font-extrabold gradient-text">{result?.best_model_name ?? 'Unknown'}</p>
            </div>
            <div className="flex flex-wrap gap-2 mb-4">
              {Object.entries(bestEntry || {})
                .filter(([key]) => !['rank', 'model', 'is_best', 'training_time'].includes(key))
                .map(([key, value]) => (
                  <span
                    key={key}
                    className="text-xs font-semibold bg-white/10 text-slate-100 px-3 py-1.5 rounded-full border border-white/10"
                  >
                    {key}: <span className="text-indigo-300">{value ?? '—'}</span>
                  </span>
                ))}
            </div>
            {Object.keys(bestParams || {}).length > 0 && (
              <div>
                <p className="text-xs text-slate-500 mb-1.5">Best hyperparameters</p>
                <pre className="text-xs text-emerald-300 bg-black/40 border border-white/10 rounded-lg p-3 overflow-x-auto font-mono">
                  {JSON.stringify(bestParams, null, 2)}
                </pre>
              </div>
            )}
          </div>
        </div>
      )}

      <Card title="Preprocessing Log">
        {preprocessingLog.length === 0 ? (
          <p className="text-sm text-slate-500">No preprocessing steps recorded.</p>
        ) : (
          <ol className="list-decimal list-inside space-y-1 text-sm text-slate-300">
            {preprocessingLog.map((step, i) => (
              <li key={i}>{step}</li>
            ))}
          </ol>
        )}
      </Card>
    </div>
  )
}

export default ModelResults
