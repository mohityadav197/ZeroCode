import { useEffect, useRef } from 'react'
import { animate, motion } from 'framer-motion'
import { AlertTriangle, Hash, Lightbulb, Rows3, Target as TargetIcon } from 'lucide-react'
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

function qualityColor(score) {
  if (score > 80) return '#10b981'
  if (score >= 50) return '#f59e0b'
  return '#ef4444'
}

function QualityRing({ score }) {
  const textRef = useRef(null)
  const circleRef = useRef(null)
  const radius = 52
  const circumference = 2 * Math.PI * radius
  const color = qualityColor(score)

  useEffect(() => {
    const controls = animate(0, score, {
      duration: 1.2,
      ease: 'easeOut',
      onUpdate(value) {
        if (textRef.current) textRef.current.textContent = Math.round(value)
        if (circleRef.current) {
          const offset = circumference - (Math.min(value, 100) / 100) * circumference
          circleRef.current.style.strokeDashoffset = offset
        }
      },
    })
    return () => controls.stop()
  }, [score, circumference])

  return (
    <div className="relative w-32 h-32 shrink-0">
      <svg viewBox="0 0 120 120" className="w-full h-full -rotate-90">
        <circle cx="60" cy="60" r={radius} fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="10" />
        <circle
          ref={circleRef}
          cx="60"
          cy="60"
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={circumference}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span ref={textRef} className="text-3xl font-extrabold" style={{ color }}>
          0
        </span>
        <span className="text-[10px] text-slate-500">/ 100</span>
      </div>
    </div>
  )
}

function MetricCard({ icon: Icon, value, label, color }) {
  const isLong = String(value).length > 10
  return (
    <div className="glass-card p-3 flex flex-col gap-2 min-w-0">
      <div className="w-9 h-9 rounded-xl flex items-center justify-center shrink-0" style={{ backgroundColor: `${color}22` }}>
        <Icon className="w-4.5 h-4.5" style={{ color }} />
      </div>
      <div className="min-w-0">
        <p className={`font-bold text-white leading-snug ${isLong ? 'text-[11px]' : 'text-sm'}`} title={String(value)}>
          {value}
        </p>
        <p className="text-[11px] text-slate-500">{label}</p>
      </div>
    </div>
  )
}

function Card({ title, children }) {
  return (
    <div className="glass-card p-5">
      <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wide mb-3">{title}</h3>
      {children}
    </div>
  )
}

const INSIGHT_COLORS = [
  { bg: 'bg-indigo-500/10', border: 'border-indigo-500/20', icon: 'text-indigo-400' },
  { bg: 'bg-purple-500/10', border: 'border-purple-500/20', icon: 'text-purple-400' },
  { bg: 'bg-cyan-500/10', border: 'border-cyan-500/20', icon: 'text-cyan-400' },
]

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div className="glass-card px-3 py-2 text-xs" style={{ background: '#1a1a2e' }}>
      <p className="text-slate-300 font-semibold mb-0.5">{label}</p>
      <p className="text-indigo-300">{payload[0].value}%</p>
    </div>
  )
}

function AnalysisResults({ result }) {
  if (!result) return null

  const profile = result.profile || {}
  const targetSuggestion = result.target_suggestion || {}
  const problemType = result.problem_type || {}
  const insights = result.insights || []
  const outliers = result.outliers?.outliers || {}
  const targetBalance = result.target_balance || {}

  const columnNames = Object.keys(profile.column_types || {})
  const constantSet = new Set(profile.constant_columns || [])
  const idLikeSet = new Set(profile.id_like_columns || [])

  const balanceData = Object.entries(targetBalance.distribution || {}).map(([name, value]) => ({
    name: name.replace(/^class_/, ''),
    value,
  }))

  const outlierEntries = Object.entries(outliers).filter(([, info]) => info.count > 0)

  const missingBarColor = (pct) => {
    if (pct > 50) return '#ef4444'
    if (pct >= 10) return '#f59e0b'
    return '#10b981'
  }

  return (
    <div className="space-y-5">
      <Card title="Dataset Profile">
        <div className="flex justify-center sm:justify-start mb-5">
          <QualityRing score={profile.quality_score ?? 0} />
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <MetricCard icon={Rows3} value={profile.rows ?? '—'} label="Rows" color="#3b82f6" />
          <MetricCard icon={Hash} value={profile.columns ?? '—'} label="Columns" color="#8b5cf6" />
          <MetricCard
            icon={TargetIcon}
            value={problemType.problem_type ? problemType.problem_type.replaceAll('_', ' ') : '—'}
            label="Problem Type"
            color="#06b6d4"
          />
          <MetricCard
            icon={TargetIcon}
            value={targetSuggestion.suggested_target ?? '—'}
            label={`Target (${targetSuggestion.confidence ?? '—'})`}
            color="#10b981"
          />
        </div>
        {targetSuggestion.reason && <p className="text-xs text-slate-500 mt-4">{targetSuggestion.reason}</p>}
      </Card>

      <Card title="Key Insights">
        {insights.length === 0 ? (
          <p className="text-sm text-slate-500">No insights available.</p>
        ) : (
          <div className="space-y-2">
            {insights.map((insight, i) => {
              const palette = INSIGHT_COLORS[i % INSIGHT_COLORS.length]
              return (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, x: -12 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.35, delay: i * 0.1 }}
                  className={`flex items-start gap-2 text-sm text-slate-200 rounded-xl border px-3 py-2 ${palette.bg} ${palette.border}`}
                >
                  <Lightbulb className={`w-4 h-4 shrink-0 mt-0.5 ${palette.icon}`} />
                  <span>{insight}</span>
                </motion.div>
              )
            })}
          </div>
        )}
      </Card>

      <Card title="Column Overview">
        <div className="overflow-x-auto">
          <table className="min-w-full text-sm">
            <thead>
              <tr className="text-left text-slate-500 border-b border-white/10">
                <th className="py-2 pr-3">Column</th>
                <th className="py-2 pr-3">Type</th>
                <th className="py-2 pr-3">Missing %</th>
                <th className="py-2 pr-3">Unique</th>
              </tr>
            </thead>
            <tbody>
              {columnNames.map((col, i) => {
                const isId = idLikeSet.has(col)
                const isConstant = constantSet.has(col)
                const missingPct = profile.missing?.[col] ?? 0
                return (
                  <tr
                    key={col}
                    className={`border-b border-white/5 hover:bg-white/5 transition-colors ${i % 2 === 0 ? 'bg-white/[0.015]' : ''}`}
                  >
                    <td className={`py-2 pr-3 font-medium ${isId ? 'text-slate-200' : isConstant ? 'text-slate-200' : 'text-slate-200'}`}>
                      {col}
                      {isId && (
                        <span className="ml-1.5 text-[10px] font-bold bg-orange-500/20 text-orange-300 px-1.5 py-0.5 rounded-full">
                          ID-LIKE
                        </span>
                      )}
                      {isConstant && (
                        <span className="ml-1.5 text-[10px] font-bold bg-orange-500/20 text-orange-300 px-1.5 py-0.5 rounded-full">
                          CONSTANT
                        </span>
                      )}
                    </td>
                    <td className="py-2 pr-3 text-slate-400">{profile.column_types[col]}</td>
                    <td className="py-2 pr-3">
                      <div className="flex items-center gap-2 w-28">
                        <div className="flex-1 h-1.5 bg-white/10 rounded-full overflow-hidden">
                          <div
                            className="h-full rounded-full"
                            style={{ width: `${Math.min(missingPct, 100)}%`, backgroundColor: missingBarColor(missingPct) }}
                          />
                        </div>
                        <span className="text-xs text-slate-500 w-9 text-right">{missingPct}%</span>
                      </div>
                    </td>
                    <td className="py-2 pr-3 text-slate-400">{profile.unique_counts?.[col] ?? '—'}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        <Card title="Target Balance">
          {balanceData.length === 0 ? (
            <p className="text-sm text-slate-500">Not applicable for this problem type.</p>
          ) : (
            <>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={balanceData}>
                  <defs>
                    <linearGradient id="balanceGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#8b5cf6" stopOpacity={1} />
                      <stop offset="100%" stopColor="#6366f1" stopOpacity={0.6} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.08)" />
                  <XAxis dataKey="name" tick={{ fontSize: 12, fill: '#94a3b8' }} />
                  <YAxis tick={{ fontSize: 12, fill: '#94a3b8' }} unit="%" />
                  <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.05)' }} />
                  <Bar dataKey="value" fill="url(#balanceGradient)" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
              {targetBalance.warning && (
                <div className="mt-3 flex items-start gap-2 text-sm text-amber-300 bg-amber-500/10 border border-amber-500/20 rounded-lg p-2">
                  <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                  <span>{targetBalance.warning}</span>
                </div>
              )}
            </>
          )}
        </Card>

        <Card title="Outliers">
          {outlierEntries.length === 0 ? (
            <p className="text-sm text-slate-500">No significant outliers detected.</p>
          ) : (
            <ul className="space-y-2 text-sm">
              {outlierEntries.map(([col, info]) => (
                <li key={col} className="flex justify-between border-b border-white/5 pb-1">
                  <span className="text-slate-200 font-medium">{col}</span>
                  <span className="text-slate-500">
                    {info.count} ({info.percentage}%)
                  </span>
                </li>
              ))}
            </ul>
          )}
        </Card>
      </div>
    </div>
  )
}

export default AnalysisResults
