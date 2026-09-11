import { useEffect, useMemo, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import {
  AlertTriangle,
  ArrowRight,
  Check,
  ChevronDown,
  Lock,
  RefreshCw,
  Ruler,
  Sparkles,
  TrendingDown,
  Wrench,
} from 'lucide-react'

const ACTION_LABELS = {
  none: 'None',
  drop: 'Drop',
  fill_median: 'Fill Median',
  fill_mean: 'Fill Mean',
  fill_zero: 'Fill Zero',
  fill_mode: 'Fill Mode',
  fill_unknown: 'Fill Unknown',
  binary_encode: 'Binary Encode',
  onehot_encode: 'One-Hot Encode',
  label_encode: 'Label Encode',
  keep: 'Keep',
  cap_iqr: 'Cap (IQR)',
  log_transform: 'Log Transform',
  standard_scale: 'Standard Scale',
  minmax_scale: 'MinMax Scale',
  target: 'Target',
}

const REC_META = {
  missing: { icon: Wrench, label: 'Missing Values' },
  encoding: { icon: RefreshCw, label: 'Encoding' },
  outliers: { icon: TrendingDown, label: 'Outliers' },
  scaling: { icon: Ruler, label: 'Scaling' },
}

const REC_ORDER = ['missing', 'encoding', 'outliers', 'scaling']

const SECTION_META = {
  warnings: { label: '⚠️ Warnings', color: '#f59e0b' },
  drop: { label: '🗑️ Columns to Drop', color: '#ef4444' },
  encode: { label: '🔄 Columns to Encode', color: '#8b5cf6' },
  scale: { label: '📏 Columns to Scale', color: '#3b82f6' },
  none: { label: '✅ No Action Needed', color: '#10b981' },
}
const SECTION_ORDER = ['warnings', 'drop', 'encode', 'scale', 'none']

function actionLabel(action) {
  return ACTION_LABELS[action] || action
}

function warningReason(colData) {
  const recs = colData.recommendations || {}
  const entry = REC_ORDER.map((t) => recs[t]).find((r) => r?.warning)
  return entry?.reason || 'Needs attention'
}

function categorize(colData) {
  const recs = colData.recommendations || {}
  const hasWarning = Object.values(recs).some((r) => r?.warning)
  if (hasWarning) return 'warnings'
  if (recs.missing?.action === 'drop' || colData.is_id_like || colData.is_constant) return 'drop'
  if (recs.encoding && recs.encoding.action !== 'none') return 'encode'
  if (recs.scaling && recs.scaling.action !== 'none') return 'scale'
  return 'none'
}

function StatCard({ label, value, color }) {
  return (
    <div className="glass-card p-3 flex flex-col gap-1" style={{ borderColor: `${color}33` }}>
      <span className="text-2xl font-extrabold" style={{ color }}>
        {value}
      </span>
      <span className="text-xs text-slate-500">{label}</span>
    </div>
  )
}

function OptionButtonGroup({ options, current, aiAction, disabled, onSelect }) {
  if (!options || options.length === 0) return null
  return (
    <div className="flex flex-wrap gap-1.5">
      {options.map((opt) => {
        const isSelected = current === opt
        const isAi = opt === aiAction
        return (
          <button
            key={opt}
            type="button"
            disabled={disabled}
            onClick={() => onSelect(opt)}
            className={`relative text-xs font-semibold px-2.5 py-1.5 rounded-lg border transition-colors ${
              disabled
                ? 'opacity-40 cursor-not-allowed border-white/10 text-slate-500'
                : isSelected
                ? 'bg-gradient-to-r from-indigo-500 to-purple-500 border-transparent text-white'
                : 'border-white/10 text-slate-300 hover:border-white/25 hover:text-white'
            }`}
          >
            {isAi && (
              <span className="absolute -top-2 -right-1.5 text-[8px] font-bold bg-cyan-500 text-white px-1 py-0.5 rounded-full leading-none">
                AI
              </span>
            )}
            {actionLabel(opt)}
            {isSelected && <Check className="inline w-3 h-3 ml-1 -mt-0.5" />}
          </button>
        )
      })}
    </div>
  )
}

function RecommendationRow({ recType, rec, current, onSelect }) {
  if (!rec) return null
  const meta = REC_META[recType]
  const Icon = meta.icon

  return (
    <div className="py-2.5 border-t border-white/5 first:border-t-0 first:pt-0">
      <div className="flex items-center gap-2 mb-1">
        <Icon className={`w-3.5 h-3.5 ${rec.warning ? 'text-amber-400' : 'text-indigo-400'}`} />
        <span className="text-xs font-semibold text-slate-300">{meta.label}</span>
        {rec.warning && (
          <span className="text-[10px] font-bold bg-amber-500/20 text-amber-300 px-1.5 py-0.5 rounded-full">
            Warning
          </span>
        )}
      </div>
      <p className="text-xs text-slate-500 mb-2">{rec.reason}</p>
      {rec.options?.length > 0 ? (
        <OptionButtonGroup options={rec.options} current={current ?? rec.action} aiAction={rec.action} onSelect={(opt) => onSelect(recType, opt, rec.action)} />
      ) : (
        <span className="text-xs font-semibold text-slate-400">{actionLabel(rec.action)}</span>
      )}
    </div>
  )
}

function ColumnCard({ name, data, choices, onOverride }) {
  const recs = data.recommendations || {}

  if (data.locked) {
    const reason = recs.missing?.reason || recs.target?.reason || ''
    return (
      <div className="glass-card p-4 opacity-70" title="Auto-managed by ZeroCode">
        <div className="flex items-center gap-2 mb-1.5">
          <Lock className="w-3.5 h-3.5 text-slate-500" />
          <span className="font-bold text-slate-300">{name}</span>
          <span className="text-[10px] font-bold uppercase bg-white/10 text-slate-400 px-1.5 py-0.5 rounded-full">
            {data.dtype}
          </span>
          <span className="text-[10px] font-bold bg-slate-500/20 text-slate-400 px-1.5 py-0.5 rounded-full ml-auto">
            {data.is_target ? 'Target' : 'Drop'}
          </span>
        </div>
        <p className="text-xs text-slate-500">{reason}</p>
      </div>
    )
  }

  return (
    <div className="glass-card p-4">
      <div className="flex items-center gap-2 flex-wrap mb-1">
        <span className="font-bold text-white">{name}</span>
        <span className="text-[10px] font-bold uppercase bg-white/10 text-slate-400 px-1.5 py-0.5 rounded-full">
          {data.dtype}
        </span>
        {data.missing_pct > 0 && (
          <span className="text-[10px] font-bold bg-orange-500/15 text-orange-300 px-1.5 py-0.5 rounded-full">
            {data.missing_pct}% missing
          </span>
        )}
        {data.outlier_pct > 0 && (
          <span className="text-[10px] font-bold bg-purple-500/15 text-purple-300 px-1.5 py-0.5 rounded-full">
            {data.outlier_pct}% outliers
          </span>
        )}
      </div>
      <div>
        {REC_ORDER.filter((t) => recs[t]).map((recType) => (
          <RecommendationRow
            key={recType}
            recType={recType}
            rec={recs[recType]}
            current={choices?.[recType]}
            onSelect={onOverride}
          />
        ))}
      </div>
    </div>
  )
}

function PreprocessingPanel({ recommendations, onChoicesChange, onApply, loading }) {
  const columns = recommendations?.columns || {}
  const summary = recommendations?.summary || {}

  const [choices, setChoices] = useState({})
  const [expanded, setExpanded] = useState(false)

  useEffect(() => {
    setChoices({})
    setExpanded(false)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [recommendations])

  useEffect(() => {
    onChoicesChange?.(choices)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [choices])

  const handleOverride = (colName) => (recType, action, aiAction) => {
    setChoices((prev) => {
      const colChoices = { ...(prev[colName] || {}) }
      if (action === aiAction) {
        delete colChoices[recType]
      } else {
        colChoices[recType] = action
      }
      const next = { ...prev }
      if (Object.keys(colChoices).length === 0) {
        delete next[colName]
      } else {
        next[colName] = colChoices
      }
      return next
    })
  }

  const resetAll = () => setChoices({})

  const acceptAll = () => {
    setChoices({})
    onApply?.()
  }

  const overrideList = useMemo(() => {
    const entries = []
    for (const [colName, colChoices] of Object.entries(choices)) {
      const recs = columns[colName]?.recommendations || {}
      for (const [recType, action] of Object.entries(colChoices)) {
        entries.push({ colName, recType, action, wasAction: recs[recType]?.action })
      }
    }
    return entries
  }, [choices, columns])

  const sections = useMemo(() => {
    const grouped = { warnings: [], drop: [], encode: [], scale: [], none: [] }
    for (const [name, data] of Object.entries(columns)) {
      if (data.is_target) continue
      grouped[categorize(data)].push([name, data])
    }
    return grouped
  }, [columns])

  const warningColumns = sections.warnings

  if (!recommendations) return null

  return (
    <div className="glass-card p-5 space-y-5">
      <div>
        <h3 className="text-lg font-bold text-white">⚙️ Smart Preprocessing</h3>
        <p className="text-sm text-slate-500">AI has analyzed each column. Review and customize before training.</p>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <StatCard label="Columns to Drop" value={summary.columns_to_drop ?? 0} color="#ef4444" />
        <StatCard label="Columns to Encode" value={summary.columns_to_encode ?? 0} color="#8b5cf6" />
        <StatCard label="Columns to Scale" value={summary.columns_to_scale ?? 0} color="#3b82f6" />
        <StatCard label="Warnings" value={summary.warnings ?? 0} color="#f59e0b" />
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <button
          onClick={acceptAll}
          disabled={loading}
          className="flex items-center gap-1.5 text-sm font-semibold px-3.5 py-2 rounded-lg bg-emerald-500/15 text-emerald-300 border border-emerald-500/30 hover:bg-emerald-500/25 transition-colors"
        >
          <Sparkles className="w-4 h-4" />
          Accept All AI Recommendations
        </button>
        <button
          onClick={() => setExpanded((v) => !v)}
          className="flex items-center gap-1.5 text-sm font-semibold px-3.5 py-2 rounded-lg glass-card text-slate-200 hover:text-white transition-colors"
        >
          Customize
          <ChevronDown className={`w-4 h-4 transition-transform duration-200 ${expanded ? 'rotate-180' : ''}`} />
        </button>
      </div>

      {!expanded && (
        <div className="space-y-3">
          {warningColumns.length > 0 && (
            <div className="space-y-2">
              {warningColumns.map(([name, data]) => (
                <div key={name} className="flex items-start gap-2 text-sm bg-amber-500/10 border border-amber-500/20 rounded-lg px-3 py-2">
                  <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
                  <span className="text-slate-200">
                    <strong>{name}</strong>: {warningReason(data)}
                  </span>
                </div>
              ))}
            </div>
          )}
          <p className="text-sm text-slate-400">
            {summary.columns_to_drop ?? 0} columns will be dropped, {summary.columns_to_encode ?? 0} encoded,{' '}
            {summary.columns_to_scale ?? 0} scaled based on AI recommendations.
          </p>
          <button
            onClick={() => setExpanded(true)}
            className="text-sm font-semibold text-indigo-300 hover:text-indigo-200 transition-colors"
          >
            Review Details ▼
          </button>
        </div>
      )}

      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25 }}
            className="overflow-hidden"
          >
            <div className="space-y-6">
              {SECTION_ORDER.map((key) => {
                const cols = sections[key]
                if (!cols || cols.length === 0) return null
                return (
                  <div key={key}>
                    <h4 className="text-sm font-semibold text-slate-300 mb-3">{SECTION_META[key].label}</h4>
                    <div className="space-y-3">
                      {cols.map(([name, data]) => (
                        <ColumnCard
                          key={name}
                          name={name}
                          data={data}
                          choices={choices[name]}
                          onOverride={handleOverride(name)}
                        />
                      ))}
                    </div>
                  </div>
                )
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="pt-3 border-t border-white/10 space-y-3">
        {overrideList.length > 0 && (
          <div>
            <p className="text-sm font-semibold text-slate-200 mb-1.5">
              You changed {overrideList.length} AI recommendation{overrideList.length === 1 ? '' : 's'}
            </p>
            <ul className="space-y-1">
              {overrideList.map(({ colName, recType, action, wasAction }) => (
                <li key={`${colName}-${recType}`} className="text-xs text-slate-400">
                  <span className="text-slate-200 font-medium">{colName}</span> ({REC_META[recType].label}):{' '}
                  {actionLabel(action)} <span className="text-slate-600">(was: {actionLabel(wasAction)})</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
          <button
            onClick={resetAll}
            disabled={overrideList.length === 0 || loading}
            className="flex items-center gap-1.5 text-sm font-semibold text-slate-400 hover:text-slate-200 transition-colors disabled:opacity-40"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            Reset to AI Recommendations
          </button>
          <button
            onClick={() => onApply?.()}
            disabled={loading}
            className="gradient-btn flex items-center justify-center gap-2 rounded-full shrink-0"
          >
            Apply &amp; Continue
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  )
}

export default PreprocessingPanel
