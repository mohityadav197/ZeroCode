import { useEffect, useMemo, useRef, useState } from 'react'
import { animate, motion } from 'framer-motion'
import { ArrowRight } from 'lucide-react'

const MODEL_STYLE = {
  'Random Forest': { color: '#10b981', initial: 'R' },
  'Random Forest Regressor': { color: '#10b981', initial: 'R' },
  XGBoost: { color: '#f97316', initial: 'X' },
  'XGBoost Regressor': { color: '#f97316', initial: 'X' },
  'Logistic Regression': { color: '#3b82f6', initial: 'L' },
  'Linear Regression': { color: '#3b82f6', initial: 'L' },
  SVM: { color: '#8b5cf6', initial: 'S' },
  SVR: { color: '#8b5cf6', initial: 'S' },
  'Decision Tree': { color: '#06b6d4', initial: 'D' },
  'Decision Tree Regressor': { color: '#06b6d4', initial: 'D' },
  KNN: { color: '#ec4899', initial: 'K' },
  'KNN Regressor': { color: '#ec4899', initial: 'K' },
  AdaBoost: { color: '#f59e0b', initial: 'A' },
  'AdaBoost Regressor': { color: '#f59e0b', initial: 'A' },
}

function styleFor(name) {
  return MODEL_STYLE[name] || { color: '#6366f1', initial: name.charAt(0).toUpperCase() }
}

function AnimatedCount({ value }) {
  const ref = useRef(null)
  const prevValue = useRef(value)

  useEffect(() => {
    const controls = animate(prevValue.current, value, {
      duration: 0.4,
      ease: 'easeOut',
      onUpdate(v) {
        if (ref.current) ref.current.textContent = Math.round(v)
      },
    })
    prevValue.current = value
    return () => controls.stop()
  }, [value])

  return <span ref={ref}>{value}</span>
}

function Toggle({ checked, onChange, activeColor }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      onClick={onChange}
      className="relative w-11 h-6 rounded-full shrink-0 transition-colors duration-200"
      style={{ backgroundColor: checked ? activeColor : 'rgba(255,255,255,0.15)' }}
    >
      <motion.span
        layout
        transition={{ type: 'spring', stiffness: 500, damping: 32 }}
        className="absolute top-0.5 w-5 h-5 rounded-full bg-white shadow"
        style={{ left: checked ? '22px' : '2px' }}
      />
    </button>
  )
}

function ModelCard({ model, checked, onToggle, index, glowColor, showBadge }) {
  const { color, initial } = styleFor(model.name)

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, delay: index * 0.06 }}
      whileHover={{ y: -3 }}
      className="relative glass-card p-4 flex items-center gap-4 border transition-shadow duration-200"
      style={{
        borderColor: checked ? glowColor : 'rgba(255,255,255,0.1)',
        boxShadow: checked ? `0 0 20px ${glowColor}55` : 'none',
      }}
    >
      {showBadge && (
        <span className="absolute -top-2.5 -right-2.5 text-[10px] font-bold bg-gradient-to-r from-amber-400 to-yellow-500 text-black px-2 py-0.5 rounded-full shadow">
          ⭐ Recommended
        </span>
      )}
      <div
        className="w-10 h-10 rounded-full flex items-center justify-center font-bold text-white shrink-0"
        style={{ backgroundColor: color }}
      >
        {initial}
      </div>
      <div className="flex-1 min-w-0">
        <p className="font-bold text-white truncate">{model.name}</p>
        <p className="text-xs text-slate-400 mt-0.5">{model.reason}</p>
      </div>
      <Toggle checked={checked} onChange={onToggle} activeColor={glowColor} />
    </motion.div>
  )
}

function Pill({ label, color }) {
  return (
    <span
      className="text-xs font-semibold px-2.5 py-1 rounded-full border"
      style={{ backgroundColor: `${color}1a`, color, borderColor: `${color}40` }}
    >
      {label}
    </span>
  )
}

function ModelRecommendations({ recommendations, onSelectionChange, onContinue, loading, confirmed }) {
  const recommended = recommendations?.recommended || []
  const optional = recommendations?.optional || []
  const summary = recommendations?.data_summary || {}

  const [selected, setSelected] = useState(() => new Set(recommended.filter((m) => m.pre_selected).map((m) => m.name)))

  useEffect(() => {
    setSelected(new Set(recommended.filter((m) => m.pre_selected).map((m) => m.name)))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [recommendations])

  useEffect(() => {
    onSelectionChange?.(Array.from(selected))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected])

  const toggle = (name) => {
    setSelected((prev) => {
      const next = new Set(prev)
      if (next.has(name)) next.delete(name)
      else next.add(name)
      return next
    })
  }

  const selectedCount = selected.size
  const estimatedSeconds = useMemo(() => selectedCount * 15, [selectedCount])

  return (
    <div className="glass-card p-5 space-y-6">
      <div>
        <h3 className="text-lg font-bold text-white">🤖 Model Recommendations</h3>
        <p className="text-sm text-slate-500 mb-3">Based on your data characteristics</p>
        <div className="flex flex-wrap gap-2">
          {summary.rows != null && <Pill label={`${summary.rows} Rows`} color="#3b82f6" />}
          {summary.size_category && (
            <Pill label={`${summary.size_category.charAt(0).toUpperCase()}${summary.size_category.slice(1)} Dataset`} color="#8b5cf6" />
          )}
          {summary.has_outliers && <Pill label="Outliers Detected" color="#f97316" />}
          {summary.is_imbalanced && <Pill label="Imbalanced Target" color="#ef4444" />}
        </div>
      </div>

      {recommended.length > 0 && (
        <div>
          <h4 className="text-sm font-semibold text-slate-300 mb-3">⭐ Recommended for Your Data</h4>
          <div className="space-y-3">
            {recommended.map((model, i) => (
              <ModelCard
                key={model.name}
                model={model}
                index={i}
                checked={selected.has(model.name)}
                onToggle={() => toggle(model.name)}
                glowColor="#10b981"
                showBadge
              />
            ))}
          </div>
        </div>
      )}

      {optional.length > 0 && (
        <div>
          <h4 className="text-sm font-semibold text-slate-300 mb-3">➕ Add More Models</h4>
          <div className="space-y-3">
            {optional.map((model, i) => (
              <ModelCard
                key={model.name}
                model={model}
                index={i}
                checked={selected.has(model.name)}
                onToggle={() => toggle(model.name)}
                glowColor="#8b5cf6"
                showBadge={false}
              />
            ))}
          </div>
        </div>
      )}

      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 pt-2 border-t border-white/10">
        <div className="text-sm text-slate-400">
          <span className="text-white font-semibold">
            <AnimatedCount value={selectedCount} /> model{selectedCount === 1 ? '' : 's'} selected
          </span>
          <span className="mx-2 text-slate-600">·</span>
          Estimated training time: ~{estimatedSeconds} seconds
        </div>
        <button
          onClick={() => onContinue?.(Array.from(selected))}
          disabled={selectedCount === 0 || loading || confirmed}
          className="gradient-btn flex items-center justify-center gap-2 rounded-full shrink-0"
        >
          {confirmed ? 'Models Confirmed' : 'Continue'}
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>
    </div>
  )
}

export default ModelRecommendations
