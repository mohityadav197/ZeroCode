import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { Check, Lock } from 'lucide-react'

const CORE_CLASSIFICATION = [
  { key: 'accuracy', label: 'Accuracy' },
  { key: 'f1', label: 'F1 Score' },
  { key: 'precision', label: 'Precision' },
  { key: 'recall', label: 'Recall' },
]
const CORE_REGRESSION = [
  { key: 'mae', label: 'MAE' },
  { key: 'rmse', label: 'RMSE' },
  { key: 'r2', label: 'R² Score' },
]

const EXTENDED_CLASSIFICATION = [
  { key: 'roc_auc', label: 'ROC-AUC Score', description: 'Area under the ROC curve', tooltip: 'Best for binary classification and imbalanced datasets' },
  { key: 'log_loss', label: 'Log Loss', description: 'Probability calibration measure', tooltip: 'Use when prediction confidence matters, not just right/wrong' },
  { key: 'mcc', label: 'Matthews Correlation (MCC)', description: 'Reliable metric for imbalanced data', tooltip: 'Use when classes are very imbalanced' },
  { key: 'cohen_kappa', label: "Cohen's Kappa", description: 'Agreement beyond chance', tooltip: 'Use to see how much better than random guessing the model is' },
  { key: 'balanced_accuracy', label: 'Balanced Accuracy', description: 'Better than accuracy for imbalanced', tooltip: 'Use instead of accuracy when one class dominates the dataset' },
  { key: 'confusion_matrix', label: 'Confusion Matrix', description: 'Visual breakdown of predictions', tooltip: 'Use to see exactly which classes get confused with each other' },
]

const EXTENDED_REGRESSION = [
  { key: 'mape', label: 'MAPE', description: 'Mean absolute percentage error', tooltip: 'Use for errors expressed as a percentage rather than raw units' },
  { key: 'explained_variance', label: 'Explained Variance', description: 'Variance explained by model', tooltip: 'Use alongside R² to spot bias in predictions' },
  { key: 'max_error', label: 'Max Error', description: 'Worst case prediction error', tooltip: 'Use when the single worst mistake matters most' },
  { key: 'median_ae', label: 'Median Absolute Error', description: 'Robust to outliers', tooltip: 'Use when extreme outliers are skewing MAE' },
  { key: 'adjusted_r2', label: 'Adjusted R²', description: 'R² penalized for extra features', tooltip: 'Use when comparing models with different numbers of features' },
]

function CoreMetricRow({ label }) {
  return (
    <div className="flex items-center justify-between glass-card px-3 py-2.5 opacity-80">
      <span className="text-sm font-medium text-slate-300 flex items-center gap-2">
        <Lock className="w-3.5 h-3.5 text-slate-500" />
        {label}
      </span>
      <div className="relative w-9 h-5 rounded-full bg-white/10 shrink-0">
        <span className="absolute top-0.5 left-4 w-4 h-4 rounded-full bg-slate-500" />
      </div>
    </div>
  )
}

function ExtendedMetricCard({ metric, checked, onToggle, index }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: index * 0.05 }}
      onClick={onToggle}
      className={`group relative glass-card p-3 cursor-pointer transition-shadow duration-200 ${
        checked
          ? 'border-purple-400/60 shadow-[0_0_16px_rgba(139,92,246,0.35)] bg-purple-500/[0.06]'
          : 'hover:shadow-[0_6px_20px_rgba(99,102,241,0.12)]'
      }`}
    >
      <div className="flex items-start gap-3">
        <span
          className={`mt-0.5 w-5 h-5 rounded-md border flex items-center justify-center shrink-0 transition-colors ${
            checked ? 'bg-gradient-to-br from-indigo-500 to-purple-500 border-transparent' : 'border-white/20'
          }`}
        >
          {checked && <Check className="w-3.5 h-3.5 text-white" />}
        </span>
        <div className="min-w-0">
          <p className="text-sm font-semibold text-slate-100">{metric.label}</p>
          <p className="text-xs text-slate-500 mt-0.5">{metric.description}</p>
        </div>
      </div>

      <div className="pointer-events-none absolute left-3 right-3 bottom-full mb-2 opacity-0 group-hover:opacity-100 transition-opacity z-10">
        <div className="glass-card px-2.5 py-1.5 text-[11px] text-slate-300" style={{ backgroundColor: '#14142a' }}>
          {metric.tooltip}
        </div>
      </div>
    </motion.div>
  )
}

function MetricsSelector({ problemType, onSelectionChange }) {
  const isRegression = problemType === 'regression'
  const coreMetrics = isRegression ? CORE_REGRESSION : CORE_CLASSIFICATION
  const extendedMetrics = isRegression ? EXTENDED_REGRESSION : EXTENDED_CLASSIFICATION

  const [selected, setSelected] = useState(new Set())

  useEffect(() => {
    setSelected(new Set())
  }, [problemType])

  useEffect(() => {
    onSelectionChange?.(Array.from(selected))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected])

  const toggle = (key) => {
    setSelected((prev) => {
      const next = new Set(prev)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })
  }

  const selectAll = () => setSelected(new Set(extendedMetrics.map((m) => m.key)))
  const clearAll = () => setSelected(new Set())

  return (
    <div className="glass-card p-5 space-y-4">
      <div>
        <h3 className="text-lg font-bold text-white">📊 Evaluation Metrics</h3>
        <p className="text-sm text-slate-500">Choose how to evaluate your models</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-500 mb-2.5">Core — Always Evaluated</p>
          <div className="space-y-2">
            {coreMetrics.map((m) => (
              <CoreMetricRow key={m.key} label={m.label} />
            ))}
          </div>
        </div>

        <div>
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-500 mb-2.5">Extended — Select What You Need</p>
          <div className="space-y-2">
            {extendedMetrics.map((m, i) => (
              <ExtendedMetricCard
                key={m.key}
                metric={m}
                index={i}
                checked={selected.has(m.key)}
                onToggle={() => toggle(m.key)}
              />
            ))}
          </div>
        </div>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3 pt-3 border-t border-white/10">
        <div className="flex items-center gap-4">
          <button onClick={selectAll} className="text-xs font-semibold text-indigo-300 hover:text-indigo-200 transition-colors">
            Select All Extended
          </button>
          <button onClick={clearAll} className="text-xs font-semibold text-slate-400 hover:text-slate-300 transition-colors">
            Clear All
          </button>
        </div>
        <p className="text-xs text-slate-500">
          {coreMetrics.length} core + {selected.size} extended metric{selected.size === 1 ? '' : 's'} selected
        </p>
      </div>
    </div>
  )
}

export default MetricsSelector
