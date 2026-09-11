import { useMemo, useState } from 'react'
import { ArrowDown, ArrowUp, Trophy } from 'lucide-react'

const METRIC_LABELS = {
  accuracy: 'Accuracy',
  f1: 'F1',
  precision: 'Precision',
  recall: 'Recall',
  roc_auc: 'ROC-AUC',
  log_loss: 'Log Loss',
  mcc: 'MCC',
  cohen_kappa: "Cohen's Kappa",
  balanced_accuracy: 'Balanced Acc.',
  mae: 'MAE',
  rmse: 'RMSE',
  r2: 'R2',
  mape: 'MAPE',
  explained_variance: 'Expl. Variance',
  max_error: 'Max Error',
  median_ae: 'Median AE',
  adjusted_r2: 'Adj. R2',
}

const CLASSIFICATION_METRIC_ORDER = [
  'accuracy', 'f1', 'precision', 'recall', 'roc_auc', 'log_loss', 'mcc', 'cohen_kappa', 'balanced_accuracy',
]
const REGRESSION_METRIC_ORDER = ['mae', 'rmse', 'r2', 'mape', 'explained_variance', 'max_error', 'median_ae', 'adjusted_r2']

// Higher-is-better, roughly 0-1 bounded metrics — eligible for green/yellow/red pills.
// Error-scale metrics (mae, rmse, log_loss, mape, max_error, median_ae) are left neutral
// since "> 0.8 = good" would be meaningless (and often backwards) on their scale.
const SCORE_METRICS = new Set([
  'accuracy', 'f1', 'precision', 'recall', 'r2', 'roc_auc', 'mcc', 'cohen_kappa', 'balanced_accuracy',
  'explained_variance', 'adjusted_r2',
])

function metricPillClass(key, value) {
  if (!SCORE_METRICS.has(key) || typeof value !== 'number') {
    return 'bg-white/10 text-slate-300'
  }
  if (value > 0.8) return 'bg-emerald-500/20 text-emerald-300'
  if (value >= 0.6) return 'bg-amber-500/20 text-amber-300'
  return 'bg-red-500/20 text-red-300'
}

function Leaderboard({ leaderboard = [] }) {
  const [sortKey, setSortKey] = useState(null)
  const [sortDir, setSortDir] = useState('desc')

  const isRegression = leaderboard.length > 0 && 'r2' in leaderboard[0]
  const metricOrder = isRegression ? REGRESSION_METRIC_ORDER : CLASSIFICATION_METRIC_ORDER
  const presentMetrics = metricOrder.filter((key) => leaderboard.some((row) => row[key] !== undefined))

  const columns = [
    { key: 'rank', label: 'Rank' },
    { key: 'model', label: 'Model' },
    ...presentMetrics.map((key) => ({ key, label: METRIC_LABELS[key] || key })),
    { key: 'training_time', label: 'Time' },
  ]

  const sortedData = useMemo(() => {
    if (!sortKey) return leaderboard
    const copy = [...leaderboard]
    copy.sort((a, b) => {
      const av = a[sortKey]
      const bv = b[sortKey]
      if (typeof av === 'string') {
        return sortDir === 'asc' ? av.localeCompare(bv) : bv.localeCompare(av)
      }
      return sortDir === 'asc' ? av - bv : bv - av
    })
    return copy
  }, [leaderboard, sortKey, sortDir])

  const handleSort = (key) => {
    if (sortKey === key) {
      setSortDir((prev) => (prev === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortKey(key)
      setSortDir('desc')
    }
  }

  if (leaderboard.length === 0) {
    return <p className="text-sm text-slate-500">No models were successfully trained.</p>
  }

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full text-sm border-separate border-spacing-y-1">
        <thead>
          <tr className="text-left text-slate-500">
            {columns.map((col) => (
              <th
                key={col.key}
                onClick={() => handleSort(col.key)}
                className="py-2 pr-4 cursor-pointer select-none hover:text-white transition-colors"
              >
                <span className="inline-flex items-center gap-1">
                  {col.label}
                  {sortKey === col.key &&
                    (sortDir === 'asc' ? (
                      <ArrowUp className="w-3 h-3 text-indigo-400 transition-transform duration-200" />
                    ) : (
                      <ArrowDown className="w-3 h-3 text-indigo-400 transition-transform duration-200" />
                    ))}
                </span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sortedData.map((row) => (
            <tr
              key={row.model}
              className={`transition-transform duration-200 hover:-translate-y-0.5 ${
                row.is_best
                  ? 'bg-gradient-to-r from-emerald-500/25 to-emerald-500/5 font-semibold'
                  : 'bg-white/[0.03]'
              }`}
            >
              {columns.map((col) => (
                <td key={col.key} className="py-2.5 pr-4 first:rounded-l-lg last:rounded-r-lg">
                  {col.key === 'model' ? (
                    <span className="text-slate-100">
                      {row.is_best && <Trophy className="inline w-4 h-4 text-yellow-400 mr-1" />}
                      {row[col.key]}
                    </span>
                  ) : col.key === 'rank' || col.key === 'training_time' ? (
                    <span className="text-slate-400">{row[col.key]}</span>
                  ) : (
                    <span className={`text-xs font-semibold px-2 py-1 rounded-full ${metricPillClass(col.key, row[col.key])}`}>
                      {row[col.key] ?? '—'}
                    </span>
                  )}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default Leaderboard
