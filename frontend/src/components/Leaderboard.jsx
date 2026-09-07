import { useMemo, useState } from 'react'
import { ArrowDown, ArrowUp, Trophy } from 'lucide-react'

function metricPillClass(key, value) {
  if (!['accuracy', 'f1', 'precision', 'recall', 'r2'].includes(key)) {
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

  const columns = isRegression
    ? [
        { key: 'rank', label: 'Rank' },
        { key: 'model', label: 'Model' },
        { key: 'mae', label: 'MAE' },
        { key: 'rmse', label: 'RMSE' },
        { key: 'r2', label: 'R2' },
        { key: 'training_time', label: 'Time' },
      ]
    : [
        { key: 'rank', label: 'Rank' },
        { key: 'model', label: 'Model' },
        { key: 'accuracy', label: 'Accuracy' },
        { key: 'f1', label: 'F1' },
        { key: 'precision', label: 'Precision' },
        { key: 'recall', label: 'Recall' },
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
                      {row[col.key]}
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
