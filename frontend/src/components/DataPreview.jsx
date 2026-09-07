function DataPreview({ data }) {
  if (!data) return null

  const { rows, columns, column_names: columnNames = [], preview = [] } = data

  return (
    <div className="w-full">
      <div className="flex gap-6 mb-3 text-sm text-slate-400">
        <span>
          <strong className="text-white">{rows}</strong> rows
        </span>
        <span>
          <strong className="text-white">{columns}</strong> columns
        </span>
      </div>
      <div className="overflow-x-auto border border-white/10 rounded-lg">
        <table className="min-w-full text-sm">
          <thead className="bg-white/5">
            <tr>
              {columnNames.map((col) => (
                <th
                  key={col}
                  className="px-3 py-2 text-left font-semibold text-slate-300 whitespace-nowrap border-b border-white/10"
                >
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {preview.slice(0, 5).map((row, i) => (
              <tr key={i} className={i % 2 === 0 ? 'bg-transparent' : 'bg-white/[0.03]'}>
                {columnNames.map((col) => (
                  <td key={col} className="px-3 py-2 whitespace-nowrap text-slate-400 border-b border-white/5">
                    {String(row[col] ?? '')}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export default DataPreview
