import { useState } from 'react'
import { createPortal } from 'react-dom'
import { AnimatePresence, motion } from 'framer-motion'
import { ArrowRight, BarChart2, Check, GitBranch, Microscope, PieChart, Target } from 'lucide-react'

const SECTIONS = [
  {
    title: 'Distribution Analysis',
    icon: BarChart2,
    options: [
      { id: 'histograms', label: 'Histograms', description: 'Distribution of numeric columns', checked: true },
      { id: 'boxplots', label: 'Box Plots', description: 'Outlier detection', checked: true },
      { id: 'bar_charts', label: 'Bar Charts', description: 'Categorical column frequencies', checked: true },
    ],
  },
  {
    title: 'Relationships',
    icon: GitBranch,
    options: [
      { id: 'correlation_heatmap', label: 'Correlation Heatmap', description: 'Feature relationships', checked: true },
      { id: 'pair_plot', label: 'Pair Plot', description: 'Slow for many columns', checked: false },
    ],
  },
  {
    title: 'Target Analysis',
    icon: Target,
    options: [
      { id: 'target_distribution', label: 'Target Distribution', description: 'Class balance check', checked: true },
      { id: 'target_vs_features', label: 'Target vs Features', description: 'Predictive power', checked: true },
    ],
  },
  {
    title: 'Advanced',
    icon: Microscope,
    options: [
      { id: 'missing_heatmap', label: 'Missing Values Heatmap', description: 'Visualize gaps', checked: false },
      { id: 'skewness_chart', label: 'Skewness Chart', description: 'Distribution skew', checked: false },
    ],
  },
]

const ALL_IDS = SECTIONS.flatMap((s) => s.options.map((o) => o.id))
const DEFAULT_SELECTED = new Set(SECTIONS.flatMap((s) => s.options.filter((o) => o.checked).map((o) => o.id)))

function Checkbox({ checked, onChange }) {
  return (
    <button
      type="button"
      onClick={onChange}
      className={`w-5 h-5 rounded-md border-2 flex items-center justify-center shrink-0 transition-all duration-200 ${
        checked
          ? 'bg-gradient-to-br from-indigo-500 to-purple-500 border-transparent'
          : 'bg-transparent border-slate-600'
      }`}
    >
      {checked && <Check className="w-3.5 h-3.5 text-white" strokeWidth={3} />}
    </button>
  )
}

function VisualChecklist({ open, onRunAnalysis }) {
  const [selected, setSelected] = useState(DEFAULT_SELECTED)

  const toggle = (id) => {
    setSelected((prev) => {
      const next = new Set(prev)
      next.has(id) ? next.delete(id) : next.add(id)
      return next
    })
  }

  const selectAll = () => setSelected(new Set(ALL_IDS))
  const clearAll = () => setSelected(new Set())

  return createPortal(
    <AnimatePresence>
      {open && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 flex items-end sm:items-center justify-center p-4"
          style={{ background: 'rgba(0,0,0,0.7)' }}
        >
          <motion.div
            initial={{ opacity: 0, y: 60 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 40 }}
            transition={{ duration: 0.35, ease: 'easeOut' }}
            className="glass-card w-full max-w-[600px] max-h-[85vh] overflow-y-auto p-6"
          >
            <div className="flex items-center gap-2 mb-1">
              <PieChart className="w-5 h-5 text-indigo-400" />
              <h2 className="text-2xl font-bold text-white">Choose Your Visualisations</h2>
            </div>
            <p className="text-sm text-slate-400 mb-6">Select what you want to explore about your data</p>

            <div className="space-y-4">
              {SECTIONS.map((section) => {
                const SectionIcon = section.icon
                return (
                  <div key={section.title} className="glass-card p-4">
                    <div className="flex items-center gap-2 mb-3">
                      <SectionIcon className="w-4 h-4 text-indigo-300" />
                      <h3 className="text-xs font-bold uppercase tracking-wide text-indigo-300">{section.title}</h3>
                    </div>
                    <div className="space-y-1">
                      {section.options.map((option) => (
                        <div
                          key={option.id}
                          onClick={() => toggle(option.id)}
                          className="flex items-center gap-3 px-2 py-2 rounded-lg hover:bg-purple-500/10 transition-colors cursor-pointer"
                        >
                          <Checkbox checked={selected.has(option.id)} onChange={() => toggle(option.id)} />
                          <div className="min-w-0">
                            <p className="text-sm text-slate-200">{option.label}</p>
                            <p className="text-xs text-slate-500">{option.description}</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )
              })}
            </div>

            <div className="mt-6 flex items-center justify-between flex-wrap gap-3">
              <div className="flex items-center gap-4 text-xs">
                <button onClick={selectAll} className="text-indigo-300 hover:text-indigo-200 font-medium">
                  Select All
                </button>
                <button onClick={clearAll} className="text-slate-400 hover:text-slate-300 font-medium">
                  Clear All
                </button>
              </div>
              <button
                onClick={() => onRunAnalysis?.(Array.from(selected))}
                disabled={selected.size === 0}
                className="gradient-btn flex items-center gap-2 rounded-full text-sm"
              >
                Run Analysis {selected.size > 0 ? `(${selected.size} selected)` : ''}
                <ArrowRight className="w-4 h-4" />
              </button>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>,
    document.body
  )
}

export default VisualChecklist
