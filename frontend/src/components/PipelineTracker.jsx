import { motion } from 'framer-motion'
import { Check, Loader2, X } from 'lucide-react'

const PILL_STYLES = {
  waiting: 'border-dashed border-slate-600 bg-white/[0.03] text-slate-400',
  running: 'border-solid border-indigo-400 bg-indigo-500/10 text-indigo-200 pulse-glow',
  done: 'border-solid border-emerald-400 bg-emerald-500/20 text-emerald-200',
  error: 'border-solid border-red-400 bg-red-500/10 text-red-200',
}

function StepIcon({ status, index }) {
  if (status === 'done') return <Check className="w-5 h-5" />
  if (status === 'running') return <Loader2 className="w-5 h-5 animate-spin" />
  if (status === 'error') return <X className="w-5 h-5" />
  return <span className="text-sm font-bold">{index + 1}</span>
}

function PipelineTracker({ steps = [], onRetry, onStepClick }) {
  const doneCount = steps.filter((s) => s.status === 'done').length
  const progressPct = steps.length > 1 ? (doneCount / (steps.length - 1)) * 100 : 0

  return (
    <div className="glass-card p-6">
      <div className="relative flex items-start justify-between">
        <div className="absolute top-7 left-0 right-0 h-0.5 bg-white/10 mx-7">
          <motion.div
            className="h-full bg-gradient-to-r from-indigo-400 to-emerald-400"
            initial={{ width: 0 }}
            animate={{ width: `${Math.min(progressPct, 100)}%` }}
            transition={{ duration: 0.6, ease: 'easeOut' }}
          />
        </div>

        {steps.map((step, idx) => (
          <div
            key={step.name}
            onClick={() => step.status === 'done' && onStepClick?.(idx)}
            className={`relative z-10 flex flex-col items-center text-center flex-1 px-1 ${
              step.status === 'done' ? 'cursor-pointer group' : ''
            }`}
          >
            <div
              className={`w-14 h-14 rounded-full border-2 flex items-center justify-center transition-all ${
                PILL_STYLES[step.status] || PILL_STYLES.waiting
              } ${step.status === 'done' ? 'group-hover:scale-110 group-hover:shadow-[0_0_20px_rgba(16,185,129,0.5)]' : ''}`}
            >
              <StepIcon status={step.status} index={idx} />
            </div>
            <p
              className={`mt-3 text-sm font-semibold ${
                step.status === 'error'
                  ? 'text-red-300'
                  : step.status === 'done'
                  ? 'text-emerald-300'
                  : step.status === 'running'
                  ? 'text-indigo-200'
                  : 'text-slate-400'
              }`}
            >
              {step.name}
            </p>
            {step.description && <p className="text-xs text-slate-500 mt-0.5 max-w-[140px]">{step.description}</p>}
            {step.status === 'error' && (
              <button
                onClick={() => onRetry?.(idx)}
                className="mt-2 text-xs font-medium text-red-300 border border-red-400/40 rounded-full px-3 py-1 hover:bg-red-500/10 transition-colors"
              >
                Retry
              </button>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

export default PipelineTracker
