import { useEffect, useState } from 'react'
import { createPortal } from 'react-dom'
import { AnimatePresence, motion } from 'framer-motion'
import { Maximize2, X } from 'lucide-react'

const BASE_URL = 'http://localhost:8000'

function toTitle(name) {
  return name.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
}

function toSrc(path) {
  const relativePath = path.startsWith('outputs/') ? path.slice('outputs/'.length) : path
  return `${BASE_URL}/files/${relativePath}`
}

function ChartCard({ name, path, onExpand }) {
  const [status, setStatus] = useState('loading')
  const title = toTitle(name)
  const src = toSrc(path)

  return (
    <div
      onClick={() => status === 'loaded' && onExpand({ name, path })}
      className="group relative glass-card p-3 cursor-pointer hover:z-50 hover:scale-[1.08] hover:shadow-[0_20px_60px_rgba(99,102,241,0.4)]"
      style={{ transition: 'all 0.3s cubic-bezier(0.4,0,0.2,1)' }}
    >
      <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity bg-black/50 rounded-md p-1">
        <Maximize2 className="w-3.5 h-3.5 text-white" />
      </div>
      <div className="aspect-video bg-white/5 rounded-lg overflow-hidden flex items-center justify-center relative">
        {status === 'loading' && <div className="absolute inset-0 shimmer rounded-lg" />}
        {status === 'error' ? (
          <div className="text-xs text-slate-500 text-center px-4">Chart unavailable</div>
        ) : (
          <motion.img
            src={src}
            alt={title}
            onLoad={() => setStatus('loaded')}
            onError={() => setStatus('error')}
            initial={{ opacity: 0 }}
            animate={{ opacity: status === 'loaded' ? 1 : 0 }}
            transition={{ duration: 0.4 }}
            className="w-full h-full object-contain"
          />
        )}
      </div>
      <p className="text-xs text-slate-400 text-center mt-2">{title}</p>
    </div>
  )
}

function ChartDisplay({ charts = {} }) {
  const entries = Object.entries(charts)
  const [expanded, setExpanded] = useState(null)

  useEffect(() => {
    if (!expanded) return
    const onKeyDown = (e) => {
      if (e.key === 'Escape') setExpanded(null)
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [expanded])

  if (entries.length === 0) {
    return <p className="text-sm text-slate-500">No charts generated yet.</p>
  }

  return (
    <>
      <div className="grid grid-cols-2 gap-4">
        {entries.map(([name, path]) => (
          <ChartCard key={name} name={name} path={path} onExpand={setExpanded} />
        ))}
      </div>

      {createPortal(
        <AnimatePresence>
          {expanded && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.25 }}
              onClick={() => setExpanded(null)}
              className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-black/85 backdrop-blur-sm p-6"
            >
              <div className="w-full flex items-center justify-between max-w-5xl mb-3">
                <h3 className="text-xl font-bold gradient-text">{toTitle(expanded.name)}</h3>
                <button
                  onClick={() => setExpanded(null)}
                  className="glass-card w-9 h-9 flex items-center justify-center text-slate-300 hover:text-white transition-colors"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
              <img
                src={toSrc(expanded.path)}
                alt={toTitle(expanded.name)}
                onClick={(e) => e.stopPropagation()}
                className="max-w-[90vw] max-h-[85vh] object-contain rounded-lg shadow-2xl"
              />
            </motion.div>
          )}
        </AnimatePresence>,
        document.body
      )}
    </>
  )
}

export default ChartDisplay
