import { motion } from 'framer-motion'
import { ArrowUpRight, Code2, Database, Download, Package } from 'lucide-react'
import { getDownloadUrl } from '../services/api'

function DownloadPanel({ reportReady, modelReady }) {
  const items = [
    {
      key: 'eda-code',
      label: 'EDA Code',
      description: 'Standalone Python script for exploratory analysis',
      badge: '.py',
      icon: Code2,
      color: '#3b82f6',
      ready: modelReady,
    },
    {
      key: 'preprocessing-code',
      label: 'Preprocessing Code',
      description: 'Reproduces the cleaning and encoding steps',
      badge: '.py',
      icon: Code2,
      color: '#f97316',
      ready: modelReady,
    },
    {
      key: 'model-code',
      label: 'Model Training Code',
      description: 'Trains the best model independently',
      badge: '.py',
      icon: Code2,
      color: '#8b5cf6',
      ready: modelReady,
    },
    {
      key: 'model',
      label: 'Best Model',
      description: 'The trained, tuned model artifact',
      badge: '.pkl',
      icon: Package,
      color: '#10b981',
      ready: modelReady,
    },
    {
      key: 'cleaned-data',
      label: 'Cleaned Data',
      description: 'Fully preprocessed dataset',
      badge: '.csv',
      icon: Database,
      color: '#06b6d4',
      ready: modelReady,
    },
    {
      key: 'report',
      label: 'Full Report',
      description: 'Complete analysis and model report',
      badge: '.html',
      icon: Download,
      color: '#ec4899',
      ready: reportReady,
    },
  ]

  const handleDownload = (item, ready) => {
    if (!ready) return
    window.open(getDownloadUrl(item), '_blank')
  }

  return (
    <div>
      <div className="flex items-center gap-2 mb-4">
        <Download className="w-5 h-5 text-indigo-400" />
        <h3 className="text-lg font-bold text-white">Your Downloads</h3>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {items.map(({ key, label, description, badge, icon: Icon, color, ready }, i) => (
          <motion.div
            key={key}
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.35, delay: i * 0.06 }}
            whileHover={ready ? { y: -4 } : {}}
            title={!ready ? 'Not ready yet' : undefined}
            className={`glass-card p-4 flex flex-col gap-3 transition-shadow ${
              ready ? 'hover:shadow-[0_10px_40px_rgba(99,102,241,0.25)]' : 'opacity-40'
            }`}
          >
            <div className="flex items-start justify-between">
              <div className="w-10 h-10 rounded-xl flex items-center justify-center" style={{ backgroundColor: `${color}22` }}>
                <Icon className="w-5 h-5" style={{ color }} />
              </div>
              <span className="text-[10px] font-bold uppercase tracking-wide bg-white/10 text-slate-300 px-2 py-1 rounded-full">
                {badge}
              </span>
            </div>
            <div>
              <p className="text-sm font-bold text-white">{label}</p>
              <p className="text-xs text-slate-500 mt-0.5">{description}</p>
            </div>
            <button
              onClick={() => handleDownload(key, ready)}
              disabled={!ready}
              className={`mt-auto flex items-center justify-center gap-1.5 text-sm font-semibold px-3 py-2 rounded-lg transition-shadow ${
                ready
                  ? 'bg-gradient-to-r from-indigo-500 to-purple-500 text-white hover:shadow-[0_0_20px_rgba(139,92,246,0.6)]'
                  : 'bg-white/5 text-slate-500 cursor-not-allowed'
              }`}
            >
              {ready ? 'Download' : 'Not ready'}
              {ready && <ArrowUpRight className="w-4 h-4" />}
            </button>
          </motion.div>
        ))}
      </div>
    </div>
  )
}

export default DownloadPanel
