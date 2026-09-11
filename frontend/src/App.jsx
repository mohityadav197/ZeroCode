import { useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { Toaster } from 'react-hot-toast'
import { ArrowLeft } from 'lucide-react'
import './App.css'
import Landing from './pages/Landing'
import Dashboard from './pages/Dashboard'
import DomainBadge from './components/DomainBadge'

function App() {
  const [showDashboard, setShowDashboard] = useState(false)
  const [datasetName, setDatasetName] = useState(null)
  const [domainInfo, setDomainInfo] = useState(null)
  const dashboardRef = useRef(null)

  return (
    <div className="h-screen overflow-hidden bg-[#0f0f1a]">
      <Toaster
        position="top-right"
        toastOptions={{
          style: {
            background: 'rgba(20, 20, 35, 0.95)',
            color: '#f1f5f9',
            border: '1px solid rgba(255,255,255,0.1)',
          },
        }}
      />

      <AnimatePresence>
        {showDashboard && (
          <motion.header
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
            className="fixed top-0 left-0 right-0 z-40 flex items-center justify-between px-6 py-4 border-b border-white/10"
            style={{
              backdropFilter: 'blur(20px)',
              WebkitBackdropFilter: 'blur(20px)',
              background: 'rgba(15,15,26,0.8)',
            }}
          >
            <div className="flex items-center gap-3">
              <button
                onClick={() => dashboardRef.current?.requestNewSession()}
                className="flex items-center gap-1 text-xs font-semibold text-slate-400 hover:text-indigo-300 transition-colors"
              >
                <ArrowLeft className="w-3.5 h-3.5" />
                New Session
              </button>
              <span className="w-px h-4 bg-white/10" />
              <h1 className="text-xl font-extrabold tracking-tight gradient-text">ZeroCode</h1>
              {datasetName && (
                <span className="text-xs font-medium text-slate-300 bg-white/5 border border-white/10 px-2.5 py-1 rounded-full">
                  {datasetName}
                </span>
              )}
              {domainInfo && domainInfo.confidence !== 'none' && <DomainBadge domainInfo={domainInfo} />}
            </div>
            <p className="text-xs text-slate-400">Zero Code. Full Insight.</p>
          </motion.header>
        )}
      </AnimatePresence>

      <main className={`h-full overflow-y-auto flex ${showDashboard ? 'pt-[68px]' : ''}`}>
        <AnimatePresence mode="wait">
          {!showDashboard ? (
            <motion.div
              key="landing"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.4 }}
              className="w-full"
            >
              <Landing onGetStarted={() => setShowDashboard(true)} />
            </motion.div>
          ) : (
            <motion.div
              key="dashboard"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.4 }}
              className="w-full flex flex-1 min-h-0"
            >
              <Dashboard ref={dashboardRef} onDatasetChange={setDatasetName} onDomainChange={setDomainInfo} />
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  )
}

export default App
