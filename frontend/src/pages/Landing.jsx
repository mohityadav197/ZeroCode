import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import {
  BarChart2,
  Brain,
  ChevronDown,
  Cpu,
  Download,
  Layers,
  MessageCircle,
  ScanSearch,
  Upload,
} from 'lucide-react'

const TAGLINES = [
  'Drop a CSV. Get Intelligence.',
  'No Code. Just Results.',
  'Your Data. Our Agents.',
  'Upload. Analyze. Deploy.',
]

const STATS = [
  { value: '6', label: 'AI Agents' },
  { value: '15+', label: 'Visualisations' },
  { value: 'Zero', label: 'Code' },
]

const FEATURES = [
  {
    icon: ScanSearch,
    color: '#8b5cf6',
    title: 'Smart Profiling',
    description:
      'Instantly understands your dataset — column types, missing values, quality score, and target detection. All automatic.',
  },
  {
    icon: BarChart2,
    color: '#06b6d4',
    title: '15+ Chart Types',
    description: 'Choose from histograms, heatmaps, box plots, pair plots and more. Interactive and zoomable.',
  },
  {
    icon: Cpu,
    color: '#6366f1',
    title: 'AutoML Training',
    description:
      'Trains Random Forest, XGBoost, SVM and more simultaneously. Picks the winner automatically with reasoning.',
  },
  {
    icon: Brain,
    color: '#10b981',
    title: 'SHAP Explainability',
    description:
      'Understand WHY the model decides what it decides. Feature importance, impact plots, and plain English explanations.',
  },
  {
    icon: MessageCircle,
    color: '#f97316',
    title: 'AI Chat Assistant',
    description: 'Ask anything about your data in plain English. Generate new charts on demand. Get recommendations instantly.',
  },
  {
    icon: Download,
    color: '#ec4899',
    title: 'Download Everything',
    description: 'Take your trained model, cleaned data, Python code and full report. Yours to keep and deploy.',
  },
]

const STEPS = [
  { icon: Upload, title: 'Upload', description: 'Drop your CSV file' },
  { icon: Layers, title: 'Configure', description: 'Choose visualisations and target' },
  { icon: Cpu, title: 'Train', description: 'AI agents analyze and train models' },
  { icon: Download, title: 'Download', description: 'Get everything in one click' },
]

const SHAP_MOCK = [
  { label: 'Sex', value: 72, positive: true },
  { label: 'Pclass', value: 55, positive: true },
  { label: 'Age', value: -38, positive: false },
  { label: 'Fare', value: 28, positive: true },
]

const stepsContainer = { hidden: {}, visible: { transition: { staggerChildren: 0.2 } } }
const stepsItem = { hidden: { opacity: 0, y: 30 }, visible: { opacity: 1, y: 0, transition: { duration: 0.5 } } }

function useTypewriter(phrases, typingSpeed = 55, deletingSpeed = 30, pause = 1400) {
  const [text, setText] = useState('')
  const [phraseIndex, setPhraseIndex] = useState(0)
  const [deleting, setDeleting] = useState(false)

  useEffect(() => {
    const current = phrases[phraseIndex % phrases.length]
    let timeout

    if (!deleting && text === current) {
      timeout = setTimeout(() => setDeleting(true), pause)
    } else if (deleting && text === '') {
      setDeleting(false)
      setPhraseIndex((i) => (i + 1) % phrases.length)
    } else {
      timeout = setTimeout(
        () => setText((t) => (deleting ? current.slice(0, t.length - 1) : current.slice(0, t.length + 1))),
        deleting ? deletingSpeed : typingSpeed
      )
    }

    return () => clearTimeout(timeout)
  }, [text, deleting, phraseIndex, phrases, typingSpeed, deletingSpeed, pause])

  return text
}

function Landing({ onGetStarted }) {
  const typed = useTypewriter(TAGLINES)

  return (
    <div className="min-h-screen text-slate-100 overflow-x-hidden" style={{ backgroundColor: '#050510' }}>
      {/* ============ HERO ============ */}
      <section className="relative min-h-screen flex flex-col items-center justify-center overflow-hidden px-6">
        <div className="absolute inset-0 grid-overlay" />
        <div className="absolute -top-20 -left-20 w-[420px] h-[420px] rounded-full blur-3xl mesh-blob-1" style={{ background: 'rgba(99,102,241,0.15)' }} />
        <div className="absolute -top-10 -right-20 w-[380px] h-[380px] rounded-full blur-3xl mesh-blob-2" style={{ background: 'rgba(6,182,212,0.1)' }} />
        <div className="absolute -bottom-24 -right-10 w-[460px] h-[460px] rounded-full blur-3xl mesh-blob-3" style={{ background: 'rgba(139,92,246,0.12)' }} />

        <div className="relative z-10 text-center max-w-4xl mx-auto">
          <div className="flex flex-nowrap items-baseline justify-center">
            <motion.span
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.7, delay: 0.2, ease: 'easeOut' }}
              style={{
                fontSize: 'clamp(5rem, 12vw, 10rem)',
                fontWeight: 900,
                letterSpacing: '-0.02em',
                lineHeight: 1,
                color: '#ffffff',
              }}
            >
              Zero
            </motion.span>
            <motion.span
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.7, delay: 0.4, ease: 'easeOut' }}
              className="relative inline-block gradient-text"
              style={{
                fontSize: 'clamp(5rem, 12vw, 10rem)',
                fontWeight: 900,
                letterSpacing: '-0.02em',
                lineHeight: 1,
              }}
            >
              Code
              <motion.span
                initial={{ width: '0%' }}
                animate={{ width: '100%' }}
                transition={{ duration: 1, delay: 1.0, ease: 'easeInOut' }}
                className="absolute left-0 -bottom-2 h-[6px] rounded-full"
                style={{ background: 'linear-gradient(90deg, #6366f1, #8b5cf6, #06b6d4)' }}
              />
            </motion.span>
          </div>

          <p className="mt-8 font-medium h-8" style={{ color: 'rgba(255,255,255,0.7)', fontSize: '1.4rem' }}>
            {typed}
            <span className="animate-pulse">|</span>
          </p>

          <motion.p
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.8 }}
            className="mt-6 mx-auto text-base leading-relaxed"
            style={{ color: 'rgba(255,255,255,0.5)', maxWidth: 600 }}
          >
            ZeroCode is an AI-powered AutoML platform that automatically analyzes your data, trains machine
            learning models, explains predictions with SHAP analysis, and delivers everything — code, models, and
            insights — in minutes. No programming required.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 1.0 }}
            className="mt-10 flex flex-wrap items-center justify-center gap-4"
          >
            <button onClick={onGetStarted} className="hero-btn-primary">
              Start For Free →
            </button>
            <a href="#how-it-works" className="hero-btn-secondary">
              See How It Works ↓
            </a>
          </motion.div>
        </div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 2 }}
          className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-1.5 text-xs text-slate-500"
        >
          <span>Scroll to explore</span>
          <motion.div animate={{ y: [0, 8, 0] }} transition={{ duration: 1.6, repeat: Infinity, ease: 'easeInOut' }}>
            <ChevronDown className="w-4 h-4" />
          </motion.div>
        </motion.div>
      </section>

      {/* ============ STATS BAR ============ */}
      <section className="relative py-10 px-6 border-y border-white/10">
        <div className="glass-card max-w-5xl mx-auto grid grid-cols-1 sm:grid-cols-3 divide-y sm:divide-y-0 sm:divide-x divide-white/10 py-8">
          {STATS.map((stat, i) => (
            <motion.div
              key={stat.label}
              initial={{ opacity: 0, y: 16 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: i * 0.1 }}
              className="flex flex-col items-center justify-center py-4 sm:py-0"
            >
              <span className="text-3xl sm:text-4xl font-extrabold gradient-text">{stat.value}</span>
              <span className="text-sm text-slate-400 mt-1">{stat.label}</span>
            </motion.div>
          ))}
        </div>
      </section>

      {/* ============ FEATURES ============ */}
      <section className="relative py-24 px-6 max-w-6xl mx-auto">
        <motion.h2
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-3xl sm:text-4xl font-bold text-center gradient-text"
        >
          Everything You Need
        </motion.h2>
        <motion.p
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.1 }}
          className="text-center text-slate-400 mt-3 mb-14"
        >
          One platform. Complete ML pipeline.
        </motion.p>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
          {FEATURES.map((feature, i) => {
            const Icon = feature.icon
            return (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 24 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: (i % 3) * 0.08 }}
                whileHover={{ y: -6 }}
                className="group glass-card p-6 transition-shadow duration-300 hover:shadow-[0_8px_40px_rgba(99,102,241,0.25)]"
              >
                <div
                  className="w-12 h-12 rounded-xl flex items-center justify-center mb-4"
                  style={{ backgroundColor: `${feature.color}22` }}
                >
                  <Icon className="w-6 h-6" style={{ color: feature.color }} />
                </div>
                <h3 className="text-lg font-bold text-white mb-2">{feature.title}</h3>
                <p className="text-sm text-slate-400">{feature.description}</p>
                <span
                  className="mt-3 inline-block text-sm font-semibold opacity-0 group-hover:opacity-100 transition-opacity"
                  style={{ color: feature.color }}
                >
                  Learn more →
                </span>
              </motion.div>
            )
          })}
        </div>
      </section>

      {/* ============ HOW IT WORKS ============ */}
      <section id="how-it-works" className="relative py-24 px-6 max-w-6xl mx-auto">
        <motion.h2
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-3xl sm:text-4xl font-bold text-center mb-16"
        >
          From CSV to Insights in Minutes
        </motion.h2>

        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          variants={stepsContainer}
          className="relative grid grid-cols-1 md:grid-cols-4 gap-12"
        >
          <div className="hidden md:block absolute top-9 left-[12.5%] right-[12.5%] h-[2px] dashed-line-anim" />

          {STEPS.map((step, i) => {
            const Icon = step.icon
            return (
              <motion.div key={step.title} variants={stepsItem} className="relative flex flex-col items-center text-center">
                <span className="text-5xl font-extrabold gradient-text leading-none mb-2">{i + 1}</span>
                <div className="relative z-10 w-14 h-14 rounded-full glass-card flex items-center justify-center mb-4 border-2 border-indigo-500/50">
                  <Icon className="w-6 h-6 text-indigo-400" />
                </div>
                <h3 className="font-bold text-white mb-1">{step.title}</h3>
                <p className="text-sm text-slate-400">{step.description}</p>
              </motion.div>
            )
          })}
        </motion.div>
      </section>

      {/* ============ SHAP HIGHLIGHT ============ */}
      <section className="relative py-24 px-6" style={{ background: 'linear-gradient(180deg, transparent, rgba(99,102,241,0.05), transparent)' }}>
        <div className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-14 items-center">
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
          >
            <span className="text-xs font-bold tracking-[0.2em] uppercase" style={{ color: '#06b6d4' }}>
              Explainable AI
            </span>
            <h2 className="text-3xl sm:text-4xl font-bold text-white mt-3 mb-5">Know Why, Not Just What</h2>
            <p className="text-slate-400 leading-relaxed mb-7 max-w-md">
              Unlike black-box ML tools, ZeroCode uses SHAP analysis to show you exactly which features drive every
              prediction. Build trust in your models.
            </p>
            <button onClick={onGetStarted} className="hero-btn-primary">
              See SHAP in Action →
            </button>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="glass-card p-6"
          >
            <p className="text-sm font-semibold text-slate-300 mb-5">Why Did The Model Predict This?</p>
            <div className="space-y-4">
              {SHAP_MOCK.map((bar, i) => (
                <div key={bar.label} className="flex items-center gap-3">
                  <span className="w-14 text-xs text-slate-400 shrink-0">{bar.label}</span>
                  <div className="flex-1 h-3 rounded-full bg-white/5 overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      whileInView={{ width: `${Math.abs(bar.value)}%` }}
                      viewport={{ once: true }}
                      transition={{ duration: 0.9, delay: i * 0.15, ease: 'easeOut' }}
                      className="h-full rounded-full"
                      style={{ backgroundColor: bar.positive ? '#10b981' : '#ef4444' }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        </div>
      </section>

      {/* ============ FOOTER ============ */}
      <footer className="border-t border-white/10 py-10 text-center" style={{ backgroundColor: '#050510' }}>
        <h3 className="text-xl font-extrabold gradient-text">ZeroCode</h3>
        <p className="text-sm text-slate-500 mt-2">Zero Code. Full Insight.</p>
        <div className="max-w-xs mx-auto border-t border-white/10 my-5" />
        <p className="text-xs text-slate-600">© 2025 ZeroCode. Built with AI.</p>
      </footer>
    </div>
  )
}

export default Landing
