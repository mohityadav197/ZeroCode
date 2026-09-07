import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { Cat, CheckCircle2, Mountain, Network, Percent, Scissors, TreePine, TrendingUp, Waypoints, Zap } from 'lucide-react'

const CLASSIFICATION_MODELS = [
  { name: 'KNN', icon: Waypoints, description: 'Classifies by nearest neighbors' },
  { name: 'AdaBoost', icon: TrendingUp, description: 'Boosts weak learners iteratively' },
  { name: 'Extra Trees', icon: TreePine, description: 'Randomized ensemble of trees' },
  { name: 'LightGBM', icon: Zap, description: 'Fast gradient boosting' },
  { name: 'CatBoost', icon: Cat, description: 'Boosting tuned for categoricals' },
  { name: 'Naive Bayes', icon: Percent, description: 'Probabilistic baseline classifier' },
]

const REGRESSION_MODELS = [
  { name: 'KNN Regressor', icon: Waypoints, description: 'Predicts by nearest neighbors' },
  { name: 'Lasso', icon: Scissors, description: 'Linear model with L1 regularization' },
  { name: 'Ridge', icon: Mountain, description: 'Linear model with L2 regularization' },
  { name: 'ElasticNet', icon: Network, description: 'Combines L1 and L2 regularization' },
  { name: 'LightGBM Regressor', icon: Zap, description: 'Fast gradient boosting' },
  { name: 'CatBoost Regressor', icon: Cat, description: 'Boosting tuned for categoricals' },
]

function ModelSelector({ problemType, onSelectionChange }) {
  const isRegression = problemType === 'regression'
  const options = isRegression ? REGRESSION_MODELS : CLASSIFICATION_MODELS
  const [selected, setSelected] = useState([])

  useEffect(() => {
    setSelected([])
  }, [problemType])

  const toggle = (name) => {
    setSelected((prev) => {
      const next = prev.includes(name) ? prev.filter((m) => m !== name) : [...prev, name]
      onSelectionChange?.(next)
      return next
    })
  }

  return (
    <div className="glass-card p-5">
      <h3 className="text-lg font-bold text-white">Enhance Your Training</h3>
      <p className="text-sm text-slate-500 mb-4">Add more algorithms to compare</p>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {options.map(({ name, icon: Icon, description }, i) => {
          const isChecked = selected.includes(name)
          return (
            <motion.div
              key={name}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: i * 0.05 }}
              whileHover={{ y: -3 }}
              onClick={() => toggle(name)}
              className={`relative glass-card p-3 cursor-pointer transition-shadow duration-200 ${
                isChecked
                  ? 'border-purple-400/70 shadow-[0_0_20px_rgba(139,92,246,0.35)] bg-gradient-to-br from-indigo-500/15 to-purple-500/15'
                  : 'hover:shadow-[0_6px_24px_rgba(99,102,241,0.15)]'
              }`}
            >
              {isChecked && (
                <CheckCircle2 className="absolute -top-2 -right-2 w-5 h-5 text-purple-400 bg-[#0f0f1a] rounded-full" />
              )}
              <div className="flex items-center gap-2 mb-1">
                <Icon className={`w-4 h-4 ${isChecked ? 'text-purple-300' : 'text-indigo-400'}`} />
                <span className="text-sm font-semibold text-slate-100">{name}</span>
              </div>
              <p className="text-xs text-slate-500">{description}</p>
            </motion.div>
          )
        })}
      </div>
    </div>
  )
}

export default ModelSelector
