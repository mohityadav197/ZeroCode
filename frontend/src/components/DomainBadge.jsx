export const CONFIDENCE_STYLES = {
  high: { label: 'High Confidence', className: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30' },
  medium: { label: 'Medium Confidence', className: 'bg-amber-500/20 text-amber-300 border-amber-500/30' },
  low: { label: 'Low Confidence', className: 'bg-orange-500/20 text-orange-300 border-orange-500/30' },
  none: { label: 'General Dataset', className: 'bg-white/10 text-slate-400 border-white/10' },
}

function DomainBadge({ domainInfo, showConfidence = false, className = '' }) {
  if (!domainInfo) return null

  const shortName = domainInfo.display_name?.split(' / ')[0] || domainInfo.display_name
  const confidence = CONFIDENCE_STYLES[domainInfo.confidence] || CONFIDENCE_STYLES.none

  return (
    <span
      className={`inline-flex items-center gap-1.5 text-xs font-medium text-slate-300 bg-white/5 border border-white/10 px-2.5 py-1 rounded-full ${className}`}
    >
      <span>{domainInfo.icon}</span>
      <span>{shortName}</span>
      {showConfidence && (
        <span className={`ml-1 text-[10px] font-bold px-1.5 py-0.5 rounded-full border ${confidence.className}`}>
          {confidence.label}
        </span>
      )}
    </span>
  )
}

export default DomainBadge
