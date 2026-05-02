import ClaimsTable from './ClaimsTable.jsx'

const RISK_BADGE = {
  high:   'risk-high',
  medium: 'risk-medium',
  low:    'risk-low',
}

const SUB_CLASS = {
  strong:   'sub-strong',
  moderate: 'sub-moderate',
  weak:     'sub-weak',
  none:     'sub-none',
}

function StatCard({ label, value, mono = false }) {
  return (
    <div className="bg-white border border-gray-200 rounded-lg px-4 py-4">
      <p className="text-xs text-gray-500 uppercase tracking-wider mb-1">{label}</p>
      <p className={`text-xl font-semibold text-gray-900 ${mono ? 'mono' : ''}`}>{value}</p>
    </div>
  )
}

function BreakdownRow({ label, count, total, colorClass }) {
  const pct = total > 0 ? Math.round((count / total) * 100) : 0
  return (
    <div className="flex items-center gap-3 text-sm">
      <span className="w-28 text-gray-600 text-xs">{label}</span>
      <div className="flex-1 bg-gray-100 rounded-full h-2">
        <div
          className={`h-2 rounded-full ${colorClass}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="mono text-xs text-gray-500 w-12 text-right">{count} ({pct}%)</span>
    </div>
  )
}

export default function ResultsView({ report, onReset }) {
  const { summary, claims, filename } = report

  const riskColors = { high: 'bg-red-500', medium: 'bg-yellow-400', low: 'bg-green-500' }
  const subColors  = { strong: 'bg-green-500', moderate: 'bg-yellow-400', weak: 'bg-orange-400', none: 'bg-red-500' }

  return (
    <div className="max-w-6xl mx-auto px-6 py-10">
      {/* Page header */}
      <div className="flex items-start justify-between mb-8">
        <div>
          <p className="text-xs mono text-gray-400 mb-1">Analysis complete</p>
          <h1 className="text-xl font-semibold text-gray-900 tracking-tight">{filename}</h1>
        </div>
        <button
          onClick={onReset}
          className="text-sm mono text-gray-500 border border-gray-300 px-3 py-1.5 rounded hover:bg-gray-100 transition-colors"
        >
          ← New analysis
        </button>
      </div>

      {/* Overall risk banner */}
      <div className={`mb-6 px-5 py-4 rounded-lg flex items-center gap-3 ${
        summary.overall_risk_level === 'high'   ? 'bg-red-50 border border-red-200' :
        summary.overall_risk_level === 'medium' ? 'bg-yellow-50 border border-yellow-200' :
                                                   'bg-green-50 border border-green-200'
      }`}>
        <span className={`inline-block px-2.5 py-0.5 rounded text-xs font-semibold mono ${RISK_BADGE[summary.overall_risk_level] ?? ''}`}>
          {summary.overall_risk_level.toUpperCase()} RISK
        </span>
        <span className="text-sm text-gray-700">
          Overall document risk level based on {summary.total_claims} identified claims
        </span>
      </div>

      {/* Stat cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <StatCard label="Total Claims" value={summary.total_claims} mono />
        <StatCard label="High Risk" value={summary.by_risk_level?.high ?? 0} mono />
        <StatCard label="Unsubstantiated" value={(summary.by_substantiation_level?.weak ?? 0) + (summary.by_substantiation_level?.none ?? 0)} mono />
        <StatCard label="Well Substantiated" value={summary.by_substantiation_level?.strong ?? 0} mono />
      </div>

      {/* Breakdowns */}
      <div className="grid md:grid-cols-3 gap-6 mb-8">
        {/* ESG Category */}
        <div className="bg-white border border-gray-200 rounded-lg p-5">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-4">By ESG Category</h3>
          <div className="flex flex-col gap-2.5">
            {Object.entries(summary.by_esg_category ?? {}).map(([cat, count]) => (
              <BreakdownRow key={cat} label={cat} count={count} total={summary.total_claims} colorClass="bg-blue-400" />
            ))}
          </div>
        </div>

        {/* Risk Level */}
        <div className="bg-white border border-gray-200 rounded-lg p-5">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-4">By Risk Level</h3>
          <div className="flex flex-col gap-2.5">
            {Object.entries(summary.by_risk_level ?? {}).map(([level, count]) => (
              <BreakdownRow key={level} label={level} count={count} total={summary.total_claims} colorClass={riskColors[level] ?? 'bg-gray-400'} />
            ))}
          </div>
        </div>

        {/* Substantiation */}
        <div className="bg-white border border-gray-200 rounded-lg p-5">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-4">By Substantiation</h3>
          <div className="flex flex-col gap-2.5">
            {Object.entries(summary.by_substantiation_level ?? {}).map(([level, count]) => (
              <BreakdownRow key={level} label={level} count={count} total={summary.total_claims} colorClass={subColors[level] ?? 'bg-gray-400'} />
            ))}
          </div>
        </div>
      </div>

      {/* Key findings */}
      {summary.key_findings?.length > 0 && (
        <div className="bg-white border border-gray-200 rounded-lg p-5 mb-8">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-3">Key Findings</h3>
          <ul className="flex flex-col gap-2">
            {summary.key_findings.map((f, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-gray-700">
                <span className="mono text-gray-300 flex-shrink-0">{String(i + 1).padStart(2, '0')}.</span>
                {f}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Claims table */}
      <ClaimsTable claims={claims} />
    </div>
  )
}
