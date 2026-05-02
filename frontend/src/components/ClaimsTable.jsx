import { useState, useMemo } from 'react'

const RISK_BADGE = {
  high:   'risk-high',
  medium: 'risk-medium',
  low:    'risk-low',
}

const SUB_LABEL = {
  strong:   { cls: 'text-green-700 bg-green-50 border border-green-200',   label: 'Strong' },
  moderate: { cls: 'text-yellow-700 bg-yellow-50 border border-yellow-200', label: 'Moderate' },
  weak:     { cls: 'text-orange-700 bg-orange-50 border border-orange-200', label: 'Weak' },
  none:     { cls: 'text-red-700 bg-red-50 border border-red-100',           label: 'None' },
}

const CAT_COLORS = {
  environmental: 'text-emerald-700 bg-emerald-50 border border-emerald-200',
  social:        'text-violet-700 bg-violet-50 border border-violet-200',
  governance:    'text-blue-700 bg-blue-50 border border-blue-200',
  unknown:       'text-gray-600 bg-gray-100 border border-gray-200',
}

function Pill({ label, cls }) {
  return (
    <span className={`inline-block px-2 py-0.5 rounded text-xs mono font-medium ${cls}`}>
      {label}
    </span>
  )
}

function EvidenceBlock({ evidence }) {
  if (!evidence || evidence.length === 0) {
    return <p className="text-xs text-gray-400 italic">No supporting evidence retrieved.</p>
  }
  return (
    <div className="flex flex-col gap-2">
      {evidence.map((ev, i) => (
        <div key={ev.evidence_id ?? i} className="bg-gray-50 border border-gray-200 rounded px-3 py-2">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs mono text-gray-400">{ev.evidence_type}</span>
            <span className="text-xs mono text-gray-400">·</span>
            <span className="text-xs mono text-gray-500">relevance {(ev.relevance_score * 100).toFixed(0)}%</span>
          </div>
          <p className="text-sm text-gray-700 leading-relaxed">{ev.text}</p>
        </div>
      ))}
    </div>
  )
}

function ClaimRow({ item, index }) {
  const [open, setOpen] = useState(false)
  const { claim, evidence, substantiation_level, risk_level, gap_explanation, confidence } = item
  const sub = SUB_LABEL[substantiation_level] ?? { cls: 'text-gray-500', label: substantiation_level }
  const catCls = CAT_COLORS[claim.esg_category] ?? CAT_COLORS.unknown

  return (
    <>
      <tr
        className={`border-b border-gray-100 cursor-pointer hover:bg-gray-50 transition-colors ${open ? 'bg-gray-50' : ''}`}
        onClick={() => setOpen(o => !o)}
      >
        {/* # */}
        <td className="py-3 pl-4 pr-2 text-xs mono text-gray-400 w-10">{index + 1}</td>

        {/* Claim text */}
        <td className="py-3 px-3 text-sm text-gray-800 max-w-sm">
          <p className="line-clamp-2 leading-snug">{claim.text}</p>
          {claim.framework_tags?.length > 0 && (
            <div className="flex flex-wrap gap-1 mt-1">
              {claim.framework_tags.map(t => (
                <span key={t} className="text-xs mono text-gray-400 bg-gray-100 rounded px-1.5 py-0.5">{t}</span>
              ))}
            </div>
          )}
        </td>

        {/* ESG Category */}
        <td className="py-3 px-3 w-32">
          <Pill label={claim.esg_category} cls={catCls} />
        </td>

        {/* Risk */}
        <td className="py-3 px-3 w-24">
          <Pill label={risk_level} cls={`${RISK_BADGE[risk_level] ?? ''}`} />
        </td>

        {/* Substantiation */}
        <td className="py-3 px-3 w-28">
          <Pill label={sub.label} cls={sub.cls} />
        </td>

        {/* Confidence */}
        <td className="py-3 px-3 w-20 mono text-xs text-gray-500">
          {confidence != null ? `${(confidence * 100).toFixed(0)}%` : '—'}
        </td>

        {/* Expand chevron */}
        <td className="py-3 pr-4 w-6 text-gray-400 text-xs">
          {open ? '▲' : '▼'}
        </td>
      </tr>

      {/* Expanded detail */}
      {open && (
        <tr className="bg-blue-50 border-b border-blue-100">
          <td colSpan={7} className="px-6 py-5">
            <div className="grid md:grid-cols-2 gap-6">
              {/* Gap explanation */}
              <div>
                <h4 className="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-2">Gap Explanation</h4>
                <p className="text-sm text-gray-700 leading-relaxed">{gap_explanation}</p>
                {claim.page_reference && (
                  <p className="text-xs mono text-gray-400 mt-2">Source: {claim.page_reference}</p>
                )}
              </div>

              {/* Supporting evidence */}
              <div>
                <h4 className="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-2">
                  Supporting Evidence
                  <span className="ml-2 font-normal normal-case text-gray-400">({evidence?.length ?? 0} passages)</span>
                </h4>
                <EvidenceBlock evidence={evidence} />
              </div>
            </div>
          </td>
        </tr>
      )}
    </>
  )
}

export default function ClaimsTable({ claims }) {
  const [search, setSearch] = useState('')
  const [filterRisk, setFilterRisk] = useState('all')
  const [filterCat, setFilterCat] = useState('all')
  const [filterSub, setFilterSub] = useState('all')

  const filtered = useMemo(() => {
    return claims.filter(item => {
      const text = item.claim.text.toLowerCase()
      const gap  = (item.gap_explanation ?? '').toLowerCase()
      const q    = search.toLowerCase()

      if (q && !text.includes(q) && !gap.includes(q)) return false
      if (filterRisk !== 'all' && item.risk_level !== filterRisk) return false
      if (filterCat  !== 'all' && item.claim.esg_category !== filterCat) return false
      if (filterSub  !== 'all' && item.substantiation_level !== filterSub) return false
      return true
    })
  }, [claims, search, filterRisk, filterCat, filterSub])

  const selectCls = "border border-gray-200 rounded px-2.5 py-1.5 text-xs mono bg-white text-gray-700 focus:outline-none focus:border-blue-400"

  return (
    <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
      {/* Table toolbar */}
      <div className="px-4 py-3 border-b border-gray-100 flex flex-wrap items-center gap-3">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-gray-400 mr-auto">
          Claims
          <span className="ml-2 font-normal normal-case text-gray-400">
            {filtered.length} / {claims.length}
          </span>
        </h3>

        {/* Search */}
        <input
          type="text"
          placeholder="Search claims…"
          value={search}
          onChange={e => setSearch(e.target.value)}
          className="border border-gray-200 rounded px-3 py-1.5 text-xs mono w-44 focus:outline-none focus:border-blue-400"
        />

        {/* Risk filter */}
        <select value={filterRisk} onChange={e => setFilterRisk(e.target.value)} className={selectCls}>
          <option value="all">All risk</option>
          <option value="high">High</option>
          <option value="medium">Medium</option>
          <option value="low">Low</option>
        </select>

        {/* Category filter */}
        <select value={filterCat} onChange={e => setFilterCat(e.target.value)} className={selectCls}>
          <option value="all">All categories</option>
          <option value="environmental">Environmental</option>
          <option value="social">Social</option>
          <option value="governance">Governance</option>
          <option value="unknown">Unknown</option>
        </select>

        {/* Substantiation filter */}
        <select value={filterSub} onChange={e => setFilterSub(e.target.value)} className={selectCls}>
          <option value="all">All substantiation</option>
          <option value="strong">Strong</option>
          <option value="moderate">Moderate</option>
          <option value="weak">Weak</option>
          <option value="none">None</option>
        </select>

        {/* Clear filters */}
        {(search || filterRisk !== 'all' || filterCat !== 'all' || filterSub !== 'all') && (
          <button
            onClick={() => { setSearch(''); setFilterRisk('all'); setFilterCat('all'); setFilterSub('all') }}
            className="text-xs mono text-gray-400 hover:text-gray-600 underline"
          >
            Clear
          </button>
        )}
      </div>

      {/* Table */}
      {filtered.length === 0 ? (
        <div className="py-16 text-center text-sm text-gray-400 mono">No claims match the current filters.</div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-left">
            <thead className="bg-gray-50 border-b border-gray-100">
              <tr>
                <th className="py-2 pl-4 pr-2 text-xs mono font-medium text-gray-400">#</th>
                <th className="py-2 px-3 text-xs mono font-medium text-gray-400">Claim</th>
                <th className="py-2 px-3 text-xs mono font-medium text-gray-400">Category</th>
                <th className="py-2 px-3 text-xs mono font-medium text-gray-400">Risk</th>
                <th className="py-2 px-3 text-xs mono font-medium text-gray-400">Substantiation</th>
                <th className="py-2 px-3 text-xs mono font-medium text-gray-400">Confidence</th>
                <th className="py-2 pr-4 w-6" />
              </tr>
            </thead>
            <tbody>
              {filtered.map((item, i) => (
                <ClaimRow key={item.claim.claim_id} item={item} index={i} />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
