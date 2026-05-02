import { useState, useEffect } from 'react'

const STAGES = [
  { key: 'chunking',   label: 'Stage 1 — Chunking',          desc: 'Splitting document into overlapping text windows' },
  { key: 'extracting', label: 'Stage 2 — Claim Extraction',  desc: 'LLM identifying ESG claims in each chunk' },
  { key: 'analyzing',  label: 'Stage 3 — Claim Analysis',    desc: 'Scoring each claim for substantiation and greenwashing risk' },
  { key: 'retrieving', label: 'Stage 4 — Evidence Retrieval',desc: 'Vector search for supporting passages via ChromaDB' },
]

// Heuristic time budgets (seconds) — gives users a realistic sense of progress
// These don't reflect actual server state; the server is a black box for now.
const STAGE_BUDGETS = [15, 600, 600, 60]

export default function AnalyzingView({ filename }) {
  const [stageIndex, setStageIndex] = useState(0)
  const [elapsed, setElapsed] = useState(0)

  // Elapsed timer
  useEffect(() => {
    const id = setInterval(() => setElapsed(s => s + 1), 1000)
    return () => clearInterval(id)
  }, [])

  // Advance stages based on cumulative budget
  useEffect(() => {
    const cumulative = STAGE_BUDGETS.slice(0, stageIndex + 1).reduce((a, b) => a + b, 0)
    if (elapsed >= cumulative && stageIndex < STAGES.length - 1) {
      setStageIndex(i => i + 1)
    }
  }, [elapsed, stageIndex])

  function fmt(s) {
    const m = Math.floor(s / 60)
    const sec = s % 60
    return m > 0 ? `${m}m ${sec}s` : `${sec}s`
  }

  return (
    <div className="max-w-xl mx-auto mt-20 px-6">
      <div className="mb-8">
        <p className="text-xs mono text-gray-400 mb-1">Analyzing</p>
        <h2 className="text-lg font-semibold text-gray-900 tracking-tight truncate">{filename}</h2>
        <p className="text-sm text-gray-500 mt-1">
          Elapsed: <span className="mono font-medium text-gray-700">{fmt(elapsed)}</span>
          <span className="text-gray-400"> · Typical runtime: 20–30 min</span>
        </p>
      </div>

      {/* Stage list */}
      <div className="flex flex-col gap-3">
        {STAGES.map((stage, i) => {
          const state = i < stageIndex ? 'done' : i === stageIndex ? 'active' : 'pending'
          return (
            <div
              key={stage.key}
              className={`px-4 py-3 rounded-lg transition-all stage-${state}`}
            >
              <div className="flex items-center gap-3">
                {state === 'done' && <span className="text-green-600 text-sm">✓</span>}
                {state === 'active' && (
                  <svg className="spin w-4 h-4 text-blue-600 flex-shrink-0" viewBox="0 0 24 24" fill="none">
                    <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" strokeDasharray="31.4" strokeDashoffset="10" strokeLinecap="round"/>
                  </svg>
                )}
                {state === 'pending' && <span className="w-4 h-4 rounded-full border border-gray-300 flex-shrink-0" />}
                <div>
                  <p className={`text-sm font-medium ${state === 'done' ? 'text-green-700' : state === 'active' ? 'text-blue-700' : 'text-gray-400'}`}>
                    {stage.label}
                  </p>
                  <p className={`text-xs ${state === 'pending' ? 'text-gray-400' : 'text-gray-500'}`}>
                    {stage.desc}
                  </p>
                </div>
              </div>
            </div>
          )
        })}
      </div>

      <p className="text-xs text-gray-400 mt-8 mono">
        All inference is running locally via Ollama. Do not close this tab.
      </p>
    </div>
  )
}
