import { useState, useEffect } from 'react'

async function fetchHealth() {
  try {
    const res = await fetch('/api/v1/health', { signal: AbortSignal.timeout(4000) })
    if (!res.ok) return { backend: false, ollama: false }
    const data = await res.json()
    return { backend: true, ollama: data.ollama_reachable === true }
  } catch {
    return { backend: false, ollama: false }
  }
}

function Dot({ on, pulsing = false }) {
  return (
    <span
      className={`w-2 h-2 rounded-full inline-block flex-shrink-0 ${on ? 'status-dot-green' : 'status-dot-red'} ${pulsing || !on ? 'pulse' : ''}`}
    />
  )
}

export default function HealthBar() {
  const [status, setStatus] = useState(null) // null = loading

  useEffect(() => {
    let cancelled = false

    async function poll() {
      const s = await fetchHealth()
      if (!cancelled) setStatus(s)
    }

    poll()
    const id = setInterval(poll, 15000)
    return () => { cancelled = true; clearInterval(id) }
  }, [])

  if (status === null) {
    return (
      <div className="flex items-center gap-2 text-xs mono text-gray-500">
        <span className="w-2 h-2 rounded-full bg-gray-600 pulse inline-block" />
        connecting…
      </div>
    )
  }

  return (
    <div className="flex items-center gap-4 text-xs mono">
      <div className="flex items-center gap-1.5">
        <Dot on={status.backend} />
        <span className={status.backend ? 'text-green-400' : 'text-red-400'}>
          api
        </span>
      </div>
      <div className="flex items-center gap-1.5">
        <Dot on={status.ollama} pulsing={!status.ollama} />
        <span className={status.ollama ? 'text-green-400' : 'text-red-400'}>
          ollama
        </span>
      </div>
    </div>
  )
}
