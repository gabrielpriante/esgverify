const BASE = '/api/v1'

/**
 * POST /api/v1/analyze
 * Submits a file for ESG analysis. Returns an AnalysisReport JSON object.
 * @param {File} file
 * @returns {Promise<object>} AnalysisReport
 */
export async function analyzeDocument(file) {
  const form = new FormData()
  form.append('file', file)

  const res = await fetch(`${BASE}/analyze`, {
    method: 'POST',
    body: form,
  })

  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(detail.detail ?? `HTTP ${res.status}`)
  }

  return res.json()
}

/**
 * GET /api/v1/health
 * Returns { status: "ok" } when the backend is reachable.
 * @returns {Promise<boolean>} true if healthy
 */
export async function checkHealth() {
  try {
    const res = await fetch(`${BASE}/health`, { signal: AbortSignal.timeout(3000) })
    return res.ok
  } catch {
    return false
  }
}
