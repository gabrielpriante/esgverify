import { useState, useRef } from 'react'
import { analyzeDocument } from '../api.js'

const ACCEPTED = ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/plain']
const ACCEPTED_EXT = ['.pdf', '.docx', '.txt']
const MAX_MB = 50

export default function UploadView({ onStart, onComplete }) {
  const [dragging, setDragging] = useState(false)
  const [file, setFile] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const inputRef = useRef()

  function validate(f) {
    if (!f) return 'No file selected.'
    if (f.size > MAX_MB * 1024 * 1024) return `File exceeds ${MAX_MB} MB limit.`
    const ext = f.name.split('.').pop().toLowerCase()
    if (!['pdf', 'docx', 'txt'].includes(ext)) return `Unsupported type. Accepted: PDF, DOCX, TXT.`
    return ''
  }

  function handleSelect(f) {
    const err = validate(f)
    setError(err)
    setFile(err ? null : f)
  }

  function onDrop(e) {
    e.preventDefault()
    setDragging(false)
    handleSelect(e.dataTransfer.files[0])
  }

  async function handleSubmit() {
    if (!file || loading) return
    setLoading(true)
    setError('')
    onStart(file.name)
    try {
      const report = await analyzeDocument(file)
      onComplete(report)
    } catch (err) {
      setError(err.message)
      setLoading(false)
    }
  }

  return (
    <div className="max-w-2xl mx-auto mt-20 px-6">
      {/* Title block */}
      <div className="mb-10">
        <h1 className="text-2xl font-semibold text-gray-900 tracking-tight">
          ESG Claim Analysis
        </h1>
        <p className="text-gray-500 mt-1 text-sm">
          Upload a corporate sustainability document. The pipeline extracts ESG claims
          and scores each for greenwashing risk — entirely locally, no data leaves your machine.
        </p>
      </div>

      {/* Drop zone */}
      <div
        onDragOver={e => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        className={`border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition-colors
          ${dragging ? 'border-blue-400 bg-blue-50' : 'border-gray-300 bg-white hover:border-gray-400 hover:bg-gray-50'}`}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".pdf,.docx,.txt"
          className="hidden"
          onChange={e => handleSelect(e.target.files[0])}
        />
        <div className="text-3xl mb-3 select-none">📄</div>
        {file ? (
          <div>
            <p className="font-medium text-gray-800 mono text-sm">{file.name}</p>
            <p className="text-gray-400 text-xs mt-1">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
          </div>
        ) : (
          <div>
            <p className="text-gray-600 font-medium text-sm">Drop a file here, or click to browse</p>
            <p className="text-gray-400 text-xs mt-1">PDF · DOCX · TXT · max {MAX_MB} MB</p>
          </div>
        )}
      </div>

      {/* Error */}
      {error && (
        <p className="mt-3 text-red-600 text-sm mono">{error}</p>
      )}

      {/* Submit */}
      <button
        onClick={handleSubmit}
        disabled={!file || loading}
        className={`mt-5 w-full py-3 rounded-lg text-sm font-medium transition-colors mono
          ${file && !loading
            ? 'bg-gray-900 text-white hover:bg-gray-700 cursor-pointer'
            : 'bg-gray-200 text-gray-400 cursor-not-allowed'}`}
      >
        Run Analysis
      </button>

      {/* Disclaimer */}
      <p className="text-xs text-gray-400 mt-6 leading-relaxed">
        ESGVerify assists with early screening only. Results do not constitute legal advice,
        regulatory compliance assessment, or certification of any kind.
      </p>
    </div>
  )
}
