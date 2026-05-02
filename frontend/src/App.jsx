import { useState } from 'react'
import HealthBar from './components/HealthBar.jsx'
import UploadView from './components/UploadView.jsx'
import AnalyzingView from './components/AnalyzingView.jsx'
import ResultsView from './components/ResultsView.jsx'

// View states: 'upload' | 'analyzing' | 'results'
export default function App() {
  const [view, setView] = useState('upload')
  const [report, setReport] = useState(null)
  const [filename, setFilename] = useState('')

  function handleAnalysisStart(name) {
    setFilename(name)
    setView('analyzing')
  }

  function handleAnalysisComplete(data) {
    setReport(data)
    setView('results')
  }

  function handleReset() {
    setReport(null)
    setFilename('')
    setView('upload')
  }

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      {/* Header */}
      <header className="bg-gray-900 text-white px-6 py-3 flex items-center justify-between border-b border-gray-700">
        <div className="flex items-center gap-3">
          <span className="mono text-green-400 text-lg font-medium tracking-tight">ESGVerify</span>
          <span className="text-gray-500 text-xs mono">v0.2.0</span>
        </div>
        <HealthBar />
      </header>

      {/* Main content */}
      <main className="flex-1">
        {view === 'upload' && (
          <UploadView
            onStart={handleAnalysisStart}
            onComplete={handleAnalysisComplete}
          />
        )}
        {view === 'analyzing' && (
          <AnalyzingView
            filename={filename}
            onComplete={handleAnalysisComplete}
          />
        )}
        {view === 'results' && (
          <ResultsView
            report={report}
            onReset={handleReset}
          />
        )}
      </main>
    </div>
  )
}
