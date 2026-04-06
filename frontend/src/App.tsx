import { useCallback, useRef, useState } from 'react'
import { PanelLeftClose, PanelLeftOpen, Box, Database, Cog, Play, BarChart3, MessageSquare, Info } from 'lucide-react'
import './App.css'
import { ConnectionStatus } from './components/ConnectionStatus'
import { ControlPanel } from './components/ControlPanel'
import { ProgressPanel } from './components/ProgressPanel'
import { CommentaryPanel } from './components/CommentaryPanel'
import { GenerationPanel } from './components/GenerationPanel'
import { AboutModal } from './components/AboutModal'
import { useAppState } from './lib/state'
import type { PipelineState } from './types'

const PIPELINE_STEPS: { key: string; label: string; icon: typeof Box; stages: PipelineState['stage'][] }[] = [
  { key: 'load', label: 'Load Model', icon: Box, stages: ['model_loading', 'model_loaded'] },
  { key: 'prepare', label: 'Prepare Data', icon: Database, stages: ['dataset_preparing', 'dataset_ready'] },
  { key: 'train', label: 'Train', icon: Cog, stages: ['training', 'trained'] },
  { key: 'evaluate', label: 'Evaluate', icon: BarChart3, stages: ['evaluating', 'evaluated'] },
  { key: 'generate', label: 'Generate', icon: MessageSquare, stages: ['generating'] },
]

function PipelineStepper({ currentStage }: { currentStage: PipelineState['stage'] }) {
  // Determine which step index we're at
  const currentIdx = PIPELINE_STEPS.findIndex((s) => s.stages.includes(currentStage))

  return (
    <nav aria-label="Pipeline progress" className="flex items-center gap-1 overflow-x-auto px-4 py-2">
      {PIPELINE_STEPS.map((step, i) => {
        const isActive = step.stages.includes(currentStage)
        const isCompleted = i < currentIdx
        const Icon = step.icon

        return (
          <div key={step.key} className="flex items-center">
            <div className={`flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium transition-colors ${
              isActive
                ? 'bg-primary text-primary-foreground'
                : isCompleted
                  ? 'bg-primary/20 text-primary'
                  : 'bg-muted text-muted-foreground'
            }`}>
              <Icon className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">{step.label}</span>
            </div>
            {i < PIPELINE_STEPS.length - 1 && (
              <div className={`mx-1 h-0.5 w-4 rounded-full transition-colors ${
                isCompleted ? 'bg-primary/40' : 'bg-muted'
              }`} />
            )}
          </div>
        )
      })}
    </nav>
  )
}

function LoadingIndicator({ message }: { message: string }) {
  return (
    <div
      data-testid="loading-indicator"
      className="flex items-center gap-3 rounded-lg border border-border bg-muted/50 px-4 py-3"
      role="status"
      aria-live="polite"
    >
      <svg
        className="h-5 w-5 animate-spin text-primary"
        xmlns="http://www.w3.org/2000/svg"
        fill="none"
        viewBox="0 0 24 24"
        aria-hidden="true"
      >
        <circle
          className="opacity-25"
          cx="12"
          cy="12"
          r="10"
          stroke="currentColor"
          strokeWidth="4"
        />
        <path
          className="opacity-75"
          fill="currentColor"
          d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
        />
      </svg>
      <div className="flex flex-col">
        <span className="text-sm font-medium">{message}</span>
        <div className="mt-1.5 h-1.5 w-48 overflow-hidden rounded-full bg-muted">
          <div className="h-full animate-pulse rounded-full bg-primary/60" style={{ width: '60%' }} />
        </div>
      </div>
    </div>
  )
}

function App() {
  const { state } = useAppState()
  const stage = state.pipeline.stage
  const highlight = state.pipeline.demoHighlight

  const isModelLoading = stage === 'model_loading'
  const isDatasetPreparing = stage === 'dataset_preparing'

  const highlightClass = "ring-2 ring-primary rounded-lg transition-shadow"

  // Collapsible + resizable sidebar
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [sidebarWidth, setSidebarWidth] = useState(420)
  const [aboutOpen, setAboutOpen] = useState(false)
  const isResizing = useRef(false)

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    isResizing.current = true

    const startX = e.clientX
    const startWidth = sidebarWidth

    const onMouseMove = (ev: MouseEvent) => {
      if (!isResizing.current) return
      const newWidth = Math.max(240, Math.min(600, startWidth + ev.clientX - startX))
      setSidebarWidth(newWidth)
    }

    const onMouseUp = () => {
      isResizing.current = false
      document.removeEventListener('mousemove', onMouseMove)
      document.removeEventListener('mouseup', onMouseUp)
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
    }

    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
    document.addEventListener('mousemove', onMouseMove)
    document.addEventListener('mouseup', onMouseUp)
  }, [sidebarWidth])

  const collapsedWidth = 48

  return (
    <div className="flex h-screen flex-col bg-gradient-to-br from-background via-background to-primary/5">
      {/* Top bar */}
      <header
        data-testid="top-bar"
        className="flex shrink-0 items-center justify-between border-b border-white/10 bg-card/70 backdrop-blur-xl px-4 py-2"
      >
        <div className="flex items-center gap-2">
          <button
            onClick={() => setSidebarOpen((v) => !v)}
            className="hidden lg:inline-flex items-center justify-center rounded-md p-1.5 text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
            aria-label={sidebarOpen ? 'Collapse sidebar' : 'Expand sidebar'}
          >
            {sidebarOpen ? <PanelLeftClose className="h-5 w-5" /> : <PanelLeftOpen className="h-5 w-5" />}
          </button>
          <h1 className="text-base font-semibold">GPT Text Generator</h1>
          <span className="text-xs text-muted-foreground">© Virgil Ennes</span>
          <button
            onClick={() => setAboutOpen(true)}
            className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
            aria-label="About this application"
          >
            <Info className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">About</span>
          </button>
        </div>
        <ConnectionStatus />
      </header>

      {/* Pipeline stepper */}
      <div className="shrink-0 border-b border-white/10 bg-card/40 backdrop-blur-lg">
        <PipelineStepper currentStage={stage} />
      </div>

      {/* Loading indicators */}
      {(isModelLoading || isDatasetPreparing) && (
        <div className="shrink-0 border-b border-white/10 bg-card/40 backdrop-blur-lg px-4 py-2">
          <LoadingIndicator
            message={
              isModelLoading
                ? 'Loading model and tokenizer…'
                : 'Preparing dataset…'
            }
          />
        </div>
      )}

      {/* Main content: collapsible sidebar + right column (progress + bottom panels) */}
      <div className="flex min-h-0 flex-1 flex-col lg:flex-row">
        {/* Left sidebar — ControlPanel (spans full height) */}
        <div
          className={`shrink-0 overflow-hidden border-b border-white/10 lg:border-b-0 lg:border-r lg:border-white/10 bg-card/50 backdrop-blur-xl transition-[width] duration-300 ease-in-out ${highlight === "ControlPanel" ? highlightClass : ""}`}
          style={{ width: window.innerWidth >= 1024 ? (sidebarOpen ? sidebarWidth : collapsedWidth) : undefined }}
        >
          {sidebarOpen ? (
            <div className="h-full overflow-y-auto">
              <ControlPanel />
            </div>
          ) : (
            <div className="flex h-full flex-col items-center gap-3 py-4">
              <button
                onClick={() => setSidebarOpen(true)}
                className="rounded-md p-1.5 text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
                aria-label="Expand sidebar"
              >
                <PanelLeftOpen className="h-5 w-5" />
              </button>
            </div>
          )}
        </div>

        {/* Resize handle (desktop only, visible when sidebar is open) */}
        {sidebarOpen && (
          <div
            className="hidden lg:flex w-1.5 cursor-col-resize items-center justify-center hover:bg-primary/20 active:bg-primary/30 transition-colors"
            onMouseDown={handleMouseDown}
            role="separator"
            aria-orientation="vertical"
            aria-label="Resize sidebar"
          >
            <div className="h-8 w-0.5 rounded-full bg-border" />
          </div>
        )}

        {/* Right column: progress panel + bottom panels */}
        <div className="flex min-h-0 flex-1 flex-col">
          {/* Main area — ProgressPanel */}
          <div className={`min-h-0 flex-1 overflow-y-auto ${highlight === "ProgressPanel" ? highlightClass : ""}`}>
            <ProgressPanel />
          </div>

          {/* Bottom panels */}
          <div className="shrink-0 border-t border-white/10 bg-card/40 backdrop-blur-lg">
            <CommentaryPanel />
            <div className={highlight === "GenerationPanel" ? highlightClass : ""}>
              <GenerationPanel />
            </div>
          </div>
        </div>
      </div>

      {/* About modal */}
      <AboutModal open={aboutOpen} onClose={() => setAboutOpen(false)} />
    </div>
  )
}

export default App
