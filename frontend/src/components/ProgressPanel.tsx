import { useEffect, useRef, useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import {
  AreaChart,
  Area,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { useAppState } from "@/lib/state";
import { ModelSummaryDisplay } from "./ModelSummaryDisplay";
import { DatasetStatsDisplay } from "./DatasetStatsDisplay";
import { EvaluationDisplay } from "./EvaluationDisplay";
import type { TrainingMetrics } from "@/types";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

function TrainingLog({ metricsHistory }: { metricsHistory: TrainingMetrics[] }) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [metricsHistory.length]);

  return (
    <div data-testid="training-log" className="rounded-lg border border-border bg-card p-4">
      <h3 className="mb-2 text-sm font-semibold">Training Output</h3>
      <div
        ref={scrollRef}
        className="max-h-60 overflow-y-auto rounded-md bg-zinc-950 p-3 font-mono text-xs text-green-400"
      >
        {metricsHistory.map((m, i) => (
          <div key={i} className="leading-relaxed">
            <span className="text-zinc-500">[Step {m.step}]</span>{" "}
            epoch={m.epoch.toFixed(2)} | train_loss={m.trainLoss.toFixed(4)}
            {m.valLoss != null && <> | val_loss={m.valLoss.toFixed(4)}</>}
            {" "}| lr={m.learningRate.toExponential(2)}
            {" "}| elapsed={formatTime(m.elapsedSeconds)}
            {m.estimatedRemainingSeconds != null && <> | eta={formatTime(m.estimatedRemainingSeconds)}</>}
          </div>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function ProgressPanel() {
  const { state } = useAppState();
  const {
    metricsHistory,
    modelSummary,
    datasetStats,
    evalResult,
    comparisonResult,
    pipeline,
    progress,
  } = state;

  const latest = metricsHistory.length > 0
    ? metricsHistory[metricsHistory.length - 1]
    : null;

  const [progressCollapsed, setProgressCollapsed] = useState(false);

  // Chart data — prepend a zero-point so the line starts from origin
  const lossData = metricsHistory.length > 0
    ? [{ step: 0, trainLoss: metricsHistory[0].trainLoss, valLoss: null as number | null },
       ...metricsHistory.map((m) => ({
         step: m.step,
         trainLoss: m.trainLoss,
         valLoss: m.valLoss,
       }))]
    : [];

  const lrData = metricsHistory.length > 0
    ? [{ step: 0, learningRate: 0 },
       ...metricsHistory.map((m) => ({
         step: m.step,
         learningRate: m.learningRate,
       }))]
    : [];

  const maxStep = metricsHistory.length > 0
    ? metricsHistory[metricsHistory.length - 1].step
    : 0;

  // Epoch progress
  const currentEpoch = latest ? latest.epoch : 0;
  // We approximate total epochs from the max epoch seen so far (rounded up)
  const epochProgress = currentEpoch > 0 ? Math.min((currentEpoch / Math.ceil(currentEpoch)) * 100, 100) : 0;

  return (
    <div data-testid="progress-panel" className="flex flex-1 flex-col gap-4 overflow-y-auto p-4">
      <button
        onClick={() => setProgressCollapsed((v) => !v)}
        className="flex items-center gap-1.5 text-lg font-semibold hover:text-primary transition-colors text-left"
        aria-expanded={!progressCollapsed}
      >
        {progressCollapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
        Progress
      </button>

      {!progressCollapsed && (<>

      {/* Model Summary */}
      {modelSummary && <div className="animate-fade-in-up"><ModelSummaryDisplay data={modelSummary} /></div>}

      {/* Dataset Stats */}
      {datasetStats && <div className="animate-fade-in-up"><DatasetStatsDisplay data={datasetStats} /></div>}

      {/* Epoch progress bar + ETA */}
      {/* Training progress + Training output side by side */}
      {(pipeline.stage === "training" || metricsHistory.length > 0) && latest && (
        <div className="flex flex-col gap-4 lg:flex-row lg:items-start animate-fade-in-up">
          {/* Training Progress ring + stats */}
          <div data-testid="epoch-progress" className="shrink-0 rounded-lg border border-border bg-card p-4">
            <h3 className="mb-3 text-sm font-semibold">Training Progress</h3>
            <div className="flex items-center gap-6">
              {/* Progress ring */}
              <div className="relative shrink-0" role="progressbar" aria-valuenow={epochProgress} aria-valuemin={0} aria-valuemax={100}>
                <svg width="80" height="80" viewBox="0 0 80 80" className="-rotate-90">
                  <circle cx="40" cy="40" r="34" fill="none" stroke="currentColor" strokeWidth="6" className="text-muted" />
                  <circle
                    cx="40" cy="40" r="34" fill="none"
                    stroke="#2563eb"
                    strokeWidth="6"
                    strokeLinecap="round"
                    strokeDasharray={`${2 * Math.PI * 34}`}
                    strokeDashoffset={`${2 * Math.PI * 34 * (1 - epochProgress / 100)}`}
                    className="transition-all duration-500"
                  />
                </svg>
                <span className="absolute inset-0 flex items-center justify-center text-sm font-semibold">
                  {Math.round(epochProgress)}%
                </span>
              </div>
              {/* Stats */}
              <div className="flex flex-col gap-1.5 text-sm">
                <div className="flex gap-3">
                  <span className="text-muted-foreground">Epoch</span>
                  <span data-testid="epoch-current" className="font-medium">{latest.epoch.toFixed(2)}</span>
                </div>
                <div className="flex gap-3">
                  <span className="text-muted-foreground">Step</span>
                  <span data-testid="step-current" className="font-medium">{latest.step}</span>
                </div>
                <div className="flex gap-3">
                  <span className="text-muted-foreground">Elapsed</span>
                  <span data-testid="elapsed-time" className="font-medium">{formatTime(latest.elapsedSeconds)}</span>
                </div>
                <div className="flex gap-3">
                  <span className="text-muted-foreground">ETA</span>
                  <span data-testid="eta-time" className="font-medium">
                    {latest.estimatedRemainingSeconds != null
                      ? formatTime(latest.estimatedRemainingSeconds)
                      : "—"}
                  </span>
                </div>
                <div className="flex gap-3">
                  <span className="text-muted-foreground">Learning Rate</span>
                  <span data-testid="current-lr" className="font-medium">{latest.learningRate.toExponential(2)}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Training output log */}
          {metricsHistory.length > 0 && (
            <div className="min-w-0 flex-1">
              <TrainingLog metricsHistory={metricsHistory} />
            </div>
          )}
        </div>
      )}

      {/* Loss chart */}
      {lossData.length > 0 && (
        <div data-testid="loss-chart" className="rounded-lg border border-border bg-card p-4 animate-fade-in-up">
          <h3 className="mb-2 text-sm font-semibold">Loss Curves</h3>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={lossData}>
                <defs>
                  <linearGradient id="gradTrain" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#2563eb" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#2563eb" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="gradVal" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#dc2626" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#dc2626" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
                <XAxis dataKey="step" type="number" domain={[0, maxStep || 'auto']} />
                <YAxis />
                <Tooltip />
                <Legend />
                <Area
                  type="linear"
                  dataKey="trainLoss"
                  name="Train Loss"
                  stroke="#2563eb"
                  strokeWidth={2}
                  fill="url(#gradTrain)"
                  dot={false}
                  isAnimationActive={false}
                />
                <Area
                  type="linear"
                  dataKey="valLoss"
                  name="Val Loss"
                  stroke="#dc2626"
                  strokeWidth={2}
                  fill="url(#gradVal)"
                  dot={false}
                  connectNulls
                  isAnimationActive={false}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Learning rate schedule chart */}
      {lrData.length > 0 && (
        <div data-testid="lr-chart" className="rounded-lg border border-border bg-card p-4 animate-fade-in-up">
          <h3 className="mb-2 text-sm font-semibold">Learning Rate Schedule</h3>
          <div className="h-40">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={lrData}>
                <defs>
                  <linearGradient id="gradLR" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#2563eb" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#2563eb" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
                <XAxis dataKey="step" type="number" domain={[0, maxStep || 'auto']} />
                <YAxis tickFormatter={(v: number) => v.toExponential(1)} />
                <Tooltip formatter={(v: number) => v.toExponential(3)} />
                <Area
                  type="linear"
                  dataKey="learningRate"
                  name="Learning Rate"
                  stroke="#2563eb"
                  strokeWidth={2}
                  fill="url(#gradLR)"
                  dot={false}
                  isAnimationActive={false}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Evaluation results */}
      {evalResult && (
        <div className="animate-fade-in-up">
          <EvaluationDisplay
            evalResult={evalResult}
            comparisonResult={comparisonResult}
            metricsHistory={metricsHistory}
          />
        </div>
      )}

      {/* Progress message */}
      {progress && (
        <div className="rounded-lg border border-border bg-card p-4">
          <p className="text-sm text-muted-foreground">{progress}</p>
        </div>
      )}

      </>)}

      {/* Empty state */}
      {!modelSummary && !datasetStats && metricsHistory.length === 0 && !evalResult && (
        <div className="flex flex-1 items-center justify-center text-muted-foreground">
          <p>Load a model to get started.</p>
        </div>
      )}
    </div>
  );
}
