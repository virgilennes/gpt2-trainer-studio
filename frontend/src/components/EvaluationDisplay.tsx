import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import type { EvalResult, ComparisonResult, TrainingMetrics } from "@/types";

interface EvaluationDisplayProps {
  evalResult: EvalResult;
  comparisonResult: ComparisonResult | null;
  metricsHistory: TrainingMetrics[];
}

export function EvaluationDisplay({
  evalResult,
  comparisonResult,
  metricsHistory,
}: EvaluationDisplayProps) {
  // Build perplexity trend from validation losses in metrics history
  const perplexityTrend = metricsHistory
    .filter((m) => m.valLoss !== null)
    .map((m, i) => ({
      index: i + 1,
      perplexity: Math.exp(m.valLoss!),
    }));

  return (
    <div data-testid="evaluation-display" className="rounded-lg border border-border bg-card p-4">
      <h3 className="mb-3 text-sm font-semibold">Evaluation Results</h3>

      {/* Perplexity value */}
      <div className="mb-4 flex items-baseline gap-2">
        <span className="text-muted-foreground text-sm">Perplexity:</span>
        <span data-testid="eval-perplexity" className="text-2xl font-bold">
          {evalResult.perplexity.toFixed(2)}
        </span>
      </div>

      {/* Baseline comparison */}
      {comparisonResult && (
        <div data-testid="eval-comparison" className="mb-4 rounded-md bg-muted/50 p-3 text-sm">
          <div className="flex justify-between">
            <span className="text-muted-foreground">Baseline Perplexity</span>
            <span data-testid="eval-baseline-perplexity">
              {comparisonResult.baselinePerplexity.toFixed(2)}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Trained Perplexity</span>
            <span data-testid="eval-trained-perplexity">
              {comparisonResult.trainedPerplexity.toFixed(2)}
            </span>
          </div>
          <div className="flex justify-between font-medium">
            <span className="text-muted-foreground">Improvement</span>
            <span data-testid="eval-improvement">
              {comparisonResult.improvementPct.toFixed(1)}%
            </span>
          </div>
        </div>
      )}

      {/* Perplexity trend chart */}
      {perplexityTrend.length > 0 && (
        <div data-testid="eval-trend-chart" className="h-48">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={perplexityTrend}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
              <XAxis dataKey="index" label={{ value: "Eval Point", position: "insideBottom", offset: -5 }} />
              <YAxis label={{ value: "Perplexity", angle: -90, position: "insideLeft" }} />
              <Tooltip />
              <Line type="linear" dataKey="perplexity" stroke="#2563eb" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
