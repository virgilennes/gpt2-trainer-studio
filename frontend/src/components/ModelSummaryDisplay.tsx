import type { ModelSummary } from "@/types";

interface ModelSummaryDisplayProps {
  data: ModelSummary;
}

export function ModelSummaryDisplay({ data }: ModelSummaryDisplayProps) {
  return (
    <div data-testid="model-summary" className="rounded-lg border border-border bg-card p-4">
      <h3 className="mb-3 text-sm font-semibold">Model Summary</h3>
      <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
        <dt className="text-muted-foreground">Name</dt>
        <dd data-testid="model-name">{data.name}</dd>

        <dt className="text-muted-foreground">Layers</dt>
        <dd data-testid="model-layers">{data.numLayers}</dd>

        <dt className="text-muted-foreground">Parameters</dt>
        <dd data-testid="model-parameters">{data.numParameters.toLocaleString()}</dd>

        <dt className="text-muted-foreground">Hidden Size</dt>
        <dd data-testid="model-hidden-size">{data.hiddenSize}</dd>

        <dt className="text-muted-foreground">Vocab Size</dt>
        <dd data-testid="model-vocab-size">{data.vocabSize.toLocaleString()}</dd>
      </dl>
    </div>
  );
}
