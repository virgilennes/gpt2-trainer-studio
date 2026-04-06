import type { DatasetStats } from "@/types";

interface DatasetStatsDisplayProps {
  data: DatasetStats;
}

export function DatasetStatsDisplay({ data }: DatasetStatsDisplayProps) {
  return (
    <div data-testid="dataset-stats" className="rounded-lg border border-border bg-card p-4">
      <h3 className="mb-3 text-sm font-semibold">Dataset Statistics</h3>
      <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
        <dt className="text-muted-foreground">Vocabulary Size</dt>
        <dd data-testid="dataset-vocab-size">{data.vocabSize.toLocaleString()}</dd>

        <dt className="text-muted-foreground">Sequence Length</dt>
        <dd data-testid="dataset-block-size">{data.blockSize}</dd>

        <dt className="text-muted-foreground">Training Samples</dt>
        <dd data-testid="dataset-train-samples">{data.trainSamples.toLocaleString()}</dd>

        <dt className="text-muted-foreground">Validation Samples</dt>
        <dd data-testid="dataset-val-samples">{data.valSamples.toLocaleString()}</dd>
      </dl>
    </div>
  );
}
