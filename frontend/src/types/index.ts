export interface TrainingConfig {
  learningRate: number;
  batchSize: number;
  numEpochs: number;
  warmupSteps: number;
  weightDecay: number;
}

export interface GenerationParams {
  prompt: string;
  temperature: number;
  topK: number;
  topP: number;
  maxLength: number;
}

export type DemoHighlight = "ControlPanel" | "ProgressPanel" | "GenerationPanel" | null;

export interface PipelineState {
  stage:
    | "idle"
    | "model_loading"
    | "model_loaded"
    | "dataset_preparing"
    | "dataset_ready"
    | "training"
    | "trained"
    | "evaluating"
    | "evaluated"
    | "generating";
  isDemo: boolean;
  demoSpeed: "fast" | "medium" | "slow";
  demoPaused: boolean;
  demoHighlight: DemoHighlight;
}

export interface TrainingMetrics {
  epoch: number;
  step: number;
  trainLoss: number;
  valLoss: number | null;
  learningRate: number;
  elapsedSeconds: number;
  estimatedRemainingSeconds: number | null;
}

export interface ModelSummary {
  name: string;
  numLayers: number;
  numParameters: number;
  hiddenSize: number;
  vocabSize: number;
}

export interface DatasetStats {
  trainSamples: number;
  valSamples: number;
  vocabSize: number;
  blockSize: number;
}

export interface EvalResult {
  perplexity: number;
  valLoss: number;
}

export interface ComparisonResult {
  baselinePerplexity: number;
  trainedPerplexity: number;
  improvementPct: number;
}

export interface GenerationResult {
  text: string;
  tokensGenerated: number;
}

export interface WSMessage {
  type:
    | "progress"
    | "metrics"
    | "commentary"
    | "error"
    | "state_change"
    | "demo_step";
  payload: Record<string, unknown>;
  timestamp: string;
}
