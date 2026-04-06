import type {
  TrainingConfig,
  GenerationParams,
  ModelSummary,
  DatasetStats,
  EvalResult,
  GenerationResult,
} from "../types";

const BASE_URL = "http://localhost:8000";

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

export interface ApiError {
  errorCode: string;
  message: string;
  details?: string | null;
}

export class ApiRequestError extends Error {
  errorCode: string;
  details: string | null;

  constructor(errorCode: string, message: string, details?: string | null) {
    super(message);
    this.name = "ApiRequestError";
    this.errorCode = errorCode;
    this.details = details ?? null;
  }
}

async function request<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const url = `${BASE_URL}${path}`;
  const { headers: extraHeaders, ...rest } = options;
  let res: Response;
  try {
    res = await fetch(url, {
      ...rest,
      headers: {
        "Content-Type": "application/json",
        ...(extraHeaders as Record<string, string> | undefined),
      },
    });
  } catch (err) {
    throw new ApiRequestError(
      "NETWORK_ERROR",
      `Network error: ${(err as Error).message}`,
    );
  }

  if (!res.ok) {
    let body: ApiError | undefined;
    try {
      body = (await res.json()) as ApiError;
    } catch {
      // response wasn't JSON
    }
    throw new ApiRequestError(
      body?.errorCode ?? `HTTP_${res.status}`,
      body?.message ?? res.statusText,
      body?.details,
    );
  }

  return (await res.json()) as T;
}

// ---------------------------------------------------------------------------
// Model endpoints
// ---------------------------------------------------------------------------

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function toModelSummary(raw: any): ModelSummary {
  return {
    name: raw.name,
    numLayers: raw.num_layers ?? raw.numLayers,
    numParameters: raw.num_parameters ?? raw.numParameters,
    hiddenSize: raw.hidden_size ?? raw.hiddenSize,
    vocabSize: raw.vocab_size ?? raw.vocabSize,
  };
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function toDatasetStats(raw: any): DatasetStats {
  return {
    trainSamples: raw.train_samples ?? raw.trainSamples,
    valSamples: raw.val_samples ?? raw.valSamples,
    vocabSize: raw.vocab_size ?? raw.vocabSize,
    blockSize: raw.block_size ?? raw.blockSize,
  };
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function toEvalResult(raw: any): EvalResult {
  return {
    perplexity: raw.perplexity,
    valLoss: raw.val_loss ?? raw.valLoss,
  };
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function toGenerationResult(raw: any): GenerationResult {
  return {
    text: raw.text,
    tokensGenerated: raw.tokens_generated ?? raw.tokensGenerated,
  };
}

export async function loadModel(): Promise<ModelSummary> {
  // Response shape: { summary: {...}, tokenizer_info: {...} }
  const data = await request<{ summary: Record<string, unknown> }>("/api/model/load", { method: "POST" });
  return toModelSummary(data.summary);
}

export async function getModelSummary(): Promise<ModelSummary> {
  const raw = await request<Record<string, unknown>>("/api/model/summary");
  return toModelSummary(raw);
}

// ---------------------------------------------------------------------------
// Dataset endpoints
// ---------------------------------------------------------------------------

export async function prepareDataset(): Promise<DatasetStats> {
  // Response shape: { stats: {...} }
  const data = await request<{ stats: Record<string, unknown> }>("/api/dataset/prepare", { method: "POST" });
  return toDatasetStats(data.stats);
}

export async function getDatasetStats(): Promise<DatasetStats> {
  const raw = await request<Record<string, unknown>>("/api/dataset/stats");
  return toDatasetStats(raw);
}

// ---------------------------------------------------------------------------
// Training endpoints
// ---------------------------------------------------------------------------

export async function startTraining(
  config: TrainingConfig,
): Promise<{ status: string }> {
  return request<{ status: string }>("/api/training/start", {
    method: "POST",
    body: JSON.stringify({
      learning_rate: config.learningRate,
      batch_size: config.batchSize,
      num_epochs: config.numEpochs,
      warmup_steps: config.warmupSteps,
      weight_decay: config.weightDecay,
    }),
  });
}

export async function stopTraining(): Promise<{ status: string }> {
  return request<{ status: string }>("/api/training/stop", { method: "POST" });
}

// ---------------------------------------------------------------------------
// Evaluation endpoints
// ---------------------------------------------------------------------------

export async function runEvaluation(): Promise<EvalResult> {
  const raw = await request<Record<string, unknown>>("/api/evaluation/run", { method: "POST" });
  return toEvalResult(raw);
}

// ---------------------------------------------------------------------------
// Generation endpoints
// ---------------------------------------------------------------------------

export async function generateText(
  params: GenerationParams,
): Promise<GenerationResult> {
  const raw = await request<Record<string, unknown>>("/api/generation/generate", {
    method: "POST",
    body: JSON.stringify({
      prompt: params.prompt,
      temperature: params.temperature,
      top_k: params.topK,
      top_p: params.topP,
      max_length: params.maxLength,
    }),
  });
  return toGenerationResult(raw);
}

export interface CompareGenerationResult {
  baselineText: string;
  trainedText: string;
  prompt: string;
}

export async function compareGeneration(
  params: GenerationParams,
): Promise<CompareGenerationResult> {
  const raw = await request<Record<string, unknown>>("/api/generation/compare", {
    method: "POST",
    body: JSON.stringify({
      prompt: params.prompt,
      temperature: params.temperature,
      top_k: params.topK,
      top_p: params.topP,
      max_length: params.maxLength,
    }),
  });
  return {
    baselineText: (raw.baseline_text ?? raw.baselineText) as string,
    trainedText: (raw.trained_text ?? raw.trainedText) as string,
    prompt: raw.prompt as string,
  };
}

// ---------------------------------------------------------------------------
// Demo endpoints
// ---------------------------------------------------------------------------

export interface DemoConfig {
  speed: "fast" | "medium" | "slow";
}

export async function startDemo(
  config: DemoConfig,
): Promise<{ status: string }> {
  return request<{ status: string }>("/api/demo/start", {
    method: "POST",
    body: JSON.stringify(config),
  });
}

export async function pauseDemo(): Promise<{ status: string }> {
  return request<{ status: string }>("/api/demo/pause", { method: "POST" });
}

export async function resumeDemo(): Promise<{ status: string }> {
  return request<{ status: string }>("/api/demo/resume", { method: "POST" });
}

export async function skipDemoStep(): Promise<{ status: string }> {
  return request<{ status: string }>("/api/demo/skip", { method: "POST" });
}

// ---------------------------------------------------------------------------
// Status endpoint
// ---------------------------------------------------------------------------

export interface StatusResponse {
  status: string;
  stage: string;
  [key: string]: unknown;
}

export async function getStatus(): Promise<StatusResponse> {
  return request<StatusResponse>("/api/status");
}
