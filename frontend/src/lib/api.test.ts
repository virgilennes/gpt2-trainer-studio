import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  loadModel,
  getModelSummary,
  prepareDataset,
  getDatasetStats,
  startTraining,
  stopTraining,
  runEvaluation,
  generateText,
  compareGeneration,
  startDemo,
  pauseDemo,
  resumeDemo,
  skipDemoStep,
  getStatus,
  ApiRequestError,
} from "./api";

// Mock global fetch
let mockFetch: ReturnType<typeof vi.fn>;

beforeEach(() => {
  mockFetch = vi.fn();
  vi.stubGlobal("fetch", mockFetch);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

function jsonResponse(data: unknown, status = 200) {
  return Promise.resolve({
    ok: status >= 200 && status < 300,
    status,
    statusText: status === 200 ? "OK" : "Error",
    json: () => Promise.resolve(data),
  } as Response);
}

describe("API client", () => {
  it("loadModel sends POST and returns ModelSummary", async () => {
    const summary = { name: "gpt2", numLayers: 12, numParameters: 124000000, hiddenSize: 768, vocabSize: 50257 };
    mockFetch.mockReturnValueOnce(jsonResponse(summary));

    const result = await loadModel();
    expect(result).toEqual(summary);
    expect(mockFetch).toHaveBeenCalledWith(
      "http://localhost:8000/api/model/load",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("getModelSummary sends GET", async () => {
    const summary = { name: "gpt2", numLayers: 12, numParameters: 124000000, hiddenSize: 768, vocabSize: 50257 };
    mockFetch.mockReturnValueOnce(jsonResponse(summary));

    const result = await getModelSummary();
    expect(result).toEqual(summary);
  });

  it("prepareDataset sends POST", async () => {
    const stats = { trainSamples: 100, valSamples: 20, vocabSize: 50257, blockSize: 128 };
    mockFetch.mockReturnValueOnce(jsonResponse(stats));

    const result = await prepareDataset();
    expect(result).toEqual(stats);
  });

  it("getDatasetStats sends GET", async () => {
    const stats = { trainSamples: 100, valSamples: 20, vocabSize: 50257, blockSize: 128 };
    mockFetch.mockReturnValueOnce(jsonResponse(stats));

    const result = await getDatasetStats();
    expect(result).toEqual(stats);
  });

  it("startTraining maps camelCase config to snake_case body", async () => {
    mockFetch.mockReturnValueOnce(jsonResponse({ status: "started" }));

    await startTraining({
      learningRate: 5e-5,
      batchSize: 8,
      numEpochs: 3,
      warmupSteps: 100,
      weightDecay: 0.01,
    });

    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body).toEqual({
      learning_rate: 5e-5,
      batch_size: 8,
      num_epochs: 3,
      warmup_steps: 100,
      weight_decay: 0.01,
    });
  });

  it("stopTraining sends POST", async () => {
    mockFetch.mockReturnValueOnce(jsonResponse({ status: "stopped" }));
    const result = await stopTraining();
    expect(result.status).toBe("stopped");
  });

  it("runEvaluation sends POST", async () => {
    const evalResult = { perplexity: 25.3, valLoss: 3.2 };
    mockFetch.mockReturnValueOnce(jsonResponse(evalResult));
    const result = await runEvaluation();
    expect(result).toEqual(evalResult);
  });

  it("generateText maps params to snake_case", async () => {
    mockFetch.mockReturnValueOnce(jsonResponse({ text: "Hello world", tokensGenerated: 10 }));

    await generateText({
      prompt: "Hello",
      temperature: 0.8,
      topK: 50,
      topP: 0.9,
      maxLength: 100,
    });

    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body).toEqual({
      prompt: "Hello",
      temperature: 0.8,
      top_k: 50,
      top_p: 0.9,
      max_length: 100,
    });
  });

  it("compareGeneration sends POST", async () => {
    const compareResult = { baselineText: "a", trainedText: "b", prompt: "Hello" };
    mockFetch.mockReturnValueOnce(jsonResponse(compareResult));

    const result = await compareGeneration({
      prompt: "Hello",
      temperature: 1.0,
      topK: 50,
      topP: 1.0,
      maxLength: 100,
    });
    expect(result).toEqual(compareResult);
  });

  it("startDemo sends POST with config", async () => {
    mockFetch.mockReturnValueOnce(jsonResponse({ status: "started" }));
    await startDemo({ speed: "fast" });
    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body).toEqual({ speed: "fast" });
  });

  it("pauseDemo sends POST", async () => {
    mockFetch.mockReturnValueOnce(jsonResponse({ status: "paused" }));
    const result = await pauseDemo();
    expect(result.status).toBe("paused");
  });

  it("resumeDemo sends POST", async () => {
    mockFetch.mockReturnValueOnce(jsonResponse({ status: "resumed" }));
    const result = await resumeDemo();
    expect(result.status).toBe("resumed");
  });

  it("skipDemoStep sends POST", async () => {
    mockFetch.mockReturnValueOnce(jsonResponse({ status: "skipped" }));
    const result = await skipDemoStep();
    expect(result.status).toBe("skipped");
  });

  it("getStatus sends GET", async () => {
    mockFetch.mockReturnValueOnce(jsonResponse({ status: "ok", stage: "idle" }));
    const result = await getStatus();
    expect(result.stage).toBe("idle");
  });
});

describe("API error handling", () => {
  it("throws ApiRequestError with error body on non-ok response", async () => {
    mockFetch.mockReturnValueOnce(
      jsonResponse({ errorCode: "MODEL_LOAD_FAILED", message: "OOM", details: null }, 500),
    );

    await expect(loadModel()).rejects.toThrow(ApiRequestError);
  });

  it("throws ApiRequestError with HTTP status when body is not JSON", async () => {
    mockFetch.mockReturnValueOnce(
      Promise.resolve({
        ok: false,
        status: 502,
        statusText: "Bad Gateway",
        json: () => Promise.reject(new Error("not json")),
      } as Response),
    );

    await expect(loadModel()).rejects.toThrow("Bad Gateway");
  });

  it("throws ApiRequestError on network failure", async () => {
    mockFetch.mockRejectedValueOnce(new Error("Failed to fetch"));

    await expect(loadModel()).rejects.toThrow("Network error");
  });
});
