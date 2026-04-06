import { describe, it, expect } from "vitest";
import { appReducer, initialState, type AppState, type AppAction } from "./state";
import type { WSMessage, TrainingMetrics } from "../types";

describe("appReducer", () => {
  it("returns initial state for unknown action", () => {
    const result = appReducer(initialState, { type: "UNKNOWN" } as unknown as AppAction);
    expect(result).toBe(initialState);
  });

  it("SET_PIPELINE_STAGE updates stage and clears error", () => {
    const stateWithError: AppState = { ...initialState, error: "old error" };
    const result = appReducer(stateWithError, {
      type: "SET_PIPELINE_STAGE",
      stage: "model_loading",
    });
    expect(result.pipeline.stage).toBe("model_loading");
    expect(result.error).toBeNull();
  });

  it("SET_CONNECTION_STATUS updates connectionStatus", () => {
    const result = appReducer(initialState, {
      type: "SET_CONNECTION_STATUS",
      status: "connected",
    });
    expect(result.connectionStatus).toBe("connected");
  });

  it("SET_DEMO_STATE updates demo fields", () => {
    const result = appReducer(initialState, {
      type: "SET_DEMO_STATE",
      isDemo: true,
      demoSpeed: "fast",
      demoPaused: false,
    });
    expect(result.pipeline.isDemo).toBe(true);
    expect(result.pipeline.demoSpeed).toBe("fast");
    expect(result.pipeline.demoPaused).toBe(false);
  });

  it("SET_MODEL_SUMMARY stores model summary", () => {
    const summary = {
      name: "gpt2",
      numLayers: 12,
      numParameters: 124000000,
      hiddenSize: 768,
      vocabSize: 50257,
    };
    const result = appReducer(initialState, {
      type: "SET_MODEL_SUMMARY",
      summary,
    });
    expect(result.modelSummary).toEqual(summary);
  });

  it("SET_DATASET_STATS stores dataset stats", () => {
    const stats = { trainSamples: 100, valSamples: 20, vocabSize: 50257, blockSize: 128 };
    const result = appReducer(initialState, { type: "SET_DATASET_STATS", stats });
    expect(result.datasetStats).toEqual(stats);
  });

  it("ADD_METRICS appends to metricsHistory", () => {
    const m: TrainingMetrics = {
      epoch: 1,
      step: 10,
      trainLoss: 3.5,
      valLoss: null,
      learningRate: 5e-5,
      elapsedSeconds: 30,
      estimatedRemainingSeconds: 60,
    };
    const result = appReducer(initialState, { type: "ADD_METRICS", metrics: m });
    expect(result.metricsHistory).toHaveLength(1);
    expect(result.metricsHistory[0]).toEqual(m);
  });

  it("CLEAR_METRICS resets metricsHistory", () => {
    const stateWithMetrics: AppState = {
      ...initialState,
      metricsHistory: [
        { epoch: 1, step: 1, trainLoss: 3, valLoss: null, learningRate: 5e-5, elapsedSeconds: 1, estimatedRemainingSeconds: null },
      ],
    };
    const result = appReducer(stateWithMetrics, { type: "CLEAR_METRICS" });
    expect(result.metricsHistory).toHaveLength(0);
  });

  it("ADD_COMMENTARY appends text", () => {
    const result = appReducer(initialState, { type: "ADD_COMMENTARY", text: "Hello" });
    expect(result.commentary).toEqual(["Hello"]);
  });

  it("SET_ERROR sets error message", () => {
    const result = appReducer(initialState, { type: "SET_ERROR", message: "fail" });
    expect(result.error).toBe("fail");
  });

  it("SET_PROGRESS sets progress message", () => {
    const result = appReducer(initialState, { type: "SET_PROGRESS", message: "loading..." });
    expect(result.progress).toBe("loading...");
  });
});

describe("appReducer WS_MESSAGE handling", () => {
  it("state_change message updates pipeline stage", () => {
    const msg: WSMessage = {
      type: "state_change",
      payload: { stage: "model_loaded" },
      timestamp: new Date().toISOString(),
    };
    const result = appReducer(initialState, { type: "WS_MESSAGE", message: msg });
    expect(result.pipeline.stage).toBe("model_loaded");
    expect(result.error).toBeNull();
  });

  it("metrics message appends to metricsHistory", () => {
    const msg: WSMessage = {
      type: "metrics",
      payload: {
        epoch: 1,
        step: 5,
        trainLoss: 2.5,
        valLoss: null,
        learningRate: 5e-5,
        elapsedSeconds: 10,
        estimatedRemainingSeconds: 50,
      },
      timestamp: new Date().toISOString(),
    };
    const result = appReducer(initialState, { type: "WS_MESSAGE", message: msg });
    expect(result.metricsHistory).toHaveLength(1);
    expect(result.metricsHistory[0].trainLoss).toBe(2.5);
  });

  it("progress message sets progress", () => {
    const msg: WSMessage = {
      type: "progress",
      payload: { message: "Downloading model..." },
      timestamp: new Date().toISOString(),
    };
    const result = appReducer(initialState, { type: "WS_MESSAGE", message: msg });
    expect(result.progress).toBe("Downloading model...");
  });

  it("commentary message appends text", () => {
    const msg: WSMessage = {
      type: "commentary",
      payload: { text: "GPT-2 is a decoder-only transformer." },
      timestamp: new Date().toISOString(),
    };
    const result = appReducer(initialState, { type: "WS_MESSAGE", message: msg });
    expect(result.commentary).toEqual(["GPT-2 is a decoder-only transformer."]);
  });

  it("error message sets error", () => {
    const msg: WSMessage = {
      type: "error",
      payload: { error_code: "MODEL_LOAD_FAILED", message: "OOM" },
      timestamp: new Date().toISOString(),
    };
    const result = appReducer(initialState, { type: "WS_MESSAGE", message: msg });
    expect(result.error).toBe("OOM");
  });

  it("demo_step message updates pipeline stage and paused state", () => {
    const demoState: AppState = {
      ...initialState,
      pipeline: { ...initialState.pipeline, isDemo: true },
    };
    const msg: WSMessage = {
      type: "demo_step",
      payload: { stage: "training", paused: false },
      timestamp: new Date().toISOString(),
    };
    const result = appReducer(demoState, { type: "WS_MESSAGE", message: msg });
    expect(result.pipeline.stage).toBe("training");
    expect(result.pipeline.demoPaused).toBe(false);
  });

  it("metrics history accumulates across multiple messages", () => {
    let state = initialState;
    for (let i = 0; i < 5; i++) {
      const msg: WSMessage = {
        type: "metrics",
        payload: {
          epoch: 1,
          step: i,
          trainLoss: 3 - i * 0.1,
          valLoss: null,
          learningRate: 5e-5,
          elapsedSeconds: i * 10,
          estimatedRemainingSeconds: null,
        },
        timestamp: new Date().toISOString(),
      };
      state = appReducer(state, { type: "WS_MESSAGE", message: msg });
    }
    expect(state.metricsHistory).toHaveLength(5);
  });
});
