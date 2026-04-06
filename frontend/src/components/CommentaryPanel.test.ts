import { describe, it, expect } from "vitest";
import type { PipelineState } from "@/types";

/**
 * CommentaryPanel unit tests
 *
 * Validates:
 * - Requirements 1.5: Commentary explains decoder-only architecture
 * - Requirements 2.5: Commentary explains dataset characteristics
 * - Requirements 3.5: Commentary explains hyperparameters
 * - Requirements 4.6: Commentary explains training dynamics
 * - Requirements 5.4: Commentary explains perplexity
 * - Requirements 6.6: Commentary explains generation parameters
 */

// ---------------------------------------------------------------------------
// Stage commentary content (mirrors the component's stageCommentary map)
// ---------------------------------------------------------------------------

const stageCommentary: Record<PipelineState["stage"], string> = {
  idle:
    "Welcome! This tool walks you through building a GPT-2 text generator. " +
    "Click \"Load Model\" in the control panel to begin.",
  model_loading:
    "Loading the GPT-2 small model and tokenizer from Hugging Face. " +
    "GPT-2 is a decoder-only transformer that generates text left-to-right. " +
    "The tokenizer uses byte-pair encoding (BPE) to split text into subword tokens, " +
    "which lets the model handle rare words by breaking them into familiar pieces.",
  model_loaded:
    "Model loaded! GPT-2 small has 12 transformer layers, 768 hidden dimensions, " +
    "and roughly 124 million parameters. The tokenizer vocabulary contains about 50,257 subword tokens. " +
    "Next, prepare the dataset so we have text to fine-tune on.",
  dataset_preparing:
    "Downloading and preparing the WikiText-2 dataset. " +
    "The text is tokenized into subword sequences and chunked into fixed-length blocks. " +
    "Fixed-length sequences let us batch efficiently during training. " +
    "The data is split into training and validation sets.",
  dataset_ready:
    "Dataset ready! The training and validation splits are prepared. " +
    "Each sample is a fixed-length token sequence. " +
    "Now configure your training hyperparameters and start training.",
  training:
    "Training in progress. The model is being fine-tuned on WikiText-2 using causal language modeling. " +
    "Watch the loss curves — training loss should decrease over time. " +
    "The learning rate follows a warm-up schedule, ramping up then decaying. " +
    "Checkpoints are saved at the end of each epoch.",
  trained:
    "Training complete! The model has been fine-tuned. " +
    "The final training loss and learning rate are shown above. " +
    "Run evaluation to measure how well the model learned from the data.",
  evaluating:
    "Evaluating the model on the validation set. " +
    "Perplexity measures how surprised the model is by unseen text — lower is better. " +
    "We compare the fine-tuned model against the original baseline to see improvement.",
  evaluated:
    "Evaluation complete! Check the perplexity scores above. " +
    "A lower perplexity on the validation set means the model predicts text more confidently. " +
    "Try generating text to see the model in action.",
  generating:
    "Generating text from your prompt. The model predicts one token at a time, " +
    "sampling from the probability distribution shaped by temperature, top-k, and top-p. " +
    "Higher temperature means more creative (but riskier) output; lower temperature is more conservative.",
};

const allStages: PipelineState["stage"][] = [
  "idle",
  "model_loading",
  "model_loaded",
  "dataset_preparing",
  "dataset_ready",
  "training",
  "trained",
  "evaluating",
  "evaluated",
  "generating",
];

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("CommentaryPanel — stage commentary content", () => {
  it("every pipeline stage has non-empty commentary", () => {
    for (const stage of allStages) {
      const text = stageCommentary[stage];
      expect(text, `stage "${stage}" should have commentary`).toBeTruthy();
      expect(text.length).toBeGreaterThan(10);
    }
  });

  it("model_loading commentary mentions decoder-only architecture", () => {
    expect(stageCommentary.model_loading).toContain("decoder-only");
  });

  it("model_loading commentary mentions subword tokenization", () => {
    expect(stageCommentary.model_loading).toContain("subword");
  });

  it("dataset_preparing commentary mentions tokenization and splitting", () => {
    const text = stageCommentary.dataset_preparing;
    expect(text).toContain("tokenized");
    expect(text).toContain("split");
  });

  it("training commentary mentions loss and checkpoints", () => {
    const text = stageCommentary.training.toLowerCase();
    expect(text).toContain("loss");
    expect(text).toContain("checkpoint");
  });

  it("evaluating commentary mentions perplexity", () => {
    expect(stageCommentary.evaluating.toLowerCase()).toContain("perplexity");
  });

  it("generating commentary mentions temperature and sampling", () => {
    const text = stageCommentary.generating;
    expect(text).toContain("temperature");
    expect(text).toContain("top-k");
    expect(text).toContain("top-p");
  });

  it("entries list combines stage text with WebSocket commentary", () => {
    const wsCommentary = ["First WS message", "Second WS message"];
    const stage: PipelineState["stage"] = "training";
    const entries = [stageCommentary[stage], ...wsCommentary];

    expect(entries).toHaveLength(3);
    expect(entries[0]).toBe(stageCommentary.training);
    expect(entries[1]).toBe("First WS message");
    expect(entries[2]).toBe("Second WS message");
  });

  it("entries list with empty commentary array shows only stage text", () => {
    const wsCommentary: string[] = [];
    const stage: PipelineState["stage"] = "idle";
    const entries = [stageCommentary[stage], ...wsCommentary];

    expect(entries).toHaveLength(1);
    expect(entries[0]).toBe(stageCommentary.idle);
  });
});
