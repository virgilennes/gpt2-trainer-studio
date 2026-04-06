import { useCallback, useEffect, useRef, useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import { useAppState } from "@/lib/state";
import type { PipelineState } from "@/types";

// ---------------------------------------------------------------------------
// Default commentary for each pipeline stage
// ---------------------------------------------------------------------------

const stageCommentary: Record<PipelineState["stage"], string> = {
  idle:
    "Welcome! This tool walks you through building a GPT-2 text generator. " +
    'Click "Load Model" in the control panel to begin.',
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
    "Watch the loss curves \u2014 training loss should decrease over time. " +
    "The learning rate follows a warm-up schedule, ramping up then decaying. " +
    "Checkpoints are saved at the end of each epoch.",
  trained:
    "Training complete! The model has been fine-tuned. " +
    "The final training loss and learning rate are shown above. " +
    "Run evaluation to measure how well the model learned from the data.",
  evaluating:
    "Evaluating the model on the validation set. " +
    "Perplexity measures how surprised the model is by unseen text \u2014 lower is better. " +
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

// ---------------------------------------------------------------------------
// Stage-to-color mapping for left-border accents
// ---------------------------------------------------------------------------

const STAGE_COLORS: Record<string, string> = {
  idle: "border-l-zinc-500",
  model_loading: "border-l-blue-500",
  model_loaded: "border-l-blue-400",
  dataset_preparing: "border-l-emerald-500",
  dataset_ready: "border-l-emerald-400",
  training: "border-l-amber-500",
  trained: "border-l-amber-400",
  evaluating: "border-l-purple-500",
  evaluated: "border-l-purple-400",
  generating: "border-l-rose-500",
};

const STAGE_BG: Record<string, string> = {
  idle: "bg-zinc-500/5",
  model_loading: "bg-blue-500/10",
  model_loaded: "bg-blue-400/10",
  dataset_preparing: "bg-emerald-500/10",
  dataset_ready: "bg-emerald-400/10",
  training: "bg-amber-500/10",
  trained: "bg-amber-400/10",
  evaluating: "bg-purple-500/10",
  evaluated: "bg-purple-400/10",
  generating: "bg-rose-500/10",
};

function extractStage(text: string): string {
  const match = text.match(/^\[(\w+)\]\s/);
  return match ? match[1] : "idle";
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const MIN_HEIGHT = 80;
const MAX_HEIGHT = 500;
const DEFAULT_HEIGHT = 192;

export function CommentaryPanel() {
  const { state } = useAppState();
  const { commentary, pipeline } = state;
  const scrollRef = useRef<HTMLDivElement>(null);
  const [height, setHeight] = useState(DEFAULT_HEIGHT);
  const isResizing = useRef(false);
  const [collapsed, setCollapsed] = useState(false);
  const [stageHistory, setStageHistory] = useState<string[]>([]);
  const lastStageRef = useRef<string | null>(null);

  // Track stage changes and accumulate stage commentary into history
  useEffect(() => {
    const currentStage = pipeline.stage;
    if (currentStage !== lastStageRef.current) {
      lastStageRef.current = currentStage;
      const text = stageCommentary[currentStage];
      if (text) {
        setStageHistory((prev) => [...prev, `[${currentStage}] ${text}`]);
      }
    }
  }, [pipeline.stage]);

  // Merge stage history + WebSocket commentary in chronological order
  const entries: string[] = [...stageHistory, ...commentary];

  // Auto-scroll to latest entry
  useEffect(() => {
    const el = scrollRef.current;
    if (el) {
      el.scrollTop = el.scrollHeight;
    }
  }, [stageHistory.length, commentary.length]);

  // Resize handle
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    isResizing.current = true;
    const startY = e.clientY;
    const startHeight = height;

    const onMouseMove = (ev: MouseEvent) => {
      if (!isResizing.current) return;
      const newHeight = Math.max(MIN_HEIGHT, Math.min(MAX_HEIGHT, startHeight + startY - ev.clientY));
      setHeight(newHeight);
    };

    const onMouseUp = () => {
      isResizing.current = false;
      document.removeEventListener("mousemove", onMouseMove);
      document.removeEventListener("mouseup", onMouseUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };

    document.body.style.cursor = "row-resize";
    document.body.style.userSelect = "none";
    document.addEventListener("mousemove", onMouseMove);
    document.addEventListener("mouseup", onMouseUp);
  }, [height]);

  return (
    <div data-testid="commentary-panel" className="flex flex-col border-t border-white/10">
      {/* Resize handle */}
      <div
        className="flex h-1.5 cursor-row-resize items-center justify-center hover:bg-primary/20 active:bg-primary/30 transition-colors"
        onMouseDown={handleMouseDown}
        role="separator"
        aria-orientation="horizontal"
        aria-label="Resize commentary panel"
      >
        <div className="w-8 h-0.5 rounded-full bg-border" />
      </div>

      <button
        onClick={() => setCollapsed((v) => !v)}
        className="flex w-full items-center gap-2 border-b border-white/10 px-4 py-2 hover:bg-white/5 transition-colors text-left"
        aria-expanded={!collapsed}
      >
        {collapsed ? <ChevronRight className="h-3.5 w-3.5 text-muted-foreground" /> : <ChevronDown className="h-3.5 w-3.5 text-muted-foreground" />}
        <span className="text-sm font-semibold" aria-hidden="true">📋</span>
        <h2 className="text-sm font-semibold">Activity Log</h2>
      </button>
      {!collapsed && (
      <div
        ref={scrollRef}
        className="flex flex-col gap-2 overflow-y-auto p-4"
        style={{ height }}
        role="log"
        aria-live="polite"
        aria-label="Pipeline commentary"
      >
        {entries.map((text, i) => {
          const stage = extractStage(text);
          const borderColor = STAGE_COLORS[stage] || "border-l-zinc-500";
          const bgColor = STAGE_BG[stage] || "bg-zinc-500/5";
          const isLatest = i === entries.length - 1;

          return (
            <div
              key={`${i}-${text.slice(0, 30)}`}
              className={`animate-fade-in-up rounded-r-md border-l-3 pl-3 pr-2 py-2 ${borderColor} ${bgColor} transition-colors`}
              data-testid={i < stageHistory.length ? "stage-commentary" : "ws-commentary"}
            >
              <p className={`text-sm leading-relaxed ${isLatest ? "font-medium text-foreground" : "text-muted-foreground"}`}>
                {text}
              </p>
            </div>
          );
        })}
      </div>
      )}
    </div>
  );
}
