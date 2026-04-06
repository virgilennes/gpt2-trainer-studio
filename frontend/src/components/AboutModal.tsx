import { useEffect, useRef } from "react";
import { X, Box, Database, Cog, Play, BarChart3, MessageSquare, Sparkles, Users, Eye } from "lucide-react";

interface AboutModalProps {
  open: boolean;
  onClose: () => void;
}

export function AboutModal({ open, onClose }: AboutModalProps) {
  const dialogRef = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const el = dialogRef.current;
    if (!el) return;
    if (open && !el.open) el.showModal();
    else if (!open && el.open) el.close();
  }, [open]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    if (open) document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <dialog
      ref={dialogRef}
      className="fixed inset-0 z-50 m-auto h-[85vh] w-[90vw] max-w-3xl rounded-xl border border-white/10 bg-zinc-900/95 backdrop-blur-xl p-0 shadow-2xl text-white"
      onClick={(e) => { if (e.target === dialogRef.current) onClose(); }}
    >
      <div className="flex h-full flex-col">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-white/10 px-6 py-4">
          <div className="flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-blue-400" />
            <h2 className="text-lg font-semibold">About GPT Text Generator</h2>
          </div>
          <button
            onClick={onClose}
            className="rounded-md p-1.5 text-zinc-400 hover:bg-zinc-800 hover:text-white transition-colors"
            aria-label="Close"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Scrollable content */}
        <div className="flex-1 overflow-y-auto px-6 py-6 space-y-6">

          {/* Intro */}
          <div className="rounded-lg border border-white/10 bg-white/5 p-5">
            <p className="text-sm leading-relaxed text-zinc-200">
              An educational full-stack application that walks you through the complete
              machine learning pipeline for building, training, and testing a
              <span className="font-semibold text-blue-400"> GPT-2 language model</span>.
              See how modern generative AI works — from raw data to generated text.
            </p>
          </div>

          {/* Use Case + Benefits side by side */}
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="rounded-lg border border-white/10 bg-white/5 p-4 space-y-2">
              <div className="flex items-center gap-2">
                <Users className="h-4 w-4 text-emerald-400" />
                <h3 className="text-sm font-semibold">Who is it for?</h3>
              </div>
              <p className="text-xs leading-relaxed text-zinc-300">
                Developers, students, and ML enthusiasts who want to understand how large
                language models are fine-tuned — without treating the model as a black box.
              </p>
            </div>
            <div className="rounded-lg border border-white/10 bg-white/5 p-4 space-y-2">
              <div className="flex items-center gap-2">
                <Eye className="h-4 w-4 text-amber-400" />
                <h3 className="text-sm font-semibold">What you get</h3>
              </div>
              <ul className="text-xs leading-relaxed text-zinc-300 space-y-1 list-none pl-0">
                <li>✦ Hands-on Transformer &amp; GPT-2 understanding</li>
                <li>✦ Real-time loss curves &amp; LR schedule visualization</li>
                <li>✦ Interactive generation parameter experimentation</li>
                <li>✦ Baseline vs. fine-tuned model comparison</li>
                <li>✦ Automated demo mode with commentary</li>
              </ul>
            </div>
          </div>

          {/* Pipeline Steps */}
          <div>
            <h3 className="text-sm font-semibold mb-3 text-zinc-100">Pipeline Steps</h3>
            <div className="space-y-3">

              {/* Step 1 */}
              <div className="flex gap-3 rounded-lg border border-blue-500/20 bg-blue-500/5 p-4">
                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-blue-500/20 text-blue-400">
                  <Box className="h-4 w-4" />
                </div>
                <div className="space-y-1">
                  <h4 className="text-sm font-semibold text-blue-300">1. Load Model &amp; Tokenizer</h4>
                  <p className="text-xs leading-relaxed text-zinc-300">
                    Downloads GPT-2 small (124M parameters, 12 layers) from Hugging Face.
                    The tokenizer uses Byte-Pair Encoding (BPE) to split text into ~50,257
                    subword tokens. This decoder-only Transformer generates text
                    autoregressively, one token at a time.
                  </p>
                </div>
              </div>

              {/* Step 2 */}
              <div className="flex gap-3 rounded-lg border border-emerald-500/20 bg-emerald-500/5 p-4">
                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-emerald-500/20 text-emerald-400">
                  <Database className="h-4 w-4" />
                </div>
                <div className="space-y-1">
                  <h4 className="text-sm font-semibold text-emerald-300">2. Prepare Dataset</h4>
                  <p className="text-xs leading-relaxed text-zinc-300">
                    Downloads WikiText-2, a clean Wikipedia-derived corpus. The text is
                    tokenized into a single long sequence, then chunked into fixed-length
                    blocks (default: 128 tokens). Split into training and validation sets.
                  </p>
                </div>
              </div>

              {/* Step 3 */}
              <div className="flex gap-3 rounded-lg border border-amber-500/20 bg-amber-500/5 p-4">
                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-amber-500/20 text-amber-400">
                  <Cog className="h-4 w-4" />
                </div>
                <div className="space-y-1">
                  <h4 className="text-sm font-semibold text-amber-300">3. Configure Training</h4>
                  <p className="text-xs leading-relaxed text-zinc-300">
                    Set hyperparameters: learning rate, batch size, epochs, warmup steps,
                    and weight decay. The data collator uses
                    <code className="mx-1 rounded bg-white/10 px-1 py-0.5 text-amber-200 text-[11px]">MLM=False</code>
                    for causal language modeling.
                  </p>
                </div>
              </div>

              {/* Step 4 */}
              <div className="flex gap-3 rounded-lg border border-orange-500/20 bg-orange-500/5 p-4">
                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-orange-500/20 text-orange-400">
                  <Play className="h-4 w-4" />
                </div>
                <div className="space-y-1">
                  <h4 className="text-sm font-semibold text-orange-300">4. Train</h4>
                  <p className="text-xs leading-relaxed text-zinc-300">
                    Fine-tune with HF Trainer + AdamW. Each step: forward pass → cross-entropy
                    loss → backpropagation → weight update. Metrics stream via WebSocket.
                    Checkpoints saved per epoch.
                  </p>
                </div>
              </div>

              {/* Step 5 */}
              <div className="flex gap-3 rounded-lg border border-purple-500/20 bg-purple-500/5 p-4">
                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-purple-500/20 text-purple-400">
                  <BarChart3 className="h-4 w-4" />
                </div>
                <div className="space-y-1">
                  <h4 className="text-sm font-semibold text-purple-300">5. Evaluate</h4>
                  <p className="text-xs leading-relaxed text-zinc-300">
                    Compute perplexity on the validation set:
                    <code className="mx-1 rounded bg-white/10 px-1 py-0.5 text-purple-200 text-[11px]">exp(avg loss)</code>.
                    Lower = better predictions. Compare fine-tuned vs. baseline GPT-2.
                  </p>
                </div>
              </div>

              {/* Step 6 */}
              <div className="flex gap-3 rounded-lg border border-rose-500/20 bg-rose-500/5 p-4">
                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-rose-500/20 text-rose-400">
                  <MessageSquare className="h-4 w-4" />
                </div>
                <div className="space-y-1">
                  <h4 className="text-sm font-semibold text-rose-300">6. Generate Text</h4>
                  <p className="text-xs leading-relaxed text-zinc-300">
                    Autoregressive sampling from a prompt. Control output with:
                  </p>
                  <div className="mt-1 grid grid-cols-2 gap-x-4 gap-y-1 text-[11px] text-zinc-400">
                    <span><span className="text-rose-300 font-medium">Temperature</span> — randomness</span>
                    <span><span className="text-rose-300 font-medium">Top-k</span> — top k tokens</span>
                    <span><span className="text-rose-300 font-medium">Top-p</span> — nucleus sampling</span>
                    <span><span className="text-rose-300 font-medium">Max length</span> — output size</span>
                  </div>
                </div>
              </div>

            </div>
          </div>

          {/* Architecture */}
          <div className="rounded-lg border border-white/10 bg-white/5 p-4 space-y-2">
            <h3 className="text-sm font-semibold text-zinc-100">Architecture</h3>
            <div className="flex flex-wrap gap-2 text-[11px]">
              <span className="rounded-full bg-blue-500/15 px-2.5 py-1 text-blue-300">React + TypeScript</span>
              <span className="rounded-full bg-emerald-500/15 px-2.5 py-1 text-emerald-300">Vite</span>
              <span className="rounded-full bg-amber-500/15 px-2.5 py-1 text-amber-300">Shadcn/ui</span>
              <span className="rounded-full bg-purple-500/15 px-2.5 py-1 text-purple-300">FastAPI</span>
              <span className="rounded-full bg-orange-500/15 px-2.5 py-1 text-orange-300">PyTorch</span>
              <span className="rounded-full bg-rose-500/15 px-2.5 py-1 text-rose-300">HF Transformers</span>
              <span className="rounded-full bg-zinc-500/15 px-2.5 py-1 text-zinc-300">WebSocket</span>
            </div>
            <p className="text-xs leading-relaxed text-zinc-400">
              REST endpoints initiate operations. A WebSocket connection streams real-time
              progress, metrics, and commentary. A pipeline state machine enforces correct
              operation ordering.
            </p>
          </div>

          {/* Footer */}
          <div className="border-t border-white/10 pt-4 text-center text-xs text-zinc-500">
            © Virgil Ennes
          </div>

        </div>
      </div>
    </dialog>
  );
}
