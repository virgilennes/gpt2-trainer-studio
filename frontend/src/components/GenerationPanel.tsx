import { useState, useCallback, type FormEvent } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tooltip } from "@/components/ui/tooltip";
import { GenerationParamsSchema } from "@/lib/validation";
import { useAppState } from "@/lib/state";
import * as api from "@/lib/api";
import type { GenerationParams } from "@/types";
import type { CompareGenerationResult } from "@/lib/api";

// ---------------------------------------------------------------------------
// Parameter tooltips — exported for property testing (sub-task 16.5)
// ---------------------------------------------------------------------------

export const GENERATION_PARAM_HELP: Record<
  keyof Omit<GenerationParams, "prompt">,
  string
> = {
  temperature:
    "Controls randomness of predictions. Lower values (e.g. 0.2) make output more focused and deterministic; higher values (e.g. 1.5) make it more creative and diverse. Range: 0.1 to 2.0.",
  topK:
    "Limits sampling to the top K most probable next tokens. Lower values restrict output to high-confidence words; higher values allow more variety. Range: 1 to 100.",
  topP:
    "Nucleus sampling: considers the smallest set of tokens whose cumulative probability exceeds P. Lower values produce more focused text; 1.0 considers all tokens. Range: 0.1 to 1.0.",
  maxLength:
    "Maximum number of tokens to generate. Longer values produce more text but take longer to generate. Range: 10 to 500.",
};

// ---------------------------------------------------------------------------
// Sample prompt library (sub-task 16.2)
// ---------------------------------------------------------------------------

export const SAMPLE_PROMPTS: { label: string; text: string }[] = [
  { label: "Once upon a time", text: "Once upon a time in a land far away," },
  {
    label: "Science explanation",
    text: "The theory of relativity explains that",
  },
  {
    label: "Code documentation",
    text: "The following function implements a",
  },
  { label: "News article", text: "Breaking news: scientists have discovered" },
  {
    label: "Creative writing",
    text: "The old lighthouse stood at the edge of the cliff,",
  },
  {
    label: "Philosophy",
    text: "The meaning of life, according to ancient philosophers,",
  },
];

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------

const DEFAULTS: Omit<GenerationParams, "prompt"> = {
  temperature: 1.0,
  topK: 50,
  topP: 1.0,
  maxLength: 100,
};

// ---------------------------------------------------------------------------
// Component (sub-tasks 16.1 – 16.5)
// ---------------------------------------------------------------------------

export function GenerationPanel() {
  const { state, dispatch } = useAppState();
  const stage = state.pipeline.stage;

  // Form state
  const [prompt, setPrompt] = useState("");
  const [params, setParams] = useState({
    temperature: String(DEFAULTS.temperature),
    topK: String(DEFAULTS.topK),
    topP: String(DEFAULTS.topP),
    maxLength: String(DEFAULTS.maxLength),
  });

  const [errors, setErrors] = useState<
    Partial<Record<keyof GenerationParams, string>>
  >({});
  const [generatedText, setGeneratedText] = useState<string | null>(null);
  const [compareResult, setCompareResult] =
    useState<CompareGenerationResult | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isComparing, setIsComparing] = useState(false);
  const [mode, setMode] = useState<"generate" | "compare">("generate");

  // --- helpers ---

  const canGenerate =
    (stage === "trained" || stage === "evaluated") && !isGenerating && !isComparing;

  const setParam = useCallback(
    (name: keyof Omit<GenerationParams, "prompt">, value: string) => {
      setParams((prev) => ({ ...prev, [name]: value }));
      setErrors((prev) => {
        if (!prev[name]) return prev;
        const next = { ...prev };
        delete next[name];
        return next;
      });
    },
    [],
  );

  const buildParams = useCallback((): GenerationParams | null => {
    const raw: GenerationParams = {
      prompt,
      temperature: Number(params.temperature),
      topK: Number(params.topK),
      topP: Number(params.topP),
      maxLength: Number(params.maxLength),
    };

    const result = GenerationParamsSchema.safeParse(raw);
    if (result.success) {
      setErrors({});
      return result.data;
    }

    const fieldErrors: Partial<Record<keyof GenerationParams, string>> = {};
    for (const issue of result.error.issues) {
      const key = issue.path[0] as keyof GenerationParams | undefined;
      if (key && !fieldErrors[key]) {
        fieldErrors[key] = issue.message;
      }
    }
    setErrors(fieldErrors);
    return null;
  }, [prompt, params]);

  // --- action handlers ---

  const handleGenerate = useCallback(
    async (e: FormEvent) => {
      e.preventDefault();
      const p = buildParams();
      if (!p) return;

      try {
        setIsGenerating(true);
        setGeneratedText(null);
        setCompareResult(null);
        dispatch({ type: "SET_PIPELINE_STAGE", stage: "generating" });
        const result = await api.generateText(p);
        setGeneratedText(result.text);
        dispatch({ type: "SET_GENERATION_RESULT", result });
      } catch (err) {
        dispatch({
          type: "SET_ERROR",
          message:
            err instanceof Error ? err.message : "Failed to generate text",
        });
      } finally {
        setIsGenerating(false);
        // Return to previous stage
        if (state.pipeline.stage === "generating") {
          dispatch({
            type: "SET_PIPELINE_STAGE",
            stage: state.evalResult ? "evaluated" : "trained",
          });
        }
      }
    },
    [buildParams, dispatch, state.pipeline.stage, state.evalResult],
  );

  const handleCompare = useCallback(
    async (e: FormEvent) => {
      e.preventDefault();
      const p = buildParams();
      if (!p) return;

      try {
        setIsComparing(true);
        setGeneratedText(null);
        setCompareResult(null);
        dispatch({ type: "SET_PIPELINE_STAGE", stage: "generating" });
        const result = await api.compareGeneration(p);
        setCompareResult(result);
      } catch (err) {
        dispatch({
          type: "SET_ERROR",
          message:
            err instanceof Error
              ? err.message
              : "Failed to compare generation",
        });
      } finally {
        setIsComparing(false);
        if (state.pipeline.stage === "generating") {
          dispatch({
            type: "SET_PIPELINE_STAGE",
            stage: state.evalResult ? "evaluated" : "trained",
          });
        }
      }
    },
    [buildParams, dispatch, state.pipeline.stage, state.evalResult],
  );

  // --- render helpers ---

  function renderParamField(
    name: keyof Omit<GenerationParams, "prompt">,
    label: string,
    inputProps?: React.ComponentProps<"input">,
  ) {
    const error = errors[name];
    const help = GENERATION_PARAM_HELP[name];
    const id = `gen-${name}`;

    return (
      <div className="space-y-1" key={name}>
        <div className="flex items-center gap-1.5">
          <Label htmlFor={id}>{label}</Label>
          <Tooltip content={help}>
            <button
              type="button"
              aria-label={`Help for ${label}`}
              className="inline-flex h-4 w-4 items-center justify-center rounded-full bg-muted text-muted-foreground text-[10px] leading-none hover:bg-muted-foreground/20"
            >
              ?
            </button>
          </Tooltip>
        </div>
        <Input
          id={id}
          value={params[name]}
          onChange={(e) => setParam(name, e.target.value)}
          aria-invalid={!!error}
          aria-describedby={error ? `${id}-error` : `${id}-help`}
          disabled={isGenerating || isComparing}
          {...inputProps}
        />
        {error ? (
          <p id={`${id}-error`} className="text-xs text-destructive" role="alert">
            {error}
          </p>
        ) : (
          <p id={`${id}-help`} className="text-xs text-muted-foreground">
            {help}
          </p>
        )}
      </div>
    );
  }

  return (
    <section
      data-testid="generation-panel"
      className="flex flex-col gap-4 border-t border-border bg-card p-4"
    >
      <h2 className="text-lg font-semibold">Text Generation</h2>

      {!canGenerate && !isGenerating && !isComparing && (
        <p className="text-sm text-muted-foreground">
          Train or evaluate the model before generating text.
        </p>
      )}

      <form
        onSubmit={mode === "compare" ? handleCompare : handleGenerate}
        className="flex flex-col gap-4"
      >
        {/* ---- Prompt input (sub-task 16.1) ---- */}
        <div className="space-y-1.5">
          <Label htmlFor="gen-prompt">Prompt</Label>
          <Input
            id="gen-prompt"
            value={prompt}
            onChange={(e) => {
              setPrompt(e.target.value);
              setErrors((prev) => {
                if (!prev.prompt) return prev;
                const next = { ...prev };
                delete next.prompt;
                return next;
              });
            }}
            placeholder="Enter a text prompt…"
            aria-invalid={!!errors.prompt}
            aria-describedby={errors.prompt ? "gen-prompt-error" : undefined}
            disabled={isGenerating || isComparing}
          />
          {errors.prompt && (
            <p
              id="gen-prompt-error"
              className="text-xs text-destructive"
              role="alert"
            >
              {errors.prompt}
            </p>
          )}
        </div>

        {/* ---- Sample prompt library (sub-task 16.2) ---- */}
        <div className="space-y-1.5">
          <Label>Sample Prompts</Label>
          <div className="flex flex-wrap gap-1.5" data-testid="sample-prompts">
            {SAMPLE_PROMPTS.map((sp) => (
              <Button
                key={sp.label}
                type="button"
                variant="outline"
                size="xs"
                onClick={() => setPrompt(sp.text)}
                disabled={isGenerating || isComparing}
              >
                {sp.label}
              </Button>
            ))}
          </div>
        </div>

        {/* ---- Parameter controls (sub-task 16.1 + 16.5 tooltips) ---- */}
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          {renderParamField("temperature", "Temperature", {
            type: "number",
            step: "0.1",
            min: "0.1",
            max: "2.0",
          })}
          {renderParamField("topK", "Top-K", {
            type: "number",
            step: "1",
            min: "1",
            max: "100",
          })}
          {renderParamField("topP", "Top-P", {
            type: "number",
            step: "0.1",
            min: "0.1",
            max: "1.0",
          })}
          {renderParamField("maxLength", "Max Length", {
            type: "number",
            step: "1",
            min: "10",
            max: "500",
          })}
        </div>

        {/* ---- Mode toggle + submit buttons ---- */}
        <div className="flex items-center gap-2">
          <Button
            type="button"
            variant={mode === "generate" ? "default" : "outline"}
            size="sm"
            onClick={() => setMode("generate")}
          >
            Generate
          </Button>
          <Button
            type="button"
            variant={mode === "compare" ? "default" : "outline"}
            size="sm"
            onClick={() => setMode("compare")}
          >
            Compare
          </Button>
          <Button
            type="submit"
            disabled={!canGenerate}
            size="sm"
            className="ml-auto"
            data-testid="btn-generate"
          >
            {isGenerating || isComparing
              ? "Generating…"
              : mode === "compare"
                ? "Compare Models"
                : "Generate Text"}
          </Button>
        </div>
      </form>

      {/* ---- Generated text output area (sub-task 16.3) ---- */}
      {generatedText !== null && mode === "generate" && (
        <div className="space-y-1.5" data-testid="generation-output">
          <Label>Generated Text</Label>
          <div className="min-h-[100px] rounded-md border border-border bg-muted/50 p-3 text-sm whitespace-pre-wrap">
            {generatedText}
          </div>
        </div>
      )}

      {/* ---- Side-by-side comparison view (sub-task 16.4) ---- */}
      {compareResult !== null && mode === "compare" && (
        <div className="space-y-1.5" data-testid="comparison-output">
          <Label>Side-by-Side Comparison</Label>
          <p className="text-xs text-muted-foreground">
            Prompt: &ldquo;{compareResult.prompt}&rdquo;
          </p>
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            <div className="space-y-1">
              <span className="text-xs font-medium text-muted-foreground">
                Baseline Model
              </span>
              <div className="min-h-[100px] rounded-md border border-border bg-muted/50 p-3 text-sm whitespace-pre-wrap">
                {compareResult.baselineText}
              </div>
            </div>
            <div className="space-y-1">
              <span className="text-xs font-medium text-muted-foreground">
                Trained Model
              </span>
              <div className="min-h-[100px] rounded-md border border-border bg-muted/50 p-3 text-sm whitespace-pre-wrap">
                {compareResult.trainedText}
              </div>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
