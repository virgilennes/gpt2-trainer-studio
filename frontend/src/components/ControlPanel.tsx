import { useState, useCallback, type FormEvent } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tooltip } from "@/components/ui/tooltip";
import { TrainingConfigSchema } from "@/lib/validation";
import { useAppState } from "@/lib/state";
import * as api from "@/lib/api";
import type { TrainingConfig } from "@/types";

// ---------------------------------------------------------------------------
// Help text for every config field — exported for property testing
// ---------------------------------------------------------------------------

export const CONFIG_FIELD_HELP: Record<keyof TrainingConfig, string> = {
  learningRate:
    "Controls how much the model weights are updated each step. Smaller values train slower but more stably. Range: 1e-6 to 1e-2.",
  batchSize:
    "Number of training examples processed together. Larger batches use more memory but give smoother gradients. Range: 1 to 64.",
  numEpochs:
    "Number of complete passes through the training dataset. More epochs can improve results but risk overfitting. Range: 1 to 20.",
  warmupSteps:
    "Number of steps during which the learning rate linearly increases from 0 to the target value. Helps stabilise early training. Range: 0 to 1000.",
  weightDecay:
    "L2 regularisation factor that penalises large weights, helping prevent overfitting. Range: 0 to 1.",
};

// ---------------------------------------------------------------------------
// Default values
// ---------------------------------------------------------------------------

const DEFAULTS: TrainingConfig = {
  learningRate: 5e-5,
  batchSize: 8,
  numEpochs: 3,
  warmupSteps: 0,
  weightDecay: 0,
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function ControlPanel() {
  const { state, dispatch } = useAppState();
  const stage = state.pipeline.stage;

  // Form field state (strings so the user can type freely)
  const [fields, setFields] = useState({
    learningRate: String(DEFAULTS.learningRate),
    batchSize: String(DEFAULTS.batchSize),
    numEpochs: String(DEFAULTS.numEpochs),
    warmupSteps: String(DEFAULTS.warmupSteps),
    weightDecay: String(DEFAULTS.weightDecay),
  });

  const [errors, setErrors] = useState<Partial<Record<keyof TrainingConfig, string>>>({});

  // --- helpers ---

  const setField = useCallback(
    (name: keyof TrainingConfig, value: string) => {
      setFields((prev) => ({ ...prev, [name]: value }));
      // Clear field error on change
      setErrors((prev) => {
        if (!prev[name]) return prev;
        const next = { ...prev };
        delete next[name];
        return next;
      });
    },
    [],
  );

  const validate = useCallback((): TrainingConfig | null => {
    const raw = {
      learningRate: Number(fields.learningRate),
      batchSize: Number(fields.batchSize),
      numEpochs: Number(fields.numEpochs),
      warmupSteps: Number(fields.warmupSteps),
      weightDecay: Number(fields.weightDecay),
    };

    const result = TrainingConfigSchema.safeParse(raw);
    if (result.success) {
      setErrors({});
      return result.data;
    }

    const fieldErrors: Partial<Record<keyof TrainingConfig, string>> = {};
    for (const issue of result.error.issues) {
      const key = issue.path[0] as keyof TrainingConfig | undefined;
      if (key && !fieldErrors[key]) {
        fieldErrors[key] = issue.message;
      }
    }
    setErrors(fieldErrors);
    return null;
  }, [fields]);

  // --- action handlers ---

  const handleLoadModel = useCallback(async () => {
    try {
      dispatch({ type: "SET_PIPELINE_STAGE", stage: "model_loading" });
      const summary = await api.loadModel();
      dispatch({ type: "SET_MODEL_SUMMARY", summary });
      dispatch({ type: "SET_PIPELINE_STAGE", stage: "model_loaded" });
    } catch (err) {
      dispatch({
        type: "SET_ERROR",
        message: err instanceof Error ? err.message : "Failed to load model",
      });
      dispatch({ type: "SET_PIPELINE_STAGE", stage: "idle" });
    }
  }, [dispatch]);

  const handlePrepareDataset = useCallback(async () => {
    try {
      dispatch({ type: "SET_PIPELINE_STAGE", stage: "dataset_preparing" });
      const stats = await api.prepareDataset();
      dispatch({ type: "SET_DATASET_STATS", stats });
      dispatch({ type: "SET_PIPELINE_STAGE", stage: "dataset_ready" });
    } catch (err) {
      dispatch({
        type: "SET_ERROR",
        message: err instanceof Error ? err.message : "Failed to prepare dataset",
      });
      dispatch({ type: "SET_PIPELINE_STAGE", stage: "model_loaded" });
    }
  }, [dispatch]);

  const handleStartTraining = useCallback(
    async (e: FormEvent) => {
      e.preventDefault();
      const config = validate();
      if (!config) return;

      try {
        dispatch({ type: "SET_PIPELINE_STAGE", stage: "training" });
        dispatch({ type: "CLEAR_METRICS" });
        await api.startTraining(config);
        dispatch({ type: "SET_PIPELINE_STAGE", stage: "trained" });
      } catch (err) {
        dispatch({
          type: "SET_ERROR",
          message: err instanceof Error ? err.message : "Failed to start training",
        });
        dispatch({ type: "SET_PIPELINE_STAGE", stage: "dataset_ready" });
      }
    },
    [validate, dispatch],
  );

  const handleRunEvaluation = useCallback(async () => {
    try {
      dispatch({ type: "SET_PIPELINE_STAGE", stage: "evaluating" });
      const result = await api.runEvaluation();
      dispatch({ type: "SET_EVAL_RESULT", result });
      dispatch({ type: "SET_PIPELINE_STAGE", stage: "evaluated" });
    } catch (err) {
      dispatch({
        type: "SET_ERROR",
        message: err instanceof Error ? err.message : "Failed to run evaluation",
      });
      dispatch({ type: "SET_PIPELINE_STAGE", stage: "trained" });
    }
  }, [dispatch]);

  // --- button enable/disable logic ---

  const isBusy = [
    "model_loading",
    "dataset_preparing",
    "training",
    "evaluating",
    "generating",
  ].includes(stage);

  const isDemo = state.pipeline.isDemo;

  const canLoadModel = stage === "idle" && !isBusy && !isDemo;
  const canPrepareDataset = stage === "model_loaded" && !isBusy && !isDemo;
  const canStartTraining = stage === "dataset_ready" && !isBusy && !isDemo;
  const canRunEvaluation =
    (stage === "trained" || stage === "evaluated") && !isBusy && !isDemo;
  const controlsDisabled = isBusy || isDemo;

  // --- render helpers ---

  function renderField(
    name: keyof TrainingConfig,
    label: string,
    inputProps?: React.ComponentProps<"input">,
  ) {
    const error = errors[name];
    const help = CONFIG_FIELD_HELP[name];
    const id = `config-${name}`;

    return (
      <div className="space-y-1.5" key={name}>
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
          value={fields[name]}
          onChange={(e) => setField(name, e.target.value)}
          aria-invalid={!!error}
          aria-describedby={error ? `${id}-error` : `${id}-help`}
          disabled={controlsDisabled}
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
    <aside
      data-testid="control-panel"
      className="flex w-full flex-col gap-6 border-r border-border bg-card p-4 lg:w-80"
    >
      <h2 className="text-lg font-semibold">Control Panel</h2>

      {/* ---- Action Buttons ---- */}
      <div className="flex flex-col gap-2">
        <Button
          onClick={handleLoadModel}
          disabled={!canLoadModel}
          variant="outline"
          size="sm"
          data-testid="btn-load-model"
        >
          Load Model
        </Button>
        <Button
          onClick={handlePrepareDataset}
          disabled={!canPrepareDataset}
          variant="outline"
          size="sm"
          data-testid="btn-prepare-dataset"
        >
          Prepare Dataset
        </Button>
      </div>

      {/* ---- Training Config Form ---- */}
      <form onSubmit={handleStartTraining} className="flex flex-col gap-4">
        <h3 className="text-sm font-medium text-muted-foreground">
          Training Configuration
        </h3>

        {renderField("learningRate", "Learning Rate", {
          type: "number",
          step: "0.00001",
          min: "0.000001",
          max: "0.01",
        })}
        {renderField("batchSize", "Batch Size", {
          type: "number",
          step: "1",
          min: "1",
          max: "64",
        })}
        {renderField("numEpochs", "Epochs", {
          type: "number",
          step: "1",
          min: "1",
          max: "20",
        })}
        {renderField("warmupSteps", "Warmup Steps", {
          type: "number",
          step: "1",
          min: "0",
          max: "1000",
        })}
        {renderField("weightDecay", "Weight Decay", {
          type: "number",
          step: "0.01",
          min: "0",
          max: "1",
        })}

        <div className="flex gap-2">
          <Button
            type="submit"
            disabled={!canStartTraining}
            size="sm"
            className="flex-1"
            data-testid="btn-start-training"
          >
            Start Training
          </Button>
          <Button
            type="button"
            onClick={handleRunEvaluation}
            disabled={!canRunEvaluation}
            variant="outline"
            size="sm"
            className="flex-1"
            data-testid="btn-run-evaluation"
          >
            Run Evaluation
          </Button>
        </div>
      </form>

      {/* ---- Error display ---- */}
      {state.error && (
        <p className="text-sm text-destructive" role="alert">
          {state.error}
        </p>
      )}
    </aside>
  );
}
