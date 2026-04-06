import { useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { useAppState } from "@/lib/state";
import * as api from "@/lib/api";
import type { PipelineState } from "@/types";

const SPEED_OPTIONS: { value: PipelineState["demoSpeed"]; label: string }[] = [
  { value: "fast", label: "Fast" },
  { value: "medium", label: "Medium" },
  { value: "slow", label: "Slow" },
];

export function DemoControls() {
  const { state, dispatch } = useAppState();
  const { isDemo, demoSpeed, demoPaused } = state.pipeline;

  const [selectedSpeed, setSelectedSpeed] =
    useState<PipelineState["demoSpeed"]>(demoSpeed);
  const [loading, setLoading] = useState(false);

  const handleStartDemo = useCallback(async () => {
    try {
      setLoading(true);
      await api.startDemo({ speed: selectedSpeed });
      dispatch({
        type: "SET_DEMO_STATE",
        isDemo: true,
        demoSpeed: selectedSpeed,
        demoPaused: false,
      });
    } catch (err) {
      dispatch({
        type: "SET_ERROR",
        message: err instanceof Error ? err.message : "Failed to start demo",
      });
    } finally {
      setLoading(false);
    }
  }, [selectedSpeed, dispatch]);

  const handlePause = useCallback(async () => {
    try {
      await api.pauseDemo();
      dispatch({
        type: "SET_DEMO_STATE",
        isDemo: true,
        demoPaused: true,
      });
    } catch (err) {
      dispatch({
        type: "SET_ERROR",
        message: err instanceof Error ? err.message : "Failed to pause demo",
      });
    }
  }, [dispatch]);

  const handleResume = useCallback(async () => {
    try {
      await api.resumeDemo();
      dispatch({
        type: "SET_DEMO_STATE",
        isDemo: true,
        demoPaused: false,
      });
    } catch (err) {
      dispatch({
        type: "SET_ERROR",
        message: err instanceof Error ? err.message : "Failed to resume demo",
      });
    }
  }, [dispatch]);

  const handleSkip = useCallback(async () => {
    try {
      await api.skipDemoStep();
    } catch (err) {
      dispatch({
        type: "SET_ERROR",
        message:
          err instanceof Error ? err.message : "Failed to skip demo step",
      });
    }
  }, [dispatch]);

  return (
    <div data-testid="demo-controls" className="flex flex-col gap-3">
      <h3 className="text-sm font-medium text-muted-foreground">Demo Mode</h3>

      {/* Speed selector */}
      {!isDemo && (
        <div className="space-y-1.5">
          <Label htmlFor="demo-speed">Speed</Label>
          <div className="flex gap-1" data-testid="demo-speed-selector">
            {SPEED_OPTIONS.map((opt) => (
              <Button
                key={opt.value}
                type="button"
                variant={selectedSpeed === opt.value ? "default" : "outline"}
                size="xs"
                onClick={() => setSelectedSpeed(opt.value)}
                data-testid={`demo-speed-${opt.value}`}
              >
                {opt.label}
              </Button>
            ))}
          </div>
        </div>
      )}

      {/* Action buttons */}
      <div className="flex flex-wrap gap-2">
        {!isDemo && (
          <Button
            onClick={handleStartDemo}
            disabled={loading}
            variant="default"
            size="sm"
            data-testid="btn-start-demo"
          >
            {loading ? "Starting…" : "Start Demo"}
          </Button>
        )}

        {isDemo && !demoPaused && (
          <Button
            onClick={handlePause}
            variant="outline"
            size="sm"
            data-testid="btn-pause-demo"
          >
            Pause
          </Button>
        )}

        {isDemo && demoPaused && (
          <Button
            onClick={handleResume}
            variant="outline"
            size="sm"
            data-testid="btn-resume-demo"
          >
            Resume
          </Button>
        )}

        {isDemo && (
          <Button
            onClick={handleSkip}
            variant="ghost"
            size="sm"
            data-testid="btn-skip-step"
          >
            Skip Step
          </Button>
        )}
      </div>
    </div>
  );
}
