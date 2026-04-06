"""Demo Orchestrator for automated pipeline walkthrough.

Executes the full ML pipeline sequentially with pauses and commentary,
allowing users to observe and learn from each step without manual
intervention.

**Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5, 7.7**
"""

from __future__ import annotations

import asyncio
import logging
from enum import Enum
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)


# The ordered sequence of demo steps
DEMO_STEPS: list[str] = [
    "model_loading",
    "dataset_preparation",
    "training_configuration",
    "training_execution",
    "evaluation",
    "text_generation",
]

# Speed settings: maps speed name to pause duration in seconds
SPEED_SETTINGS: dict[str, float] = {
    "fast": 2.0,
    "medium": 5.0,
    "slow": 10.0,
}

# Commentary text for each pipeline step
STEP_COMMENTARY: dict[str, str] = {
    "model_loading": (
        "Loading the GPT-2 small model from Hugging Face. This decoder-only "
        "transformer has 124M parameters with 12 layers and 768 hidden dimensions. "
        "The tokenizer uses byte-pair encoding for subword tokenization."
    ),
    "dataset_preparation": (
        "Preparing the WikiText-2 dataset. The text corpus is tokenized into "
        "subword tokens and chunked into fixed-length sequences for training. "
        "The data is split into training and validation sets."
    ),
    "training_configuration": (
        "Configuring training hyperparameters. The learning rate, batch size, "
        "number of epochs, warmup steps, and weight decay all affect how the "
        "model learns. The data collator is set with MLM=False for decoder-only training."
    ),
    "training_execution": (
        "Training the model on the prepared dataset. Watch the loss curves to "
        "see how the model improves over time. Lower loss means the model is "
        "getting better at predicting the next token."
    ),
    "evaluation": (
        "Evaluating the trained model using perplexity on the validation set. "
        "Perplexity measures how well the model predicts unseen text — lower "
        "values indicate better performance."
    ),
    "text_generation": (
        "Generating text with the trained model. Temperature, top-k, and top-p "
        "parameters control the randomness and diversity of the output. Compare "
        "the baseline model output with the fine-tuned model."
    ),
}

# Maps each step to the UI component that should be highlighted
STEP_HIGHLIGHT: dict[str, str] = {
    "model_loading": "ProgressPanel",
    "dataset_preparation": "ProgressPanel",
    "training_configuration": "ControlPanel",
    "training_execution": "ProgressPanel",
    "evaluation": "ProgressPanel",
    "text_generation": "GenerationPanel",
}


class DemoState(str, Enum):
    """Possible states for the demo orchestrator."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    COMPLETED = "completed"


class DemoOrchestrator:
    """Orchestrates an automated walkthrough of the full ML pipeline.

    The orchestrator executes steps sequentially, pausing between each
    step to display commentary and highlight the active UI component.
    It supports pause/resume/skip controls and handles step errors
    gracefully.
    """

    def __init__(self) -> None:
        self._state: DemoState = DemoState.IDLE
        self._current_step_index: int = 0
        self._speed: str = "medium"
        self._pause_event: asyncio.Event = asyncio.Event()
        self._pause_event.set()  # Not paused initially
        self._ws_callback: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None
        self._step_executor: Callable[[str], Awaitable[None]] | None = None
        self._task: asyncio.Task[None] | None = None
        self._executed_steps: list[str] = []
        self._error_step: str | None = None
        self._skip_requested: bool = False
        # Override pause duration for testing (None = use speed settings)
        self._pause_duration_override: float | None = None

    # ------------------------------------------------------------------
    # Read-only accessors
    # ------------------------------------------------------------------

    @property
    def state(self) -> DemoState:
        """Return the current demo state."""
        return self._state

    @property
    def current_step_index(self) -> int:
        """Return the index of the current step."""
        return self._current_step_index

    @property
    def current_step(self) -> str | None:
        """Return the name of the current step, or None if not running."""
        if 0 <= self._current_step_index < len(DEMO_STEPS):
            return DEMO_STEPS[self._current_step_index]
        return None

    @property
    def executed_steps(self) -> list[str]:
        """Return the list of steps that have been executed so far."""
        return list(self._executed_steps)

    @property
    def speed(self) -> str:
        """Return the current speed setting."""
        return self._speed

    def get_status(self) -> dict[str, Any]:
        """Return a JSON-serialisable status dict."""
        return {
            "state": self._state.value,
            "current_step": self.current_step,
            "current_step_index": self._current_step_index,
            "speed": self._speed,
            "executed_steps": list(self._executed_steps),
            "total_steps": len(DEMO_STEPS),
        }

    # ------------------------------------------------------------------
    # Control methods
    # ------------------------------------------------------------------

    async def start_demo(
        self,
        speed: str = "medium",
        ws_callback: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None,
        step_executor: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        """Start the automated demo.

        Parameters
        ----------
        speed:
            One of "fast", "medium", "slow".
        ws_callback:
            Async callback ``(msg_type, payload)`` for sending WebSocket messages.
        step_executor:
            Optional async callback that performs the actual ML operation for
            each step. If not provided, steps are simulated (commentary only).
        """
        if self._state == DemoState.RUNNING:
            return  # Already running

        self._speed = speed if speed in SPEED_SETTINGS else "medium"
        self._ws_callback = ws_callback
        self._step_executor = step_executor
        self._current_step_index = 0
        self._executed_steps = []
        self._state = DemoState.RUNNING
        self._pause_event.set()
        self._skip_requested = False
        self._error_step = None

        self._task = asyncio.create_task(self._run_pipeline())

    async def pause(self) -> None:
        """Pause the demo at the current step."""
        if self._state == DemoState.RUNNING:
            self._state = DemoState.PAUSED
            self._pause_event.clear()
            if self._ws_callback:
                await self._ws_callback("demo_step", {
                    "action": "paused",
                    "step": self.current_step,
                    "step_index": self._current_step_index,
                })

    async def resume(self) -> None:
        """Resume the demo from where it was paused."""
        if self._state in (DemoState.PAUSED, DemoState.ERROR):
            self._state = DemoState.RUNNING
            self._pause_event.set()
            if self._ws_callback:
                await self._ws_callback("demo_step", {
                    "action": "resumed",
                    "step": self.current_step,
                    "step_index": self._current_step_index,
                })

    async def skip_step(self) -> None:
        """Skip the current step and move to the next one."""
        if self._state in (DemoState.PAUSED, DemoState.ERROR, DemoState.RUNNING):
            self._skip_requested = True
            # If paused or in error, resume to let the loop advance
            if self._state in (DemoState.PAUSED, DemoState.ERROR):
                self._state = DemoState.RUNNING
                self._pause_event.set()

    async def stop(self) -> None:
        """Stop the demo entirely."""
        self._state = DemoState.IDLE
        self._pause_event.set()  # Unblock if paused
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    # ------------------------------------------------------------------
    # Internal pipeline execution
    # ------------------------------------------------------------------

    async def _run_pipeline(self) -> None:
        """Execute all demo steps sequentially."""
        try:
            while self._current_step_index < len(DEMO_STEPS):
                if self._state == DemoState.IDLE:
                    return  # Stopped

                step_name = DEMO_STEPS[self._current_step_index]

                # Wait if paused
                await self._pause_event.wait()
                if self._state == DemoState.IDLE:
                    return

                # Check for skip before executing
                if self._skip_requested:
                    self._skip_requested = False
                    self._executed_steps.append(step_name)
                    self._current_step_index += 1
                    continue

                # Send commentary and highlight for this step
                await self._send_step_start(step_name)

                # Execute the actual step operation
                try:
                    if self._step_executor:
                        await self._step_executor(step_name)
                except Exception as exc:
                    await self._handle_step_error(step_name, exc)
                    # Wait for resume or skip
                    await self._pause_event.wait()
                    if self._state == DemoState.IDLE:
                        return
                    # If skip was requested during error, move on
                    if self._skip_requested:
                        self._skip_requested = False
                        self._executed_steps.append(step_name)
                        self._current_step_index += 1
                        continue
                    # Otherwise retry the step (resume was called)
                    continue

                self._executed_steps.append(step_name)

                # Pause between steps (respecting speed setting)
                if self._pause_duration_override is not None:
                    pause_duration = self._pause_duration_override
                else:
                    pause_duration = SPEED_SETTINGS.get(self._speed, 5.0)
                await self._interruptible_sleep(pause_duration)

                if self._state == DemoState.IDLE:
                    return

                # Check for skip after pause
                if self._skip_requested:
                    self._skip_requested = False

                self._current_step_index += 1

            # All steps completed
            self._state = DemoState.COMPLETED
            if self._ws_callback:
                await self._ws_callback("demo_step", {
                    "action": "completed",
                    "executed_steps": list(self._executed_steps),
                })

        except asyncio.CancelledError:
            self._state = DemoState.IDLE
            raise

    async def _send_step_start(self, step_name: str) -> None:
        """Send commentary and highlight messages for a step."""
        if self._ws_callback is None:
            return

        commentary = STEP_COMMENTARY.get(step_name, "")
        highlight = STEP_HIGHLIGHT.get(step_name, "")

        await self._ws_callback("commentary", {
            "step": step_name,
            "text": commentary,
        })

        await self._ws_callback("demo_step", {
            "action": "step_start",
            "step": step_name,
            "step_index": self._current_step_index,
            "highlight": highlight,
            "commentary": commentary,
        })

    async def _handle_step_error(self, step_name: str, error: Exception) -> None:
        """Handle an error during a step: pause and notify."""
        self._state = DemoState.ERROR
        self._error_step = step_name
        self._pause_event.clear()

        logger.error("Demo step '%s' failed: %s", step_name, error)

        if self._ws_callback:
            await self._ws_callback("commentary", {
                "step": step_name,
                "text": f"Error during {step_name}: {error}. You can retry or skip this step.",
            })
            await self._ws_callback("error", {
                "step": step_name,
                "error_code": "DEMO_STEP_FAILED",
                "message": str(error),
            })

    async def _interruptible_sleep(self, duration: float) -> None:
        """Sleep for *duration* seconds, but wake early if paused or stopped."""
        try:
            await asyncio.wait_for(self._wait_for_interrupt(), timeout=duration)
        except asyncio.TimeoutError:
            pass  # Normal: sleep completed without interruption

    async def _wait_for_interrupt(self) -> None:
        """Block forever — used as the 'thing to cancel' in interruptible sleep."""
        # This will only return if cancelled or if we add interrupt logic
        await asyncio.Event().wait()


# Module-level singleton
demo_orchestrator = DemoOrchestrator()
