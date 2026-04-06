"""Pipeline state machine for tracking ML pipeline stages.

Manages the lifecycle of the GPT-2 training pipeline with states from
idle through model loading, dataset preparation, training, evaluation,
and generation. Enforces transition guards to prevent invalid operations
(e.g., training without a prepared dataset) and broadcasts state changes
via WebSocket.
"""

from __future__ import annotations

import asyncio
from enum import Enum
from typing import Any

from backend.app.api.websocket import manager as ws_manager


class PipelineStage(str, Enum):
    """All valid pipeline stages."""

    IDLE = "idle"
    MODEL_LOADING = "model_loading"
    MODEL_LOADED = "model_loaded"
    DATASET_PREPARING = "dataset_preparing"
    DATASET_READY = "dataset_ready"
    TRAINING = "training"
    TRAINED = "trained"
    EVALUATING = "evaluating"
    EVALUATED = "evaluated"
    GENERATING = "generating"


# Valid transitions: maps (current_state, action) -> next_state
_TRANSITIONS: dict[tuple[PipelineStage, str], PipelineStage] = {
    # Model loading
    (PipelineStage.IDLE, "load_model"): PipelineStage.MODEL_LOADING,
    (PipelineStage.MODEL_LOADING, "model_loaded"): PipelineStage.MODEL_LOADED,
    (PipelineStage.MODEL_LOADING, "error"): PipelineStage.IDLE,
    # Dataset preparation
    (PipelineStage.MODEL_LOADED, "prepare_dataset"): PipelineStage.DATASET_PREPARING,
    (PipelineStage.DATASET_PREPARING, "dataset_ready"): PipelineStage.DATASET_READY,
    (PipelineStage.DATASET_PREPARING, "error"): PipelineStage.MODEL_LOADED,
    # Training
    (PipelineStage.DATASET_READY, "start_training"): PipelineStage.TRAINING,
    (PipelineStage.TRAINING, "training_complete"): PipelineStage.TRAINED,
    (PipelineStage.TRAINING, "error"): PipelineStage.DATASET_READY,
    (PipelineStage.TRAINING, "stop"): PipelineStage.DATASET_READY,
    # Evaluation
    (PipelineStage.TRAINED, "run_evaluation"): PipelineStage.EVALUATING,
    (PipelineStage.EVALUATED, "run_evaluation"): PipelineStage.EVALUATING,
    (PipelineStage.EVALUATING, "evaluation_complete"): PipelineStage.EVALUATED,
    (PipelineStage.EVALUATING, "error"): PipelineStage.TRAINED,
    # Generation (from trained or evaluated)
    (PipelineStage.TRAINED, "generate"): PipelineStage.GENERATING,
    (PipelineStage.EVALUATED, "generate"): PipelineStage.GENERATING,
    (PipelineStage.GENERATING, "generation_complete_to_trained"): PipelineStage.TRAINED,
    (PipelineStage.GENERATING, "generation_complete_to_evaluated"): PipelineStage.EVALUATED,
}


class InvalidTransitionError(Exception):
    """Raised when a requested state transition is not allowed."""

    def __init__(self, current: PipelineStage, action: str) -> None:
        self.current = current
        self.action = action
        super().__init__(
            f"Cannot perform '{action}' from state '{current.value}'"
        )


class PipelineStateMachine:
    """Tracks the current pipeline stage and enforces valid transitions.

    State changes are broadcast to all connected WebSocket clients via
    the shared :data:`ws_manager`.
    """

    def __init__(self) -> None:
        self._stage: PipelineStage = PipelineStage.IDLE
        self._previous_stage: PipelineStage | None = None

    # ------------------------------------------------------------------
    # Read-only accessors
    # ------------------------------------------------------------------

    @property
    def stage(self) -> PipelineStage:
        """Return the current pipeline stage."""
        return self._stage

    @property
    def previous_stage(self) -> PipelineStage | None:
        """Return the stage before the last transition."""
        return self._previous_stage

    # ------------------------------------------------------------------
    # Transition logic
    # ------------------------------------------------------------------

    def can_transition(self, action: str) -> bool:
        """Return ``True`` if *action* is valid from the current stage."""
        return (self._stage, action) in _TRANSITIONS

    async def transition(self, action: str) -> PipelineStage:
        """Attempt to transition via *action*.

        Raises :class:`InvalidTransitionError` if the transition is not
        allowed from the current stage.  On success the new stage is
        broadcast over WebSocket and returned.
        """
        key = (self._stage, action)
        if key not in _TRANSITIONS:
            raise InvalidTransitionError(self._stage, action)

        self._previous_stage = self._stage
        self._stage = _TRANSITIONS[key]

        await self._broadcast_state_change(action)
        return self._stage

    def transition_sync(self, action: str) -> PipelineStage:
        """Synchronous transition — broadcasts in a fire-and-forget task.

        Useful when called from synchronous code that cannot ``await``.
        """
        key = (self._stage, action)
        if key not in _TRANSITIONS:
            raise InvalidTransitionError(self._stage, action)

        self._previous_stage = self._stage
        self._stage = _TRANSITIONS[key]

        # Schedule the broadcast without blocking.
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._broadcast_state_change(action))
        except RuntimeError:
            pass  # No running loop — skip broadcast (e.g., in tests)

        return self._stage

    # ------------------------------------------------------------------
    # Guard helpers
    # ------------------------------------------------------------------

    def require_model_loaded(self) -> None:
        """Raise if the model has not been loaded yet."""
        if self._stage in (PipelineStage.IDLE, PipelineStage.MODEL_LOADING):
            raise InvalidTransitionError(self._stage, "requires_model_loaded")

    def require_dataset_ready(self) -> None:
        """Raise if the dataset has not been prepared yet."""
        ready_stages = {
            PipelineStage.DATASET_READY,
            PipelineStage.TRAINING,
            PipelineStage.TRAINED,
            PipelineStage.EVALUATING,
            PipelineStage.EVALUATED,
            PipelineStage.GENERATING,
        }
        if self._stage not in ready_stages:
            raise InvalidTransitionError(self._stage, "requires_dataset_ready")

    def require_trained(self) -> None:
        """Raise if the model has not been trained yet."""
        trained_stages = {
            PipelineStage.TRAINED,
            PipelineStage.EVALUATING,
            PipelineStage.EVALUATED,
            PipelineStage.GENERATING,
        }
        if self._stage not in trained_stages:
            raise InvalidTransitionError(self._stage, "requires_trained")

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Return a JSON-serialisable status dict."""
        return {
            "stage": self._stage.value,
            "previous_stage": self._previous_stage.value if self._previous_stage else None,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _broadcast_state_change(self, action: str) -> None:
        """Push the new state to all WebSocket clients."""
        payload = {
            "stage": self._stage.value,
            "previous_stage": self._previous_stage.value if self._previous_stage else None,
            "action": action,
        }
        ws_manager.set_pipeline_state(payload)
        await ws_manager.broadcast("state_change", payload)


# Module-level singleton
pipeline = PipelineStateMachine()
