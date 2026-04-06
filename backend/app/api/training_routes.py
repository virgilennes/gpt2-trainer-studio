"""REST endpoints for training.

POST /api/training/start  – start training with config
POST /api/training/stop   – stop training gracefully
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter

from backend.app.api.error_handler import AppError
from backend.app.api.model_routes import get_state as get_model_state
from backend.app.api.dataset_routes import get_state as get_dataset_state
from backend.app.api.websocket import manager as ws_manager
from backend.app.engines.training_engine import (
    TrainingResult,
    request_stop,
    start_training,
)
from backend.app.models.schemas import TrainingConfig, TrainingMetrics
from backend.app.pipeline import pipeline

router = APIRouter(prefix="/api/training", tags=["training"])

# Module-level state for training
_state: dict = {
    "is_training": False,
    "last_result": None,
}


@router.post("/start")
async def start_training_endpoint(config: TrainingConfig) -> dict:
    """Start model training with the provided configuration.

    Requires model and dataset to be loaded/prepared first.
    """
    if _state["is_training"]:
        raise AppError(
            error_code="TRAINING_ALREADY_RUNNING",
            message="A training run is already in progress.",
        )

    model_state = get_model_state()
    dataset_state = get_dataset_state()

    model = model_state.get("model")
    tokenizer = model_state.get("tokenizer")
    train_dataset = dataset_state.get("train_dataset")
    val_dataset = dataset_state.get("val_dataset")

    if model is None or tokenizer is None:
        raise AppError(
            error_code="MODEL_NOT_LOADED",
            message="Model and tokenizer must be loaded first. Call POST /api/model/load.",
        )

    if train_dataset is None:
        raise AppError(
            error_code="DATASET_NOT_PREPARED",
            message="Dataset must be prepared first. Call POST /api/dataset/prepare.",
        )

    _state["is_training"] = True
    # Sync with centralised shared state
    from backend.app.main import app_state

    app_state.is_training = True

    # Advance pipeline to training
    if pipeline.can_transition("start_training"):
        await pipeline.transition("start_training")

    # Create a callback that broadcasts metrics via WebSocket
    loop = asyncio.get_running_loop()

    def ws_metrics_callback(metrics: TrainingMetrics) -> None:
        """Broadcast training metrics to all WebSocket clients."""
        payload = {
            "epoch": metrics.epoch,
            "step": metrics.step,
            "trainLoss": metrics.train_loss,
            "valLoss": metrics.val_loss,
            "learningRate": metrics.learning_rate,
            "elapsedSeconds": metrics.elapsed_seconds,
            "estimatedRemainingSeconds": metrics.estimated_remaining_seconds,
        }
        future = asyncio.run_coroutine_threadsafe(
            ws_manager.broadcast("metrics", payload), loop
        )
        # Wait briefly to ensure the broadcast actually happens
        try:
            future.result(timeout=1.0)
        except Exception:
            pass

    _completed = False
    try:
        # Run training in a thread so the event loop stays free for WebSocket
        result: TrainingResult = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: start_training(
                model=model,
                train_dataset=train_dataset,
                config=config,
                tokenizer=tokenizer,
                eval_dataset=val_dataset,
                ws_callback=ws_metrics_callback,
            ),
        )
        _state["last_result"] = result

        # Sync with centralised shared state
        from backend.app.main import app_state

        app_state.last_training_result = result

        # Advance pipeline to trained
        if pipeline.can_transition("training_complete"):
            await pipeline.transition("training_complete")

        _completed = True
        return {
            "success": result.success,
            "epochs_completed": result.epochs_completed,
            "final_train_loss": result.final_train_loss,
            "checkpoint_dir": result.checkpoint_dir,
        }
    finally:
        _state["is_training"] = False
        # Sync training flag with centralised state
        from backend.app.main import app_state as _app_state

        _app_state.is_training = False
        # If training was stopped or errored mid-run, revert pipeline stage
        if not _completed:
            if pipeline.can_transition("stop"):
                await pipeline.transition("stop")
            elif pipeline.can_transition("error"):
                await pipeline.transition("error")


@router.post("/stop")
async def stop_training_endpoint() -> dict:
    """Gracefully stop the current training run."""
    if not _state["is_training"]:
        raise AppError(
            error_code="TRAINING_ERROR",
            message="No training run is currently in progress.",
        )

    request_stop()
    return {"message": "Stop signal sent. Training will stop after the current step."}


def get_state() -> dict:
    """Expose module state for other modules."""
    return _state
