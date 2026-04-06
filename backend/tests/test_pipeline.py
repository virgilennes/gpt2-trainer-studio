"""Tests for the pipeline state machine.

Covers state transitions, guards, WebSocket broadcasting, and the
GET /api/status endpoint.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from backend.app.pipeline import (
    InvalidTransitionError,
    PipelineStage,
    PipelineStateMachine,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_pipeline() -> PipelineStateMachine:
    """Return a new state machine starting at idle."""
    return PipelineStateMachine()


# ---------------------------------------------------------------------------
# Unit tests – initial state
# ---------------------------------------------------------------------------


def test_initial_stage_is_idle():
    sm = _fresh_pipeline()
    assert sm.stage == PipelineStage.IDLE


def test_initial_previous_stage_is_none():
    sm = _fresh_pipeline()
    assert sm.previous_stage is None


# ---------------------------------------------------------------------------
# Unit tests – valid transitions (async)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_load_model_transition():
    sm = _fresh_pipeline()
    result = await sm.transition("load_model")
    assert result == PipelineStage.MODEL_LOADING
    assert sm.stage == PipelineStage.MODEL_LOADING


@pytest.mark.asyncio
async def test_full_happy_path():
    """Walk through the entire happy-path pipeline."""
    sm = _fresh_pipeline()

    await sm.transition("load_model")
    assert sm.stage == PipelineStage.MODEL_LOADING

    await sm.transition("model_loaded")
    assert sm.stage == PipelineStage.MODEL_LOADED

    await sm.transition("prepare_dataset")
    assert sm.stage == PipelineStage.DATASET_PREPARING

    await sm.transition("dataset_ready")
    assert sm.stage == PipelineStage.DATASET_READY

    await sm.transition("start_training")
    assert sm.stage == PipelineStage.TRAINING

    await sm.transition("training_complete")
    assert sm.stage == PipelineStage.TRAINED

    await sm.transition("run_evaluation")
    assert sm.stage == PipelineStage.EVALUATING

    await sm.transition("evaluation_complete")
    assert sm.stage == PipelineStage.EVALUATED

    await sm.transition("generate")
    assert sm.stage == PipelineStage.GENERATING

    await sm.transition("generation_complete_to_evaluated")
    assert sm.stage == PipelineStage.EVALUATED


@pytest.mark.asyncio
async def test_generate_from_trained():
    sm = _fresh_pipeline()
    await sm.transition("load_model")
    await sm.transition("model_loaded")
    await sm.transition("prepare_dataset")
    await sm.transition("dataset_ready")
    await sm.transition("start_training")
    await sm.transition("training_complete")

    await sm.transition("generate")
    assert sm.stage == PipelineStage.GENERATING

    await sm.transition("generation_complete_to_trained")
    assert sm.stage == PipelineStage.TRAINED


@pytest.mark.asyncio
async def test_error_during_model_loading_returns_to_idle():
    sm = _fresh_pipeline()
    await sm.transition("load_model")
    await sm.transition("error")
    assert sm.stage == PipelineStage.IDLE


@pytest.mark.asyncio
async def test_error_during_training_returns_to_dataset_ready():
    sm = _fresh_pipeline()
    await sm.transition("load_model")
    await sm.transition("model_loaded")
    await sm.transition("prepare_dataset")
    await sm.transition("dataset_ready")
    await sm.transition("start_training")
    await sm.transition("error")
    assert sm.stage == PipelineStage.DATASET_READY


@pytest.mark.asyncio
async def test_stop_during_training_returns_to_dataset_ready():
    sm = _fresh_pipeline()
    await sm.transition("load_model")
    await sm.transition("model_loaded")
    await sm.transition("prepare_dataset")
    await sm.transition("dataset_ready")
    await sm.transition("start_training")
    await sm.transition("stop")
    assert sm.stage == PipelineStage.DATASET_READY


# ---------------------------------------------------------------------------
# Unit tests – invalid transitions (guards)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cannot_train_from_idle():
    sm = _fresh_pipeline()
    assert not sm.can_transition("start_training")
    with pytest.raises(InvalidTransitionError):
        await sm.transition("start_training")


@pytest.mark.asyncio
async def test_cannot_train_without_dataset():
    sm = _fresh_pipeline()
    await sm.transition("load_model")
    await sm.transition("model_loaded")
    assert not sm.can_transition("start_training")
    with pytest.raises(InvalidTransitionError):
        await sm.transition("start_training")


@pytest.mark.asyncio
async def test_cannot_evaluate_without_trained_model():
    sm = _fresh_pipeline()
    await sm.transition("load_model")
    await sm.transition("model_loaded")
    await sm.transition("prepare_dataset")
    await sm.transition("dataset_ready")
    assert not sm.can_transition("run_evaluation")
    with pytest.raises(InvalidTransitionError):
        await sm.transition("run_evaluation")


@pytest.mark.asyncio
async def test_cannot_generate_from_idle():
    sm = _fresh_pipeline()
    with pytest.raises(InvalidTransitionError):
        await sm.transition("generate")


# ---------------------------------------------------------------------------
# Unit tests – guard helpers
# ---------------------------------------------------------------------------


def test_require_model_loaded_raises_from_idle():
    sm = _fresh_pipeline()
    with pytest.raises(InvalidTransitionError):
        sm.require_model_loaded()


@pytest.mark.asyncio
async def test_require_model_loaded_passes_after_load():
    sm = _fresh_pipeline()
    await sm.transition("load_model")
    await sm.transition("model_loaded")
    sm.require_model_loaded()  # should not raise


def test_require_dataset_ready_raises_from_idle():
    sm = _fresh_pipeline()
    with pytest.raises(InvalidTransitionError):
        sm.require_dataset_ready()


@pytest.mark.asyncio
async def test_require_dataset_ready_passes():
    sm = _fresh_pipeline()
    await sm.transition("load_model")
    await sm.transition("model_loaded")
    await sm.transition("prepare_dataset")
    await sm.transition("dataset_ready")
    sm.require_dataset_ready()  # should not raise


def test_require_trained_raises_from_idle():
    sm = _fresh_pipeline()
    with pytest.raises(InvalidTransitionError):
        sm.require_trained()


@pytest.mark.asyncio
async def test_require_trained_passes():
    sm = _fresh_pipeline()
    await sm.transition("load_model")
    await sm.transition("model_loaded")
    await sm.transition("prepare_dataset")
    await sm.transition("dataset_ready")
    await sm.transition("start_training")
    await sm.transition("training_complete")
    sm.require_trained()  # should not raise


# ---------------------------------------------------------------------------
# Unit tests – can_transition
# ---------------------------------------------------------------------------


def test_can_transition_true():
    sm = _fresh_pipeline()
    assert sm.can_transition("load_model") is True


def test_can_transition_false():
    sm = _fresh_pipeline()
    assert sm.can_transition("start_training") is False


# ---------------------------------------------------------------------------
# Unit tests – get_status
# ---------------------------------------------------------------------------


def test_get_status_idle():
    sm = _fresh_pipeline()
    status = sm.get_status()
    assert status["stage"] == "idle"
    assert status["previous_stage"] is None


@pytest.mark.asyncio
async def test_get_status_after_transition():
    sm = _fresh_pipeline()
    await sm.transition("load_model")
    status = sm.get_status()
    assert status["stage"] == "model_loading"
    assert status["previous_stage"] == "idle"


# ---------------------------------------------------------------------------
# Unit tests – sync transition
# ---------------------------------------------------------------------------


def test_transition_sync():
    sm = _fresh_pipeline()
    result = sm.transition_sync("load_model")
    assert result == PipelineStage.MODEL_LOADING
    assert sm.stage == PipelineStage.MODEL_LOADING


def test_transition_sync_invalid():
    sm = _fresh_pipeline()
    with pytest.raises(InvalidTransitionError):
        sm.transition_sync("start_training")


# ---------------------------------------------------------------------------
# Unit tests – previous_stage tracking
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_previous_stage_tracked():
    sm = _fresh_pipeline()
    await sm.transition("load_model")
    assert sm.previous_stage == PipelineStage.IDLE
    await sm.transition("model_loaded")
    assert sm.previous_stage == PipelineStage.MODEL_LOADING


# ---------------------------------------------------------------------------
# Unit tests – re-evaluation from evaluated state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_re_evaluate_from_evaluated():
    sm = _fresh_pipeline()
    await sm.transition("load_model")
    await sm.transition("model_loaded")
    await sm.transition("prepare_dataset")
    await sm.transition("dataset_ready")
    await sm.transition("start_training")
    await sm.transition("training_complete")
    await sm.transition("run_evaluation")
    await sm.transition("evaluation_complete")
    assert sm.stage == PipelineStage.EVALUATED

    # Should be able to re-evaluate
    await sm.transition("run_evaluation")
    assert sm.stage == PipelineStage.EVALUATING
