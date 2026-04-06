"""REST endpoints for dataset preparation.

POST /api/dataset/prepare  – triggers dataset download + tokenization
GET  /api/dataset/stats    – returns DatasetStats if dataset is prepared
"""

from __future__ import annotations

from fastapi import APIRouter

from backend.app.api.error_handler import AppError
from backend.app.api.model_routes import get_state as get_model_state
from backend.app.engines.dataset_preparer import prepare_dataset
from backend.app.models.schemas import DatasetStats
from backend.app.pipeline import pipeline

router = APIRouter(prefix="/api/dataset", tags=["dataset"])

# Module-level state for prepared dataset artefacts
_state: dict = {
    "train_dataset": None,
    "val_dataset": None,
    "stats": None,
}


@router.post("/prepare")
async def prepare_dataset_endpoint() -> dict:
    """Download WikiText-2, tokenize, and create train/val splits.

    Requires the tokenizer to be loaded first via ``POST /api/model/load``.
    """
    model_state = get_model_state()
    tokenizer = model_state.get("tokenizer")
    if tokenizer is None:
        raise AppError(
            error_code="MODEL_NOT_LOADED",
            message="Tokenizer has not been loaded yet. Call POST /api/model/load first.",
        )

    train_ds, val_ds, stats = prepare_dataset(tokenizer)

    _state["train_dataset"] = train_ds
    _state["val_dataset"] = val_ds
    _state["stats"] = stats

    # Sync with centralised shared state
    from backend.app.main import app_state

    app_state.train_dataset = train_ds
    app_state.val_dataset = val_ds
    app_state.dataset_stats = stats

    # Advance pipeline state machine
    if pipeline.can_transition("prepare_dataset"):
        await pipeline.transition("prepare_dataset")
    if pipeline.can_transition("dataset_ready"):
        await pipeline.transition("dataset_ready")

    return {"stats": stats.model_dump()}


@router.get("/stats", response_model=DatasetStats)
async def get_dataset_stats() -> DatasetStats:
    """Return statistics about the prepared dataset.

    Raises ``DATASET_NOT_PREPARED`` if the dataset has not been prepared yet.
    """
    if _state["stats"] is None:
        raise AppError(
            error_code="DATASET_NOT_PREPARED",
            message="Dataset has not been prepared yet. Call POST /api/dataset/prepare first.",
        )
    return _state["stats"]


def get_state() -> dict:
    """Expose module state for other modules that need the prepared datasets."""
    return _state
