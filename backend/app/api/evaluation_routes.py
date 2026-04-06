"""REST endpoints for model evaluation.

POST /api/evaluation/run  – run evaluation on validation set
"""

from __future__ import annotations

from fastapi import APIRouter

from backend.app.api.error_handler import AppError
from backend.app.api.model_routes import get_state as get_model_state
from backend.app.api.dataset_routes import get_state as get_dataset_state
from backend.app.engines.evaluation_engine import evaluate

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])


@router.post("/run")
async def run_evaluation_endpoint() -> dict:
    """Run evaluation on the validation set.

    Requires model, tokenizer, and dataset to be loaded/prepared first.
    Returns perplexity and validation loss.
    """
    model_state = get_model_state()
    dataset_state = get_dataset_state()

    model = model_state.get("model")
    tokenizer = model_state.get("tokenizer")
    val_dataset = dataset_state.get("val_dataset")

    if model is None or tokenizer is None:
        raise AppError(
            error_code="MODEL_NOT_LOADED",
            message="Model and tokenizer must be loaded first. Call POST /api/model/load.",
        )

    if val_dataset is None:
        raise AppError(
            error_code="DATASET_NOT_PREPARED",
            message="Dataset must be prepared first. Call POST /api/dataset/prepare.",
        )

    result = evaluate(model, tokenizer, val_dataset)
    return {
        "perplexity": result.perplexity,
        "val_loss": result.val_loss,
    }
