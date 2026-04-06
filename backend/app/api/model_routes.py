"""REST endpoints for model and tokenizer loading.

POST /api/model/load  – triggers model + tokenizer loading
GET  /api/model/summary – returns the ModelSummary if model is loaded
"""

from __future__ import annotations

from fastapi import APIRouter

from backend.app.api.error_handler import AppError
from backend.app.engines.cache_manager import CacheManager
from backend.app.engines.model_loader import load_model, load_tokenizer
from backend.app.models.schemas import ModelSummary
from backend.app.pipeline import pipeline

router = APIRouter(prefix="/api/model", tags=["model"])

# Shared state – populated after a successful load
_state: dict = {
    "model": None,
    "tokenizer": None,
    "summary": None,
    "tokenizer_info": None,
}

_cache_manager = CacheManager()


@router.post("/load")
async def load_model_endpoint() -> dict:
    """Initiate model and tokenizer loading.

    Downloads GPT-2 small (or loads from cache) and stores the model,
    tokenizer, and summary in module-level state so that subsequent
    endpoints can access them.  Also updates the centralised AppState.
    """
    model, summary = load_model(cache_manager=_cache_manager)
    tokenizer, tokenizer_info = load_tokenizer(cache_manager=_cache_manager)

    _state["model"] = model
    _state["tokenizer"] = tokenizer
    _state["summary"] = summary
    _state["tokenizer_info"] = tokenizer_info

    # Sync with centralised shared state
    from backend.app.main import app_state

    app_state.model = model
    app_state.tokenizer = tokenizer
    app_state.model_summary = summary
    app_state.tokenizer_info = tokenizer_info

    # Advance pipeline state machine
    if pipeline.can_transition("load_model"):
        await pipeline.transition("load_model")
    if pipeline.can_transition("model_loaded"):
        await pipeline.transition("model_loaded")

    return {
        "summary": summary.model_dump(),
        "tokenizer_info": tokenizer_info,
    }


@router.get("/summary", response_model=ModelSummary)
async def get_model_summary() -> ModelSummary:
    """Return the architecture summary of the loaded model.

    Raises ``MODEL_NOT_LOADED`` if the model has not been loaded yet.
    """
    if _state["summary"] is None:
        raise AppError(
            error_code="MODEL_NOT_LOADED",
            message="Model has not been loaded yet. Call POST /api/model/load first.",
        )
    return _state["summary"]


def get_state() -> dict:
    """Expose module state for other modules that need the loaded model/tokenizer."""
    return _state
