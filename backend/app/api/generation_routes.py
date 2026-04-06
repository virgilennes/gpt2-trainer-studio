"""REST endpoints for text generation.

POST /api/generation/generate  – generate text from a prompt
POST /api/generation/compare   – side-by-side generation from baseline and trained models
"""

from __future__ import annotations

from fastapi import APIRouter

from backend.app.api.error_handler import AppError
from backend.app.api.model_routes import get_state as get_model_state
from backend.app.engines.generation_engine import compare_generation, generate
from backend.app.models.schemas import GenerationParams

router = APIRouter(prefix="/api/generation", tags=["generation"])

# Module-level state for baseline model used in comparisons
_state: dict = {
    "baseline_model": None,
}


@router.post("/generate")
async def generate_endpoint(params: GenerationParams) -> dict:
    """Generate text from a prompt using the loaded model.

    Requires model and tokenizer to be loaded first via ``POST /api/model/load``.
    """
    model_state = get_model_state()
    model = model_state.get("model")
    tokenizer = model_state.get("tokenizer")

    if model is None or tokenizer is None:
        raise AppError(
            error_code="MODEL_NOT_LOADED",
            message="Model and tokenizer must be loaded first. Call POST /api/model/load.",
        )

    result = generate(model, tokenizer, params.prompt, params)
    return result.model_dump()


@router.post("/compare")
async def compare_endpoint(params: GenerationParams) -> dict:
    """Generate text from both baseline and trained models for comparison.

    Requires model and tokenizer to be loaded. Loads a fresh baseline
    GPT-2 on first call and caches it for subsequent requests.
    """
    model_state = get_model_state()
    model = model_state.get("model")
    tokenizer = model_state.get("tokenizer")

    if model is None or tokenizer is None:
        raise AppError(
            error_code="MODEL_NOT_LOADED",
            message="Model and tokenizer must be loaded first. Call POST /api/model/load.",
        )

    # Lazy-load baseline model for comparisons
    if _state["baseline_model"] is None:
        try:
            from transformers import AutoModelForCausalLM

            device = next(model.parameters()).device
            _state["baseline_model"] = AutoModelForCausalLM.from_pretrained("gpt2").to(device)

            # Sync with centralised shared state
            from backend.app.main import app_state

            app_state.baseline_model = _state["baseline_model"]
        except Exception as exc:
            raise AppError(
                error_code="GENERATION_ERROR",
                message="Failed to load baseline model for comparison",
                details=str(exc),
            ) from exc

    result = compare_generation(
        _state["baseline_model"], model, tokenizer, params.prompt, params
    )
    return result.model_dump()


def get_state() -> dict:
    """Expose module state for other modules."""
    return _state
