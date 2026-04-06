"""Generation Engine for GPT-2 text generation.

Generates text from a prompt using configurable parameters (temperature,
top_k, top_p, max_length) and supports side-by-side comparison between
a baseline and trained model.

**Validates: Requirements 6.2, 6.4, 6.7**
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from backend.app.api.error_handler import AppError
from backend.app.models.schemas import (
    CompareGenerationResult,
    GenerationParams,
    GenerationResult,
)

logger = logging.getLogger(__name__)


def generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    params: GenerationParams,
) -> GenerationResult:
    """Generate text from a prompt with the given parameters.

    Parameters
    ----------
    model:
        A Hugging Face causal LM (e.g. GPT2LMHeadModel).
    tokenizer:
        The tokenizer instance used for encoding/decoding.
    prompt:
        The text prompt to continue from.
    params:
        Generation parameters (temperature, top_k, top_p, max_length).

    Returns
    -------
    GenerationResult
        Contains the generated ``text`` and ``tokens_generated`` count.

    Raises
    ------
    AppError
        With ``error_code="GENERATION_ERROR"`` on any failure.
    """
    try:
        model.eval()
        device = next(model.parameters()).device

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        input_length = input_ids.shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                do_sample=True,
                temperature=params.temperature,
                top_k=params.top_k,
                top_p=params.top_p,
                max_length=params.max_length,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        tokens_generated = output_ids.shape[1] - input_length

        logger.info(
            "Generation complete: %d new tokens from prompt of %d tokens",
            tokens_generated,
            input_length,
        )

        return GenerationResult(text=generated_text, tokens_generated=tokens_generated)

    except AppError:
        raise
    except Exception as exc:
        raise AppError(
            error_code="GENERATION_ERROR",
            message="Text generation failed",
            details=str(exc),
        ) from exc


def compare_generation(
    baseline_model: Any,
    trained_model: Any,
    tokenizer: Any,
    prompt: str,
    params: GenerationParams,
) -> CompareGenerationResult:
    """Generate text from both baseline and trained models for comparison.

    Parameters
    ----------
    baseline_model:
        The original (untrained) Hugging Face causal LM.
    trained_model:
        The fine-tuned Hugging Face causal LM.
    tokenizer:
        The shared tokenizer instance.
    prompt:
        The text prompt to continue from.
    params:
        Generation parameters applied to both models.

    Returns
    -------
    CompareGenerationResult
        Contains ``baseline_text``, ``trained_text``, and the ``prompt``.

    Raises
    ------
    AppError
        With ``error_code="GENERATION_ERROR"`` on any failure.
    """
    try:
        baseline_result = generate(baseline_model, tokenizer, prompt, params)
        trained_result = generate(trained_model, tokenizer, prompt, params)

        logger.info("Comparison generation complete for prompt: %s...", prompt[:40])

        return CompareGenerationResult(
            baseline_text=baseline_result.text,
            trained_text=trained_result.text,
            prompt=prompt,
        )

    except AppError:
        raise
    except Exception as exc:
        raise AppError(
            error_code="GENERATION_ERROR",
            message="Comparison generation failed",
            details=str(exc),
        ) from exc
