"""Evaluation Engine for GPT-2 model assessment.

Calculates perplexity on the validation set and compares trained model
performance against the baseline GPT-2 model.

**Validates: Requirements 5.1, 5.3, 5.5**
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
from torch.utils.data import DataLoader

from backend.app.api.error_handler import AppError
from backend.app.models.schemas import ComparisonResult, EvalResult

logger = logging.getLogger(__name__)


def evaluate(model: Any, tokenizer: Any, val_dataset: Any) -> EvalResult:
    """Calculate perplexity on the validation set.

    Iterates over the validation dataset, computes cross-entropy loss
    using the model, and returns exp(avg_loss) as perplexity.

    Parameters
    ----------
    model:
        A Hugging Face causal LM (e.g. GPT2LMHeadModel).
    tokenizer:
        The tokenizer instance (used for pad token setup).
    val_dataset:
        Validation dataset — each item is a tensor of token IDs or a
        dict with ``input_ids`` and ``labels``.

    Returns
    -------
    EvalResult
        Contains ``perplexity`` (exp of average loss) and ``val_loss``.

    Raises
    ------
    AppError
        With ``error_code="EVALUATION_ERROR"`` on any failure.
    """
    try:
        model.eval()
        device = next(model.parameters()).device

        total_loss = 0.0
        total_steps = 0

        dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, torch.Tensor):
                    input_ids = batch.to(device)
                    labels = input_ids.clone()
                elif isinstance(batch, dict):
                    input_ids = torch.tensor(batch["input_ids"]).to(device) if not isinstance(batch["input_ids"], torch.Tensor) else batch["input_ids"].to(device)
                    labels = torch.tensor(batch["labels"]).to(device) if not isinstance(batch["labels"], torch.Tensor) else batch["labels"].to(device)
                else:
                    raise ValueError(f"Unexpected batch type: {type(batch)}")

                outputs = model(input_ids=input_ids, labels=labels)
                total_loss += outputs.loss.item()
                total_steps += 1

        if total_steps == 0:
            raise ValueError("Validation dataset is empty — no batches to evaluate.")

        avg_loss = total_loss / total_steps
        perplexity = math.exp(avg_loss)

        logger.info("Evaluation complete: perplexity=%.4f, val_loss=%.4f", perplexity, avg_loss)

        return EvalResult(perplexity=perplexity, val_loss=avg_loss)

    except AppError:
        raise
    except Exception as exc:
        raise AppError(
            error_code="EVALUATION_ERROR",
            message="Evaluation failed",
            details=str(exc),
        ) from exc


def compare_baseline(model: Any, tokenizer: Any, val_dataset: Any) -> ComparisonResult:
    """Compare trained model perplexity against a fresh GPT-2 baseline.

    Loads a fresh GPT-2 model, evaluates both the trained and baseline
    models on the same validation set, and computes the improvement
    percentage.

    Parameters
    ----------
    model:
        The trained Hugging Face causal LM.
    tokenizer:
        The tokenizer instance.
    val_dataset:
        Validation dataset for evaluation.

    Returns
    -------
    ComparisonResult
        Contains ``baseline_perplexity``, ``trained_perplexity``, and
        ``improvement_pct`` (positive means trained is better).

    Raises
    ------
    AppError
        With ``error_code="EVALUATION_ERROR"`` on any failure.
    """
    try:
        from transformers import AutoModelForCausalLM

        # Evaluate the trained model
        trained_result = evaluate(model, tokenizer, val_dataset)

        # Load a fresh baseline GPT-2
        baseline_model = AutoModelForCausalLM.from_pretrained("gpt2")
        device = next(model.parameters()).device
        baseline_model = baseline_model.to(device)

        # Evaluate the baseline
        baseline_result = evaluate(baseline_model, tokenizer, val_dataset)

        # Improvement: positive means trained is better (lower perplexity)
        if baseline_result.perplexity > 0:
            improvement_pct = (
                (baseline_result.perplexity - trained_result.perplexity)
                / baseline_result.perplexity
            ) * 100.0
        else:
            improvement_pct = 0.0

        logger.info(
            "Comparison: baseline=%.4f, trained=%.4f, improvement=%.2f%%",
            baseline_result.perplexity,
            trained_result.perplexity,
            improvement_pct,
        )

        return ComparisonResult(
            baseline_perplexity=baseline_result.perplexity,
            trained_perplexity=trained_result.perplexity,
            improvement_pct=improvement_pct,
        )

    except AppError:
        raise
    except Exception as exc:
        raise AppError(
            error_code="EVALUATION_ERROR",
            message="Baseline comparison failed",
            details=str(exc),
        ) from exc
