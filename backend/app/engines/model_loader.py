"""Model and tokenizer loading for GPT-2.

Downloads/caches GPT-2 small via Hugging Face Transformers and extracts
architecture summary information.

**Validates: Requirements 1.1, 1.2, 1.3, 1.6**
"""

from __future__ import annotations

import logging
from typing import Any

from transformers import AutoModelForCausalLM, AutoTokenizer

from backend.app.api.error_handler import AppError
from backend.app.engines.cache_manager import CacheManager
from backend.app.models.schemas import ModelSummary

logger = logging.getLogger(__name__)

_MODEL_NAME = "gpt2"


def load_model(cache_manager: CacheManager | None = None) -> tuple[Any, ModelSummary]:
    """Download / load GPT-2 small and return the model with its summary.

    Parameters
    ----------
    cache_manager:
        Optional :class:`CacheManager` used to check whether the model is
        already cached locally.

    Returns
    -------
    tuple[model, ModelSummary]
        The loaded ``AutoModelForCausalLM`` instance and a
        :class:`ModelSummary` describing its architecture.

    Raises
    ------
    AppError
        With ``error_code="MODEL_LOAD_FAILED"`` when the download or
        loading process fails for any reason.
    """
    try:
        cached = False
        if cache_manager is not None:
            cached = cache_manager.is_cached(_MODEL_NAME)
            if cached:
                logger.info("Model '%s' found in cache.", _MODEL_NAME)
            else:
                logger.info("Model '%s' not cached – downloading.", _MODEL_NAME)

        model = AutoModelForCausalLM.from_pretrained(_MODEL_NAME)

        config = model.config
        num_parameters = sum(p.numel() for p in model.parameters())

        summary = ModelSummary(
            name=_MODEL_NAME,
            num_layers=config.n_layer,
            num_parameters=num_parameters,
            hidden_size=config.n_embd,
            vocab_size=config.vocab_size,
        )

        logger.info(
            "Model loaded: %s (%d parameters, %d layers)",
            _MODEL_NAME,
            num_parameters,
            config.n_layer,
        )
        return model, summary

    except AppError:
        raise
    except Exception as exc:
        raise AppError(
            error_code="MODEL_LOAD_FAILED",
            message=f"Failed to load model '{_MODEL_NAME}'",
            details=str(exc),
        ) from exc


def load_tokenizer(cache_manager: CacheManager | None = None) -> tuple[Any, dict]:
    """Download / load the GPT-2 tokenizer and return it with vocab info.

    Parameters
    ----------
    cache_manager:
        Optional :class:`CacheManager` for cache-awareness logging.

    Returns
    -------
    tuple[tokenizer, dict]
        The loaded ``AutoTokenizer`` instance and a dictionary with keys
        ``vocab_size`` (int) and ``examples`` (list of dicts showing
        sample tokenisations).

    Raises
    ------
    AppError
        With ``error_code="MODEL_LOAD_FAILED"`` when the tokenizer
        cannot be loaded.
    """
    try:
        if cache_manager is not None:
            cached = cache_manager.is_cached(_MODEL_NAME)
            if cached:
                logger.info("Tokenizer '%s' found in cache.", _MODEL_NAME)
            else:
                logger.info("Tokenizer '%s' not cached – downloading.", _MODEL_NAME)

        tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)

        # Build a few tokenisation examples for the frontend
        sample_texts = [
            "Hello, world!",
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is fascinating.",
        ]
        examples = []
        for text in sample_texts:
            token_ids = tokenizer.encode(text)
            tokens = tokenizer.convert_ids_to_tokens(token_ids)
            examples.append({"text": text, "tokens": tokens, "token_ids": token_ids})

        info = {
            "vocab_size": tokenizer.vocab_size,
            "examples": examples,
        }

        logger.info("Tokenizer loaded: vocab_size=%d", tokenizer.vocab_size)
        return tokenizer, info

    except AppError:
        raise
    except Exception as exc:
        raise AppError(
            error_code="MODEL_LOAD_FAILED",
            message=f"Failed to load tokenizer for '{_MODEL_NAME}'",
            details=str(exc),
        ) from exc
