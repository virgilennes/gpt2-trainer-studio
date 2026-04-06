"""Dataset preparation for GPT-2 fine-tuning.

Downloads WikiText-2 via Hugging Face ``datasets``, tokenizes the corpus,
and creates fixed-length training/validation splits.

**Validates: Requirements 2.1, 2.2, 2.3, 2.6**
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch.utils.data import Dataset

from backend.app.api.error_handler import AppError
from backend.app.models.schemas import DatasetStats

logger = logging.getLogger(__name__)

_DATASET_NAME = "wikitext"
_DATASET_CONFIG = "wikitext-2-raw-v1"


class TextDataset(Dataset):
    """Tokenizes corpus text and chunks into fixed-length sequences.

    The raw text is encoded into a single long token-ID sequence, then
    split into non-overlapping segments of exactly ``block_size`` tokens.
    Any trailing tokens that don't fill a complete block are discarded.

    Implements ``__len__`` and ``__getitem__`` for PyTorch DataLoader
    compatibility.
    """

    def __init__(self, text: str, tokenizer: Any, block_size: int = 128) -> None:
        token_ids = tokenizer.encode(text)
        # Chunk into non-overlapping segments of exactly block_size
        num_complete_blocks = len(token_ids) // block_size
        total_tokens = num_complete_blocks * block_size
        token_ids = token_ids[:total_tokens]

        self.examples: list[torch.Tensor] = []
        for i in range(0, total_tokens, block_size):
            self.examples.append(torch.tensor(token_ids[i : i + block_size], dtype=torch.long))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.examples[idx]


def prepare_dataset(
    tokenizer: Any,
    block_size: int = 128,
) -> tuple["TextDataset", "TextDataset", DatasetStats]:
    """Download WikiText-2, tokenize, and create train/val splits.

    Parameters
    ----------
    tokenizer:
        A Hugging Face tokenizer instance (e.g. GPT-2).
    block_size:
        Number of tokens per training sequence.

    Returns
    -------
    tuple[TextDataset, TextDataset, DatasetStats]
        Training dataset, validation dataset, and statistics.

    Raises
    ------
    AppError
        With ``error_code="DATASET_DOWNLOAD_FAILED"`` when the download
        or preparation fails.
    """
    try:
        from datasets import load_dataset

        logger.info("Downloading WikiText-2 dataset…")
        raw = load_dataset(_DATASET_NAME, _DATASET_CONFIG)

        train_text = "\n".join(line for line in raw["train"]["text"] if line.strip())
        val_text = "\n".join(line for line in raw["validation"]["text"] if line.strip())

        logger.info("Tokenizing training split (block_size=%d)…", block_size)
        train_dataset = TextDataset(train_text, tokenizer, block_size=block_size)

        logger.info("Tokenizing validation split (block_size=%d)…", block_size)
        val_dataset = TextDataset(val_text, tokenizer, block_size=block_size)

        vocab_size: int = tokenizer.vocab_size

        stats = DatasetStats(
            train_samples=len(train_dataset),
            val_samples=len(val_dataset),
            vocab_size=vocab_size,
            block_size=block_size,
        )

        logger.info(
            "Dataset ready: %d train samples, %d val samples, vocab=%d, block=%d",
            stats.train_samples,
            stats.val_samples,
            stats.vocab_size,
            stats.block_size,
        )
        return train_dataset, val_dataset, stats

    except AppError:
        raise
    except Exception as exc:
        raise AppError(
            error_code="DATASET_DOWNLOAD_FAILED",
            message="Failed to download or prepare the WikiText-2 dataset",
            details=str(exc),
        ) from exc
