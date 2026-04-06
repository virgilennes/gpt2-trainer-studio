"""Property tests for dataset preparation.

Property 1:  TextDataset fixed-length sequences
Property 16: Dataset split non-empty

**Validates: Requirements 2.2, 2.3**
"""

from __future__ import annotations

from hypothesis import given, settings, strategies as st, assume
from unittest.mock import MagicMock

from backend.app.engines.dataset_preparer import TextDataset, prepare_dataset


# ---------------------------------------------------------------------------
# Helpers — lightweight fake tokenizer
# ---------------------------------------------------------------------------

class _FakeTokenizer:
    """Minimal tokenizer stub that splits text into character-level token IDs."""

    vocab_size: int = 256

    def encode(self, text: str) -> list[int]:
        return [ord(c) % self.vocab_size for c in text]


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Corpus text: non-empty strings with enough characters to form at least one block.
# We use printable text to keep things readable.
_corpus_text = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
    min_size=1,
    max_size=2000,
)

_block_size = st.integers(min_value=1, max_value=256)


# ---------------------------------------------------------------------------
# Property 1 — TextDataset fixed-length sequences
# ---------------------------------------------------------------------------

class TestTextDatasetFixedLength:
    """Property 1 — for any corpus and block_size, all sequences have
    exactly ``block_size`` tokens.

    **Validates: Requirements 2.3**
    """

    @given(text=_corpus_text, block_size=_block_size)
    @settings(max_examples=100)
    def test_all_sequences_have_exact_block_size(self, text: str, block_size: int) -> None:
        """**Validates: Requirements 2.3**

        Every tensor returned by TextDataset must have length == block_size.
        """
        tokenizer = _FakeTokenizer()
        ds = TextDataset(text, tokenizer, block_size=block_size)

        for i in range(len(ds)):
            seq = ds[i]
            assert seq.shape[0] == block_size, (
                f"Sequence {i} has length {seq.shape[0]}, expected {block_size}"
            )

    @given(text=_corpus_text, block_size=_block_size)
    @settings(max_examples=100)
    def test_no_tokens_lost_beyond_remainder(self, text: str, block_size: int) -> None:
        """**Validates: Requirements 2.3**

        The number of samples times block_size should equal the total
        tokens minus the discarded remainder.
        """
        tokenizer = _FakeTokenizer()
        token_ids = tokenizer.encode(text)
        ds = TextDataset(text, tokenizer, block_size=block_size)

        expected_samples = len(token_ids) // block_size
        assert len(ds) == expected_samples


# ---------------------------------------------------------------------------
# Property 16 — Dataset split non-empty
# ---------------------------------------------------------------------------

class TestDatasetSplitNonEmpty:
    """Property 16 — for any successful preparation, both train and val
    sets are non-empty.

    **Validates: Requirements 2.2**
    """

    @given(
        train_text=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
            min_size=200,
            max_size=2000,
        ),
        val_text=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
            min_size=200,
            max_size=2000,
        ),
        block_size=st.integers(min_value=1, max_value=64),
    )
    @settings(max_examples=100, deadline=None)
    def test_both_splits_non_empty(
        self, train_text: str, val_text: str, block_size: int
    ) -> None:
        """**Validates: Requirements 2.2**

        When prepare_dataset succeeds, both train and val datasets must
        contain at least one sample, and total equals train + val.
        """
        tokenizer = _FakeTokenizer()

        # Ensure enough tokens for at least one block in each split
        train_tokens = tokenizer.encode(train_text)
        val_tokens = tokenizer.encode(val_text)
        assume(len(train_tokens) >= block_size)
        assume(len(val_tokens) >= block_size)

        # Mock the HF datasets download to return our generated text
        import unittest.mock as mock

        fake_dataset = {
            "train": {"text": [train_text]},
            "validation": {"text": [val_text]},
        }

        # prepare_dataset does `from datasets import load_dataset` inside the
        # try block, so we patch the module-level datasets.load_dataset.
        with mock.patch("datasets.load_dataset", return_value=fake_dataset):
            train_ds, val_ds, stats = prepare_dataset(tokenizer, block_size=block_size)

        assert len(train_ds) > 0, "Training set must be non-empty"
        assert len(val_ds) > 0, "Validation set must be non-empty"
        assert stats.train_samples == len(train_ds)
        assert stats.val_samples == len(val_ds)
        assert stats.train_samples + stats.val_samples == len(train_ds) + len(val_ds)
        assert stats.block_size == block_size
        assert stats.vocab_size == tokenizer.vocab_size
