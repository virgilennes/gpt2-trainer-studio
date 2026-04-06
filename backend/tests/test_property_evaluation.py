"""Property test for the Evaluation Engine.

Property 5: Evaluation produces valid perplexity — for any model and
validation set, perplexity is a positive finite number.

**Validates: Requirements 5.1**
"""

from __future__ import annotations

import math

import torch
from hypothesis import given, settings, strategies as st
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config

from backend.app.engines.evaluation_engine import evaluate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tiny_model(vocab_size: int = 50) -> AutoModelForCausalLM:
    """Create a tiny GPT-2 model for fast testing."""
    config = GPT2Config(
        vocab_size=vocab_size,
        n_embd=16,
        n_layer=1,
        n_head=1,
        n_positions=32,
    )
    return AutoModelForCausalLM.from_config(config)


def _make_val_dataset(num_samples: int, seq_len: int, vocab_size: int) -> list[torch.Tensor]:
    """Create a random validation dataset of token-ID tensors."""
    return [torch.randint(0, vocab_size, (seq_len,)) for _ in range(num_samples)]


# ---------------------------------------------------------------------------
# Property 5 — Valid perplexity
# ---------------------------------------------------------------------------

class TestValidPerplexity:
    """Property 5 — for any model and validation set, perplexity is a
    positive finite number.

    **Validates: Requirements 5.1**
    """

    @given(
        num_samples=st.integers(min_value=1, max_value=8),
        seq_len=st.integers(min_value=4, max_value=32),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=20, deadline=None)
    def test_perplexity_is_positive_finite(
        self, num_samples: int, seq_len: int, seed: int
    ) -> None:
        """**Validates: Requirements 5.1**

        For any model and validation set, perplexity must be a positive
        finite number (> 0, not infinity, not NaN).
        """
        torch.manual_seed(seed)
        vocab_size = 50

        model = _make_tiny_model(vocab_size=vocab_size)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        val_dataset = _make_val_dataset(num_samples, seq_len, vocab_size)

        result = evaluate(model, tokenizer, val_dataset)

        assert result.perplexity > 0, (
            f"Perplexity must be positive, got {result.perplexity}"
        )
        assert math.isfinite(result.perplexity), (
            f"Perplexity must be finite, got {result.perplexity}"
        )
        assert result.val_loss >= 0, (
            f"Validation loss must be non-negative, got {result.val_loss}"
        )
        assert math.isfinite(result.val_loss), (
            f"Validation loss must be finite, got {result.val_loss}"
        )
        # Verify perplexity = exp(val_loss)
        expected_perplexity = math.exp(result.val_loss)
        assert abs(result.perplexity - expected_perplexity) < 1e-4, (
            f"Perplexity should equal exp(val_loss): "
            f"{result.perplexity} != exp({result.val_loss}) = {expected_perplexity}"
        )
