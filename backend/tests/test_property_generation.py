"""Property test for the Generation Engine.

Property 6: Generation produces non-empty output for valid inputs — for any
valid prompt and generation parameters, the output is non-empty and contains
at least the original prompt.

**Validates: Requirements 6.2**
"""

from __future__ import annotations

import torch
from hypothesis import given, settings, strategies as st
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config

from backend.app.engines.generation_engine import generate
from backend.app.models.schemas import GenerationParams


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tiny_model(vocab_size: int = 50257) -> AutoModelForCausalLM:
    """Create a tiny GPT-2 model for fast testing."""
    config = GPT2Config(
        vocab_size=vocab_size,
        n_embd=16,
        n_layer=1,
        n_head=1,
        n_positions=128,
    )
    return AutoModelForCausalLM.from_config(config)


_tokenizer = AutoTokenizer.from_pretrained("gpt2")
_model = _make_tiny_model(vocab_size=_tokenizer.vocab_size)


# ---------------------------------------------------------------------------
# Property 6 — Generation non-empty output
# ---------------------------------------------------------------------------

class TestGenerationNonEmptyOutput:
    """Property 6 — for any valid prompt and params, output is non-empty
    and contains the prompt.

    **Validates: Requirements 6.2**
    """

    @given(
        prompt=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
            min_size=1,
            max_size=20,
        ).filter(lambda s: s.strip() != ""),
        temperature=st.floats(min_value=0.1, max_value=2.0),
        top_k=st.integers(min_value=1, max_value=100),
        top_p=st.floats(min_value=0.1, max_value=1.0),
        max_length=st.integers(min_value=50, max_value=100),
    )
    @settings(max_examples=20, deadline=None)
    def test_generation_non_empty_and_contains_prompt(
        self,
        prompt: str,
        temperature: float,
        top_k: int,
        top_p: float,
        max_length: int,
    ) -> None:
        """**Validates: Requirements 6.2**

        For any valid prompt and generation parameters, the generated text
        is non-empty and contains the original prompt.
        """
        torch.manual_seed(42)

        params = GenerationParams(
            prompt=prompt,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_length=max_length,
        )

        result = generate(_model, _tokenizer, prompt, params)

        assert result.text, "Generated text must be non-empty"
        assert len(result.text) > 0, "Generated text length must be > 0"
        assert result.tokens_generated >= 0, (
            f"tokens_generated must be non-negative, got {result.tokens_generated}"
        )
        # The generated text should start with / contain the prompt
        assert prompt in result.text, (
            f"Generated text must contain the prompt.\n"
            f"Prompt: {prompt!r}\n"
            f"Generated: {result.text!r}"
        )
