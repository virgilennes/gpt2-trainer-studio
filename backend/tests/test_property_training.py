"""Property tests for the Training Engine.

Property 3:  Config round-trip to training arguments
Property 4:  Checkpoint saved per epoch
Property 12: Checkpoint save/load round-trip

**Validates: Requirements 3.4, 4.5, 10.2**
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from hypothesis import given, settings, strategies as st, assume
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config

from backend.app.engines.training_engine import config_to_training_args
from backend.app.models.schemas import TrainingConfig


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_valid_training_config = st.builds(
    TrainingConfig,
    learning_rate=st.floats(min_value=1e-6, max_value=1e-2, allow_nan=False, allow_infinity=False),
    batch_size=st.integers(min_value=1, max_value=64),
    num_epochs=st.integers(min_value=1, max_value=20),
    warmup_steps=st.integers(min_value=0, max_value=1000),
    weight_decay=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)


# ---------------------------------------------------------------------------
# Property 3 — Config round-trip to training arguments
# ---------------------------------------------------------------------------

class TestConfigRoundTrip:
    """Property 3 — for any valid TrainingConfig, converting to
    TrainingArguments and reading back produces equivalent values.

    **Validates: Requirements 3.4**
    """

    @given(config=_valid_training_config)
    @settings(max_examples=100)
    def test_config_round_trip(self, config: TrainingConfig) -> None:
        """**Validates: Requirements 3.4**

        Converting TrainingConfig to TrainingArguments and reading back
        the fields should produce equivalent values.
        """
        with tempfile.TemporaryDirectory() as tmp:
            args = config_to_training_args(config, tmp)

            assert abs(args.learning_rate - config.learning_rate) < 1e-12, (
                f"learning_rate mismatch: {args.learning_rate} != {config.learning_rate}"
            )
            assert args.per_device_train_batch_size == config.batch_size, (
                f"batch_size mismatch: {args.per_device_train_batch_size} != {config.batch_size}"
            )
            assert args.num_train_epochs == config.num_epochs, (
                f"num_epochs mismatch: {args.num_train_epochs} != {config.num_epochs}"
            )
            assert args.warmup_steps == config.warmup_steps, (
                f"warmup_steps mismatch: {args.warmup_steps} != {config.warmup_steps}"
            )
            assert abs(args.weight_decay - config.weight_decay) < 1e-12, (
                f"weight_decay mismatch: {args.weight_decay} != {config.weight_decay}"
            )



# ---------------------------------------------------------------------------
# Property 4 — Checkpoint saved per epoch
# ---------------------------------------------------------------------------

class TestCheckpointPerEpoch:
    """Property 4 — for any N-epoch training run, exactly N checkpoints
    exist after completion.

    **Validates: Requirements 4.5**
    """

    @given(num_epochs=st.integers(min_value=1, max_value=5))
    @settings(max_examples=5, deadline=None)
    def test_n_checkpoints_for_n_epochs(self, num_epochs: int) -> None:
        """**Validates: Requirements 4.5**

        After training for N epochs, exactly N checkpoint directories
        should exist in the output directory.
        """
        from backend.app.engines.training_engine import start_training

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = str(Path(tmp) / "checkpoints")

            # Create a tiny model for fast training
            tiny_config = GPT2Config(
                vocab_size=50,
                n_embd=16,
                n_layer=1,
                n_head=1,
                n_positions=32,
            )
            model = AutoModelForCausalLM.from_config(tiny_config)

            # Create a tiny tokenizer-like object with pad_token set
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token

            # Create a minimal dataset: list of dicts with input_ids and labels
            # All same length to avoid padding issues
            dataset = []
            for _ in range(16):
                ids = torch.randint(0, 50, (32,)).tolist()
                dataset.append({"input_ids": ids, "labels": ids})

            config = TrainingConfig(
                learning_rate=5e-4,
                batch_size=4,
                num_epochs=num_epochs,
                warmup_steps=0,
                weight_decay=0.0,
            )

            result = start_training(
                model=model,
                train_dataset=dataset,
                config=config,
                tokenizer=tokenizer,
                output_dir=output_dir,
            )

            assert result.success, f"Training should succeed, got error: {result.error_message}"

            # Count checkpoint directories
            checkpoint_dirs = sorted(Path(output_dir).glob("checkpoint-*"))
            assert len(checkpoint_dirs) == num_epochs, (
                f"Expected {num_epochs} checkpoints, found {len(checkpoint_dirs)}: "
                f"{[d.name for d in checkpoint_dirs]}"
            )


# ---------------------------------------------------------------------------
# Property 12 — Checkpoint save/load round-trip
# ---------------------------------------------------------------------------

class TestCheckpointRoundTrip:
    """Property 12 — for any saved checkpoint, loading it produces a model
    with identical outputs for the same input.

    **Validates: Requirements 10.2**
    """

    @given(seed=st.integers(min_value=0, max_value=10000))
    @settings(max_examples=3, deadline=None)
    def test_checkpoint_save_load_produces_identical_outputs(self, seed: int) -> None:
        """**Validates: Requirements 10.2**

        Save a model checkpoint, load it back, and verify that both
        produce identical outputs for the same input.
        """
        torch.manual_seed(seed)

        with tempfile.TemporaryDirectory() as tmp:
            # Create a tiny model
            tiny_config = GPT2Config(
                vocab_size=50,
                n_embd=16,
                n_layer=1,
                n_head=1,
                n_positions=32,
            )
            model = AutoModelForCausalLM.from_config(tiny_config)
            model.eval()

            # Save checkpoint
            checkpoint_dir = str(Path(tmp) / "checkpoint")
            model.save_pretrained(checkpoint_dir)

            # Load checkpoint
            loaded_model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
            loaded_model.eval()

            # Generate identical input
            input_ids = torch.randint(0, 50, (1, 16))

            with torch.no_grad():
                original_output = model(input_ids).logits
                loaded_output = loaded_model(input_ids).logits

            assert torch.allclose(original_output, loaded_output, atol=1e-6), (
                "Loaded checkpoint should produce identical outputs. "
                f"Max diff: {(original_output - loaded_output).abs().max().item()}"
            )
