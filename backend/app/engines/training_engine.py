"""Training Engine for GPT-2 fine-tuning.

Runs training via Hugging Face Trainer with custom callback for
streaming metrics via WebSocket.

**Validates: Requirements 3.3, 3.4, 4.1, 4.5, 4.7**
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from backend.app.api.error_handler import AppError
from backend.app.models.schemas import TrainingConfig, TrainingMetrics

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Result returned after training completes or fails."""

    success: bool
    epochs_completed: float = 0.0
    final_train_loss: float | None = None
    checkpoint_dir: str | None = None
    error_message: str | None = None


# Module-level flag for graceful stop
_stop_requested: bool = False


def request_stop() -> None:
    """Signal the training loop to stop gracefully."""
    global _stop_requested
    _stop_requested = True


def reset_stop() -> None:
    """Reset the stop flag (call before starting a new run)."""
    global _stop_requested
    _stop_requested = False


def config_to_training_args(config: TrainingConfig, output_dir: str, has_eval_dataset: bool = False) -> TrainingArguments:
    """Convert a TrainingConfig to HF TrainingArguments.

    Parameters
    ----------
    config:
        Application-level training configuration.
    output_dir:
        Directory where checkpoints will be saved.
    has_eval_dataset:
        If True, enable per-epoch evaluation so val_loss is reported.

    Returns
    -------
    TrainingArguments
        Hugging Face TrainingArguments with equivalent values.
    """
    return TrainingArguments(
        output_dir=output_dir,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        num_train_epochs=config.num_epochs,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        save_strategy="epoch",
        eval_strategy="epoch" if has_eval_dataset else "no",
        logging_steps=1,
        report_to="none",
        use_cpu=True,
    )



class MetricsCallback(TrainerCallback):
    """Custom TrainerCallback that streams TrainingMetrics via a callback.

    On each logging step, constructs a TrainingMetrics object and invokes
    the provided ``ws_callback`` so the frontend receives real-time updates.
    """

    def __init__(self, ws_callback: Callable[[TrainingMetrics], None] | None = None) -> None:
        super().__init__()
        self._ws_callback = ws_callback
        self._start_time: float = 0.0

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        self._start_time = time.time()

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if self._ws_callback is None or logs is None:
            return

        elapsed = time.time() - self._start_time
        current_epoch = state.epoch or 0.0
        total_epochs = args.num_train_epochs
        estimated_remaining = None
        if current_epoch > 0:
            estimated_remaining = (elapsed / current_epoch) * (total_epochs - current_epoch)

        metrics = TrainingMetrics(
            epoch=current_epoch,
            step=state.global_step,
            train_loss=logs.get("loss", 0.0),
            val_loss=logs.get("eval_loss", None),
            learning_rate=logs.get("learning_rate", 0.0),
            elapsed_seconds=elapsed,
            estimated_remaining_seconds=estimated_remaining,
        )
        self._ws_callback(metrics)

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Check for stop request and signal trainer to stop."""
        global _stop_requested
        if _stop_requested:
            control.should_training_stop = True


class _StopCallback(TrainerCallback):
    """Checks the module-level stop flag and halts training."""

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        global _stop_requested
        if _stop_requested:
            control.should_training_stop = True


def start_training(
    model: Any,
    train_dataset: Any,
    config: TrainingConfig,
    tokenizer: Any,
    output_dir: str = "./checkpoints",
    ws_callback: Callable[[TrainingMetrics], None] | None = None,
    eval_dataset: Any | None = None,
) -> TrainingResult:
    """Run training via HF Trainer.

    Parameters
    ----------
    model:
        A Hugging Face model (e.g. GPT2LMHeadModel).
    train_dataset:
        Training dataset (torch Dataset).
    config:
        Training hyperparameters.
    tokenizer:
        Tokenizer instance (needed for DataCollator).
    output_dir:
        Directory for saving checkpoints.
    ws_callback:
        Optional callback invoked with TrainingMetrics on each log step.
    eval_dataset:
        Optional validation dataset.

    Returns
    -------
    TrainingResult
        Result with success status, epochs completed, and checkpoint info.
    """
    reset_stop()

    training_args = config_to_training_args(config, output_dir, has_eval_dataset=eval_dataset is not None)

    # GPT-2 tokenizer doesn't have a pad token by default; set it to eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    metrics_cb = MetricsCallback(ws_callback=ws_callback)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[metrics_cb],
    )

    try:
        train_output = trainer.train()

        # Determine final checkpoint directory
        checkpoint_dirs = sorted(Path(output_dir).glob("checkpoint-*"))
        last_checkpoint = str(checkpoint_dirs[-1]) if checkpoint_dirs else None

        return TrainingResult(
            success=True,
            epochs_completed=train_output.metrics.get("epoch", config.num_epochs),
            final_train_loss=train_output.metrics.get("train_loss"),
            checkpoint_dir=last_checkpoint,
        )

    except Exception as exc:
        # Graceful error handling: try to save last checkpoint
        last_checkpoint = None
        try:
            emergency_dir = str(Path(output_dir) / "checkpoint-emergency")
            trainer.save_model(emergency_dir)
            last_checkpoint = emergency_dir
            logger.info("Emergency checkpoint saved to %s", emergency_dir)
        except Exception as save_exc:
            logger.warning("Failed to save emergency checkpoint: %s", save_exc)

        raise AppError(
            error_code="TRAINING_ERROR",
            message="Training failed",
            details=str(exc),
        ) from exc
