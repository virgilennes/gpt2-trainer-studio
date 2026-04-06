"""Pydantic data models for the GPT Text Generator API."""

from pydantic import BaseModel, field_validator


class TrainingConfig(BaseModel):
    """Hyperparameter configuration for model training."""

    learning_rate: float = 5e-5
    batch_size: int = 8
    num_epochs: int = 3
    warmup_steps: int = 0
    weight_decay: float = 0.0

    @field_validator("learning_rate")
    @classmethod
    def learning_rate_in_range(cls, v: float) -> float:
        if not (1e-6 <= v <= 1e-2):
            raise ValueError("learning_rate must be between 1e-6 and 1e-2")
        return v

    @field_validator("batch_size")
    @classmethod
    def batch_size_in_range(cls, v: int) -> int:
        if not (1 <= v <= 64):
            raise ValueError("batch_size must be between 1 and 64")
        return v

    @field_validator("num_epochs")
    @classmethod
    def num_epochs_in_range(cls, v: int) -> int:
        if not (1 <= v <= 20):
            raise ValueError("num_epochs must be between 1 and 20")
        return v

    @field_validator("warmup_steps")
    @classmethod
    def warmup_steps_in_range(cls, v: int) -> int:
        if not (0 <= v <= 1000):
            raise ValueError("warmup_steps must be between 0 and 1000")
        return v

    @field_validator("weight_decay")
    @classmethod
    def weight_decay_in_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("weight_decay must be between 0 and 1")
        return v


class GenerationParams(BaseModel):
    """Parameters for text generation."""

    prompt: str
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    max_length: int = 100

    @field_validator("temperature")
    @classmethod
    def temperature_in_range(cls, v: float) -> float:
        if not (0.1 <= v <= 2.0):
            raise ValueError("temperature must be between 0.1 and 2.0")
        return v

    @field_validator("top_k")
    @classmethod
    def top_k_in_range(cls, v: int) -> int:
        if not (1 <= v <= 100):
            raise ValueError("top_k must be between 1 and 100")
        return v

    @field_validator("top_p")
    @classmethod
    def top_p_in_range(cls, v: float) -> float:
        if not (0.1 <= v <= 1.0):
            raise ValueError("top_p must be between 0.1 and 1.0")
        return v

    @field_validator("max_length")
    @classmethod
    def max_length_in_range(cls, v: int) -> int:
        if not (10 <= v <= 500):
            raise ValueError("max_length must be between 10 and 500")
        return v


class ModelSummary(BaseModel):
    """Summary of loaded model architecture."""

    name: str
    num_layers: int
    num_parameters: int
    hidden_size: int
    vocab_size: int


class DatasetStats(BaseModel):
    """Statistics about the prepared dataset."""

    train_samples: int
    val_samples: int
    vocab_size: int
    block_size: int


class TrainingMetrics(BaseModel):
    """Real-time training metrics streamed during training."""

    epoch: float
    step: int
    train_loss: float
    val_loss: float | None
    learning_rate: float
    elapsed_seconds: float
    estimated_remaining_seconds: float | None


class EvalResult(BaseModel):
    """Evaluation results after model assessment."""

    perplexity: float
    val_loss: float


class ComparisonResult(BaseModel):
    """Comparison between baseline and trained model."""

    baseline_perplexity: float
    trained_perplexity: float
    improvement_pct: float


class GenerationResult(BaseModel):
    """Result of text generation."""

    text: str
    tokens_generated: int


class CompareGenerationResult(BaseModel):
    """Side-by-side generation comparison."""

    baseline_text: str
    trained_text: str
    prompt: str


class ErrorResponse(BaseModel):
    """Structured error response for API errors."""

    error_code: str
    message: str
    details: str | None = None


class WSMessage(BaseModel):
    """WebSocket message format."""

    type: str  # progress, metrics, commentary, error, state_change, demo_step
    payload: dict
    timestamp: str


class DemoConfig(BaseModel):
    """Configuration for automated demo mode."""

    speed: str = "medium"  # fast, medium, slow
