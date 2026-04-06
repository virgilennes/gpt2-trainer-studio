"""Unit tests for model_loader and model REST endpoints."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.app.api.error_handler import AppError
from backend.app.engines.model_loader import load_model, load_tokenizer
from backend.app.main import app


client = TestClient(app)


# ------------------------------------------------------------------
# load_model tests
# ------------------------------------------------------------------


class TestLoadModel:
    """Tests for the load_model function."""

    @patch("backend.app.engines.model_loader.AutoModelForCausalLM")
    def test_returns_model_and_summary(self, mock_auto):
        """load_model returns a (model, ModelSummary) tuple with correct fields."""
        mock_model = MagicMock()
        mock_model.config.n_layer = 12
        mock_model.config.n_embd = 768
        mock_model.config.vocab_size = 50257

        # Simulate two parameters with known sizes
        p1 = MagicMock()
        p1.numel.return_value = 1000
        p2 = MagicMock()
        p2.numel.return_value = 2000
        mock_model.parameters.return_value = [p1, p2]

        mock_auto.from_pretrained.return_value = mock_model

        model, summary = load_model()

        assert model is mock_model
        assert summary.name == "gpt2"
        assert summary.num_layers == 12
        assert summary.hidden_size == 768
        assert summary.vocab_size == 50257
        assert summary.num_parameters == 3000

    @patch("backend.app.engines.model_loader.AutoModelForCausalLM")
    def test_raises_app_error_on_failure(self, mock_auto):
        """load_model wraps download failures in AppError."""
        mock_auto.from_pretrained.side_effect = OSError("network error")

        with pytest.raises(AppError) as exc_info:
            load_model()

        assert exc_info.value.error_code == "MODEL_LOAD_FAILED"
        assert "network error" in (exc_info.value.details or "")


# ------------------------------------------------------------------
# load_tokenizer tests
# ------------------------------------------------------------------


class TestLoadTokenizer:
    """Tests for the load_tokenizer function."""

    @patch("backend.app.engines.model_loader.AutoTokenizer")
    def test_returns_tokenizer_and_info(self, mock_auto_tok):
        """load_tokenizer returns (tokenizer, info_dict) with vocab_size and examples."""
        mock_tok = MagicMock()
        mock_tok.vocab_size = 50257
        mock_tok.encode.side_effect = lambda t: list(range(len(t.split())))
        mock_tok.convert_ids_to_tokens.side_effect = lambda ids: [f"tok_{i}" for i in ids]
        mock_auto_tok.from_pretrained.return_value = mock_tok

        tokenizer, info = load_tokenizer()

        assert tokenizer is mock_tok
        assert info["vocab_size"] == 50257
        assert isinstance(info["examples"], list)
        assert len(info["examples"]) == 3
        for ex in info["examples"]:
            assert "text" in ex
            assert "tokens" in ex
            assert "token_ids" in ex

    @patch("backend.app.engines.model_loader.AutoTokenizer")
    def test_raises_app_error_on_failure(self, mock_auto_tok):
        """load_tokenizer wraps failures in AppError."""
        mock_auto_tok.from_pretrained.side_effect = OSError("download failed")

        with pytest.raises(AppError) as exc_info:
            load_tokenizer()

        assert exc_info.value.error_code == "MODEL_LOAD_FAILED"


# ------------------------------------------------------------------
# REST endpoint tests
# ------------------------------------------------------------------


class TestModelEndpoints:
    """Tests for POST /api/model/load and GET /api/model/summary."""

    def test_summary_returns_error_when_not_loaded(self):
        """GET /api/model/summary returns 400 when model is not loaded."""
        # Reset state
        from backend.app.api import model_routes
        model_routes._state["summary"] = None

        response = client.get("/api/model/summary")
        assert response.status_code == 400
        body = response.json()
        assert body["error_code"] == "MODEL_NOT_LOADED"

    @patch("backend.app.api.model_routes.load_tokenizer")
    @patch("backend.app.api.model_routes.load_model")
    def test_load_endpoint_populates_state(self, mock_load_model, mock_load_tok):
        """POST /api/model/load stores model/tokenizer and returns summary."""
        from backend.app.models.schemas import ModelSummary

        summary = ModelSummary(
            name="gpt2",
            num_layers=12,
            num_parameters=124_000_000,
            hidden_size=768,
            vocab_size=50257,
        )
        mock_load_model.return_value = (MagicMock(), summary)
        mock_load_tok.return_value = (
            MagicMock(),
            {"vocab_size": 50257, "examples": []},
        )

        response = client.post("/api/model/load")
        assert response.status_code == 200
        body = response.json()
        assert body["summary"]["name"] == "gpt2"
        assert body["summary"]["num_layers"] == 12

    @patch("backend.app.api.model_routes.load_tokenizer")
    @patch("backend.app.api.model_routes.load_model")
    def test_summary_available_after_load(self, mock_load_model, mock_load_tok):
        """GET /api/model/summary returns data after a successful load."""
        from backend.app.models.schemas import ModelSummary

        summary = ModelSummary(
            name="gpt2",
            num_layers=12,
            num_parameters=124_000_000,
            hidden_size=768,
            vocab_size=50257,
        )
        mock_load_model.return_value = (MagicMock(), summary)
        mock_load_tok.return_value = (
            MagicMock(),
            {"vocab_size": 50257, "examples": []},
        )

        client.post("/api/model/load")
        response = client.get("/api/model/summary")
        assert response.status_code == 200
        body = response.json()
        assert body["name"] == "gpt2"
        assert body["num_layers"] == 12
        assert body["hidden_size"] == 768
        assert body["vocab_size"] == 50257

    @patch("backend.app.api.model_routes.load_model")
    def test_load_endpoint_returns_error_on_failure(self, mock_load_model):
        """POST /api/model/load returns structured error on failure."""
        mock_load_model.side_effect = AppError(
            error_code="MODEL_LOAD_FAILED",
            message="Failed to load model",
            details="timeout",
        )

        response = client.post("/api/model/load")
        assert response.status_code == 400
        body = response.json()
        assert body["error_code"] == "MODEL_LOAD_FAILED"
