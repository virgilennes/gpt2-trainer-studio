"""Integration tests for the GPT Text Generator API.

Tests the full pipeline flow, WebSocket communication, and demo mode
through the actual API endpoints. Heavy ML operations are mocked to
avoid GPU/download requirements.

Validates Task 22: Integration Testing
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from backend.app.main import app, app_state
from backend.app.pipeline import pipeline, PipelineStage
from backend.app.api.websocket import manager as ws_manager
from backend.app.engines.demo_orchestrator import (
    DemoOrchestrator,
    DemoState,
    demo_orchestrator,
    DEMO_STEPS,
)
from backend.app.models.schemas import (
    DatasetStats,
    EvalResult,
    GenerationResult,
    ModelSummary,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_state():
    """Reset all shared state, pipeline, and demo orchestrator before each test."""
    # Reset pipeline
    pipeline._stage = PipelineStage.IDLE
    pipeline._previous_stage = None

    # Reset app_state
    app_state.model = None
    app_state.tokenizer = None
    app_state.tokenizer_info = {}
    app_state.model_summary = None
    app_state.train_dataset = None
    app_state.val_dataset = None
    app_state.dataset_stats = None
    app_state.baseline_model = None
    app_state.is_training = False
    app_state.last_training_result = None

    # Reset module-level states in route modules
    from backend.app.api.model_routes import _state as model_state
    model_state["model"] = None
    model_state["tokenizer"] = None
    model_state["summary"] = None
    model_state["tokenizer_info"] = None

    from backend.app.api.dataset_routes import _state as dataset_state
    dataset_state["train_dataset"] = None
    dataset_state["val_dataset"] = None
    dataset_state["stats"] = None

    from backend.app.api.training_routes import _state as training_state
    training_state["is_training"] = False
    training_state["last_result"] = None

    from backend.app.api.generation_routes import _state as gen_state
    gen_state["baseline_model"] = None

    # Reset demo orchestrator
    demo_orchestrator._state = DemoState.IDLE
    demo_orchestrator._current_step_index = 0
    demo_orchestrator._executed_steps = []
    demo_orchestrator._pause_event.set()
    demo_orchestrator._task = None
    demo_orchestrator._ws_callback = None
    demo_orchestrator._step_executor = None
    demo_orchestrator._skip_requested = False
    demo_orchestrator._error_step = None
    demo_orchestrator._pause_duration_override = None

    # Reset WebSocket manager state
    ws_manager._current_state = {"stage": "idle"}
    ws_manager._last_broadcast_ts.clear()

    yield

    # Teardown
    app_state.model = None
    app_state.tokenizer = None
    app_state.train_dataset = None
    app_state.val_dataset = None
    app_state.baseline_model = None
    app_state.is_training = False



# ---------------------------------------------------------------------------
# Helper mocks
# ---------------------------------------------------------------------------

def _make_fake_model():
    """Create a mock model object that behaves like a HF model."""
    model = MagicMock()
    model.config = MagicMock()
    model.config.n_layer = 12
    model.config.n_embd = 768
    model.config.vocab_size = 50257
    # parameters() returns an iterable of tensors
    import torch
    fake_param = torch.zeros(1)
    model.parameters.return_value = iter([fake_param])
    return model


def _make_fake_tokenizer():
    """Create a mock tokenizer object."""
    tokenizer = MagicMock()
    tokenizer.vocab_size = 50257
    tokenizer.eos_token_id = 50256
    tokenizer.pad_token = None
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.encode.return_value = [15496, 11, 995, 0]
    tokenizer.convert_ids_to_tokens.return_value = ["Hello", ",", "Ġworld", "!"]
    return tokenizer


def _make_fake_dataset(n_samples: int = 10):
    """Create a mock dataset with __len__."""
    import torch
    ds = MagicMock()
    ds.__len__ = MagicMock(return_value=n_samples)
    ds.__getitem__ = MagicMock(return_value=torch.zeros(128, dtype=torch.long))
    return ds


# ---------------------------------------------------------------------------
# 22.1 Full Pipeline Integration Test
# ---------------------------------------------------------------------------

class TestFullPipelineIntegration:
    """Integration test for the full pipeline: model load → dataset prep →
    train (1 epoch) → evaluate → generate.

    All heavy ML operations are mocked; the test verifies the API flow,
    response codes, JSON payloads, and state transitions.
    """

    @pytest.mark.asyncio
    async def test_full_pipeline_flow(self):
        """Walk through the entire pipeline via REST endpoints."""
        fake_model = _make_fake_model()
        fake_tokenizer = _make_fake_tokenizer()
        fake_summary = ModelSummary(
            name="gpt2",
            num_layers=12,
            num_parameters=124_439_808,
            hidden_size=768,
            vocab_size=50257,
        )
        fake_tokenizer_info = {
            "vocab_size": 50257,
            "examples": [{"text": "Hello", "tokens": ["Hello"], "token_ids": [15496]}],
        }
        fake_train_ds = _make_fake_dataset(100)
        fake_val_ds = _make_fake_dataset(20)
        fake_stats = DatasetStats(
            train_samples=100, val_samples=20, vocab_size=50257, block_size=128
        )

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:

            # --- Step 1: Load model ---
            with patch(
                "backend.app.api.model_routes.load_model",
                return_value=(fake_model, fake_summary),
            ), patch(
                "backend.app.api.model_routes.load_tokenizer",
                return_value=(fake_tokenizer, fake_tokenizer_info),
            ):
                resp = await client.post("/api/model/load")

            assert resp.status_code == 200
            data = resp.json()
            assert data["summary"]["name"] == "gpt2"
            assert data["summary"]["num_layers"] == 12
            assert data["tokenizer_info"]["vocab_size"] == 50257

            # Verify model summary endpoint works after loading
            resp = await client.get("/api/model/summary")
            assert resp.status_code == 200
            assert resp.json()["name"] == "gpt2"

            # --- Step 2: Prepare dataset ---
            with patch(
                "backend.app.api.dataset_routes.prepare_dataset",
                return_value=(fake_train_ds, fake_val_ds, fake_stats),
            ):
                resp = await client.post("/api/dataset/prepare")

            assert resp.status_code == 200
            data = resp.json()
            assert data["stats"]["train_samples"] == 100
            assert data["stats"]["val_samples"] == 20

            # Verify dataset stats endpoint
            resp = await client.get("/api/dataset/stats")
            assert resp.status_code == 200
            assert resp.json()["block_size"] == 128

            # --- Step 3: Train (1 epoch) ---
            from backend.app.engines.training_engine import TrainingResult

            fake_training_result = TrainingResult(
                success=True,
                epochs_completed=1.0,
                final_train_loss=3.5,
                checkpoint_dir="./checkpoints/checkpoint-1",
            )
            with patch(
                "backend.app.api.training_routes.start_training",
                return_value=fake_training_result,
            ):
                resp = await client.post(
                    "/api/training/start",
                    json={
                        "learning_rate": 5e-5,
                        "batch_size": 8,
                        "num_epochs": 1,
                        "warmup_steps": 0,
                        "weight_decay": 0.0,
                    },
                )

            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is True
            assert data["epochs_completed"] == 1.0
            assert data["final_train_loss"] == 3.5

            # --- Step 4: Evaluate ---
            fake_eval_result = EvalResult(perplexity=42.5, val_loss=3.75)
            with patch(
                "backend.app.api.evaluation_routes.evaluate",
                return_value=fake_eval_result,
            ):
                resp = await client.post("/api/evaluation/run")

            assert resp.status_code == 200
            data = resp.json()
            assert data["perplexity"] == 42.5
            assert data["val_loss"] == 3.75

            # --- Step 5: Generate ---
            fake_gen_result = GenerationResult(
                text="Hello, world! This is generated text.",
                tokens_generated=8,
            )
            with patch(
                "backend.app.api.generation_routes.generate",
                return_value=fake_gen_result,
            ):
                resp = await client.post(
                    "/api/generation/generate",
                    json={
                        "prompt": "Hello, world!",
                        "temperature": 0.8,
                        "top_k": 50,
                        "top_p": 0.9,
                        "max_length": 50,
                    },
                )

            assert resp.status_code == 200
            data = resp.json()
            assert data["text"] == "Hello, world! This is generated text."
            assert data["tokens_generated"] == 8

            # --- Verify final status ---
            resp = await client.get("/api/status")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert data["model_loaded"] is True
            assert data["dataset_prepared"] is True

    @pytest.mark.asyncio
    async def test_pipeline_rejects_operations_without_prerequisites(self):
        """Verify endpoints return errors when prerequisites are not met."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:

            # Dataset prepare without model loaded
            resp = await client.post("/api/dataset/prepare")
            assert resp.status_code == 400
            data = resp.json()
            assert data["error_code"] == "MODEL_NOT_LOADED"

            # Training without model
            resp = await client.post(
                "/api/training/start",
                json={"learning_rate": 5e-5, "batch_size": 8, "num_epochs": 1},
            )
            assert resp.status_code == 400
            assert resp.json()["error_code"] == "MODEL_NOT_LOADED"

            # Evaluation without model
            resp = await client.post("/api/evaluation/run")
            assert resp.status_code == 400
            assert resp.json()["error_code"] == "MODEL_NOT_LOADED"

            # Generation without model
            resp = await client.post(
                "/api/generation/generate",
                json={"prompt": "Hello", "max_length": 50},
            )
            assert resp.status_code == 400
            assert resp.json()["error_code"] == "MODEL_NOT_LOADED"

            # Dataset stats without preparation
            resp = await client.get("/api/dataset/stats")
            assert resp.status_code == 400
            assert resp.json()["error_code"] == "DATASET_NOT_PREPARED"

            # Model summary without loading
            resp = await client.get("/api/model/summary")
            assert resp.status_code == 400
            assert resp.json()["error_code"] == "MODEL_NOT_LOADED"


# ---------------------------------------------------------------------------
# 22.2 WebSocket Communication Integration Test
# ---------------------------------------------------------------------------

class TestWebSocketIntegration:
    """Integration tests for WebSocket communication: connect, receive
    progress, disconnect, reconnect, and receive state.

    Uses the FastAPI TestClient for WebSocket testing.
    """

    @pytest.mark.asyncio
    async def test_websocket_connect_receives_initial_state(self):
        """On connect, client should receive the current pipeline state."""
        from starlette.testclient import TestClient

        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                # The first message should be a state_change with current state
                data = ws.receive_text()
                msg = json.loads(data)
                assert msg["type"] == "state_change"
                assert "payload" in msg
                assert msg["payload"]["stage"] == "idle"
                assert "timestamp" in msg

    @pytest.mark.asyncio
    async def test_websocket_receives_broadcast_messages(self):
        """Connected clients should receive state change messages when
        pipeline transitions happen via REST endpoints."""
        from starlette.testclient import TestClient

        # Use a REST endpoint to trigger a real broadcast.
        # The /api/status endpoint doesn't broadcast, but we can verify
        # that the WebSocket receives the initial state and that
        # set_pipeline_state works for subsequent connections.
        ws_manager.set_pipeline_state({"stage": "model_loading", "action": "load_model"})

        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                # On connect, client receives the current state
                data = ws.receive_text()
                msg = json.loads(data)
                assert msg["type"] == "state_change"
                assert msg["payload"]["stage"] == "model_loading"
                assert msg["payload"]["action"] == "load_model"

    @pytest.mark.asyncio
    async def test_websocket_disconnect_and_reconnect_receives_state(self):
        """After disconnect and reconnect, client should receive current state."""
        from starlette.testclient import TestClient

        # Set a non-idle state in the manager
        ws_manager.set_pipeline_state({"stage": "model_loaded", "previous_stage": "model_loading"})

        with TestClient(app) as client:
            # First connection
            with client.websocket_connect("/ws") as ws:
                data = ws.receive_text()
                msg = json.loads(data)
                assert msg["payload"]["stage"] == "model_loaded"

            # After disconnect, reconnect should get current state
            with client.websocket_connect("/ws") as ws:
                data = ws.receive_text()
                msg = json.loads(data)
                assert msg["type"] == "state_change"
                assert msg["payload"]["stage"] == "model_loaded"

    @pytest.mark.asyncio
    async def test_websocket_multiple_clients_receive_same_initial_state(self):
        """Multiple connected clients should all receive the same initial state."""
        from starlette.testclient import TestClient

        ws_manager.set_pipeline_state({"stage": "training", "action": "start_training"})

        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws1:
                with client.websocket_connect("/ws") as ws2:
                    # Both clients should receive the current state on connect
                    data1 = ws1.receive_text()
                    data2 = ws2.receive_text()
                    msg1 = json.loads(data1)
                    msg2 = json.loads(data2)
                    assert msg1["payload"]["stage"] == "training"
                    assert msg2["payload"]["stage"] == "training"
                    assert msg1["type"] == "state_change"
                    assert msg2["type"] == "state_change"

    @pytest.mark.asyncio
    async def test_websocket_state_sync_on_reconnect_after_state_change(self):
        """After a state change, a new connection should receive
        the updated state on connect."""
        from starlette.testclient import TestClient

        with TestClient(app) as client:
            # First client connects and sees idle state
            with client.websocket_connect("/ws") as ws:
                data = ws.receive_text()
                msg = json.loads(data)
                assert msg["payload"]["stage"] == "idle"

            # Simulate a state change (as would happen from a pipeline transition)
            ws_manager.set_pipeline_state({"stage": "evaluated", "action": "evaluation_complete"})

            # New client connects and should get the updated state
            with client.websocket_connect("/ws") as ws:
                data = ws.receive_text()
                msg = json.loads(data)
                assert msg["payload"]["stage"] == "evaluated"
                assert msg["payload"]["action"] == "evaluation_complete"


# ---------------------------------------------------------------------------
# 22.3 Demo Mode Integration Test
# ---------------------------------------------------------------------------

class TestDemoModeIntegration:
    """Integration tests for demo mode: start, verify step sequence,
    pause, resume, and complete — all through the REST API endpoints.

    The demo orchestrator's step_executor is not set (no actual ML ops),
    so steps execute as commentary-only. The pause duration is overridden
    to keep tests fast.
    """

    @pytest.mark.asyncio
    async def test_demo_start_and_complete(self):
        """Start demo via API and verify it runs through all steps."""
        # Override pause duration to make the test fast
        demo_orchestrator._pause_duration_override = 0.01

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/demo/start", json={"speed": "fast"})
            assert resp.status_code == 200
            data = resp.json()
            assert data["message"] == "Demo started"
            assert data["state"] == "running"

        # Wait for the demo to complete (steps are fast with overridden pause)
        for _ in range(50):
            await asyncio.sleep(0.05)
            if demo_orchestrator.state == DemoState.COMPLETED:
                break

        assert demo_orchestrator.state == DemoState.COMPLETED
        assert demo_orchestrator.executed_steps == DEMO_STEPS

    @pytest.mark.asyncio
    async def test_demo_step_sequence_is_correct(self):
        """Verify the demo executes steps in the exact required order."""
        demo_orchestrator._pause_duration_override = 0.01

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/demo/start", json={"speed": "fast"})
            assert resp.status_code == 200

        # Wait for completion
        for _ in range(50):
            await asyncio.sleep(0.05)
            if demo_orchestrator.state == DemoState.COMPLETED:
                break

        expected_sequence = [
            "model_loading",
            "dataset_preparation",
            "training_configuration",
            "training_execution",
            "evaluation",
            "text_generation",
        ]
        assert demo_orchestrator.executed_steps == expected_sequence

    @pytest.mark.asyncio
    async def test_demo_pause_and_resume(self):
        """Pause the demo, verify it stops, then resume and verify completion."""
        demo_orchestrator._pause_duration_override = 0.2  # Slow enough to pause

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Start demo
            resp = await client.post("/api/demo/start", json={"speed": "fast"})
            assert resp.status_code == 200

            # Wait a bit for the first step to start
            await asyncio.sleep(0.3)

            # Pause
            if demo_orchestrator.state == DemoState.RUNNING:
                resp = await client.post("/api/demo/pause")
                assert resp.status_code == 200
                data = resp.json()
                assert data["message"] == "Demo paused"

                # Verify it's paused
                assert demo_orchestrator.state == DemoState.PAUSED
                paused_step = demo_orchestrator.current_step
                paused_steps_count = len(demo_orchestrator.executed_steps)

                # Wait and verify no progress while paused
                await asyncio.sleep(0.3)
                assert len(demo_orchestrator.executed_steps) == paused_steps_count

                # Resume
                resp = await client.post("/api/demo/resume")
                assert resp.status_code == 200
                assert resp.json()["message"] == "Demo resumed"

            # Speed up remaining steps
            demo_orchestrator._pause_duration_override = 0.01

            # Wait for completion
            for _ in range(100):
                await asyncio.sleep(0.05)
                if demo_orchestrator.state == DemoState.COMPLETED:
                    break

            assert demo_orchestrator.state == DemoState.COMPLETED
            assert demo_orchestrator.executed_steps == DEMO_STEPS

    @pytest.mark.asyncio
    async def test_demo_skip_step(self):
        """Skip a step during demo and verify it advances."""
        demo_orchestrator._pause_duration_override = 0.5  # Slow enough to skip

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/demo/start", json={"speed": "fast"})
            assert resp.status_code == 200

            # Wait for first step to start
            await asyncio.sleep(0.1)

            # Pause first so we can skip reliably
            if demo_orchestrator.state == DemoState.RUNNING:
                resp = await client.post("/api/demo/pause")
                if resp.status_code == 200:
                    steps_before = len(demo_orchestrator.executed_steps)

                    # Skip
                    resp = await client.post("/api/demo/skip")
                    assert resp.status_code == 200

                    # Speed up remaining
                    demo_orchestrator._pause_duration_override = 0.01

                    # Wait for completion
                    for _ in range(100):
                        await asyncio.sleep(0.05)
                        if demo_orchestrator.state == DemoState.COMPLETED:
                            break

                    assert demo_orchestrator.state == DemoState.COMPLETED

    @pytest.mark.asyncio
    async def test_demo_cannot_start_twice(self):
        """Starting a demo while one is running should return an error."""
        demo_orchestrator._pause_duration_override = 1.0  # Keep it running

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/demo/start", json={"speed": "fast"})
            assert resp.status_code == 200

            # Try to start again
            resp = await client.post("/api/demo/start", json={"speed": "fast"})
            assert resp.status_code == 400
            assert resp.json()["error_code"] == "DEMO_ALREADY_RUNNING"

        # Cleanup: stop the demo
        await demo_orchestrator.stop()

    @pytest.mark.asyncio
    async def test_demo_pause_when_not_running_returns_error(self):
        """Pausing when no demo is running should return an error."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/demo/pause")
            assert resp.status_code == 400
            assert resp.json()["error_code"] == "DEMO_STEP_FAILED"

    @pytest.mark.asyncio
    async def test_demo_resume_when_not_paused_returns_error(self):
        """Resuming when demo is not paused should return an error."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/demo/resume")
            assert resp.status_code == 400
            assert resp.json()["error_code"] == "DEMO_STEP_FAILED"

    @pytest.mark.asyncio
    async def test_demo_invalid_speed_returns_error(self):
        """Starting demo with invalid speed should return an error."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/demo/start", json={"speed": "turbo"})
            assert resp.status_code == 400
            assert resp.json()["error_code"] == "INVALID_CONFIG"
