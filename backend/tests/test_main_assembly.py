"""Tests for FastAPI application assembly, lifespan, and shared state wiring.

Validates Task 21: lifespan handler, shared AppState, and shutdown cleanup.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from backend.app.main import app, app_state, AppState, _shutdown_cleanup
from backend.app.pipeline import pipeline, PipelineStage
from backend.app.api.websocket import manager as ws_manager
from backend.app.engines.demo_orchestrator import demo_orchestrator, DemoState


@pytest.fixture(autouse=True)
def _reset_state():
    """Reset shared state and pipeline before each test."""
    pipeline._stage = PipelineStage.IDLE
    pipeline._previous_stage = None
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
    yield
    # Cleanup after test
    app_state.model = None
    app_state.tokenizer = None
    app_state.train_dataset = None
    app_state.val_dataset = None
    app_state.baseline_model = None
    app_state.is_training = False


class TestAppState:
    """Tests for the centralised AppState dataclass."""

    def test_default_state_is_empty(self):
        state = AppState()
        assert state.model is None
        assert state.tokenizer is None
        assert state.train_dataset is None
        assert state.val_dataset is None
        assert state.baseline_model is None
        assert state.is_training is False
        assert state.last_training_result is None
        assert state.model_summary is None
        assert state.dataset_stats is None
        assert state.tokenizer_info == {}

    def test_state_fields_are_mutable(self):
        state = AppState()
        state.model = "fake_model"
        state.tokenizer = "fake_tokenizer"
        state.is_training = True
        assert state.model == "fake_model"
        assert state.tokenizer == "fake_tokenizer"
        assert state.is_training is True


class TestStatusEndpointWithSharedState:
    """Tests for the enhanced /api/status endpoint."""

    @pytest.mark.asyncio
    async def test_status_includes_shared_state_fields(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/status")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["stage"] == "idle"
        assert data["model_loaded"] is False
        assert data["dataset_prepared"] is False
        assert data["is_training"] is False

    @pytest.mark.asyncio
    async def test_status_reflects_model_loaded(self):
        app_state.model = "fake_model"
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/status")

        data = resp.json()
        assert data["model_loaded"] is True
        assert data["dataset_prepared"] is False

    @pytest.mark.asyncio
    async def test_status_reflects_dataset_prepared(self):
        app_state.train_dataset = "fake_dataset"
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/status")

        data = resp.json()
        assert data["dataset_prepared"] is True

    @pytest.mark.asyncio
    async def test_status_reflects_training_flag(self):
        app_state.is_training = True
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/status")

        data = resp.json()
        assert data["is_training"] is True


class TestShutdownCleanup:
    """Tests for the shutdown cleanup logic."""

    @pytest.mark.asyncio
    async def test_shutdown_clears_shared_state(self):
        app_state.model = "fake_model"
        app_state.tokenizer = "fake_tokenizer"
        app_state.train_dataset = "fake_train"
        app_state.val_dataset = "fake_val"
        app_state.baseline_model = "fake_baseline"
        app_state.is_training = False

        await _shutdown_cleanup()

        assert app_state.model is None
        assert app_state.tokenizer is None
        assert app_state.train_dataset is None
        assert app_state.val_dataset is None
        assert app_state.baseline_model is None
        assert app_state.model_summary is None
        assert app_state.dataset_stats is None
        assert app_state.is_training is False
        assert app_state.last_training_result is None

    @pytest.mark.asyncio
    async def test_shutdown_stops_demo_if_running(self):
        """If demo is running, shutdown should stop it."""
        # Manually set demo state to running
        demo_orchestrator._state = DemoState.RUNNING
        demo_orchestrator._pause_event.set()

        await _shutdown_cleanup()

        assert demo_orchestrator.state == DemoState.IDLE


class TestAppAssembly:
    """Tests that the app is properly assembled with all routes."""

    @pytest.mark.asyncio
    async def test_all_route_prefixes_registered(self):
        """Verify all expected route prefixes are registered on the app."""
        route_paths = [r.path for r in app.routes if hasattr(r, "path")]

        # Check key endpoints exist
        assert "/api/status" in route_paths
        assert "/ws" in route_paths

        # Check route prefixes by looking for at least one route per module
        path_set = set(route_paths)
        assert any(p.startswith("/api/model") for p in path_set)
        assert any(p.startswith("/api/dataset") for p in path_set)
        assert any(p.startswith("/api/training") for p in path_set)
        assert any(p.startswith("/api/evaluation") for p in path_set)
        assert any(p.startswith("/api/generation") for p in path_set)
        assert any(p.startswith("/api/demo") for p in path_set)

    def test_cors_middleware_configured(self):
        """Verify CORS middleware is present."""
        middleware_classes = [
            type(m).__name__
            for m in getattr(app, "user_middleware", [])
        ]
        # FastAPI stores user middleware as Middleware objects
        assert any("CORS" in str(m) for m in app.user_middleware)

    def test_app_has_lifespan(self):
        """Verify the app has a lifespan handler configured."""
        # The lifespan is set on the router
        assert app.router.lifespan_context is not None
