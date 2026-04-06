"""Tests for the GET /api/status endpoint."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from backend.app.main import app
from backend.app.pipeline import pipeline, PipelineStage


@pytest.mark.asyncio
async def test_status_endpoint_returns_idle():
    """GET /api/status returns the current pipeline stage."""
    # Reset pipeline to idle for a clean test
    pipeline._stage = PipelineStage.IDLE
    pipeline._previous_stage = None

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/status")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["stage"] == "idle"
    assert data["previous_stage"] is None


@pytest.mark.asyncio
async def test_status_endpoint_reflects_state_change():
    """GET /api/status reflects the pipeline stage after a transition."""
    pipeline._stage = PipelineStage.IDLE
    pipeline._previous_stage = None

    await pipeline.transition("load_model")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/status")

    assert resp.status_code == 200
    data = resp.json()
    assert data["stage"] == "model_loading"
    assert data["previous_stage"] == "idle"

    # Clean up
    pipeline._stage = PipelineStage.IDLE
    pipeline._previous_stage = None
