"""REST endpoints for demo mode control.

POST /api/demo/start   – start automated demo
POST /api/demo/pause   – pause demo
POST /api/demo/resume  – resume demo
POST /api/demo/skip    – skip current demo step
"""

from __future__ import annotations

from fastapi import APIRouter

from backend.app.api.error_handler import AppError
from backend.app.api.websocket import manager as ws_manager
from backend.app.engines.demo_orchestrator import (
    DemoState,
    demo_orchestrator,
)
from backend.app.models.schemas import DemoConfig

router = APIRouter(prefix="/api/demo", tags=["demo"])


async def _ws_callback(msg_type: str, payload: dict) -> None:
    """Bridge between the orchestrator and the WebSocket manager."""
    await ws_manager.broadcast(msg_type, payload)


@router.post("/start")
async def start_demo_endpoint(config: DemoConfig | None = None) -> dict:
    """Start the automated demo walkthrough."""
    if demo_orchestrator.state == DemoState.RUNNING:
        raise AppError(
            error_code="DEMO_ALREADY_RUNNING",
            message="A demo is already in progress.",
        )

    speed = config.speed if config else "medium"
    if speed not in ("fast", "medium", "slow"):
        raise AppError(
            error_code="INVALID_CONFIG",
            message=f"Invalid speed '{speed}'. Must be fast, medium, or slow.",
        )

    await demo_orchestrator.start_demo(speed=speed, ws_callback=_ws_callback)

    return {
        "message": "Demo started",
        **demo_orchestrator.get_status(),
    }


@router.post("/pause")
async def pause_demo_endpoint() -> dict:
    """Pause the running demo."""
    if demo_orchestrator.state != DemoState.RUNNING:
        raise AppError(
            error_code="DEMO_STEP_FAILED",
            message="Demo is not currently running.",
        )

    await demo_orchestrator.pause()
    return {
        "message": "Demo paused",
        **demo_orchestrator.get_status(),
    }


@router.post("/resume")
async def resume_demo_endpoint() -> dict:
    """Resume a paused demo."""
    if demo_orchestrator.state not in (DemoState.PAUSED, DemoState.ERROR):
        raise AppError(
            error_code="DEMO_STEP_FAILED",
            message="Demo is not paused or in error state.",
        )

    await demo_orchestrator.resume()
    return {
        "message": "Demo resumed",
        **demo_orchestrator.get_status(),
    }


@router.post("/skip")
async def skip_demo_step_endpoint() -> dict:
    """Skip the current demo step."""
    if demo_orchestrator.state not in (DemoState.RUNNING, DemoState.PAUSED, DemoState.ERROR):
        raise AppError(
            error_code="DEMO_STEP_FAILED",
            message="Demo is not active.",
        )

    await demo_orchestrator.skip_step()
    return {
        "message": "Step skip requested",
        **demo_orchestrator.get_status(),
    }
