"""FastAPI application entry point with CORS middleware and lifespan handler.

Assembles all routes, WebSocket endpoint, CORS, shared application state,
and startup/shutdown lifecycle management.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.error_handler import register_error_handlers
from backend.app.api.model_routes import router as model_router
from backend.app.api.dataset_routes import router as dataset_router
from backend.app.api.training_routes import router as training_router
from backend.app.api.evaluation_routes import router as evaluation_router
from backend.app.api.generation_routes import router as generation_router
from backend.app.api.demo_routes import router as demo_router
from backend.app.api.websocket import router as ws_router, manager as ws_manager
from backend.app.engines.demo_orchestrator import demo_orchestrator, DemoState
from backend.app.engines.training_engine import request_stop as request_training_stop
from backend.app.pipeline import pipeline

logger = logging.getLogger(__name__)


@dataclass
class AppState:
    """Centralised shared state accessible across route handlers.

    Holds references to the loaded model, tokenizer, prepared datasets,
    and pipeline state so that engine instances can share resources
    without relying solely on scattered module-level dicts.
    """

    model: Any = None
    tokenizer: Any = None
    tokenizer_info: dict = field(default_factory=dict)
    model_summary: Any = None
    train_dataset: Any = None
    val_dataset: Any = None
    dataset_stats: Any = None
    baseline_model: Any = None
    is_training: bool = False
    last_training_result: Any = None


# Module-level shared state instance
app_state = AppState()


def get_app_state() -> AppState:
    """Return the shared application state singleton."""
    return app_state


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application startup and shutdown lifecycle.

    Startup:
        - Initialise shared state
        - Log startup message

    Shutdown:
        - Stop any running training
        - Stop any running demo
        - Close all WebSocket connections
        - Clean up shared state references
    """
    # --- Startup ---
    logger.info("GPT Text Generator API starting up")
    logger.info("Pipeline state: %s", pipeline.stage.value)
    yield
    # --- Shutdown ---
    logger.info("GPT Text Generator API shutting down — cleaning up resources")
    await _shutdown_cleanup()
    logger.info("Shutdown complete")


async def _shutdown_cleanup() -> None:
    """Perform graceful cleanup of all running resources."""
    # 1. Stop training if in progress
    from backend.app.api.training_routes import get_state as get_training_state

    training_state = get_training_state()
    if training_state.get("is_training"):
        logger.info("Stopping active training run…")
        request_training_stop()

    # 2. Stop demo if running
    if demo_orchestrator.state in (DemoState.RUNNING, DemoState.PAUSED):
        logger.info("Stopping active demo…")
        await demo_orchestrator.stop()

    # 3. Close all WebSocket connections
    active_connections = ws_manager.active_connections
    if active_connections:
        logger.info("Closing %d WebSocket connection(s)…", len(active_connections))
        for ws in active_connections:
            try:
                await ws.close()
            except Exception:
                pass
        # Clear the connection list
        async with ws_manager._lock:
            ws_manager._connections.clear()

    # 4. Clear shared state references to allow GC
    app_state.model = None
    app_state.tokenizer = None
    app_state.baseline_model = None
    app_state.train_dataset = None
    app_state.val_dataset = None
    app_state.model_summary = None
    app_state.dataset_stats = None
    app_state.is_training = False
    app_state.last_training_result = None


app = FastAPI(
    title="GPT Text Generator API",
    description="Backend API for GPT-2 text generator training application",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware for frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register structured error handlers
register_error_handlers(app)

# Register route modules
app.include_router(model_router)
app.include_router(dataset_router)
app.include_router(training_router)
app.include_router(evaluation_router)
app.include_router(generation_router)
app.include_router(demo_router)
app.include_router(ws_router)


@app.get("/api/status")
async def get_status():
    """Return current pipeline state and shared application state summary."""
    return {
        "status": "ok",
        **pipeline.get_status(),
        "model_loaded": app_state.model is not None,
        "dataset_prepared": app_state.train_dataset is not None,
        "is_training": app_state.is_training,
    }
