"""WebSocket endpoint for real-time client communication.

Provides a ``/ws`` endpoint that accepts WebSocket connections, manages
client sessions via a :class:`ConnectionManager`, and supports broadcasting
typed messages to all connected clients.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.app.models.schemas import WSMessage

router = APIRouter(tags=["websocket"])


class ConnectionManager:
    """Track active WebSocket connections and broadcast messages."""

    def __init__(self) -> None:
        self._connections: list[WebSocket] = []
        self._lock = asyncio.Lock()
        # Tracks the last broadcast timestamp per message type so callers
        # can enforce a maximum update interval.
        self._last_broadcast_ts: dict[str, float] = {}
        # Minimum interval (seconds) between progress broadcasts.
        self.min_broadcast_interval: float = 2.0
        # Snapshot of the current pipeline state sent to newly-connected
        # clients so they can sync immediately.
        self._current_state: dict[str, Any] = {"stage": "idle"}

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self, ws: WebSocket) -> None:
        """Accept *ws* and register it as an active connection."""
        await ws.accept()
        async with self._lock:
            self._connections.append(ws)
        # Send current state so the client is in sync right away.
        await self._send_state_sync(ws)

    async def disconnect(self, ws: WebSocket) -> None:
        """Remove *ws* from the active connection list."""
        async with self._lock:
            if ws in self._connections:
                self._connections.remove(ws)

    @property
    def active_connections(self) -> list[WebSocket]:
        """Return a snapshot of currently active connections."""
        return list(self._connections)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def set_pipeline_state(self, state: dict[str, Any]) -> None:
        """Update the cached pipeline state used for reconnection sync."""
        self._current_state = state

    def get_pipeline_state(self) -> dict[str, Any]:
        """Return the current cached pipeline state."""
        return dict(self._current_state)

    # ------------------------------------------------------------------
    # Messaging helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_message(msg_type: str, payload: dict[str, Any]) -> WSMessage:
        return WSMessage(
            type=msg_type,
            payload=payload,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    async def _send_json(self, ws: WebSocket, message: WSMessage) -> None:
        """Send a serialised WSMessage to a single client."""
        try:
            await ws.send_text(message.model_dump_json())
        except Exception:
            await self.disconnect(ws)

    async def _send_state_sync(self, ws: WebSocket) -> None:
        """Send the current pipeline state to a single client."""
        msg = self._build_message("state_change", self._current_state)
        await self._send_json(ws, msg)

    # ------------------------------------------------------------------
    # Broadcasting
    # ------------------------------------------------------------------

    async def broadcast(self, msg_type: str, payload: dict[str, Any]) -> None:
        """Send a message to **all** connected clients.

        For ``progress`` messages the manager enforces a minimum interval
        of :attr:`min_broadcast_interval` seconds between consecutive
        broadcasts.  If a broadcast is requested too soon the call is
        silently skipped so that the WebSocket channel is not flooded.
        """
        now = time.monotonic()

        if msg_type == "progress":
            last = self._last_broadcast_ts.get("progress", 0.0)
            if now - last < self.min_broadcast_interval:
                return  # throttle – too soon
            self._last_broadcast_ts["progress"] = now

        message = self._build_message(msg_type, payload)

        # Update cached state when broadcasting state changes.
        if msg_type == "state_change":
            self._current_state = payload

        async with self._lock:
            targets = list(self._connections)

        await asyncio.gather(
            *(self._send_json(ws, message) for ws in targets),
            return_exceptions=True,
        )

    async def broadcast_progress(self, payload: dict[str, Any]) -> None:
        """Convenience wrapper for progress broadcasts with rate-limiting."""
        await self.broadcast("progress", payload)


# Module-level singleton so other modules can import and use it.
manager = ConnectionManager()


@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    """Accept a WebSocket connection and keep it alive until the client
    disconnects.  Incoming text frames are ignored for now – all
    communication is server → client via :meth:`ConnectionManager.broadcast`.
    """
    await manager.connect(ws)
    try:
        while True:
            # Keep the connection alive by reading (and discarding) frames.
            await ws.receive_text()
    except WebSocketDisconnect:
        await manager.disconnect(ws)
    except Exception:
        await manager.disconnect(ws)
